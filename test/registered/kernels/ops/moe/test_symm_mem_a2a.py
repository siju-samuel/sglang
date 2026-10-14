"""Numerics of the symmetric-memory MoE all-to-all kernels.

The ranks are simulated on one device: the test plays the transport, copying
each rank's send staging into the receivers' staging, so the kernels' layout
contract is checked without symmetric memory.
"""

import math
import sys

import pytest
import torch

from sglang.kernels.ops.moe.symm_mem_a2a import (
    symm_mem_a2a_combine_round,
    symm_mem_a2a_pack,
    symm_mem_a2a_unpack,
)
from sglang.srt.utils import get_device, is_cuda, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(
    not (is_cuda() or is_xpu()), reason="needs a CUDA or XPU device for Triton"
)


def _reference_slot(topk_ids: torch.Tensor, num_local_experts: int, world_size: int):
    slot = torch.full((topk_ids.shape[0], world_size), -1, dtype=torch.int32)
    next_row = [0] * world_size
    for token, ids in enumerate(topk_ids.tolist()):
        for peer in sorted({e // num_local_experts for e in ids if e >= 0}):
            slot[token, peer] = next_row[peer]
            next_row[peer] += 1
    return slot, torch.tensor(next_row, dtype=torch.int32)


def _make_rank_inputs(
    *, num_tokens, hidden_size, topk, num_experts, num_local_experts, dtype, seed
):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(num_tokens, hidden_size, generator=gen).to(dtype)
    ids = torch.stack(
        [torch.randperm(num_experts, generator=gen)[:topk] for _ in range(num_tokens)]
        or [torch.empty(topk, dtype=torch.int64)]
    )[:num_tokens].to(torch.int32)
    if num_tokens > 1:
        ids[-1] = -1  # a padding row that goes nowhere
    weights = torch.rand(num_tokens, topk, generator=gen)
    slot, send_counts = _reference_slot(
        ids, num_local_experts, num_experts // num_local_experts
    )
    return hidden, ids, weights, slot, send_counts


@pytest.mark.parametrize(
    "world_size,tokens_per_rank,hidden_size,round_tokens",
    [
        (2, [7, 5], 256, 64),  # one round
        (4, [9, 0, 13, 4], 320, 3),  # many rounds, an idle rank, a masked hidden tail
        (8, [6, 1, 2, 3, 4, 5, 6, 7], 2304, 2),  # several hidden blocks
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dispatch_and_combine_round_trip(
    world_size, tokens_per_rank, hidden_size, round_tokens, dtype
):
    device = get_device()
    topk, num_local_experts = 4, 2
    num_experts = world_size * num_local_experts
    ranks = [
        [
            t.to(device)
            for t in _make_rank_inputs(
                num_tokens=n,
                hidden_size=hidden_size,
                topk=topk,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                dtype=dtype,
                seed=rank,
            )
        ]
        for rank, n in enumerate(tokens_per_rank)
    ]
    counts = torch.stack([r[4] for r in ranks])  # [src, dst]
    counts_list = counts.tolist()
    num_rounds = math.ceil(counts.max().item() / round_tokens)

    def send_rows(src, dst, round_start):
        return max(0, min(round_tokens, counts_list[src][dst] - round_start))

    # Dispatch.
    recv_total = counts.sum(0).tolist()
    outs = [
        (
            torch.empty(recv_total[j], hidden_size, dtype=dtype, device=device),
            torch.empty(recv_total[j], topk, dtype=torch.int32, device=device),
            torch.empty(recv_total[j], topk, dtype=torch.float32, device=device),
        )
        for j in range(world_size)
    ]
    send = [
        (
            torch.zeros(
                world_size, round_tokens, hidden_size, dtype=dtype, device=device
            ),
            torch.zeros(
                world_size, round_tokens, 2 * topk, dtype=torch.int32, device=device
            ),
        )
        for _ in range(world_size)
    ]
    for k in range(num_rounds):
        round_start = k * round_tokens
        for src, (hidden, ids, weights, slot, _) in enumerate(ranks):
            symm_mem_a2a_pack(
                hidden_states=hidden,
                topk_ids=ids,
                topk_weights=weights,
                slot=slot,
                send_hidden=send[src][0],
                send_meta=send[src][1],
                round_start=round_start,
            )
        for dst in range(world_size):
            symm_mem_a2a_unpack(
                recv_hidden=torch.stack(
                    [send[src][0][dst] for src in range(world_size)]
                ),
                recv_meta=torch.stack([send[src][1][dst] for src in range(world_size)]),
                counts=counts,
                out_hidden=outs[dst][0],
                out_topk_ids=outs[dst][1],
                out_topk_weights=outs[dst][2],
                rank=dst,
                num_local_experts=num_local_experts,
                round_start=round_start,
            )

    for dst in range(world_size):
        rows = [r[3][:, dst] >= 0 for r in ranks]
        exp_hidden = torch.cat([r[0][m] for r, m in zip(ranks, rows)])
        exp_ids = torch.cat([r[1][m] for r, m in zip(ranks, rows)])
        exp_ids = torch.where(
            exp_ids // num_local_experts == dst,
            exp_ids - dst * num_local_experts,
            -1,
        )
        exp_weights = torch.cat([r[2][m] for r, m in zip(ranks, rows)])
        torch.testing.assert_close(outs[dst][0], exp_hidden, atol=0, rtol=0)
        torch.testing.assert_close(
            outs[dst][1], exp_ids.to(torch.int32), atol=0, rtol=0
        )
        torch.testing.assert_close(outs[dst][2], exp_weights, atol=0, rtol=0)

    # Combine: rank j's "expert" scales its rows by j + 1.
    expert_out = [(outs[j][0].float() * (j + 1)).to(dtype) for j in range(world_size)]
    recv_offset = [
        [sum(counts_list[s][j] for s in range(src)) for j in range(world_size)]
        for src in range(world_size)
    ]
    results = []
    for src, (hidden, _, _, slot, _) in enumerate(ranks):
        out = torch.empty_like(hidden)
        acc = torch.empty(hidden.shape, dtype=torch.float32, device=device)
        for k in range(num_rounds):
            round_start = k * round_tokens
            recv = torch.zeros(
                world_size, round_tokens, hidden_size, dtype=dtype, device=device
            )
            for j in range(world_size):
                n = send_rows(src, j, round_start)
                start = recv_offset[src][j] + round_start
                recv[j, :n] = expert_out[j][start : start + n]
            symm_mem_a2a_combine_round(
                recv_hidden=recv,
                slot=slot,
                acc=acc,
                out=out,
                round_start=round_start,
                first_round=k == 0,
                last_round=k == num_rounds - 1,
            )
        results.append(out)

    for src, (hidden, _, _, slot, _) in enumerate(ranks):
        expected = torch.zeros(hidden.shape, dtype=torch.float32, device=device)
        for j in range(world_size):
            sent = (slot[:, j] >= 0).unsqueeze(1)
            expected += torch.where(
                sent, (hidden.float() * (j + 1)).to(dtype).float(), 0
            )
        if hidden.shape[0] == 0:
            continue
        # One round sums in ascending rank order, exactly like the reference.
        tol = 0 if num_rounds == 1 else 1e-2
        torch.testing.assert_close(results[src], expected.to(dtype), atol=tol, rtol=tol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
