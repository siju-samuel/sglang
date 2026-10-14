"""Multi-rank correctness of the symm_mem MoE all-to-all dispatcher.

The cases live in one test function on purpose: every rank must issue the same
sequence of barriers, and a shuffled order would pair ranks on different
batches. Every rank rebuilds all ranks' inputs from seeds, so it can check its
own outputs against a local reference.

Usage::

    # Runs itself under torchrun on 2 devices (pass --world-size N for more):
    python test/registered/ep/test_symm_mem_a2a_dispatcher.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.utils import get_device, get_device_module
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci

register_cuda_ci(est_time=120, stage="nightly", runner_config="2-gpu-large")
register_xpu_ci(est_time=180, suite="nightly-xpu-kernel-main-2-gpu", nightly=True)

_HIDDEN_SIZE = 1152  # two hidden blocks, the second one masked
_TOPK = 4
_NUM_LOCAL_EXPERTS = 3
_DTYPE = torch.bfloat16


class _Group(NamedTuple):
    """The GroupCoordinator fields SymmMemDispatcher reads."""

    rank_in_group: int
    world_size: int
    cpu_group: dist.ProcessGroup
    device: torch.device


def _make_dispatcher(group: _Group):
    from sglang.srt.layers.moe.token_dispatcher.symm_mem import SymmMemDispatcher

    return SymmMemDispatcher(
        group=group,
        router_topk=_TOPK,
        num_experts=_NUM_LOCAL_EXPERTS * group.world_size,
        num_local_experts=_NUM_LOCAL_EXPERTS,
        hidden_size=_HIDDEN_SIZE,
        params_dtype=_DTYPE,
    )


def _rank_inputs(*, rank, num_tokens, world_size, seed, routed=True):
    gen = torch.Generator().manual_seed(seed * 1000 + rank)
    num_experts = _NUM_LOCAL_EXPERTS * world_size
    hidden = torch.randn(num_tokens, _HIDDEN_SIZE, generator=gen).to(_DTYPE)
    ids = torch.full((num_tokens, _TOPK), -1, dtype=torch.int32)
    if routed:
        for t in range(num_tokens):
            ids[t] = torch.randperm(num_experts, generator=gen)[:_TOPK].to(torch.int32)
        if num_tokens > 1:
            ids[-1, 1:] = -1  # a token with a single routed expert
    weights = torch.rand(num_tokens, _TOPK, generator=gen)
    return hidden, ids, weights


def _fake_experts(hidden, local_ids, weights, rank):
    """Expert e scales its input by e + 1; rows sum their local experts in fp32."""
    global_ids = torch.where(local_ids >= 0, local_ids + rank * _NUM_LOCAL_EXPERTS, -1)
    scale = torch.where(local_ids >= 0, weights * (global_ids + 1).float(), 0).sum(1)
    return (hidden.float() * scale.unsqueeze(1)).to(_DTYPE)


def _reference_combine(hidden, ids, weights, world_size):
    """Per-rank bf16 partial sums added in fp32 in ascending rank order."""
    acc = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    owner = torch.where(ids >= 0, ids // _NUM_LOCAL_EXPERTS, -1)
    for peer in range(world_size):
        on_peer = owner == peer
        scale = torch.where(on_peer, weights * (ids + 1).float(), 0).sum(1)
        partial = (hidden.float() * scale.unsqueeze(1)).to(_DTYPE).float()
        acc += torch.where(on_peer.any(1, keepdim=True), partial, 0)
    return acc.to(_DTYPE)


def _run_case(dispatcher, group, *, tokens_per_rank, seed, routed=True, exact=True):
    rank, world_size = group.rank_in_group, group.world_size
    inputs = [
        _rank_inputs(
            rank=r,
            num_tokens=tokens_per_rank[r],
            world_size=world_size,
            seed=seed,
            routed=routed,
        )
        for r in range(world_size)
    ]
    hidden, ids, weights = (t.to(group.device) for t in inputs[rank])
    out = dispatcher.dispatch(
        hidden,
        StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=None),
    )

    # Rows arrive grouped by source rank, in token order, once per token.
    owners = [torch.where(i >= 0, i // _NUM_LOCAL_EXPERTS, -1) for _, i, _ in inputs]
    sent = [(o == rank).any(1) for o in owners]
    exp_hidden = torch.cat([h[m] for (h, _, _), m in zip(inputs, sent)])
    exp_ids = torch.cat([i[m] for (_, i, _), m in zip(inputs, sent)])
    exp_ids = torch.where(
        exp_ids // _NUM_LOCAL_EXPERTS == rank, exp_ids - rank * _NUM_LOCAL_EXPERTS, -1
    )
    exp_weights = torch.cat([w[m] for (_, _, w), m in zip(inputs, sent)])
    torch.testing.assert_close(out.hidden_states.cpu(), exp_hidden, atol=0, rtol=0)
    torch.testing.assert_close(
        out.topk_output.topk_ids.cpu(), exp_ids.to(torch.int32), atol=0, rtol=0
    )
    torch.testing.assert_close(
        out.topk_output.topk_weights.cpu(), exp_weights, atol=0, rtol=0
    )

    expert_out = _fake_experts(
        out.hidden_states, out.topk_output.topk_ids, out.topk_output.topk_weights, rank
    )
    combined = dispatcher.combine(StandardCombineInput(hidden_states=expert_out))
    # On the device, like the fake experts: the weight sum order must match.
    expected = _reference_combine(hidden, ids, weights, world_size)
    tol = 0 if exact else 2e-2
    torch.testing.assert_close(combined.cpu(), expected.cpu(), atol=tol, rtol=tol)


@torch.inference_mode()
def test_dispatch_combine_matches_reference() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device_module = get_device_module()
    device_module.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    group = _Group(
        rank_in_group=dist.get_rank(),
        world_size=world_size,
        cpu_group=dist.group.WORLD,
        device=torch.device(get_device(), local_rank),
    )
    sizes = [37, 0, 64, 5, 1, 19, 8, 3][:world_size]

    layer_a, layer_b = _make_dispatcher(group), _make_dispatcher(group)
    # One round, an idle rank, and two layers sharing the staging.
    _run_case(layer_a, group, tokens_per_rank=sizes, seed=1)
    _run_case(layer_b, group, tokens_per_rank=sizes[::-1], seed=2)
    # Nothing routed: no data rounds, so consecutive layers reuse the counts
    # back to back, and a stale count would misroute the next batch.
    for seed in (3, 4, 5):
        _run_case(layer_a, group, tokens_per_rank=sizes, seed=seed, routed=False)
    _run_case(layer_b, group, tokens_per_rank=sizes, seed=6)

    # Many rounds: a token's per-rank partials can land in different rounds.
    with envs.SGLANG_SYMM_MEM_A2A_NUM_MAX_DISPATCH_TOKENS_PER_ROUND.override(5):
        many_rounds = _make_dispatcher(group)
    for seed in (7, 8):
        _run_case(many_rounds, group, tokens_per_rank=sizes, seed=seed, exact=False)
    dist.barrier()
    dist.destroy_process_group()


def _main() -> int:
    args = sys.argv[1:]
    world_size = 2
    if "--world-size" in args:
        i = args.index("--world-size")
        world_size = int(args[i + 1])
        del args[i : i + 2]
    if get_device_module().device_count() < world_size:
        print(f"needs {world_size} devices; skipping")
        return 0
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={world_size}",
            __file__,
            *args,
        ],
        env={**os.environ, "_SYMM_MEM_A2A_TEST_WORKER": "1"},
    ).returncode


if __name__ == "__main__":
    if os.environ.get("_SYMM_MEM_A2A_TEST_WORKER"):
        # no:randomly: a per-process shuffle would desynchronize the ranks.
        sys.exit(pytest.main([__file__, "-p", "no:randomly", "-x", *sys.argv[1:]]))
    sys.exit(_main())
