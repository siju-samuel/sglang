"""Triton kernels for the symmetric-memory MoE all-to-all (``--moe-a2a-backend symm_mem``).

Layouts shared with ``sglang.srt.layers.moe.token_dispatcher.symm_mem``:

- ``slot``: [num_tokens, world_size] int32, a token's row in the block it sends
  to each rank, or -1 when no top-k expert of the token lives there. Rows
  follow token order.
- ``counts``: [world_size, world_size] int32, ``counts[src, dst]`` rows sent.
- Staging: [world_size, round_tokens, ...], indexed by the peer rank; one round
  moves rows ``[round_start, round_start + round_tokens)`` of every block.
- Meta rows: [2 * topk] int32, the global top-k ids followed by the fp32 top-k
  weights bit-cast to int32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_H = 1024


@triton.jit
def _route_kernel(
    topk_ids_ptr,
    slot_ptr,
    peer_counts_ptrs,
    rank,
    num_tokens,
    num_local_experts,
    WORLD_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # One program walks the tokens in order, so each block's rows are dense.
    w_offs = tl.arange(0, BLOCK_W)
    w_mask = w_offs < WORLD_SIZE
    k_offs = tl.arange(0, BLOCK_K)
    rows_sent = tl.zeros([BLOCK_W], dtype=tl.int32)
    for start in range(0, num_tokens, BLOCK_T):
        t_offs = start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < num_tokens
        ids = tl.load(
            topk_ids_ptr + t_offs[:, None] * TOPK + k_offs[None, :],
            mask=t_mask[:, None] & (k_offs[None, :] < TOPK),
            other=-1,
        ).to(tl.int32)
        owner = tl.where(ids >= 0, ids // num_local_experts, -1)
        sends = tl.max(
            (owner[:, :, None] == w_offs[None, None, :]).to(tl.int32), axis=1
        )
        sends = tl.where(t_mask[:, None] & w_mask[None, :], sends, 0)
        rows = tl.cumsum(sends, axis=0) - 1 + rows_sent[None, :]
        tl.store(
            slot_ptr + t_offs[:, None] * WORLD_SIZE + w_offs[None, :],
            tl.where(sends > 0, rows, -1),
            mask=t_mask[:, None] & w_mask[None, :],
        )
        rows_sent += tl.sum(sends, axis=0)
    for peer in tl.static_range(WORLD_SIZE):
        peer_counts = tl.load(peer_counts_ptrs + peer).to(tl.pointer_type(tl.int32))
        tl.store(peer_counts + rank * WORLD_SIZE + w_offs, rows_sent, mask=w_mask)


def symm_mem_a2a_route(
    *,
    topk_ids: torch.Tensor,
    slot: torch.Tensor,
    peer_counts_ptrs: torch.Tensor,
    rank: int,
    num_local_experts: int,
) -> None:
    """Fill ``slot`` and write this rank's row of ``counts`` on every rank.

    ``peer_counts_ptrs`` holds each rank's counts base address (int64, see
    ``SymmMemWorkspace.peer_ptrs``); the caller barriers before reading.
    """
    num_tokens, topk = topk_ids.shape
    world_size = slot.shape[1]
    _route_kernel[(1,)](
        topk_ids,
        slot,
        peer_counts_ptrs,
        rank,
        num_tokens,
        num_local_experts,
        WORLD_SIZE=world_size,
        TOPK=topk,
        BLOCK_T=128,
        BLOCK_K=triton.next_power_of_2(topk),
        BLOCK_W=triton.next_power_of_2(world_size),
    )


@triton.jit
def _pack_round_kernel(
    hidden_ptr,
    hidden_stride,
    topk_ids_ptr,
    topk_weights_ptr,
    slot_ptr,
    send_hidden_ptr,
    send_meta_ptr,
    hidden_size,
    round_start,
    round_tokens,
    WORLD_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    h_block = tl.program_id(1)
    h_offs = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < hidden_size
    row = tl.load(hidden_ptr + token * hidden_stride + h_offs, mask=h_mask)

    k_offs = tl.arange(0, BLOCK_K)
    # Only the first hidden block writes the meta row.
    k_mask = (k_offs < TOPK) & (h_block == 0)
    ids = tl.load(topk_ids_ptr + token * TOPK + k_offs, mask=k_mask).to(tl.int32)
    weights = tl.load(topk_weights_ptr + token * TOPK + k_offs, mask=k_mask)
    weight_bits = weights.to(tl.float32).to(tl.int32, bitcast=True)

    for peer in tl.static_range(WORLD_SIZE):
        local_row = tl.load(slot_ptr + token * WORLD_SIZE + peer) - round_start
        in_round = (local_row >= 0) & (local_row < round_tokens)
        dst = (peer * round_tokens + local_row).to(tl.int64)
        tl.store(
            send_hidden_ptr + dst * hidden_size + h_offs, row, mask=h_mask & in_round
        )
        meta = send_meta_ptr + dst * (2 * TOPK)
        tl.store(meta + k_offs, ids, mask=k_mask & in_round)
        tl.store(meta + TOPK + k_offs, weight_bits, mask=k_mask & in_round)


def symm_mem_a2a_pack(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    slot: torch.Tensor,
    send_hidden: torch.Tensor,
    send_meta: torch.Tensor,
    round_start: int,
) -> None:
    """Copy each token's row into ``send_hidden[peer]`` for every peer it goes to this round.

    ``send_hidden``: [world_size, round_tokens, hidden_size];
    ``send_meta``: [world_size, round_tokens, 2 * topk] int32.
    """
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    world_size, round_tokens, _ = send_hidden.shape
    topk = topk_ids.shape[1]
    assert hidden_states.stride(1) == 1
    _pack_round_kernel[(num_tokens, triton.cdiv(hidden_size, _BLOCK_H))](
        hidden_states,
        hidden_states.stride(0),
        topk_ids,
        topk_weights,
        slot,
        send_hidden,
        send_meta,
        hidden_size,
        round_start,
        round_tokens,
        WORLD_SIZE=world_size,
        TOPK=topk,
        BLOCK_H=_BLOCK_H,
        BLOCK_K=triton.next_power_of_2(topk),
    )


@triton.jit
def _unpack_round_kernel(
    recv_hidden_ptr,
    recv_meta_ptr,
    counts_ptr,
    out_hidden_ptr,
    out_topk_ids_ptr,
    out_topk_weights_ptr,
    rank,
    first_local_expert,
    num_local_experts,
    hidden_size,
    round_start,
    round_tokens,
    WORLD_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    src = tl.program_id(0)
    local_row = tl.program_id(1)
    h_block = tl.program_id(2)

    # Received blocks are laid out in source-rank order.
    out_start = 0
    for prev in tl.static_range(WORLD_SIZE):
        prev_rows = tl.load(counts_ptr + prev * WORLD_SIZE + rank)
        out_start += tl.where(prev < src, prev_rows, 0)
    rows = tl.load(counts_ptr + src * WORLD_SIZE + rank) - round_start
    valid = local_row < rows

    in_row = (src * round_tokens + local_row).to(tl.int64)
    out_row = (out_start + round_start + local_row).to(tl.int64)

    h_offs = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = (h_offs < hidden_size) & valid
    tl.store(
        out_hidden_ptr + out_row * hidden_size + h_offs,
        tl.load(recv_hidden_ptr + in_row * hidden_size + h_offs, mask=h_mask),
        mask=h_mask,
    )

    k_offs = tl.arange(0, BLOCK_K)
    k_mask = (k_offs < TOPK) & valid & (h_block == 0)
    meta = recv_meta_ptr + in_row * (2 * TOPK)
    ids = tl.load(meta + k_offs, mask=k_mask, other=-1)
    is_local = (ids >= first_local_expert) & (
        ids < first_local_expert + num_local_experts
    )
    tl.store(
        out_topk_ids_ptr + out_row * TOPK + k_offs,
        tl.where(is_local, ids - first_local_expert, -1),
        mask=k_mask,
    )
    weight_bits = tl.load(meta + TOPK + k_offs, mask=k_mask)
    tl.store(
        out_topk_weights_ptr + out_row * TOPK + k_offs,
        weight_bits.to(tl.float32, bitcast=True),
        mask=k_mask,
    )


def symm_mem_a2a_unpack(
    *,
    recv_hidden: torch.Tensor,
    recv_meta: torch.Tensor,
    counts: torch.Tensor,
    out_hidden: torch.Tensor,
    out_topk_ids: torch.Tensor,
    out_topk_weights: torch.Tensor,
    rank: int,
    num_local_experts: int,
    round_start: int,
) -> None:
    """Move one round of received rows into the source-ordered outputs.

    Top-k ids become local expert ids, -1 for an expert on another rank.
    """
    world_size, round_tokens, hidden_size = recv_hidden.shape
    topk = out_topk_ids.shape[1]
    _unpack_round_kernel[
        (world_size, round_tokens, triton.cdiv(hidden_size, _BLOCK_H))
    ](
        recv_hidden,
        recv_meta,
        counts,
        out_hidden,
        out_topk_ids,
        out_topk_weights,
        rank,
        rank * num_local_experts,
        num_local_experts,
        hidden_size,
        round_start,
        round_tokens,
        WORLD_SIZE=world_size,
        TOPK=topk,
        BLOCK_H=_BLOCK_H,
        BLOCK_K=triton.next_power_of_2(topk),
    )


@triton.jit
def _combine_round_kernel(
    recv_hidden_ptr,
    slot_ptr,
    acc_ptr,
    out_ptr,
    hidden_size,
    round_start,
    round_tokens,
    WORLD_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    FIRST_ROUND: tl.constexpr,
    LAST_ROUND: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    h_offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < hidden_size
    if FIRST_ROUND:
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    else:
        acc = tl.load(acc_ptr + token * hidden_size + h_offs, mask=h_mask)
    # Ascending rank order, so a single-round result is identical on every run.
    for peer in tl.static_range(WORLD_SIZE):
        local_row = tl.load(slot_ptr + token * WORLD_SIZE + peer) - round_start
        in_round = (local_row >= 0) & (local_row < round_tokens)
        row = (peer * round_tokens + local_row).to(tl.int64)
        acc += tl.load(
            recv_hidden_ptr + row * hidden_size + h_offs,
            mask=h_mask & in_round,
            other=0.0,
        ).to(tl.float32)
    if LAST_ROUND:
        tl.store(
            out_ptr + token * hidden_size + h_offs,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )
    else:
        tl.store(acc_ptr + token * hidden_size + h_offs, acc, mask=h_mask)


def symm_mem_a2a_combine_round(
    *,
    recv_hidden: torch.Tensor,
    slot: torch.Tensor,
    acc: torch.Tensor | None,
    out: torch.Tensor,
    round_start: int,
    first_round: bool,
    last_round: bool,
) -> None:
    """Add one round of returned expert outputs into each token's sum.

    ``recv_hidden``: [world_size, round_tokens, hidden_size], indexed by the rank
    that computed the rows. ``acc`` (fp32, like ``out``) carries the partial sum
    between rounds and may be ``None`` for a single round.
    """
    num_tokens, hidden_size = out.shape
    if num_tokens == 0:
        return
    world_size, round_tokens, _ = recv_hidden.shape
    assert acc is not None or (first_round and last_round)
    _combine_round_kernel[(num_tokens, triton.cdiv(hidden_size, _BLOCK_H))](
        recv_hidden,
        slot,
        # Unread for a single round: FIRST_ROUND and LAST_ROUND compile both accesses out.
        acc if acc is not None else out,
        out,
        hidden_size,
        round_start,
        round_tokens,
        WORLD_SIZE=world_size,
        BLOCK_H=_BLOCK_H,
        FIRST_ROUND=first_round,
        LAST_ROUND=last_round,
    )
