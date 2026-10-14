"""MoE all-to-all over torch symmetric memory (``--moe-a2a-backend symm_mem``).

Single-node expert parallelism with no communication library beyond torch. A
rank copies the rows it routes to each peer straight into that peer's
symmetric staging buffer, one peer per step in rotation, in rounds of at most
``SGLANG_SYMM_MEM_A2A_NUM_MAX_DISPATCH_TOKENS_PER_ROUND`` rows per peer, and
orders the copies with the workspace barrier. A token reaches each rank that
hosts one of its experts once, with top-k ids made local (-1 for an expert
elsewhere), so the dispatch output is the standard format. Combine sends each
rank's partial sums back to the source rank, which adds them in fp32 in rank
order.

Every rank must issue the same sequence of barriers, so the number of rounds
comes from the exchanged counts, never from local state.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.moe.symm_mem_a2a import (
    symm_mem_a2a_combine_round,
    symm_mem_a2a_pack,
    symm_mem_a2a_route,
    symm_mem_a2a_unpack,
)
from sglang.srt.distributed.device_communicators.symm_mem_workspace import (
    SymmMemWorkspace,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopKOutput,
    TopKOutputChecker,
)
from sglang.srt.runtime_context import get_resources

logger = logging.getLogger(__name__)

_BYTE_ALIGNMENT = 256


def _align(nbytes: int) -> int:
    return (nbytes + _BYTE_ALIGNMENT - 1) // _BYTE_ALIGNMENT * _BYTE_ALIGNMENT


class _StagingLayout(msgspec.Struct, frozen=True):
    """Byte offsets of the regions in one rank's symmetric buffer."""

    world_size: int
    round_tokens: int
    hidden_size: int
    topk: int
    dtype: torch.dtype
    # Two [world_size, world_size] int32 count matrices, alternated per dispatch.
    counts_offset: int
    dispatch_hidden_offset: int
    dispatch_meta_offset: int
    combine_hidden_offset: int
    nbytes: int

    @classmethod
    def build(
        cls,
        *,
        world_size: int,
        round_tokens: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
    ) -> _StagingLayout:
        counts_bytes = _align(2 * world_size * world_size * 4)
        hidden_bytes = _align(world_size * round_tokens * hidden_size * dtype.itemsize)
        meta_bytes = _align(world_size * round_tokens * 2 * topk * 4)
        return cls(
            world_size=world_size,
            round_tokens=round_tokens,
            hidden_size=hidden_size,
            topk=topk,
            dtype=dtype,
            counts_offset=0,
            dispatch_hidden_offset=counts_bytes,
            dispatch_meta_offset=counts_bytes + hidden_bytes,
            combine_hidden_offset=counts_bytes + hidden_bytes + meta_bytes,
            nbytes=counts_bytes + 2 * hidden_bytes + meta_bytes,
        )


class _SymmMemA2AState:
    """Staging shared by every MoE layer of one geometry; creating it is collective."""

    def __init__(self, *, workspace: SymmMemWorkspace, layout: _StagingLayout):
        self.workspace = workspace
        self.layout = layout
        world_size, round_tokens = layout.world_size, layout.round_tokens
        hidden_shape = (world_size, round_tokens, layout.hidden_size)
        meta_shape = (world_size, round_tokens, 2 * layout.topk)
        ranks = range(world_size)
        # Index [rank] for the views over that rank's buffer, own rank included.
        self.dispatch_hidden = [
            workspace.peer_view(
                r,
                shape=hidden_shape,
                dtype=layout.dtype,
                byte_offset=layout.dispatch_hidden_offset,
            )
            for r in ranks
        ]
        self.dispatch_meta = [
            workspace.peer_view(
                r,
                shape=meta_shape,
                dtype=torch.int32,
                byte_offset=layout.dispatch_meta_offset,
            )
            for r in ranks
        ]
        self.combine_hidden = [
            workspace.peer_view(
                r,
                shape=hidden_shape,
                dtype=layout.dtype,
                byte_offset=layout.combine_hidden_offset,
            )
            for r in ranks
        ]
        counts_bytes = world_size * world_size * 4
        self.counts = [
            workspace.peer_view(
                workspace.rank,
                shape=(world_size, world_size),
                dtype=torch.int32,
                byte_offset=layout.counts_offset + parity * counts_bytes,
            )
            for parity in range(2)
        ]
        self.counts_peer_ptrs = [
            workspace.peer_ptrs + layout.counts_offset + parity * counts_bytes
            for parity in range(2)
        ]
        self.counts_parity = 0
        device = workspace.device
        self.send_hidden = torch.empty(hidden_shape, dtype=layout.dtype, device=device)
        self.send_meta = torch.empty(meta_shape, dtype=torch.int32, device=device)


def _get_or_create_state(
    *, group: GroupCoordinator, layout: _StagingLayout
) -> _SymmMemA2AState:
    states: Dict[_StagingLayout, _SymmMemA2AState] = get_resources().buffers.setdefault(
        "symm_mem_a2a_states", {}
    )
    state = states.get(layout)
    if state is None:
        workspace = SymmMemWorkspace.create(
            group=group.cpu_group,
            device=group.device,
            nbytes=layout.nbytes,
            owner="SymmMemDispatcher",
        )
        if workspace is None:
            raise RuntimeError(
                "--moe-a2a-backend symm_mem needs peer-mapped torch symmetric "
                "memory on a single node; see the warning above for the reason "
                "it is unavailable."
            )
        state = _SymmMemA2AState(workspace=workspace, layout=layout)
        states[layout] = state
        logger.info(
            "SymmMemDispatcher: %.1f MiB symmetric staging (world_size=%d, "
            "%d tokens per peer per round, hidden_size=%d).",
            layout.nbytes / (1 << 20),
            layout.world_size,
            layout.round_tokens,
            layout.hidden_size,
        )
    return state


class _DispatchHandle(NamedTuple):
    # [num_tokens, world_size] int32: row of each token in the block per rank, or -1.
    slot: torch.Tensor
    # counts[src][dst]: rows src sent dst.
    counts: List[List[int]]
    num_rounds: int


def _rows_in_round(rows: int, *, round_start: int, round_tokens: int) -> int:
    return max(0, min(round_tokens, rows - round_start))


class SymmMemDispatcher(BaseDispatcher):
    def __init__(
        self,
        *,
        group: GroupCoordinator,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
    ):
        super().__init__()
        self.rank = group.rank_in_group
        self.world_size = group.world_size
        if num_experts != num_local_experts * self.world_size:
            raise ValueError(
                f"symm_mem a2a needs every rank to host the same number of "
                f"experts: {num_experts} experts, {num_local_experts} per rank, "
                f"{self.world_size} ranks."
            )
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.round_tokens = (
            envs.SGLANG_SYMM_MEM_A2A_NUM_MAX_DISPATCH_TOKENS_PER_ROUND.get()
        )
        self._state = _get_or_create_state(
            group=group,
            layout=_StagingLayout.build(
                world_size=self.world_size,
                round_tokens=self.round_tokens,
                hidden_size=hidden_size,
                topk=router_topk,
                dtype=params_dtype,
            ),
        )
        self._handle: Optional[_DispatchHandle] = None

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> StandardDispatchOutput:
        assert TopKOutputChecker.format_is_standard(topk_output)
        state = self._state
        assert hidden_states.shape[1] == self.hidden_size
        assert hidden_states.dtype == state.layout.dtype

        slot, counts_dev, counts = self._route(topk_output.topk_ids)
        num_rounds = math.ceil(max(max(row) for row in counts) / self.round_tokens)

        recv_rows = sum(counts[src][self.rank] for src in range(self.world_size))
        topk = state.layout.topk
        device = hidden_states.device
        out_hidden = torch.empty(
            (recv_rows, self.hidden_size), dtype=hidden_states.dtype, device=device
        )
        out_topk_ids = torch.empty((recv_rows, topk), dtype=torch.int32, device=device)
        out_topk_weights = torch.empty(
            (recv_rows, topk), dtype=torch.float32, device=device
        )

        for k in range(num_rounds):
            round_start = k * self.round_tokens
            symm_mem_a2a_pack(
                hidden_states=hidden_states,
                topk_ids=topk_output.topk_ids,
                topk_weights=topk_output.topk_weights,
                slot=slot,
                send_hidden=state.send_hidden,
                send_meta=state.send_meta,
                round_start=round_start,
            )
            for peer in self._rotation():
                rows = _rows_in_round(
                    counts[self.rank][peer],
                    round_start=round_start,
                    round_tokens=self.round_tokens,
                )
                if rows:
                    state.dispatch_hidden[peer][self.rank, :rows].copy_(
                        state.send_hidden[peer, :rows]
                    )
                    state.dispatch_meta[peer][self.rank, :rows].copy_(
                        state.send_meta[peer, :rows]
                    )
            state.workspace.barrier()
            symm_mem_a2a_unpack(
                recv_hidden=state.dispatch_hidden[self.rank],
                recv_meta=state.dispatch_meta[self.rank],
                counts=counts_dev,
                out_hidden=out_hidden,
                out_topk_ids=out_topk_ids,
                out_topk_weights=out_topk_weights,
                rank=self.rank,
                num_local_experts=self.num_local_experts,
                round_start=round_start,
            )
            # The next round overwrites the staging every peer just read.
            if k < num_rounds - 1:
                state.workspace.barrier()

        self._handle = _DispatchHandle(slot=slot, counts=counts, num_rounds=num_rounds)
        return StandardDispatchOutput(
            hidden_states=out_hidden,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=out_topk_weights,
                topk_ids=out_topk_ids,
                router_logits=None,
            ),
        )

    def combine(self, combine_input: StandardCombineInput) -> torch.Tensor:
        (expert_output,) = combine_input
        handle, self._handle = self._handle, None
        assert handle is not None, "combine() without a matching dispatch()"
        state = self._state
        counts = handle.counts
        num_tokens = handle.slot.shape[0]
        out = torch.empty(
            (num_tokens, self.hidden_size),
            dtype=state.layout.dtype,
            device=handle.slot.device,
        )
        if handle.num_rounds == 0:
            return out.zero_()
        expert_output = expert_output.to(state.layout.dtype)
        acc = (
            torch.empty(out.shape, dtype=torch.float32, device=out.device)
            if handle.num_rounds > 1
            else None
        )
        # Received blocks were laid out in source-rank order.
        recv_offsets = [0] * self.world_size
        for src in range(1, self.world_size):
            recv_offsets[src] = recv_offsets[src - 1] + counts[src - 1][self.rank]

        for k in range(handle.num_rounds):
            round_start = k * self.round_tokens
            for peer in self._rotation():
                rows = _rows_in_round(
                    counts[peer][self.rank],
                    round_start=round_start,
                    round_tokens=self.round_tokens,
                )
                if rows:
                    start = recv_offsets[peer] + round_start
                    state.combine_hidden[peer][self.rank, :rows].copy_(
                        expert_output[start : start + rows]
                    )
            state.workspace.barrier()
            symm_mem_a2a_combine_round(
                recv_hidden=state.combine_hidden[self.rank],
                slot=handle.slot,
                acc=acc,
                out=out,
                round_start=round_start,
                first_round=k == 0,
                last_round=k == handle.num_rounds - 1,
            )
            if k < handle.num_rounds - 1:
                state.workspace.barrier()
        return out

    def _route(
        self, topk_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        """Each token's row in the block it sends each rank, and the exchanged
        counts on the device and the host."""
        state = self._state
        # Alternate matrices: a peer that finishes this layer with no rows to
        # move can publish the next one before this rank has read these counts.
        parity = state.counts_parity
        state.counts_parity ^= 1
        slot = torch.empty(
            (topk_ids.shape[0], self.world_size),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        symm_mem_a2a_route(
            topk_ids=topk_ids,
            slot=slot,
            peer_counts_ptrs=state.counts_peer_ptrs[parity],
            rank=self.rank,
            num_local_experts=self.num_local_experts,
        )
        state.workspace.barrier()
        counts_dev = state.counts[parity]
        return slot, counts_dev, counts_dev.tolist()

    def _rotation(self):
        """Peers in send order: one peer per step keeps receivers from contending."""
        return [(self.rank + step) % self.world_size for step in range(self.world_size)]
