"""XPU-specific DeepEP <-> triton MoE runner adapters.

On CUDA the DeepEP dispatcher feeds DeepGEMM directly via the
(deepep_normal/deepep_ll, deep_gemm) permute adapters.  XPU has no DeepGEMM, so
these adapters route the deepep_normal / deepep_ll dispatch layouts through the
shared triton fused-MoE runner instead.

This module only registers (deepep_*, triton) permute methods into the global
PermuteMethodPool, so every quant method that runs through the triton runner
(unquant, fp8, w8a8, ...) is covered with a single registration.  Keeping it
separate from triton.py isolates the XPU-only glue from the shared runner; it is
imported for its @register_* side effects from triton.py under an is_xpu()
guard.
"""

from __future__ import annotations

import functools
import os

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeRunnerConfig,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    TritonRunnerInput,
    TritonRunnerOutput,
)

# Match the padding the triton fused-MoE config lookup uses (w8a8 w/o block_shape).
_MOE_PADDING_SIZE = 128 if bool(int(os.getenv("SGLANG_MOE_PADDING", "0"))) else 0


@register_pre_permute("deepep_normal", "triton")
def pre_permute_deepep_normal_to_triton(
    dispatch_output,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:
    """Adapt DeepEP normal dispatch output for the triton MoE runner.

    In deepep_normal mode, tokens have already been scattered to the rank that
    owns the target experts.  hidden_states contains the received tokens and
    topk_ids / topk_weights describe which local expert each token maps to.
    We simply call moe_align_block_size to produce the sorted indices the
    triton fused-MoE kernel expects.
    """
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        moe_align_block_size,
        try_get_optimal_moe_config,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_config_dtype_str,
    )

    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_ids
    topk_weights = dispatch_output.topk_weights

    # Save for post_permute
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    num_tokens = hidden_states.shape[0]
    num_local_experts = runner_config.num_local_experts

    if (
        not (quant_info.use_fp8_w8a8 or quant_info.use_int8_w8a8)
        or quant_info.block_shape is not None
    ):
        padding_size = 0
    else:
        padding_size = _MOE_PADDING_SIZE

    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        quant_info.w13_weight.shape,
        (
            num_local_experts,
            quant_info.w2_weight.shape[1],
            quant_info.w2_weight.shape[2] - padding_size,
        ),
        topk_ids.shape[1],
        config_dtype,
        block_shape=quant_info.block_shape,
        per_channel_quant=quant_info.per_channel_quant,
    )

    config = get_config_func(num_tokens)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_local_experts
    )

    running_state["config"] = config

    return TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "deepep_normal")
def post_permute_triton_to_deepep_normal(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    return DeepEPNormalCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


# =============================================================================
# DeepEP low-latency dispatch -> triton runner
# =============================================================================
# The LL dispatch output is a masked-expert tensor:
#   hidden_states      : [num_local_experts, max_recv, hidden]
#   masked_m           : [num_local_experts]  (valid tokens per expert)
#   topk_ids / weights : the ORIGINAL sender's topk layout
#                        (shape [num_tokens_src, num_topk]); NOT used to
#                        route within this rank — each row at [e, s] is
#                        already owned by local expert e.
#
# To run through the triton fused-MoE kernel we flatten to [E*M, H] and
# synthesize a 1-wide per-row topk where each row's id = its expert.
# Out-of-range slots (s >= masked_m[e]) get an invalid sentinel id so
# moe_align_block_size drops them from the work list.
#
# Post-permute reshapes the runner output back to [E, M, H] so it can be
# fed directly to buffer.low_latency_combine.
# =============================================================================


@register_pre_permute("deepep_ll", "triton")
def pre_permute_deepep_ll_to_triton(
    dispatch_output,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:
    """
    Compact the masked LL dispatch layout into a DENSE runner input.

    Input:  hidden_states_masked [E, M, H]  where only the first masked_m[e]
            rows of each expert are live (rest are padding).
    Output: triton input with hidden_states [N_live, H] where
            N_live = sum(masked_m) across all local experts on this rank.

    Why this matters: at decode batch=8 the live rows are ~64 while E*M can
    be 512-4096.  Feeding the padded [E*M, H] tensor to the triton kernel
    would waste 8-64× of the compute on sentinel-masked rows.  The
    compaction path is bounded by the actual routing density.
    """
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        moe_align_block_size,
        try_get_optimal_moe_config,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_config_dtype_str,
    )

    hidden_states_masked = dispatch_output.hidden_states
    masked_m = dispatch_output.masked_m  # [E] int32
    E, M, H = hidden_states_masked.shape
    num_local_experts = runner_config.num_local_experts
    assert (
        E == num_local_experts
    ), f"deepep_ll dispatch E={E} != runner num_local_experts={num_local_experts}"

    device = hidden_states_masked.device

    # Build a prefix sum so we know where each expert's live rows land in
    # the dense output:  begin[e] = sum_{k<e} masked_m[k], n_live = begin[E].
    masked_m_i32 = masked_m.to(torch.int32)
    begin = torch.zeros(E + 1, device=device, dtype=torch.int32)
    torch.cumsum(masked_m_i32, dim=0, dtype=torch.int32, out=begin[1:])
    n_live_t = begin[E]
    # Need int on host to size output tensors.  This is one small D2H sync;
    # unavoidable because torch.empty size must be a Python int.  The cost
    # is a few microseconds — negligible compared to what we save.
    n_live = int(n_live_t.item())

    # Build a gather index  [n_live]  s.t.  dense[i] = flat_masked[gather[i]].
    # For expert e the live slots are e*M + 0 .. e*M + masked_m[e] - 1.
    # We compute it with a single vectorized expression:
    #   row_in_expert = arange(n_live) - begin[expert_of_row]
    #   gather        = expert_of_row * M + row_in_expert
    # where expert_of_row is derived via searchsorted.
    row_idx = torch.arange(n_live, device=device, dtype=torch.int32)
    # searchsorted(begin[1..E], row_idx, right=True) maps each row to its
    # expert index.  (right=True so that a row at boundary goes to the
    # left expert.)
    expert_of_row = torch.searchsorted(begin[1:].contiguous(), row_idx, right=True).to(
        torch.int32
    )
    row_in_expert = row_idx - begin[expert_of_row]
    gather_idx = (expert_of_row.to(torch.int64) * M) + row_in_expert.to(torch.int64)

    if not hidden_states_masked.is_contiguous():
        hidden_states_masked = hidden_states_masked.contiguous()
    flat_masked = hidden_states_masked.view(E * M, H)
    # Index-select is one contiguous read per expert row (at most n_live
    # cache lines).  This is the only copy in the pre-permute path.
    dense_hidden = flat_masked.index_select(0, gather_idx)  # [n_live, H]

    # Synthetic topk for the triton runner: each dense row maps to exactly
    # one expert (expert_of_row).  topk_weight = 1.0.
    dense_topk_ids = expert_of_row.view(n_live, 1)
    dense_topk_weights = torch.ones((n_live, 1), device=device, dtype=torch.float32)

    # Save state for post_permute scatter-back.
    running_state["ll_E"] = E
    running_state["ll_M"] = M
    running_state["ll_H"] = H
    running_state["ll_n_live"] = n_live
    # gather_idx doubles as scatter_idx: dense_out[i] goes back to
    # flat_masked[gather_idx[i]].
    running_state["ll_scatter_idx"] = gather_idx
    running_state["ll_orig_topk_ids"] = dispatch_output.topk_ids
    running_state["ll_orig_topk_weights"] = dispatch_output.topk_weights

    # Triton config lookup (mirrors standard / deepep_normal paths).
    if (
        not (quant_info.use_fp8_w8a8 or quant_info.use_int8_w8a8)
        or quant_info.block_shape is not None
    ):
        padding_size = 0
    else:
        padding_size = _MOE_PADDING_SIZE

    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        dtype=dense_hidden.dtype,
    )
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        quant_info.w13_weight.shape,
        (
            num_local_experts,
            quant_info.w2_weight.shape[1],
            quant_info.w2_weight.shape[2] - padding_size,
        ),
        1,  # topk == 1 per dense row
        config_dtype,
        block_shape=quant_info.block_shape,
        per_channel_quant=quant_info.per_channel_quant,
    )
    # n_live can be 0 (no live tokens on this rank).  moe_align_block_size
    # returns zero-sized outputs, which is fine — the triton runner skips.
    config = get_config_func(max(n_live, 1))
    running_state["config"] = config

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        dense_topk_ids, config["BLOCK_SIZE_M"], num_local_experts
    )

    return TritonRunnerInput(
        hidden_states=dense_hidden,
        topk_weights=dense_topk_weights,
        topk_ids=dense_topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


# Per-device cached [E*M, H] scratch tensor used by post_permute.  The
# combine kernel (buffer.low_latency_combine) only reads positions listed
# in layout_range — it never touches uninitialized slots — so `empty`
# suffices and we avoid a ~tens-of-MB memset per MoE layer per decode step
# (which at 25 layers × 16 decode steps dominated the LL total cost).
#
# Keyed by (device_index, dtype, E, M, H).  Buffers are pooled across
# calls; when shape changes we reallocate.  Bounded by the number of
# distinct shapes a run uses (typically 1).
_ll_post_permute_cache: dict = {}


def _get_ll_post_permute_buffer(device, dtype, E: int, M: int, H: int):
    key = (device, dtype, E, M, H)
    buf = _ll_post_permute_cache.get(key)
    if buf is None:
        buf = torch.empty(E * M, H, device=device, dtype=dtype)
        _ll_post_permute_cache[key] = buf
    return buf


@register_post_permute("triton", "deepep_ll")
def post_permute_triton_to_deepep_ll(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    """
    Scatter the dense [n_live, H] runner output back into the [E, M, H]
    masked layout that buffer.low_latency_combine expects.

    Crucial optimizations:
    - Reuse a cached [E*M, H] tensor (no per-call malloc).
    - Use `empty` (no memset).  buffer.low_latency_combine walks only the
      live positions listed in layout_range — unused slots are never read
      so their contents don't matter.
    """
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    E = running_state["ll_E"]
    M = running_state["ll_M"]
    H = running_state["ll_H"]
    n_live = running_state["ll_n_live"]
    scatter_idx = running_state["ll_scatter_idx"]
    device = runner_output.hidden_states.device
    dtype = runner_output.hidden_states.dtype

    masked_out = _get_ll_post_permute_buffer(device, dtype, E, M, H)
    if n_live > 0:
        masked_out.index_copy_(0, scatter_idx, runner_output.hidden_states)
    masked_out_3d = masked_out.view(E, M, H)

    return DeepEPLLCombineInput(
        hidden_states=masked_out_3d,
        topk_ids=running_state["ll_orig_topk_ids"],
        topk_weights=running_state["ll_orig_topk_weights"],
    )
