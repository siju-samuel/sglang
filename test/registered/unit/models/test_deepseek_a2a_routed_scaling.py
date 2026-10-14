"""DeepseekV2MoE.forward_deepep must apply routed_scaling_factor exactly once.

Regression: with ``--moe-a2a-backend symm_mem`` on XPU the routed experts run
through the same MoE runner as forward_normal, which already scales their
output, and forward_deepep scaled it a second time.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_ROUTED_SCALING_FACTOR = 2.5


def _make_moe(*, routed: torch.Tensor, shared: torch.Tensor):
    def experts(hidden_states, topk_output):
        return routed.clone()

    experts.should_fuse_routed_scaling_factor_in_topk = False
    experts.quant_method = None
    moe = deepseek_v2.DeepseekV2MoE.__new__(deepseek_v2.DeepseekV2MoE)
    torch.nn.Module.__init__(moe)
    vars(moe).update(
        layer_id=0,
        is_nextn=True,  # skips the EPLB dispatch info, which needs global state
        is_hash=False,
        _fuse_shared_experts_inside_sbo=False,
        num_fused_shared_experts=0,
        routed_scaling_factor=_ROUTED_SCALING_FACTOR,
        experts=experts,
        gate=lambda hidden_states, **kwargs: hidden_states,
        topk=lambda *args, **kwargs: None,
        alt_stream=None,
        _forward_shared_experts=lambda hidden_states, *args, **kwargs: shared.clone(),
    )
    return moe


@pytest.mark.parametrize(
    "backend,is_xpu,routed_scaled_by_runner",
    [
        # The MoE runner scales on XPU; the model must not scale again.
        (MoeA2ABackend.SYMM_MEM, True, True),
        # Where forward_normal scales in the model, symm_mem does too.
        (MoeA2ABackend.SYMM_MEM, False, False),
        # DeepEP-family experts return unscaled outputs on every platform.
        (MoeA2ABackend.DEEPEP, True, False),
    ],
)
def test_forward_deepep_scales_routed_output_once(
    backend, is_xpu, routed_scaled_by_runner
):
    routed = torch.full((3, 4), 2.0)
    shared = torch.full((3, 4), 1.0)
    moe = _make_moe(routed=routed, shared=shared)
    forward_batch = SimpleNamespace(moe_num_token_non_padded=lambda: None)
    with (
        patch.object(deepseek_v2, "get_moe_a2a_backend", return_value=backend),
        patch.object(deepseek_v2, "_is_xpu", is_xpu),
        patch.object(deepseek_v2, "_is_cuda", False),
        patch.object(deepseek_v2, "_is_musa", False),
        patch.object(deepseek_v2, "_use_aiter", False),
    ):
        out = moe.forward_deepep(torch.ones(3, 4), forward_batch)

    scale = 1.0 if routed_scaled_by_runner else _ROUTED_SCALING_FACTOR
    torch.testing.assert_close(out, shared + scale * routed)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
