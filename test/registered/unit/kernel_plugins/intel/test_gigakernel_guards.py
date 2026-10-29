"""CPU checks of --enable-intel-gigakernel's startup resolution."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.plugin_kernels.intel import spec as registry
from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.platform_hook import handle_xpu_backends
from sglang.srt.kernel_plugins.intel import binding
from sglang.srt.kernel_plugins.intel.binding import gigakernel_disable_reasons
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _intel_fakes import ARCH, ISA, server_args  # noqa: E402

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestGigakernelGuards(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)

    def test_a_disabled_flag_looks_nothing_up_and_keeps_the_forward(self):
        with patch.object(binding, "plugin_reasons") as plugin:
            binding.handle_intel_gigakernel(server_args())
        plugin.assert_not_called()
        model = SimpleNamespace()  # any attribute read would raise
        with patch.object(binding, "gigakernel_enabled", return_value=False):
            binding.install_gigakernel(
                model=model, hf_config=None, capture_tail_hooks=[], is_draft=False
            )
        self.assertEqual(vars(model), {})

    def test_an_unmet_tp_size_is_refused_and_an_empty_field_accepts_any(self):
        hf_config = SimpleNamespace(architectures=[ARCH], model_type="fake")

        def tp_reasons(path):
            registry._REGISTRY.clear()
            registry.register(architecture=ARCH, path=path)
            _, reasons = gigakernel_disable_reasons(
                cfg=server_args(tp_size=8), hf_config=hf_config, isa=ISA
            )
            return [r for r in reasons if "tp_size" in r]

        self.assertTrue(tp_reasons("_intel_fakes:SPEC"))
        self.assertEqual(tp_reasons("_intel_fakes:ANY_TP"), [])

    def test_another_isa_or_no_xpu_is_refused(self):
        registry.register(architecture=ARCH, path="_intel_fakes:SPEC")
        hf_config = SimpleNamespace(architectures=[ARCH])
        _, reasons = gigakernel_disable_reasons(
            cfg=server_args(), hf_config=hf_config, isa="pvc"
        )
        self.assertIn("not pvc", "".join(reasons))
        _, reasons = gigakernel_disable_reasons(
            cfg=server_args(), hf_config=hf_config, isa=None
        )
        self.assertEqual(reasons, ["no XPU is visible"])

    def test_another_kernel_plugin_flag_is_named_in_the_same_error(self):
        args = server_args(enable_intel_gigakernel=True, enable_amd_monokernel=True)
        with patch.object(binding, "plugin_reasons", return_value=["not installed"]):
            with self.assertRaisesRegex(ValueError, "not installed(.|\n)*monokernel"):
                binding.handle_intel_gigakernel(args)

    def test_the_flag_keeps_xpu_decode_graphs_on_unless_a_decode_backend_is_given(self):
        def decode_backend(**kw):
            args = server_args(device="xpu", cuda_graph_config=None, **kw)
            parse_cuda_graph_config(args)
            handle_xpu_backends(args)
            return resolution_result(args, "cuda_graph_config").decode.backend

        self.assertEqual(decode_backend(), Backend.DISABLED)
        self.assertEqual(decode_backend(enable_intel_gigakernel=True), Backend.FULL)
        self.assertEqual(
            decode_backend(
                enable_intel_gigakernel=True,
                cuda_graph_backend_decode=Backend.DISABLED,
            ),
            Backend.DISABLED,
        )

    def test_the_sycl_architecture_maps_to_its_name(self):
        def isa(arch):
            props = SimpleNamespace(architecture=arch)
            with (
                patch.object(torch.xpu, "is_available", return_value=True),
                patch.object(torch.xpu, "current_device", return_value=0),
                patch.object(torch.xpu, "get_device_properties", return_value=props),
            ):
                return binding.device_isa()

        # Arc Pro B60, as torch 2.14 reports it
        self.assertEqual(isa(21479031808), "bmg_g21")
        # an unknown architecture never matches a spec by accident
        self.assertEqual(isa(0x12345), "arch_0x12345")
        with patch.object(torch.xpu, "is_available", return_value=False):
            self.assertIsNone(binding.device_isa())


if __name__ == "__main__":
    unittest.main()
