"""CPU checks of the gigakernel install, dispatch and capture check."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from msgspec.structs import replace

from sglang.kernels.plugin_kernels.intel import spec as registry
from sglang.srt.kernel_plugins.intel import binding
from sglang.srt.kernel_plugins.intel.binding import GigakernelBinding
from sglang.srt.runtime_context import publish, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _intel_fakes import (  # noqa: E402
    ANY_TP,
    ARCH,
    ISA,
    NO_WIDTH,
    SPEC,
    Lm,
    batch,
    server_args,
)

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def install(*, archs=(ARCH,), gathered=None, tokens=1, hooks=None, is_draft=False):
    """install_gigakernel on an Lm as rank 0 of the published server; gathered maps
    a gathered value to what the other ranks report, default the same as this rank."""
    lm, hooks = Lm(), [] if hooks is None else hooks
    xpu = SimpleNamespace(gpu_subslice_count=20)
    with (
        patch.object(binding, "device_isa", return_value=ISA),
        patch.object(binding, "deadline_s", return_value=30.0),
        patch.object(
            binding,
            "_gather_tp",
            side_effect=gathered or (lambda x, group=None: [x]),
        ),
        patch.object(binding, "_tokens_per_request", return_value=tokens),
        patch.object(binding, "_decode_graphs_enabled", return_value=True),
        patch.object(torch.xpu, "current_device", return_value=0),
        patch.object(torch.xpu, "get_device_properties", return_value=xpu),
        patch.object(torch.xpu, "mem_get_info", return_value=(0, 0)),
    ):
        binding.install_gigakernel(
            model=lm,
            hf_config=SimpleNamespace(architectures=list(archs)),
            capture_tail_hooks=hooks,
            is_draft=is_draft,
        )
    return lm


class TestGigakernelRuntime(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        registry.register(architecture=ARCH, path="_intel_fakes:SPEC")
        publish(server_args(enable_intel_gigakernel=True), role="test")
        self.addCleanup(reset_context)
        # the running forward's view reads its attention backend, which tests lack
        view_patch = patch.object(
            GigakernelBinding, "forward_view", return_value="view"
        )
        view_patch.start()
        self.addCleanup(view_patch.stop)

    def _install_spec(self, spec, **kw):
        with patch.object(binding, "find_spec", return_value=(spec, "")):
            return install(**kw)

    def test_covered_steps_run_the_gigakernel_and_others_the_forward(self):
        lm = install()
        built = binding.built_gigakernel(lm)
        with patch.object(torch.xpu, "is_current_stream_capturing", return_value=False):
            self.assertEqual(lm(batch(4).input_ids, None, batch(4)), "gigakernel")
            self.assertEqual(lm(batch(6).input_ids, None, batch(6)), "standard")
        # only a covered step runs the gigakernel, with the running forward's view
        self.assertEqual((built.steps, built.views), ([4], ["view"]))

    def test_a_target_without_a_usable_spec_is_refused_and_a_draft_keeps_its_forward(
        self,
    ):
        # the flag must not silently run the model's own forward
        with self.assertRaisesRegex(ValueError, "in the model workers"):
            install(archs=["TargetWithoutSpec"])
        tp8 = replace(SPEC, requires=replace(SPEC.requires, tp_sizes=frozenset({8})))
        with self.assertRaisesRegex(ValueError, "tp_size 4"):
            self._install_spec(tp8)
        with self.assertRaisesRegex(ValueError, "built no width"):
            self._install_spec(NO_WIDTH)
        # an empty field accepts any value
        self.assertIsNotNone(binding.built_gigakernel(self._install_spec(ANY_TP)))
        with self.assertLogs(binding.logger, "INFO") as logs:
            lm = install(archs=["DraftWithoutSpec"], is_draft=True)
        self.assertNotIn("forward", lm.__dict__)
        self.assertIn("keeps its own forward", "".join(logs.output))

    def test_too_few_xe_cores_are_refused(self):
        wide = replace(SPEC, requires=replace(SPEC.requires, resident_blocks=32))
        with self.assertRaisesRegex(ValueError, "20 Xe cores"):
            self._install_spec(wide)

    def test_ranks_that_disagree_are_refused_on_every_rank(self):
        """A rank that differs would wait in a different collective."""

        def no_spec_on_rank_1(value, group=None):
            return [value, (value[0], "rank-1 reason", None)]

        def fewer_widths_on_rank_1(value, group=None):
            widths = isinstance(value, tuple) and all(isinstance(w, int) for w in value)
            return [value, value[:1]] if widths else [value]

        with self.assertRaisesRegex(ValueError, "rank 1: rank-1 reason"):
            install(gathered=no_spec_on_rank_1)
        with self.assertRaisesRegex(RuntimeError, "widths fake built differ"):
            install(gathered=fewer_widths_on_rank_1)

    def test_capture_check_refuses_when_no_graph_runs_the_gigakernel(self):
        def capture(sizes, capture_bs):
            hooks = []
            lm = install(hooks=hooks)
            runner = SimpleNamespace(capture_bs=capture_bs)
            with patch.object(
                torch.xpu, "is_current_stream_capturing", return_value=True
            ):
                for bs in sizes:
                    lm(batch(bs).input_ids, None, batch(bs))
                    hooks[0](runner, None, batch(bs), bs)

        capture([2, 8], capture_bs=[2, 8, 16])  # not all captured yet: no verdict
        capture([2, 8], capture_bs=[2, 8])
        with self.assertRaisesRegex(ValueError, "none of the 2 captured decode graphs"):
            capture([8, 16], capture_bs=[8, 16])


if __name__ == "__main__":
    unittest.main()
