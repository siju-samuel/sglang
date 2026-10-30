import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_module  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _barrier_enabled(*, cuda: bool, hip: bool, xpu: bool) -> bool:
    """Run the real Scheduler.run_event_loop on a stub and return the WAR barrier gate."""
    scheduler = SimpleNamespace(
        device="xpu" if xpu else "cuda",
        enable_overlap=True,
        schedule_stream=SimpleNamespace(cuda_stream=1),
        forward_stream=SimpleNamespace(cuda_stream=2),
        device_module=SimpleNamespace(StreamContext=lambda stream: nullcontext()),
        metrics_reporter=mock.Mock(),
    )
    with (
        mock.patch.object(scheduler_module, "is_cuda", return_value=cuda),
        mock.patch.object(scheduler_module, "_is_hip", hip),
        mock.patch.object(scheduler_module, "_is_xpu", xpu),
        mock.patch.object(scheduler_module, "use_mlx", return_value=False),
        mock.patch.object(scheduler_module, "triton_load_watch"),
        mock.patch.object(
            scheduler_module, "get_parallel", return_value=SimpleNamespace(pp_size=1)
        ),
        mock.patch.object(scheduler_module, "dispatch_event_loop"),
    ):
        scheduler_module.Scheduler.run_event_loop(scheduler)
    return scheduler._war_barrier_enabled


class TestSchedulerWarBarrierGate(CustomTestCase):
    def test_xpu_enables_barrier_without_env(self):
        """With the overlap scheduler on XPU, a decode graph replay still read the
        req_to_token row and SWA mapping of a request that finished by EOS while the
        scheduler freed and reassigned them, and the device was lost. XPU must get the
        WAR barrier by default, like CUDA."""
        with envs.SGLANG_ENABLE_WAR_BARRIER.override(False):
            self.assertTrue(_barrier_enabled(cuda=False, hip=False, xpu=True))

    def test_hip_needs_the_env_var(self):
        # Negative branch: the barrier stays opt-in where no platform turns it on.
        with envs.SGLANG_ENABLE_WAR_BARRIER.override(False):
            self.assertFalse(_barrier_enabled(cuda=False, hip=True, xpu=False))
        with envs.SGLANG_ENABLE_WAR_BARRIER.override(True):
            self.assertTrue(_barrier_enabled(cuda=False, hip=True, xpu=False))


if __name__ == "__main__":
    unittest.main()
