"""CPU checks of the gigakernel binding: the view is the running forward's attention
backend and KV pool, and a pool replaced after capture is caught."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.kernel_plugins.intel import binding as gb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

BACKEND = "sglang.srt.model_executor.forward_context.get_attn_backend"


def backend(pool):
    return SimpleNamespace(
        token_to_kv_pool=pool,
        forward_metadata=SimpleNamespace(page_table="pages"),
        forward=lambda *a, **k: "attention",
    )


class TestGigakernelBinding(CustomTestCase):
    def test_the_view_is_the_running_forwards_backend_and_pool(self):
        pool = object()
        with patch(BACKEND, return_value=backend(pool)):
            view = gb.GigakernelBinding(model=torch.nn.Module()).forward_view()
        self.assertEqual(view.metadata.page_table, "pages")
        self.assertIs(view.kv_pool, pool)
        self.assertEqual(view.attention(), "attention")

    def test_a_replaced_pool_fails_assert_live(self):
        binding = gb.GigakernelBinding(model=torch.nn.Module())
        attn_backend = backend(object())
        with patch(BACKEND, return_value=attn_backend):
            binding.forward_view()
            binding.assert_live()
        with patch(BACKEND, return_value=backend(object())):
            with self.assertRaisesRegex(RuntimeError, "kv_pool"):
                binding.assert_live()


if __name__ == "__main__":
    unittest.main()
