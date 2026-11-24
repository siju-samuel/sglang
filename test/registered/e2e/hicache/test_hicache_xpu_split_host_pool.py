"""HiCache on Intel XPU with a host pool larger than one pinned allocation.

One pinned buffer tops out at 16 GiB on an Arc Pro B60, so an 18 GB layer_first
pool is carved into per-layer views of several pinned chunks. KV reloaded from
those chunks must reproduce the device-tier generation.
"""

import tempfile
import unittest

import psutil

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.hicache_xpu_common import (
    COMPARE_TOKENS,
    EVICT_DEVICE_POOL_TOKENS,
    XPU_AVAILABLE,
    complete,
    load_back_tokens,
    prime_cache,
    resolve_base_url,
    shared_prefix_len,
    wait_load_back_tokens,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_xpu_ci(est_time=200, suite="extra-test-1-gpu-xpu")

HOST_POOL_GB = 18


@unittest.skipUnless(XPU_AVAILABLE, "Intel XPU not available")
@unittest.skipUnless(
    psutil.virtual_memory().available > 2 * HOST_POOL_GB << 30,
    f"needs {HOST_POOL_GB} GB of pinnable host memory with headroom",
)
class TestHiCacheXpuSplitHostPool(CustomTestCase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = resolve_base_url()
        cls.stdout = tempfile.NamedTemporaryFile("w+", suffix=".log")
        cls.stderr = tempfile.NamedTemporaryFile("w+", suffix=".log")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-hierarchical-cache",
                "--mem-fraction-static",
                0.4,
                "--max-total-tokens",
                EVICT_DEVICE_POOL_TOKENS,
                "--hicache-size",
                HOST_POOL_GB,
                "--page-size",
                16,
                "--hicache-io-backend",
                "kernel",
                "--hicache-mem-layout",
                "layer_first",
                "--enable-cache-report",
                "--enable-metrics",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
        cls.stdout.close()
        cls.stderr.close()

    def test_pool_is_split(self):
        # Read by name: the server's output threads still write through the handles.
        logs = ""
        for f in (self.stdout, self.stderr):
            with open(f.name) as log:
                logs += log.read()
        self.assertIn("XPU host KV pool split into", logs)

    def test_repeat_after_eviction_agrees(self):
        prompt = (
            "The answer to life, the universe, and everything has been debated "
            "by philosophers, novelists, and astronomers alike. " * 8
        ) + " The most widely cited answer is"
        prime_cache(self.base_url, self.model, prompt, max_tokens=COMPARE_TOKENS)
        first, first_cached = complete(
            self.base_url,
            self.model,
            prompt,
            max_tokens=COMPARE_TOKENS,
            want_cached=True,
        )
        # Filler well past the 1024-token device pool evicts the prompt's pages.
        for i in range(10):
            complete(
                self.base_url,
                self.model,
                f"Unrelated filler request {i}: " + "lorem ipsum " * 60,
                max_tokens=16,
            )
        loaded_before = load_back_tokens(self.base_url)
        second, cached = complete(
            self.base_url,
            self.model,
            prompt,
            max_tokens=COMPARE_TOKENS,
            want_cached=True,
        )
        loaded_after = wait_load_back_tokens(self.base_url, loaded_before)
        self.assertGreater(
            loaded_after,
            loaded_before,
            "the repeat was not loaded back from the host tier",
        )
        self.assertEqual(cached, first_cached)
        self.assertEqual(
            second,
            first,
            f"repeat diverged after {shared_prefix_len(second, first)} chars",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
