"""Only an XPU host pool split across pinned allocations is refused L3 storage.

Every other anchor or side pool, including ones that are not HostKVCache
(DeepSeek-V4's LogicalHostPool) and asymmetric K/V pools whose buffer has
always been a (k, v) tuple, must still attach a storage backend.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host.mha import (
    AsymmetricMHATokenToKVPoolHost,
    MHATokenToKVPoolHost,
    refuse_split_host_pool_for_storage,
)
from sglang.srt.mem_cache.pool_host.page_interleave import PageInterleaveHostPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSplitHostPoolStorageGuard(CustomTestCase):
    def test_pools_that_are_not_split_attach(self):
        for pool in (
            SimpleNamespace(layout="page_first"),
            LogicalHostPool.__new__(LogicalHostPool),
            AsymmetricMHATokenToKVPoolHost.__new__(AsymmetricMHATokenToKVPoolHost),
            MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost),
        ):
            with self.subTest(pool=type(pool).__name__):
                refuse_split_host_pool_for_storage(pool)

    def test_split_pool_is_refused(self):
        pool = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        pool.split_across_allocations = True
        wrapped = PageInterleaveHostPool.__new__(PageInterleaveHostPool)
        wrapped.host_pool = pool
        for candidate in (pool, wrapped):
            with self.subTest(pool=type(candidate).__name__):
                with self.assertRaisesRegex(
                    ValueError, "split across pinned allocations"
                ):
                    refuse_split_host_pool_for_storage(candidate)


if __name__ == "__main__":
    unittest.main()
