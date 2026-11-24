"""Tests for session references on UnifiedRadixCache."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import random
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import tree_slots_of


def make_params(enable_session: bool, size: int = 64) -> CacheInitParams:
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=size,
        page_size=1,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=size,
        dtype=dtype,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=8,
        max_context_len=128,
        device="cpu",
        enable_memory_saver=False,
    )
    return CacheInitParams(
        disable=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        eviction_policy="lru",
        enable_session_radix_cache=enable_session,
        tree_slots=tree_slots_of((ComponentType.FULL,)),
    )


def insert(cache, token_ids):
    """Insert and return the tail node; the cache boundary hands back a NodeId."""
    indices = cache.token_to_kv_pool_allocator.alloc(len(token_ids))
    node_id = cache.insert(
        InsertParams(
            key=RadixKey(array("q", token_ids)),
            value=indices.to(torch.int64),
        )
    ).last_device_node
    return cache.tree_core.node_by_id(node_id)


def match_len(cache, token_ids) -> int:
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", token_ids)))
    ).device_prefix_len


def register(cache, token_ids, session_id, generation=None):
    if generation is None:
        generation = cache.ensure_session_generation(session_id)
    leaf = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", token_ids)))
    ).last_device_node
    cache.session_refs.register_session_ref(
        SimpleNamespace(
            session_id=session_id,
            session_generation=generation,
            session=None,
            origin_input_ids=array("q", token_ids),
            output_ids=array("q"),
            extra_key=None,
        ),
        leaf=leaf,
    )


class TestSessionUnifiedRadixCache(CustomTestCase):
    def setUp(self):
        self.cache = UnifiedRadixCache(make_params(enable_session=True))
        self.full = self.cache.components[ComponentType.FULL]

    def test_explicit_rust_selection_uses_python_session_semantics(self):
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("rust"):
            cache = UnifiedRadixCache(make_params(enable_session=True))
        self.assertEqual(cache._tree_core_backend, "python")
        self.assertIsInstance(cache.tree_core, UnifiedTreeCore)
        leaf = insert(cache, [1, 2, 3, 4])
        generation = cache.open_radix_session("fallback")
        register(cache, [1, 2, 3, 4], "fallback", generation)
        full = cache.components[ComponentType.FULL]
        self.assertEqual(full.session_ref(leaf), 1)
        cache.release_radix_session("fallback")
        self.assertEqual(full.session_ref(leaf), 0)
        cache.sanity_check()

    def test_register_and_release_update_full_component_reference(self):
        leaf = insert(self.cache, [1, 2, 3, 4])
        generation = self.cache.open_radix_session("s1")

        register(self.cache, [1, 2, 3, 4], "s1", generation)
        self.assertEqual(self.full.session_ref(leaf), 1)

        self.cache.release_radix_session("s1")
        self.assertEqual(self.full.session_ref(leaf), 0)

    def test_reopen_rejects_stale_generation(self):
        leaf = insert(self.cache, [1, 2, 3, 4])
        old_generation = self.cache.open_radix_session("s1")
        self.cache.release_radix_session("s1")
        self.cache.open_radix_session("s1")

        register(self.cache, [1, 2, 3, 4], "s1", old_generation)

        self.assertEqual(self.full.session_ref(leaf), 0)

    def test_eviction_prefers_unreferenced_full_kv(self):
        referenced = insert(self.cache, [1, 2, 3, 4])
        insert(self.cache, [7, 8, 9])
        register(self.cache, [1, 2, 3, 4], "s1")

        self.cache.evict(EvictParams(num_tokens=3))

        self.assertEqual(match_len(self.cache, [7, 8, 9]), 0)
        self.assertEqual(match_len(self.cache, [1, 2, 3, 4]), 4)
        self.assertEqual(self.full.session_ref(referenced), 1)

    def _two_sessions(self):
        # "soon" is inserted first, so plain LRU evicts it before "late".
        insert(self.cache, [1, 2, 3, 4])
        register(self.cache, [1, 2, 3, 4], "soon")
        insert(self.cache, [5, 6, 7, 8])
        register(self.cache, [5, 6, 7, 8], "late")

    def _parked_resume_at(self) -> dict[int, float]:
        return self.cache.session_park.parked_resume_at(
            session_leaves=self.full.session_leaves,
            session_ref=self.full.session_ref,
            root=self.cache.tree_core.root_node,
        )

    def test_unparked_sessions_evict_in_lru_order(self):
        self._two_sessions()
        self.cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(match_len(self.cache, [1, 2, 3, 4]), 0)
        self.assertEqual(match_len(self.cache, [5, 6, 7, 8]), 4)

    def test_parked_session_returning_last_is_evicted_first(self):
        self._two_sessions()
        self.assertTrue(self.cache.park_session("soon", resume_at=100.0))
        self.assertTrue(self.cache.park_session("late", resume_at=200.0))
        self.cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(match_len(self.cache, [1, 2, 3, 4]), 4)
        self.assertEqual(match_len(self.cache, [5, 6, 7, 8]), 0)
        self.cache.sanity_check()

    def test_parked_session_goes_before_active_session(self):
        self._two_sessions()
        self.cache.park_session("late", resume_at=200.0)
        self.cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(match_len(self.cache, [1, 2, 3, 4]), 4)
        self.assertEqual(match_len(self.cache, [5, 6, 7, 8]), 0)

    def test_park_of_unknown_session_records_nothing(self):
        self.assertFalse(self.cache.park_session("nobody", resume_at=100.0))
        self.assertEqual(self.cache.session_park.num_parked_sessions(), 0)

    def test_branched_session_parks_its_whole_path(self):
        # Two frontier leaves of "late" put session_ref 2 on its own trunk.
        insert(self.cache, [1, 2, 3, 4, 5, 6])
        register(self.cache, [1, 2, 3, 4, 5, 6], "late")
        insert(self.cache, [1, 2, 3, 4, 7, 8])
        register(self.cache, [1, 2, 3, 4, 7, 8], "late")
        insert(self.cache, [9, 10, 11, 12])
        register(self.cache, [9, 10, 11, 12], "soon")
        self.cache.park_session("late", resume_at=900.0)
        self.cache.park_session("soon", resume_at=100.0)
        self.cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(match_len(self.cache, [1, 2, 3, 4, 5, 6]), 0)
        self.assertEqual(match_len(self.cache, [9, 10, 11, 12]), 4)
        self.cache.sanity_check()

    def test_prefix_shared_with_an_active_session_is_not_parked(self):
        a_tail = insert(self.cache, [1, 2, 3, 4, 5])
        insert(self.cache, [1, 2, 3, 6, 7])
        register(self.cache, [1, 2, 3, 4, 5], "a")
        register(self.cache, [1, 2, 3, 6, 7], "b")
        self.cache.park_session("a", resume_at=100.0)
        self.assertEqual(self._parked_resume_at(), {a_tail.id: 100.0})

    def test_prefix_shared_only_by_parked_sessions_takes_the_earliest_resume(self):
        a_tail = insert(self.cache, [1, 2, 3, 4, 5])
        b_tail = insert(self.cache, [1, 2, 3, 6, 7])
        register(self.cache, [1, 2, 3, 4, 5], "a")
        register(self.cache, [1, 2, 3, 6, 7], "b")
        self.cache.park_session("a", resume_at=100.0)
        self.cache.park_session("b", resume_at=200.0)
        parked = self._parked_resume_at()
        self.assertEqual(parked.pop(a_tail.id), 100.0)
        self.assertEqual(parked.pop(b_tail.id), 200.0)
        self.assertEqual(list(parked.values()), [100.0])

    def test_turn_registered_after_park_is_parked(self):
        # A park can arrive before the turn that triggered it finishes.
        insert(self.cache, [7, 8, 9, 10])
        register(self.cache, [7, 8, 9, 10], "active")
        insert(self.cache, [1, 2, 3, 4])
        register(self.cache, [1, 2, 3, 4], "s")
        self.cache.park_session("s", resume_at=900.0)
        insert(self.cache, [1, 2, 3, 4, 5, 6])
        register(self.cache, [1, 2, 3, 4, 5, 6], "s")
        self.cache.evict(EvictParams(num_tokens=6))
        self.assertEqual(match_len(self.cache, [1, 2, 3, 4, 5, 6]), 0)
        self.assertEqual(match_len(self.cache, [7, 8, 9, 10]), 4)

    def test_released_session_leaves_no_park_behind(self):
        # A release, whether a close or the session reaper, unparks; a new
        # session on the same tokens is then ordered as active.
        self._two_sessions()
        self.cache.park_session("late", resume_at=200.0)
        self.cache.release_radix_session("late")
        self.assertEqual(self.cache.session_park.num_parked_sessions(), 0)
        register(self.cache, [5, 6, 7, 8], "successor")
        self.cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(match_len(self.cache, [1, 2, 3, 4]), 0)
        self.assertEqual(match_len(self.cache, [5, 6, 7, 8]), 4)

    def test_parked_map_matches_a_fresh_derivation_after_every_change(self):
        """A rebuilt parked map must equal one derived from scratch (a node is parked
        iff every frontier leaf covering it belongs to a parked session), and the
        cached map may only lag on the safe side: never park a node the derivation
        does not, never give one a later resume time."""
        cache = UnifiedRadixCache(make_params(enable_session=True, size=512))
        full = cache.components[ComponentType.FULL]
        root = cache.tree_core.root_node
        sessions = [f"s{i}" for i in range(5)]
        resume_at: dict[str, float] = {}
        rng = random.Random(0)

        def derived() -> dict[int, float]:
            leaves = {sid: full.session_leaves(sid) for sid in sessions}
            out, stack = {}, list(root.children.values())
            while stack:
                node = stack.pop()
                stack.extend(node.children.values())
                cover = []
                for sid, sid_leaves in leaves.items():
                    for leaf in sid_leaves:
                        walk = leaf
                        while walk is not root and walk is not node:
                            walk = walk.parent
                        if walk is node:
                            cover.append(sid)
                if cover and all(sid in resume_at for sid in cover):
                    out[node.id] = min(resume_at[sid] for sid in cover)
            return out

        def insert_new(tokens: list[int]) -> None:
            # Free the matched prefix's duplicate slots, as a finished request does.
            if cache.token_to_kv_pool_allocator.available_size() < 2 * len(tokens):
                cache.evict(EvictParams(num_tokens=4 * len(tokens)))
            indices = cache.token_to_kv_pool_allocator.alloc(len(tokens))
            prefix = cache.insert(
                InsertParams(
                    key=RadixKey(array("q", tokens)), value=indices.to(torch.int64)
                )
            ).prefix_len
            cache.token_to_kv_pool_allocator.free(indices[:prefix])

        for _ in range(600):
            op = rng.random()
            sid = rng.choice(sessions)
            tokens = [rng.randrange(3)] + [rng.randrange(1, 6) for _ in range(4)]
            tokens += [rng.randrange(100) for _ in range(rng.randrange(1, 4))]
            if op < 0.35:
                insert_new(tokens)
                register(cache, tokens, sid)
            elif op < 0.45:
                # A request without a session can split a parked node.
                insert_new(tokens[: rng.randrange(2, len(tokens) + 1)])
            elif op < 0.65:
                due = float(rng.randrange(1000))
                if cache.park_session(sid, due):
                    resume_at[sid] = due
            elif op < 0.75:
                resume_at.pop(sid, None)
                cache.unpark_session(sid)
            elif op < 0.85:
                resume_at.pop(sid, None)
                cache.release_radix_session(sid)
                cache.open_radix_session(sid)
            else:
                cache.evict(EvictParams(num_tokens=rng.randrange(1, 33)))
            truth = derived()
            fresh = cache.session_park.parked_resume_at(
                session_leaves=full.session_leaves,
                session_ref=full.session_ref,
                root=root,
            )
            self.assertEqual(fresh, truth)
            full._refresh_parked_resume_at()
            live = {node.id for node in self._walk(root)}
            for node_id, due in full._parked_resume_at.items():
                if node_id in live:
                    self.assertIn(node_id, truth)
                    self.assertLessEqual(due, truth[node_id])

    @staticmethod
    def _walk(root):
        stack = list(root.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            yield node

    def test_unpark_restores_lru_order(self):
        self._two_sessions()
        self.cache.park_session("late", resume_at=200.0)
        self.cache.unpark_session("late")
        self.cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(match_len(self.cache, [1, 2, 3, 4]), 0)
        self.assertEqual(match_len(self.cache, [5, 6, 7, 8]), 4)


if __name__ == "__main__":
    unittest.main()
