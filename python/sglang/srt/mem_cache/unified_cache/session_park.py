"""Tool-call parking for session-aware radix caching (``/park_session``).

An agent that hands its turn to a tool tells the server when it expects to send the
session's next turn. Among session-referenced KV, eviction then takes the KV due back
furthest in the future first (Belady's rule with the client's ETA) instead of the
least recently used, on the device and on the host tier alike.

Only the session is recorded, not its nodes: each eviction walk derives the parked
nodes from the sessions' current frontier leaves, so later turns, branches, forks and
releases are always reflected. A node is parked when every frontier leaf covering it
belongs to a parked session; it takes the earliest resume time among them, and any
active session covering it keeps it protected. A parked session stays parked until its
next request arrives or it is released; an overdue one keeps the earliest resume time,
so it is the last parked KV evicted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode


class SessionParkTracker:
    def __init__(self) -> None:
        self._resume_at: dict[str, float] = {}
        # Bumped whenever the parked set changes, so callers can cache a map.
        self.version = 0

    def park(self, session_id: str, resume_at: float) -> None:
        self._resume_at[session_id] = resume_at
        self.version += 1

    def unpark(self, session_id: str) -> None:
        if self._resume_at.pop(session_id, None) is not None:
            self.version += 1

    def num_parked_sessions(self) -> int:
        return len(self._resume_at)

    def reset(self) -> None:
        self._resume_at.clear()
        self.version += 1

    def parked_resume_at(
        self,
        *,
        session_leaves: Callable[[str], Iterable[UnifiedTreeNode]],
        session_ref: Callable[[UnifiedTreeNode], int],
        root: UnifiedTreeNode,
    ) -> dict[int, float]:
        """Node id -> resume time of every node only parked sessions cover.

        Linear in the distinct nodes on parked paths: each path is walked up only
        to the first node already seen, then counts flow from children to parents.
        """
        if not self._resume_at:
            return {}
        nodes: dict[int, UnifiedTreeNode] = {}
        coverage: dict[int, int] = {}
        earliest: dict[int, float] = {}
        for session_id, resume_at in self._resume_at.items():
            for leaf in session_leaves(session_id):
                if leaf is root:
                    continue
                coverage[leaf.id] = coverage.get(leaf.id, 0) + 1
                earliest[leaf.id] = min(earliest.get(leaf.id, math.inf), resume_at)
                node = leaf
                while node is not root and node.id not in nodes:
                    nodes[node.id] = node
                    node = node.parent
        # Every ancestor of a seen node is seen, so the seen nodes form a forest
        # under root; fold each node into its parent once all its seen children are.
        pending = dict.fromkeys(nodes, 0)
        for node in nodes.values():
            if node.parent is not root:
                pending[node.parent.id] += 1
        ready = [node_id for node_id, count in pending.items() if count == 0]
        while ready:
            node = nodes[ready.pop()]
            parent = node.parent
            if parent is root:
                continue
            coverage[parent.id] = coverage.get(parent.id, 0) + coverage.get(node.id, 0)
            earliest[parent.id] = min(
                earliest.get(parent.id, math.inf), earliest.get(node.id, math.inf)
            )
            pending[parent.id] -= 1
            if pending[parent.id] == 0:
                ready.append(parent.id)
        # session_ref counts covering frontier leaves of every session, so equal
        # counts mean no active session's leaf covers the node.
        return {
            node_id: earliest[node_id]
            for node_id, node in nodes.items()
            if session_ref(node) == coverage.get(node_id, 0)
        }
