"""A parked session that is due back last loses its KV before one due back soon.

Under the same memory pressure, plain session eviction takes the least recently
used session instead, so both orders are checked on one live server, for
radix-native (`session_id`) and for `/open_session` sessions.
"""

import random
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=70, stage="extra-a", runner_config="1-gpu-small")

POOL_TOKENS = 4096
SESSION_TOKENS = 1024
# Three newer 768-token sessions push the pool past POOL_TOKENS once the two
# 1024-token sessions are cached, so exactly one of the two must go.
FILLER_TOKENS = 768
NUM_FILLERS = 3


def _tokens(rng: random.Random, n: int) -> list[int]:
    # Ordinary ids, inside any small test model's vocabulary.
    return [rng.randrange(1000, 30000) for _ in range(n)]


class TestSessionParkEviction(CustomTestCase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-session-radix-cache",
                "--max-total-tokens",
                POOL_TOKENS,
                "--mem-fraction-static",
                0.5,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)

    def _post(self, path: str, body: dict) -> requests.Response:
        resp = requests.post(self.base_url + path, json=body, timeout=120)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp

    def _turn(self, session: dict, input_ids: list[int]) -> dict:
        """One turn; an /open_session session sends only the new tokens."""
        body = {
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        }
        if session["opened"]:
            body["session_params"] = {"id": session["id"], "rid": session["rid"]}
        else:
            body["session_id"] = session["id"]
            body["input_ids"] = session["history"] + input_ids
        out = self._post("/generate", body).json()
        session["rid"] = out["meta_info"]["id"]
        session["history"] = body["input_ids"] + out["output_ids"]
        return out

    def _open(self, sid: str, opened: bool) -> dict:
        if opened:
            self.assertEqual(
                self._post(
                    "/open_session", {"session_id": sid, "capacity_of_str_len": 1000}
                ).json(),
                sid,
            )
        return {"id": sid, "opened": opened, "rid": None, "history": []}

    def _resume_cached_tokens(
        self, *, tag: str, park: bool, survivor: str, opened: bool
    ) -> dict[str, int]:
        """Cache sessions "soon" then "late", add pressure, return each resume's hit.

        The expected survivor resumes first: the other one's recompute allocates
        a session's worth of KV and could otherwise evict the survivor too.
        """
        rng = random.Random(0)
        self._post("/flush_cache", {})
        sessions = {
            name: self._open(f"{tag}-{name}", opened) for name in ("soon", "late")
        }
        for session in sessions.values():
            self._turn(session, _tokens(rng, SESSION_TOKENS))
        if park:
            for name, eta_s in (("soon", 5), ("late", 600)):
                self._post(
                    "/park_session",
                    {"session_id": sessions[name]["id"], "eta_s": eta_s},
                )
        fillers = [self._open(f"{tag}-filler{i}", False) for i in range(NUM_FILLERS)]
        for filler in fillers:
            self._turn(filler, _tokens(rng, FILLER_TOKENS))
        cached = {}
        other = "soon" if survivor == "late" else "late"
        out = self._turn(sessions[survivor], _tokens(rng, 32))
        cached[survivor] = out["meta_info"]["cached_tokens"]
        if park:
            # That resume unparked the survivor, so a filler that needs more than
            # the survivor's new tail must push out older active KV instead.
            fillers.append(self._open(f"{tag}-filler-extra", False))
            self._turn(fillers[-1], _tokens(rng, 2 * FILLER_TOKENS))
            out = self._turn(sessions[survivor], _tokens(rng, 32))
            cached["survivor_again"] = out["meta_info"]["cached_tokens"]
        out = self._turn(sessions[other], _tokens(rng, 32))
        cached[other] = out["meta_info"]["cached_tokens"]
        for session in [*sessions.values(), *fillers]:
            self._post("/close_session", {"session_id": session["id"]})
        return cached

    def _check(self, cached: dict[str, int], *, kept: str, lost: str) -> None:
        self.assertGreaterEqual(cached[kept], SESSION_TOKENS - 128, cached)
        self.assertLess(cached[lost], SESSION_TOKENS // 2, cached)

    def test_unparked_sessions_lose_the_oldest_first(self):
        for opened in (False, True):
            with self.subTest(opened=opened):
                cached = self._resume_cached_tokens(
                    tag=f"lru{opened:d}", park=False, survivor="late", opened=opened
                )
                self._check(cached, kept="late", lost="soon")

    def test_parked_session_due_last_loses_its_kv_first(self):
        for opened in (False, True):
            with self.subTest(opened=opened):
                cached = self._resume_cached_tokens(
                    tag=f"park{opened:d}", park=True, survivor="soon", opened=opened
                )
                self._check(cached, kept="soon", lost="late")
                self.assertGreaterEqual(
                    cached["survivor_again"], SESSION_TOKENS - 128, cached
                )


if __name__ == "__main__":
    unittest.main()
