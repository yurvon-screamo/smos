"""Model-free smoke test for ``SMOSClient``.

Exercises the adapter's parsing and orchestration logic end-to-end
(`add` → `finalize_pending` → `search`) WITHOUT a real `smos` binary,
llama-server, or NLI model. The real subprocess call (`_run`) is stubbed with
scripted outputs so the test is deterministic, fast, and cross-platform.

Run:

    python benchmarks/common/test_smos_client.py

Use pytest to run it as part of a suite (`pytest benchmarks/common/`).
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Sequence

from benchmarks.common.smos_client import SMOSClient, sanitize_user_id


class _FakeProc:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


class _ScriptedSMOSClient(SMOSClient):
    """``SMOSClient`` with ``_run`` stubbed to pop scripted outputs."""

    def __init__(self, scripts: dict[str, Sequence[_FakeProc]]) -> None:
        super().__init__(smos_binary="mock-smos", config_path="mock.toml")
        # op -> list of _FakeProc to pop FIFO per call.
        self._scripts = {op: list(queue) for op, queue in scripts.items()}
        self.calls: list[tuple[str, list[str]]] = []

    async def _run(self, smos_args, *, user_id, op, stdin_data=None):  # type: ignore[override]
        self.calls.append((op, list(smos_args)))
        queue = self._scripts.get(op)
        if not queue:
            return None
        return queue.pop(0)


def _import_raw_stdout(memory_key: str, session_id: str) -> str:
    return (
        "INFO surrealdb starting...\n"
        "=== Raw import complete ===\n"
        f"Memory key: {memory_key}\n"
        f"Session:    {session_id}\n"
        "New facts:  3\n"
    )


def _search_stdout(items: list[dict]) -> str:
    return json.dumps(items)


def test_sanitize_user_id() -> None:
    assert sanitize_user_id("beam_100K_0_ab12cd34") == "beam_100K_0_ab12cd34"
    # Strips unsafe chars, keeps the rest.
    assert sanitize_user_id("user@home/evil") == "user_home_evil"
    # Leading non-alphanumeric gets an 'u' prefix.
    assert sanitize_user_id("_leading").startswith("u")
    # '..' (path traversal) is collapsed.
    assert ".." not in sanitize_user_id("a..b")
    # Empty falls back to a safe default.
    assert sanitize_user_id("!!!").startswith("u") or sanitize_user_id("!!!") == "user"


def test_add_parses_memory_key_and_session() -> None:
    client = _ScriptedSMOSClient(
        {
            "add": [
                _FakeProc(
                    _import_raw_stdout("beam_100K_0_ab12cd34", "sess_rawimport1234")
                )
            ]
        }
    )
    result = asyncio.run(
        client.add([{"role": "user", "content": "I like rust"}], "beam_100K_0_ab12cd34")
    )
    assert result == {"results": []}, "add returns the Mem0-shaped success envelope"
    assert client._sessions["beam_100K_0_ab12cd34"] == (
        "beam_100K_0_ab12cd34",
        "sess_rawimport1234",
    ), "Memory key + Session parsed from import-raw stdout"


def test_add_empty_messages_short_circuits() -> None:
    client = _ScriptedSMOSClient({})
    result = asyncio.run(client.add([{"role": "user", "content": "   "}], "u1"))
    assert result == {"results": []}
    assert client.calls == [], "empty text must not spawn a subprocess"


def test_search_parses_json_array() -> None:
    payload = [
        {
            "id": "fact_aaaaaaaaaaaaaaaa",
            "memory": "Rust is memory-safe",
            "score": 0.95,
            "created_at": "2025-06-18T12:00:00Z",
            "confidence": 0.9,
            "status": "accepted",
            "valid_until": None,
            "conflicts_with": [],
            "memory_key": "beam_100K_0_ab12cd34",
        }
    ]
    client = _ScriptedSMOSClient({"search": [_FakeProc(_search_stdout(payload))]})
    results = asyncio.run(
        client.search("what is rust", "beam_100K_0_ab12cd34", top_k=50)
    )
    assert results == payload, "search returns the JSON array verbatim"
    op, args = client.calls[0]
    assert op == "search"
    assert "--top-k" in args and "50" in args
    assert "--person" in args


def test_search_returns_empty_on_failure() -> None:
    client = _ScriptedSMOSClient({"search": [None]})  # type: ignore[dict-item]
    results = asyncio.run(client.search("q", "u1"))
    assert results == []


def test_finalize_pending_dedupes_by_session_id() -> None:
    # Three BEAM users, all sharing the deterministic import-raw session id.
    client = _ScriptedSMOSClient(
        {
            "add": [
                _FakeProc(_import_raw_stdout("u1", "sess_shared00001")),
                _FakeProc(_import_raw_stdout("u2", "sess_shared00001")),
                _FakeProc(_import_raw_stdout("u3", "sess_shared00001")),
            ],
            "finalize": [_FakeProc('{"processed": 9}', 0)],
        }
    )
    asyncio.run(client.add([{"role": "user", "content": "a"}], "u1"))
    asyncio.run(client.add([{"role": "user", "content": "b"}], "u2"))
    asyncio.run(client.add([{"role": "user", "content": "c"}], "u3"))

    results = asyncio.run(client.finalize_pending())
    # Three users share ONE session id → exactly one finalize subprocess.
    assert list(results.keys()) == ["sess_shared00001"]
    assert results["sess_shared00001"] == 0
    finalize_calls = [c for c in client.calls if c[0] == "finalize"]
    assert len(finalize_calls) == 1, "dedupe collapses N users to one finalize"
    assert "--memory-key" not in finalize_calls[0][1], "uses the discovery path"


def test_full_round_trip_add_finalize_search() -> None:
    """The headline smoke: add → finalize_pending → search returns ≥1 fact."""
    fact = {
        "id": "fact_deadbeefdeadbee",
        "memory": "User likes rust",
        "score": 0.9,
        "created_at": "2025-06-18T12:00:00Z",
    }
    client = _ScriptedSMOSClient(
        {
            "add": [_FakeProc(_import_raw_stdout("u1", "sess_roundtrip0001"))],
            "finalize": [_FakeProc('{"processed": 1}', 0)],
            "search": [_FakeProc(_search_stdout([fact]))],
        }
    )

    async def scenario() -> list[dict]:
        await client.add([{"role": "user", "content": "I like rust"}], "u1")
        await client.finalize_pending()
        return await client.search("what does the user like", "u1")

    results = asyncio.run(scenario())
    assert len(results) >= 1, "the round-trip surfaces ≥1 fact through the adapter"
    assert results[0]["memory"] == "User likes rust"
    assert results[0]["created_at"] == "2025-06-18T12:00:00Z"


_TESTS = [
    test_sanitize_user_id,
    test_add_parses_memory_key_and_session,
    test_add_empty_messages_short_circuits,
    test_search_parses_json_array,
    test_search_returns_empty_on_failure,
    test_finalize_pending_dedupes_by_session_id,
    test_full_round_trip_add_finalize_search,
]


def main() -> int:
    failures = 0
    for test in _TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
