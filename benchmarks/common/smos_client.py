"""SMOS client — drop-in async adapter for the BEAM benchmark.

Replaces ``Mem0Client`` so the BEAM harness (``benchmarks/beam/run.py``) drives
SMOS via the CLI instead of the Mem0 HTTP API. The interface mirrors
``Mem0Client`` exactly (``add`` / ``search`` / ``delete_user`` + async context
manager) so ``run.py`` needs no edits outside the client construction.
``smos import raw`` finalizes inline by default, so the harness no longer wires
a separate Phase-1→Phase-2 finalize step (``finalize_pending`` is retained as an
explicit recovery hook only).

Each operation shells out to the unified ``smos`` binary (``import raw``,
``finalize``, ``search``) against a single benchmark config (``SMOS_CONFIG``
env var, default ``smos.bench.toml``). Subprocesses are strictly sequential —
SMOS' RocksDB store takes a single-writer lock at connect time, and the BEAM
``run.py`` loop is itself sequential, so no concurrency is needed (or safe).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from typing import Any

logger = logging.getLogger(__name__)

# Output markers printed by `smos import raw` (see raw_import_runner.rs).
_MEMORY_KEY_RE = re.compile(r"^Memory key:\s*(\S+)", re.MULTILINE)
_SESSION_RE = re.compile(r"^Session:\s*(\S+)", re.MULTILINE)

# MemoryKey validation (mirrors smos-domain::value_objects::memory_key):
# first char ASCII alphanumeric, rest in [A-Za-z0-9_.-], no "..", no path
# separators, length <= 64.
_SAFE_CHAR_RE = re.compile(r"[^A-Za-z0-9_.-]")
_MAX_MEMORY_KEY_LEN = 64


def _default_subprocess_timeout() -> float:
    """Per-subprocess timeout, overridable via ``SMOS_SUBPROCESS_TIMEOUT``.

    Default 1800 s (30 min): ``import raw`` reloads the 643 MB DeBERTa NLI
    model per chunk to run the inline finalize drain, and the explicit
    ``finalize_pending()`` recovery hook re-drains every pending fact across
    every memory_key in one shot — both can run well past a low default.
    Operators tuning for a smoke run can lower it.
    """
    raw = os.getenv("SMOS_SUBPROCESS_TIMEOUT")
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning(
                "invalid SMOS_SUBPROCESS_TIMEOUT=%r (not a number); using default 1800s",
                raw,
            )
    return 1800.0


def sanitize_user_id(user_id: str) -> str:
    """Map a BEAM ``user_id`` to a valid SMOS ``MemoryKey``.

    BEAM ids look like ``beam_100K_0_<runid>`` — already almost valid — but the
    sanitiser is defensive: it strips every char outside ``[A-Za-z0-9_.-]``,
    collapses ``..`` (path-traversal guard), guarantees an alphanumeric lead,
    and truncates to the 64-char MemoryKey ceiling.
    """
    cleaned = _SAFE_CHAR_RE.sub("_", user_id).replace("..", "_")
    if not cleaned:
        cleaned = "user"
    if not cleaned[0].isascii() or not cleaned[0].isalnum():
        cleaned = f"u{cleaned}"
    return cleaned[:_MAX_MEMORY_KEY_LEN]


class SMOSClient:
    """Async SMOS adapter presenting the ``Mem0Client`` interface to BEAM.

    Args:
        mode: Accepted for ``Mem0Client`` parity; ignored (SMOS has one mode).
        host: Accepted for parity; ignored.
        api_key: Accepted for parity; ignored.
        smos_binary: Name or path of the ``smos`` executable on PATH.
        config_path: Path to the SMOS TOML config. Defaults to the
            ``SMOS_CONFIG`` env var, then ``smos.bench.toml``.
        timeout: Per-subprocess timeout in seconds.
    """

    def __init__(
        self,
        mode: str = "oss",
        host: str | None = None,
        api_key: str | None = None,
        organization_id: str | None = None,
        project_id: str | None = None,
        max_retries: int = 5,
        retry_delay: float = 5.0,
        rpm: int = 60,
        timeout: float | None = None,
        event_poll_interval: float = 0.5,
        event_poll_timeout: float = 300.0,
        smos_binary: str | None = None,
        config_path: str | None = None,
    ) -> None:
        # Mem0Client parity kwargs are accepted and deliberately unused: the
        # BEAM harness constructs the client with a fixed kwarg set, and keeping
        # the signature a superset lets run.py swap the class with one edit.
        self.smos_binary = smos_binary or os.getenv("SMOS_BINARY", "smos")
        self.config_path = config_path or os.getenv("SMOS_CONFIG", "smos.bench.toml")
        self.timeout = timeout if timeout is not None else _default_subprocess_timeout()
        # user_id -> (memory_key, session_id) captured from `import raw` stdout.
        self._sessions: dict[str, tuple[str, str]] = {}

    async def __aenter__(self) -> "SMOSClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        # No persistent session to tear down: each operation is a fresh
        # subprocess that releases the RocksDB lock on exit.
        return None

    # ------------------------------------------------------------------
    # Add (= `smos import raw`)
    # ------------------------------------------------------------------

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """Ingest a chunk: concatenate message contents into one ``import raw``.

        Returns ``{"results": []}`` on success (SMOS does not surface the
        extracted facts on stdout — only a count) or ``None`` on failure, so
        the BEAM ingestion counter treats a non-None return as processed.
        """
        memory_key = sanitize_user_id(user_id)
        text = "\n".join(
            msg.get("content", "") for msg in messages if msg.get("content")
        ).strip()
        if not text:
            logger.warning("SMOS add: empty text for user=%s; skipping", user_id)
            return {"results": []}

        # Pipe the chunk text via --stdin so a chunk starting with `-`/`--`
        # is not misparsed by clap as an unknown flag (the positional argv
        # path is fragile; --stdin is robust to any text content).
        proc = await self._run(
            ["import", "raw", "--stdin", "--memory-key", memory_key],
            user_id=user_id,
            op="add",
            stdin_data=text,
        )
        if proc is None:
            return None

        parsed_memory_key, session_id = self._parse_import_summary(proc.stdout)
        if session_id:
            self._sessions[user_id] = (parsed_memory_key or memory_key, session_id)
        return {"results": []}

    # ------------------------------------------------------------------
    # Search (= `smos search`)
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        """Retrieve reranked accepted facts for ``query`` under ``user_id``.

        Returns the SMOS JSON array verbatim (already in descending rerank
        order). The BEAM harness re-sorts via ``format_search_results``, which
        is a no-op here because SMOS' rerank score is already higher=better.
        """
        memory_key = sanitize_user_id(user_id)
        # Pipe the query via --stdin (same hyphen-safety rationale as add()).
        proc = await self._run(
            ["search", "--stdin", "--person", memory_key, "--top-k", str(top_k)],
            user_id=user_id,
            op="search",
            stdin_data=query,
        )
        if proc is None:
            return []

        import json

        stdout = proc.stdout.strip()
        if not stdout:
            return []
        try:
            results = json.loads(stdout)
        except json.JSONDecodeError as exc:
            logger.error(
                "SMOS search returned non-JSON for user=%s: %s; stdout=%.500s",
                user_id,
                exc,
                stdout,
            )
            return []
        if not isinstance(results, list):
            return []
        return results

    # ------------------------------------------------------------------
    # Delete (= no-op; isolation via unique memory_key per BEAM user)
    # ------------------------------------------------------------------

    async def delete_user(self, user_id: str) -> bool:
        logger.info(
            "SMOS delete_user is a no-op (user=%s); each BEAM user isolates "
            "under its own sanitized memory_key",
            user_id,
        )
        return True

    # ------------------------------------------------------------------
    # Finalize — drain pending facts to Accepted between BEAM phases
    # ------------------------------------------------------------------

    async def finalize_pending(self) -> dict[str, int]:
        """Run ``smos finalize`` once per captured session_id.

        Redundant for the normal BEAM flow: since ``smos import raw`` now runs
        finalize inline by default, each chunk's pending facts are already
        promoted to Accepted (and conflicts against the accumulated Accepted
        pool are detected) as ingestion proceeds. Retained as an explicit
        recovery hook for the case where inline finalize was skipped (e.g. an
        ``--no-finalize`` run) or a prior ingest died before the drain.

        ``smos import raw`` derives a deterministic session id, so every BEAM
        user shares ONE session id; the discovery path (no ``--memory-key``)
        then finalises every memory_key that session touched in a single
        subprocess. Returns ``{session_id: exit_code}`` for observability.
        """
        results: dict[str, int] = {}
        # Dedupe by session id: the shared deterministic id means N users
        # collapse to one finalize subprocess.
        unique_sids = {sid for _mk, sid in self._sessions.values()}
        for session_id in unique_sids:
            proc = await self._run(
                ["finalize", session_id],
                user_id=session_id,
                op="finalize",
            )
            results[session_id] = 0 if proc is not None else 1
        if not unique_sids:
            logger.warning(
                "SMOS finalize_pending: no captured sessions; nothing to finalize"
            )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run(
        self,
        smos_args: list[str],
        *,
        user_id: str,
        op: str,
        stdin_data: str | None = None,
    ) -> asyncio.subprocess.Process | None:
        binary = shutil.which(self.smos_binary) or self.smos_binary
        cmd = [binary, *smos_args, "--config", self.config_path]
        logger.debug("SMOS %s: %s (user=%s)", op, " ".join(cmd[:2]), user_id)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error(
                "SMOS binary %r not found on PATH (op=%s); install smos or set "
                "SMOS_BINARY",
                self.smos_binary,
                op,
            )
            return None

        payload = stdin_data.encode("utf-8") if stdin_data is not None else None
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=payload), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            # Reap the killed child so the OS reclaims its resources and the
            # RocksDB lock is released before the next sequential subprocess.
            # Bounded wait: on Windows TerminateProcess is reliable, but a
            # pathological hang here would cascade to the next connect failing.
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("SMOS %s child did not exit 5s after kill", op)
            logger.error(
                "SMOS %s timed out after %ss (user=%s)", op, self.timeout, user_id
            )
            return None

        if proc.returncode != 0:
            logger.error(
                "SMOS %s failed exit=%s (user=%s): %s",
                op,
                proc.returncode,
                user_id,
                stderr_b.decode(errors="replace")[:500],
            )
            return None

        # Attach decoded stdout for the caller; stderr is surfaced only on
        # error above (SMOS logs RocksDB/NLI chatter to stderr).
        proc.stdout = stdout_b.decode(errors="replace")
        return proc

    @staticmethod
    def _parse_import_summary(stdout: str) -> tuple[str | None, str | None]:
        mk_match = _MEMORY_KEY_RE.search(stdout)
        sid_match = _SESSION_RE.search(stdout)
        memory_key = mk_match.group(1) if mk_match else None
        session_id = sid_match.group(1) if sid_match else None
        return memory_key, session_id
