"""
Thin wrapper around the Together Sandbox SDK.

Together provides cloud-based sandboxed VMs for executing code, file operations,
and port forwarding. Each sandbox is backed by a *snapshot* — an immutable disk
image built from a Dockerfile context.

Requires:
    pip install together-sandbox
    export TOGETHER_API_KEY=...

Optional environment variables:
    TOGETHER_LOCAL_BUILD=1   build the Dockerfile locally and upload, instead
                             of building remotely on Together's infrastructure
    TOGETHER_BASE_URL=...    override the management API endpoint

Snapshot caching
----------------
Sandboxes are created via a content-addressed alias on Together's snapshot
registry. Calling ``create()`` for a Dockerfile context that already has a
registered snapshot reuses it instead of rebuilding — this is the single most
important optimization for RL training, where many sandboxes per epoch share
the same task image.

Lifetime
--------
Together has no native VM lifetime cap, so a watchdog coroutine forces
``shutdown()`` after ``timeout`` seconds. Always call ``cleanup()`` explicitly
to release resources promptly and cancel the watchdog.

See: https://github.com/togethercomputer/together-sandbox
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from pathlib import Path
from typing import Any

try:
    import together_sandbox as ts
except ImportError:
    raise ImportError(
        "together-sandbox is required for TogetherSandboxWrapper. "
        "Install it with: uv pip install 'tinker-cookbook[together] @ "
        "git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'"
    ) from None

from tinker_cookbook.sandbox.sandbox_interface import (
    SandboxResult,
    SandboxTerminatedError,
)

logger = logging.getLogger(__name__)


# Cached snapshot aliases live under this prefix on Together's registry.
SNAPSHOT_ALIAS_PREFIX = "tinker-cookbook-harbor"

# Locks for de-duplicating concurrent snapshot builds within a single process.
# When multiple sandboxes for the same task spin up in parallel (e.g. group_size
# rollouts), only the first one builds; the others wait and reuse the result.
_snapshot_build_locks: dict[str, asyncio.Lock] = {}
_snapshot_build_locks_mu = asyncio.Lock()


def _hash_build_context(env_dir: Path) -> str:
    """Hash all files under ``env_dir`` to produce a stable snapshot alias.

    Walks files recursively in sorted order, hashing both relative path and
    bytes so that any change to the build context invalidates the cache.
    """
    h = hashlib.sha256()
    for path in sorted(env_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(env_dir).as_posix().encode()
        h.update(rel)
        h.update(b"\x00")
        h.update(path.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _is_sandbox_terminated(exc: BaseException) -> bool:
    """Check if an exception indicates the sandbox has died."""
    if isinstance(exc, ts.HttpError) and getattr(exc, "status", None) in (404, 410):
        return True
    msg = str(exc).lower()
    return any(keyword in msg for keyword in ("terminated", "died", "not found"))


async def _ensure_snapshot(sdk: "ts.TogetherSandbox", env_dir: Path, alias: str) -> str:
    """Look up a snapshot by alias; build one if it doesn't exist.

    Concurrent calls for the same alias serialize on an in-process lock so we
    don't issue duplicate builds in the same Python process.
    """
    async with _snapshot_build_locks_mu:
        lock = _snapshot_build_locks.setdefault(alias, asyncio.Lock())

    async with lock:
        # Re-check after acquiring the lock — another coroutine may have built it.
        try:
            existing = await sdk.snapshots.get_by_alias(alias)
            logger.info(
                "Reusing cached Together snapshot %s for alias %s", existing.id, alias
            )
            return existing.id
        except Exception:
            pass  # Alias not found — fall through to build

        dockerfile = env_dir / "Dockerfile"
        logger.info("Building Together snapshot for %s (alias=%s)", env_dir, alias)
        result = await sdk.snapshots.create(
            ts.CreateContextSnapshotParams(
                context=str(env_dir),
                dockerfile=str(dockerfile),
                alias=alias,
                on_progress=lambda event: logger.debug(
                    "snapshot build: %s", getattr(event, "output", event)
                ),
            )
        )
        return result.snapshot_id


class TogetherSandboxWrapper:
    """
    Persistent Together sandbox for code execution. Conforms to ``SandboxInterface``.

    Usage:
        sandbox = await TogetherSandboxWrapper.create(env_dir=path, timeout=600)
        await sandbox.write_file("/workspace/code.py", "print('hello')")
        result = await sandbox.run_command("python /workspace/code.py")
        await sandbox.cleanup()
    """

    def __init__(
        self,
        sdk: "ts.TogetherSandbox",
        sandbox: Any,
        sandbox_id: str,
        snapshot_id: str,
        timeout: int,
        max_stream_output_bytes: int = 128 * 1024,
    ) -> None:
        self._sdk = sdk
        self._sandbox = sandbox
        self._sandbox_id = sandbox_id
        self._snapshot_id = snapshot_id
        self._timeout = timeout
        self._max_stream_output_bytes = max_stream_output_bytes
        self._closed = False
        self._watchdog: asyncio.Task[None] | None = None

    @classmethod
    async def create(
        cls,
        env_dir: Path,
        timeout: int = 600,
        cpu_millicores: int | None = None,
        memory_mb: int | None = None,
        disk_gb: int | None = None,
        max_stream_output_bytes: int = 128 * 1024,
    ) -> TogetherSandboxWrapper:
        """Create a new Together sandbox from a Dockerfile build context.

        Args:
            env_dir: Directory containing ``Dockerfile`` and build context.
            timeout: Sandbox lifetime in seconds (enforced by an in-process watchdog).
            cpu_millicores: CPU allocation. ``None`` uses Together's default (1000mC).
            memory_mb: Memory allocation in MB. ``None`` uses Together's default (2048).
            disk_gb: Disk allocation in GB. ``None`` uses Together's default (10).
            max_stream_output_bytes: Default cap for command output streams.
        """
        dockerfile = env_dir / "Dockerfile"
        if not dockerfile.is_file():
            raise FileNotFoundError(
                f"Expected Dockerfile at {dockerfile}; "
                "Together backend requires environment/Dockerfile."
            )

        sdk = ts.TogetherSandbox()  # reads TOGETHER_API_KEY from env
        alias = f"{SNAPSHOT_ALIAS_PREFIX}-{_hash_build_context(env_dir)}"
        snapshot_id = await _ensure_snapshot(sdk, env_dir, alias)

        # Only include resource overrides the caller specified, so we accept
        # Together's defaults otherwise.
        create_kwargs: dict[str, Any] = {"snapshot_id": snapshot_id}
        if cpu_millicores is not None:
            create_kwargs["cpu"] = cpu_millicores
        if memory_mb is not None:
            create_kwargs["memory"] = memory_mb
        if disk_gb is not None:
            create_kwargs["disk"] = disk_gb

        sandbox_model = await sdk.sandboxes.create(**create_kwargs)
        sandbox = await sdk.sandboxes.start(sandbox_model.id)

        wrapper = cls(
            sdk=sdk,
            sandbox=sandbox,
            sandbox_id=sandbox_model.id,
            snapshot_id=snapshot_id,
            timeout=timeout,
            max_stream_output_bytes=max_stream_output_bytes,
        )
        wrapper._watchdog = asyncio.create_task(wrapper._auto_shutdown())
        return wrapper

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    async def _auto_shutdown(self) -> None:
        """Force shutdown after ``timeout`` seconds — Together has no native VM timeout."""
        try:
            await asyncio.sleep(self._timeout)
        except asyncio.CancelledError:
            return
        if self._closed:
            return
        logger.warning(
            "Together sandbox %s exceeded lifetime of %ds — forcing shutdown",
            self._sandbox_id,
            self._timeout,
        )
        with contextlib.suppress(Exception):
            await self.cleanup()

    async def send_heartbeat(self, timeout: int = 30) -> None:
        """No-op heartbeat: Together has no explicit liveness check, so we run ``true``."""
        try:
            await asyncio.wait_for(
                self._sandbox.execs.exec("true", []),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            raise

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Run a shell command in the sandbox.

        Together returns combined stdout+stderr as a single ``output`` field;
        we place it in ``SandboxResult.stdout`` and leave ``stderr`` empty.
        Apply ``max_output_bytes`` as a post-hoc truncation.
        """
        cap = (
            max_output_bytes
            if max_output_bytes is not None
            else self._max_stream_output_bytes
        )
        try:
            result = await asyncio.wait_for(
                self._sandbox.execs.exec(
                    "bash",
                    ["-lc", command],
                    cwd=workdir,
                ),
                timeout=timeout,
            )
            output = result.get("output", "") or ""
            if cap is not None and len(output) > cap:
                output = output[:cap]
            return SandboxResult(
                stdout=output,
                stderr="",
                exit_code=int(result.get("exit_code", -1) or -1),
            )
        except asyncio.TimeoutError:
            return SandboxResult(
                stdout="", stderr=f"command timed out after {timeout}s", exit_code=-1
            )
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        """Read a file from the sandbox."""
        try:
            content = await asyncio.wait_for(
                self._sandbox.files.read(path), timeout=timeout
            )
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            if max_bytes is not None and len(content) > max_bytes:
                content = content[:max_bytes]
            return SandboxResult(stdout=content, stderr="", exit_code=0)
        except asyncio.TimeoutError:
            return SandboxResult(
                stdout="", stderr=f"read_file timed out after {timeout}s", exit_code=-1
            )
        except ts.HttpError as e:
            if getattr(e, "status", None) == 404:
                return SandboxResult(
                    stdout="", stderr=f"file not found: {path}", exit_code=1
                )
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)

    async def write_file(
        self,
        path: str,
        content: str | bytes = "",
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        """Write content to a file in the sandbox.

        Together's files API has no ``executable`` flag, so when requested we
        follow up with ``chmod +x`` via exec.
        """
        try:
            await asyncio.wait_for(
                self._sandbox.files.create(path, content),
                timeout=timeout,
            )
            if executable:
                result = await asyncio.wait_for(
                    self._sandbox.execs.exec("chmod", ["+x", path]),
                    timeout=timeout,
                )
                exit_code = int(result.get("exit_code", 0) or 0)
                if exit_code != 0:
                    return SandboxResult(
                        stdout="",
                        stderr=result.get("output", "") or "",
                        exit_code=exit_code,
                    )
            return SandboxResult(stdout="", stderr="", exit_code=0)
        except asyncio.TimeoutError:
            return SandboxResult(
                stdout="", stderr=f"write_file timed out after {timeout}s", exit_code=-1
            )
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def cleanup(self) -> None:
        """Shut down the sandbox VM and cancel the lifetime watchdog.

        Snapshots are intentionally *not* deleted — they are cached for reuse
        by sibling sandboxes (same task, different rollout) and across runs.
        """
        if self._closed:
            return
        self._closed = True

        if self._watchdog is not None and not self._watchdog.done():
            self._watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog

        with contextlib.suppress(Exception):
            await self._sdk.sandboxes.shutdown(self._sandbox_id)
