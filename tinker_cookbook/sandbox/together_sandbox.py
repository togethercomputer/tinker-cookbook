"""
Thin wrapper around the Together Sandbox SDK.

Together provides cloud-based sandboxed VMs for executing code, file operations,
and port forwarding. Each sandbox is backed by a *snapshot* — an immutable disk
image built from a Dockerfile context.

Requires:
    pip install 'together-sandbox>=4.0.3'
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
the same task image. Builds additionally pass a stable ``cache_key`` so the
remote builder can reuse layers across rebuilds.

Lifetime
--------
Sandboxes are created with a server-side ``ttl``, so they are reclaimed even if
the training process dies. Always call ``cleanup()`` explicitly to release
resources promptly rather than waiting for the TTL.

See: https://github.com/togethercomputer/together-sandbox
"""

from __future__ import annotations

import asyncio
import contextlib
import getpass
import hashlib
import logging
import os
import posixpath
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


def _default_tags() -> dict[str, str]:
    """Tags attached to every sandbox and snapshot, for cost attribution.

    ``user`` defaults to the OS login name; override it with
    ``TINKER_SANDBOX_USER`` when running under a shared service account.
    """
    user = os.environ.get("TINKER_SANDBOX_USER") or getpass.getuser()
    return {"user": user, "job": "tinker", "component": "tinker-cookbook"}


# Locks for de-duplicating concurrent snapshot builds within a single process.
# When multiple sandboxes for the same task spin up in parallel (e.g. group_size
# rollouts), only the first one builds; the others wait and reuse the result.
_snapshot_build_locks: dict[str, asyncio.Lock] = {}
_snapshot_build_locks_mu = asyncio.Lock()

# One management-API client per process. Each client owns an httpx connection
# pool, so creating one per sandbox would leak sockets across an RL run.
_sdk: ts.TogetherSandbox | None = None
_sdk_mu = asyncio.Lock()


async def _get_sdk() -> ts.TogetherSandbox:
    """Return the process-wide SDK client, creating it on first use."""
    global _sdk
    async with _sdk_mu:
        if _sdk is None:
            _sdk = ts.TogetherSandbox()  # reads TOGETHER_API_KEY from env
        return _sdk


async def close_sdk() -> None:
    """Close the process-wide SDK client. Safe to call more than once."""
    global _sdk
    async with _sdk_mu:
        if _sdk is not None:
            with contextlib.suppress(Exception):
                await _sdk.close()
            _sdk = None


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
    if isinstance(exc, ts.HttpError) and exc.status in (404, 410):
        return True
    msg = str(exc).lower()
    return any(
        keyword in msg
        for keyword in (
            "terminated",
            "died",
            "not found",
            # SDK raises this when an exec's SSE stream ends without an exit
            # code, which happens when the VM goes away mid-command.
            "stream ended without an exit code",
        )
    )


async def _ensure_snapshot(
    sdk: ts.TogetherSandbox, env_dir: Path, alias: str, tags: dict[str, str]
) -> str:
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
            logger.info("Reusing cached Together snapshot %s for alias %s", existing.id, alias)
            return str(existing.id)
        except ts.HttpError as e:
            if e.status != 404:
                raise
            # Alias not registered yet — fall through to build.

        dockerfile = env_dir / "Dockerfile"
        logger.info("Building Together snapshot for %s (alias=%s)", env_dir, alias)
        result = await sdk.snapshots.create(
            ts.CreateContextSnapshotParams(
                context=str(env_dir),
                dockerfile=str(dockerfile),
                alias=alias,
                tags=tags,
                # Snapshots build under a generated image name, so without an
                # explicit key the remote builder's layer cache never hits.
                cache_key=f"{SNAPSHOT_ALIAS_PREFIX}/{alias}",
                on_progress=lambda event: logger.debug("snapshot build: %s", event.output),
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
        sdk: ts.TogetherSandbox,
        sandbox: ts.Sandbox,
        snapshot_id: str,
        timeout: int,
        max_stream_output_bytes: int = 128 * 1024,
    ) -> None:
        self._sdk = sdk
        self._sandbox = sandbox
        self._sandbox_id = sandbox.id
        self._snapshot_id = snapshot_id
        self._timeout = timeout
        self._max_stream_output_bytes = max_stream_output_bytes
        self._closed = False

    @classmethod
    async def create(
        cls,
        env_dir: Path,
        timeout: int = 600,
        cpu_cores: float | None = None,
        memory_bytes: int | None = None,
        max_stream_output_bytes: int = 128 * 1024,
        tags: dict[str, str] | None = None,
    ) -> TogetherSandboxWrapper:
        """Create a new Together sandbox from a Dockerfile build context.

        Args:
            env_dir: Directory containing ``Dockerfile`` and build context.
            timeout: Sandbox lifetime in seconds, enforced server-side as a TTL.
            cpu_cores: CPU allocation in cores (0.1–16). ``None`` uses Together's
                default (1 vCPU).
            memory_bytes: Memory allocation in bytes, between 1 GB and 8 GB per
                requested core. ``None`` uses Together's default (2 GiB).
            max_stream_output_bytes: Default cap for command output streams.
            tags: Extra key/value labels for the sandbox and its snapshot, merged
                over the defaults from ``_default_tags()``. Useful for
                attributing spend to a recipe or task.
        """
        dockerfile = env_dir / "Dockerfile"
        if not dockerfile.is_file():
            raise FileNotFoundError(
                f"Expected Dockerfile at {dockerfile}; "
                "Together backend requires environment/Dockerfile."
            )

        sdk = await _get_sdk()
        alias = f"{SNAPSHOT_ALIAS_PREFIX}-{_hash_build_context(env_dir)}"
        all_tags = {**_default_tags(), **(tags or {})}
        snapshot_id = await _ensure_snapshot(sdk, env_dir, alias, all_tags)

        # Only include resource overrides the caller specified, so we accept
        # Together's defaults otherwise. Omitting termination_policy makes the
        # sandbox ephemeral: no snapshot is kept when it goes away.
        create_kwargs: dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "ttl": timeout,
            "tags": all_tags,
        }
        if cpu_cores is not None:
            create_kwargs["cpu"] = cpu_cores
        if memory_bytes is not None:
            create_kwargs["memory_bytes"] = memory_bytes

        # create() starts the VM and returns once it is running — there is no
        # separate start step in the v4 SDK.
        sandbox = await sdk.sandboxes.create(**create_kwargs)

        return cls(
            sdk=sdk,
            sandbox=sandbox,
            snapshot_id=snapshot_id,
            timeout=timeout,
            max_stream_output_bytes=max_stream_output_bytes,
        )

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    async def send_heartbeat(self, timeout: int = 30) -> None:
        """No-op heartbeat: Together has no explicit liveness check, so we run ``true``."""
        try:
            await asyncio.wait_for(
                self._sandbox.execs.exec("true", []),
                timeout=timeout,
            )
        except TimeoutError:
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
        cap = max_output_bytes if max_output_bytes is not None else self._max_stream_output_bytes
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
            # Compare against None rather than using `or`: a successful command
            # reports exit_code 0, which is falsy.
            exit_code = result.get("exit_code")
            return SandboxResult(
                stdout=output,
                stderr="",
                exit_code=-1 if exit_code is None else int(exit_code),
            )
        except TimeoutError:
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
            content = await asyncio.wait_for(self._sandbox.files.read(path), timeout=timeout)
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            if max_bytes is not None and len(content) > max_bytes:
                content = content[:max_bytes]
            return SandboxResult(stdout=content, stderr="", exit_code=0)
        except TimeoutError:
            return SandboxResult(
                stdout="", stderr=f"read_file timed out after {timeout}s", exit_code=-1
            )
        except ts.HttpError as e:
            if e.status == 404:
                return SandboxResult(stdout="", stderr=f"file not found: {path}", exit_code=1)
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

        Creates the parent directory first, since ``files.create`` does not.
        Together's files API has no ``executable`` flag, so when requested we
        follow up with ``chmod +x`` via exec.
        """
        try:
            parent = posixpath.dirname(path)
            if parent not in ("", "/"):
                with contextlib.suppress(ts.HttpError):
                    await asyncio.wait_for(
                        self._sandbox.directories.create(parent), timeout=timeout
                    )
            await asyncio.wait_for(
                self._sandbox.files.create(path, content),
                timeout=timeout,
            )
            if executable:
                result = await asyncio.wait_for(
                    self._sandbox.execs.exec("chmod", ["+x", path]),
                    timeout=timeout,
                )
                chmod_code = result.get("exit_code")
                exit_code = 0 if chmod_code is None else int(chmod_code)
                if exit_code != 0:
                    return SandboxResult(
                        stdout="",
                        stderr=result.get("output", "") or "",
                        exit_code=exit_code,
                    )
            return SandboxResult(stdout="", stderr="", exit_code=0)
        except TimeoutError:
            return SandboxResult(
                stdout="", stderr=f"write_file timed out after {timeout}s", exit_code=-1
            )
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def cleanup(self) -> None:
        """Terminate the sandbox VM and close its agent connection.

        Snapshots are intentionally *not* retired — they are cached for reuse
        by sibling sandboxes (same task, different rollout) and across runs.
        Termination is final; the sandbox cannot be restarted afterwards.
        """
        if self._closed:
            return
        self._closed = True

        with contextlib.suppress(Exception):
            # snapshot=None makes the teardown ephemeral (no snapshot kept).
            await self._sdk.sandboxes.terminate(self._sandbox_id, snapshot=None)
        with contextlib.suppress(Exception):
            await self._sandbox.close()
