"""
Thin wrapper around Together Sandbox API.

Together Sandbox provides cloud-based sandboxed execution environments.
Requires TOGETHER_API_KEY environment variable.

See: https://docs.together.ai/docs/sandbox

Configuration via environment variables:
    TOGETHER_API_KEY: API key for authentication (required)
    TOGETHER_POOL_SIZE: Number of sandboxes in the pool (default: 32)
    TOGETHER_CREATION_RATE_LIMIT: Max sandboxes created per second (default: 4)
    TOGETHER_SNAPSHOT_ALIAS: Snapshot alias to use (default: python-numpy@latest)
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from pathlib import Path

try:
    from together_sandbox import TogetherSandbox
    from together_sandbox import Sandbox as TogetherSandboxInstance
    from together_sandbox._snapshots import CreateContextSnapshotParams
except ImportError:
    raise ImportError(
        "together_sandbox is required for TogetherSandboxPool. "
        "Install it with: uv pip install 'tinker-cookbook[together-sandbox] @ "
        "git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'"
    ) from None

from tinker_cookbook.exceptions import SandboxError
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError

logger = logging.getLogger(__name__)


class TogetherSandboxPool:
    """
    Pool of Together Sandbox instances for concurrent execution.

    Each sandbox handles one request at a time. The pool manages
    creating and shutting down sandboxes automatically.

    Configuration via environment variables:
        TOGETHER_API_KEY: Required API key
        TOGETHER_POOL_SIZE: Number of warm sandboxes to maintain (default: 32)
        TOGETHER_CREATION_RATE_LIMIT: Max sandboxes created per second (default: 4)
        TOGETHER_SNAPSHOT_ALIAS: Snapshot alias (default: python-numpy@latest)
    """

    def __init__(
        self,
        *,
        pool_size: int | None = None,
        sandbox_timeout_secs: int = 1200,
        snapshot_alias: str | None = None,
        snapshot_id: str | None = None,
        app_name: str = "tinker-cookbook-runner",
    ):
        self._pool_size = pool_size or int(os.getenv("TOGETHER_POOL_SIZE", "32"))
        self._creation_rate_limit = int(os.getenv("TOGETHER_CREATION_RATE_LIMIT", "4"))
        self._sandbox_timeout_secs = sandbox_timeout_secs
        self._snapshot_alias = snapshot_alias or os.getenv(
            "TOGETHER_SNAPSHOT_ALIAS", "python-numpy@latest"
        )
        self._snapshot_id = snapshot_id
        self._terminated = False

        self._sdk = TogetherSandbox()  # reads TOGETHER_API_KEY from env

        self._warm_pool: asyncio.Queue[TogetherSandboxInstance] = asyncio.Queue()
        self._to_shutdown: list[TogetherSandboxInstance] = []
        self._active_count = 0
        self._snapshot_ready = asyncio.Event()

        asyncio.create_task(self._ensure_snapshot())
        asyncio.create_task(self._maintain_pool())

    # Dockerfile used when auto-creating the snapshot
    _NUMPY_DOCKERFILE = "FROM python:3.12-slim\nRUN pip install numpy\n"

    async def _ensure_snapshot(self) -> None:
        """Ensure the snapshot alias exists, creating it from a Dockerfile if needed."""
        if self._snapshot_id is not None:
            # Explicit snapshot ID provided — no alias needed.
            self._snapshot_ready.set()
            return

        try:
            await self._sdk.snapshots.get_by_alias(self._snapshot_alias)
            logger.info("Together snapshot alias %r found.", self._snapshot_alias)
            self._snapshot_ready.set()
            return
        except Exception:
            pass

        logger.info(
            "Together snapshot alias %r not found — creating from Dockerfile (python:3.12-slim + numpy)...",
            self._snapshot_alias,
        )
        try:
            with tempfile.TemporaryDirectory(prefix="together-sandbox-") as tmpdir:
                context_dir = Path(tmpdir) / "python-numpy"
                context_dir.mkdir()
                (context_dir / "Dockerfile").write_text(self._NUMPY_DOCKERFILE)

                def _on_progress(p: object) -> None:
                    logger.info("Snapshot build: %s", p)

                result = await self._sdk.snapshots.create(
                    CreateContextSnapshotParams(
                        context=str(context_dir),
                        alias=self._snapshot_alias,
                        on_progress=_on_progress,
                    )
                )
            logger.info(
                "Created snapshot %r with alias %r.",
                result.snapshot_id,
                result.alias,
            )
            self._snapshot_ready.set()
        except Exception as e:
            logger.error("Failed to create Together snapshot: %s", e)
            # Still set the event so the pool doesn't hang; sandbox creation
            # will fail individually and be logged.
            self._snapshot_ready.set()

    async def _create_sandbox(self) -> TogetherSandboxInstance:
        """Create and start a new ephemeral sandbox."""
        kwargs: dict = {"ephemeral": True}
        if self._snapshot_id is not None:
            kwargs["snapshot_id"] = self._snapshot_id
        elif self._snapshot_alias is not None:
            kwargs["snapshot_alias"] = self._snapshot_alias

        model = await self._sdk.sandboxes.create(**kwargs)
        return await self._sdk.sandboxes.start(model.id)

    async def _maintain_pool(self) -> None:
        """Background task to handle all sandbox creation and shutdown."""
        await self._snapshot_ready.wait()
        while not self._terminated:
            try:
                await self._maintain_pool_step()
            except Exception as e:
                logger.error(f"Error maintaining TogetherSandboxPool: {e}")
            await asyncio.sleep(1.0)

    async def _maintain_pool_step(self) -> None:
        """Single iteration of pool maintenance: shut down used sandboxes, create new ones."""
        if self._to_shutdown:
            to_shutdown, self._to_shutdown = self._to_shutdown, []
            await asyncio.gather(
                *(self._sdk.sandboxes.shutdown(sb.id) for sb in to_shutdown),
                return_exceptions=True,
            )

        total = self._warm_pool.qsize() + self._active_count
        need = min(self._creation_rate_limit, self._pool_size - total)
        if need > 0:
            new_sandboxes = await asyncio.gather(
                *(self._create_sandbox() for _ in range(need)),
                return_exceptions=True,
            )
            for sb in new_sandboxes:
                if isinstance(sb, BaseException):
                    logger.error(f"Error creating Together sandbox: {sb}")
                else:
                    await self._warm_pool.put(sb)

    async def run_in_workdir(
        self,
        files: dict[str, str],
        command: list[str],
        timeout: int | None = None,
    ) -> SandboxResult:
        """
        Execute command with files using an available sandbox from the pool.
        If all sandboxes are busy, waits until one becomes available.

        Creates an isolated workdir, writes files, and runs the command.

        Args:
            files: Files to write {filename: content}
            command: Command and arguments (e.g., ["python", "run.py"])
            timeout: Execution timeout in seconds
        """
        if self._terminated:
            raise SandboxError("TogetherSandboxPool has been terminated.")

        sandbox = await self._warm_pool.get()
        self._active_count += 1

        try:
            workdir = f"/workspace/{uuid.uuid4().hex[:12]}"
            await sandbox.directories.create(workdir)

            if files:
                await asyncio.gather(
                    *(
                        sandbox.files.create(f"{workdir}/{filename}", content)
                        for filename, content in files.items()
                    )
                )

            exec_coro = sandbox.execs.exec(
                command[0],
                command[1:],
                cwd=workdir,
            )
            if timeout is not None:
                result = await asyncio.wait_for(exec_coro, timeout=timeout)
            else:
                result = await exec_coro

            return SandboxResult(
                stdout=result["output"],
                stderr="",
                exit_code=result["exit_code"],
            )
        except asyncio.TimeoutError:
            return SandboxResult(
                stdout="",
                stderr="Execution timed out",
                exit_code=-1,
            )
        finally:
            self._active_count -= 1
            self._to_shutdown.append(sandbox)

    async def terminate(self) -> None:
        """Exit the pool and shut down all sandboxes."""
        self._terminated = True

        # Wait for active sandboxes to finish and be added to _to_shutdown
        while self._active_count > 0:
            await asyncio.sleep(0.5)

        # Collect and shut down all sandboxes
        all_sandboxes = list(self._to_shutdown)
        while not self._warm_pool.empty():
            try:
                all_sandboxes.append(self._warm_pool.get_nowait())
            except asyncio.QueueEmpty:
                break
        await asyncio.gather(
            *(self._sdk.sandboxes.shutdown(sb.id) for sb in all_sandboxes),
            return_exceptions=True,
        )
