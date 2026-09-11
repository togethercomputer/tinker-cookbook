# Harbor RL

## Installation

```bash
uv pip install 'tinker-cookbook[together] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
export TOGETHER_API_KEY=...   # required for the Together Sandbox SDK
```

Set `TOGETHER_LOCAL_BUILD=1` if you have Docker installed locally and prefer building images on your machine instead of remotely on Together's infrastructure.

RL training on Harbor formatted tasks (e.g., Terminal Bench 2.0) with sandboxed code execution. An agent gets a bash tool inside a sandboxed container, attempts a task, and receives reward based on test results. Sandboxes are provisioned via the [Together Sandbox SDK](https://github.com/togethercomputer/together-sandbox).

## HarborTask

Harbor offers a standardized format for SWE/Terminal-Bench style task.
Adhering to this allows seperation between task creation layer and evaluation/training harness layer.
We can download the harbor datasets through `uvx harbor datasets download terminal-bench@2.0`.
By default, the task will land in `~/.cache/harbor/tasks/` with the structure

```
~/.cache/harbor/tasks/
  └── <shortuuid(task_id)>/       # deterministic hash for deduplication
      └── <task_name>/            # human-readable task directory
          ├── environment/
          │   └── Dockerfile
          ├── tests/
          │   └── test.sh
          ├── instruction.md
          ├── task.toml
          └── solution/
```

To use harbor tasks for training or evaluation, we designed the following interface

```python
@dataclass(frozen=True)
class HarborTask:
    task_name: str
    instruction: str
    task_dir: Path      # must contain environment/Dockerfile and tests/test.sh
    config: dict[str, Any] = field(default_factory=dict)
```

You can load your downloaded tasks (e.g., 89 Terminal-Bench tasks) via `load_harbor_tasks()` in `launch_terminal_bench.py`:

```python
from tinker_cookbook.recipes.harbor_rl.launch_terminal_bench import load_harbor_tasks

tasks = load_harbor_tasks()  # reads from ~/.cache/harbor/tasks/ by default
print(f"Loaded {len(tasks)} tasks")
print(tasks[0].task_name, tasks[0].task_dir)
```

The training environment is implemented against this interface.
You can customize your own task as long as they conforms to the interface above.

## Sandbox Protocol and custom backends

### The Protocol

`tinker_cookbook.sandbox.sandbox_interface` defines `SandboxInterface`:

```python
@runtime_checkable
class SandboxInterface(Protocol):
    async def run_command(self, command: str, workdir: str | None = None, timeout: int = 60, max_output_bytes: int | None = None) -> SandboxResult: ...
    async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult: ...
    async def write_file(self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60) -> SandboxResult: ...
    async def send_heartbeat(self) -> None: ...
    async def cleanup(self) -> None: ...
```

`TogetherSandboxWrapper` (in `tinker_cookbook/sandbox/together_sandbox.py`) implements this interface.

### SandboxFactory and injection

`harbor_env.py` defines a backend-agnostic factory type and a default Together implementation:

```python
SandboxFactory = Callable[..., Awaitable[SandboxInterface]]

async def default_sandbox_factory(
    env_dir: Path,
    timeout: int,
    *,
    cpu_cores: float | None = None,
    memory_bytes: int | None = None,
    tags: dict[str, str] | None = None,
) -> SandboxInterface:
    """Create a Together sandbox from a task environment directory."""
    from tinker_cookbook.sandbox.together_sandbox import TogetherSandboxWrapper
    return await TogetherSandboxWrapper.create(
        env_dir=env_dir,
        timeout=timeout,
        cpu_cores=cpu_cores,
        memory_bytes=memory_bytes,
        tags=tags,
    )
```

The first argument is the task's `environment/` directory (containing a Dockerfile and build context). Each backend converts this to its own image format internally (Together builds and registers a snapshot under a content-addressed alias, so repeated `create()` calls for the same Dockerfile reuse the cached image instead of rebuilding).

`cpu_cores` / `memory_bytes` are also exposed on `HarborEnvGroupBuilder` and `HarborDatasetBuilder` so you can override Together's defaults (1 vCPU / 2 GiB) for resource-hungry tasks like SWE-Bench. Leave them as `None` to accept the defaults. Memory must be between 1 GB and 8 GB per requested core.

Sandboxes and their snapshots are tagged for cost attribution (`user`, `job`, `component`, plus `recipe` and `task`). `user` comes from the OS login name; set `TINKER_SANDBOX_USER` to override it when running under a shared service account.

`cli_main()` accepts an optional `sandbox_factory` parameter. When `None`, it falls back to `default_sandbox_factory` (Together). The factory flows through: `cli_main` -> `HarborDatasetBuilder` -> `HarborEnvGroupBuilder.make_envs()`.

Sandboxes are created with a server-side TTL of `sandbox_timeout` seconds, so they are reclaimed even if the training process dies. Still call `cleanup()` (already wired into `HarborEnvGroupBuilder.cleanup()` and `eval.py`'s finally-blocks) to release them promptly rather than waiting out the TTL. Termination is final — a sandbox cannot be restarted.

## Running

First, download the Terminal-Bench tasks:

```bash
uvx harbor datasets download terminal-bench@2.0 -o ~/.cache/harbor/tasks/terminal-bench-2.0/
```

Then launch training:

```bash
uv run python tinker_cookbook/recipes/harbor_rl/scripts/train_terminal_bench.py \
    model_name=moonshotai/Kimi-K2.6 \
    max_tokens=8192 \
    group_size=4 \
    groups_per_batch=8 \
    learning_rate=1e-5 \
    lora_rank=32 \
    wandb_project=cookbook_harbor_rl
```

## Evaluation

Evaluate a Tinker endpoint on Harbor datasets without training.

Download datasets:

```bash
uvx harbor datasets download terminal-bench@2.0 -o ~/.cache/harbor/tasks/terminal-bench-2.0
uvx harbor datasets download swebench-verified@1.0 -o ~/.cache/harbor/tasks/swebench-verified-1.0
```

Run evaluation:

```bash
uv run python tinker_cookbook/recipes/harbor_rl/scripts/eval_harbor_rl.py \
    checkpoint_url=tinker://YOUR_CHECKPOINT/sampler_weights/final \
    benchmarks=terminal_bench,swe_bench \
    max_turns=200 \
    max_tokens=8192 \
    temperature=1.0
```

Key parameters in `EvalConfig`: `checkpoint_url`, `max_turns`, `max_tokens`, `temperature`.
`run_eval()` also accepts `sandbox_factory` for custom sandbox backends and `output_path` to control where results are written (default: `tinker_cookbook/recipes/harbor_rl/scripts/results/<timestamp>/`).

We evaluated SWE-Bench-Verified-1.0 and Terminal-Bench-2.0 at 32K context length and naive agent harness with no advanced features like context compatification that summarizes the tool calling history.

### Results: Kimi-K2.6 (32K context, no compaction)

| Benchmark              | Total | PASS        | FAIL       | ERROR       | Pass Rate |
| ---------------------- | ----- | ----------- | ---------- | ----------- | --------- |
| SWE-Bench Verified 1.0 | 500   | 145 (29.0%) | 52 (10.4%) | 303 (60.6%) | 29.0%     |
| Terminal-Bench 2.0     | 89    | 14 (15.7%)  | 31 (34.8%) | 44 (49.4%)  | 15.7%     |

**Config**: `max_turns=200, max_tokens=8192, temperature=1.0, sandbox_timeout=3600s`

All ERRORs are context window overflow (`prompt_tokens + max_tokens > 32768`).
These occur when the conversation history exceeds ~24.5K tokens, leaving insufficient room for the 8192 `max_tokens` generation budget.
