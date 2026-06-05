"""Smoke test: run Harbor RL training on a single task for 1 step."""

import asyncio

from tinker_cookbook.recipes.harbor_rl.harbor_env import default_sandbox_factory, load_harbor_tasks
from tinker_cookbook.recipes.harbor_rl.train import CLIConfig, cli_main

if __name__ == "__main__":
    tasks = load_harbor_tasks("terminal-bench-2.0")
    task_name = "filter-js-from-html"  # change this to run a specific task
    one_task = [t for t in tasks if t.task_name == task_name]
    if not one_task:
        raise ValueError(f"Task {task_name!r} not found. Available: {[t.task_name for t in tasks]}")
    print(f"Running on task: {one_task[0].task_name}")

    cli_config = CLIConfig(
        model_name="deepseek-ai/DeepSeek-V3.1",
        group_size=2,
        groups_per_batch=1,
        max_steps=1,
        eval_every=1,
        save_every=1,
        command_timeout=600,
        grader_timeout=300,
    )
    asyncio.run(cli_main(cli_config, one_task, sandbox_factory=default_sandbox_factory))
