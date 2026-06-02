import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from tinker_cookbook.recipes.harbor_rl.eval import EvalConfig, run_eval
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    default_sandbox_factory,
    load_harbor_tasks,
)

if __name__ == "__main__":
    config = EvalConfig(
        max_turns=20,
        temperature=0.1,
        max_tokens=2048,
        max_tasks=5,
        sandbox_timeout=600,
    )
    tasks = load_harbor_tasks("terminal-bench-2.0/terminal-bench")
    asyncio.run(run_eval(config, tasks, sandbox_factory=default_sandbox_factory))
