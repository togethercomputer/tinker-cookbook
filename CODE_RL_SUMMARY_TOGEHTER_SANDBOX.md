# Code RL Recipe — How It Works

## The Big Picture

This recipe teaches a language model to write better code by having it actually run its code and learn from whether it passes tests. It uses a technique called **GRPO** (Group Relative Policy Optimization) — a form of reinforcement learning where the model gets rewarded for correct solutions.

After 100 training steps, a Qwen 4B model improves from 33.8% → 42.7% pass rate on competitive programming problems.

---

## The Training Loop

Each training step looks like this:

1. **Pick problems** — sample a batch of competitive programming problems from the DeepCoder dataset (thousands of problems from TACO, PrimeIntellect, LiveCodeBench)

2. **Generate solutions** — for each problem, the model makes several independent attempts (controlled by `group_size`). Each attempt is a conversation where:
   - The model reads the problem
   - Writes Python code and calls a `check_solution` tool
   - Gets back `{"passed": true/false}` from a real code execution sandbox
   - Can revise and try again (up to 2 rounds)

3. **Score each attempt** — did the final code pass all the hidden test cases? That's the reward (1.0 = pass, 0.0 = fail, with a small -0.1 penalty for not even formatting code correctly)

4. **Update the model** — compare the attempts within each group. The ones that did better than average get reinforced; the ones that did worse get discouraged. Apply one gradient update to the LoRA weights.

5. **Repeat**

---

## Why Multiple Attempts Per Problem?

GRPO needs to compare attempts *against each other* to compute a relative signal. If all 4 attempts fail, the model learns nothing. If some pass and some fail, it learns which reasoning patterns worked. This is more stable than just rewarding/penalizing in isolation.

---

## Eval vs Training

Every N steps (configurable), the model is tested on **held-out problems it has never trained on** — one attempt per problem, no weight update. This measures whether the model is genuinely improving or just memorizing training problems.

---

## Key Numbers

| Parameter | What it means | Smoke test | Full run |
|---|---|---|---|
| `group_size` | Attempts per problem | 4 | 8 |
| `groups_per_batch` | Problems per step | 10 | 128 |
| Rollouts per step | Total model calls | 40 | 1024 |
| `max_steps` | Total weight updates | 3 | hundreds |
| Sandboxes needed | Concurrent executions | ~40 | ~1024 |

---

## The Sandbox

Generated code is executed in an isolated cloud VM (sandbox) to get the reward signal safely. Three backends are supported:

- **SandboxFusion** — runs locally in Docker (default)
- **Modal** — cloud, no Docker needed, manages its own Python+numpy image
- **Together Sandbox** — cloud, no Docker needed, uses a snapshot with Python+numpy pre-installed. Auto-creates the snapshot on first run if it doesn't exist.

The sandbox only needs Python + numpy. The model's code runs there and results come back as pass/fail.

---

## Running It

```bash
# Smoke test (3 weight updates, ~120 total code executions)
TOGETHER_POOL_SIZE=40 TOGETHER_CREATION_RATE_LIMIT=20 \
uv run python -m tinker_cookbook.recipes.code_rl.train \
    sandbox_backend=together \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    group_size=4 groups_per_batch=10 \
    max_steps=3 eval_every=3 save_every=3

# Full training
uv run python -m tinker_cookbook.recipes.code_rl.train \
    sandbox_backend=together \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    group_size=8 groups_per_batch=128 \
    learning_rate=4e-5 lora_rank=32 max_tokens=24576
```
