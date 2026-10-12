"""
Run configuration for the math agent example: the profile, the evaluator,
the case split and the optimizer settings. The CLI loads this file:

    harness-evolve run examples/math_agent/config.py --run-dir runs/math-1

Environment: MODEL (litellm model id for the agent), PROPOSER_MODEL (for the
proposer and critic; defaults to MODEL), and the provider's API key, e.g.
TOGETHER_API_KEY. PRICES_PER_MTOK="0.30,1.20" sets the agent model's price.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluator import MathEvaluator  # noqa: E402

from harness_evolve import AgentProfile, Optimizer, OptimizerConfig, litellm_llm, profiles  # noqa: E402
from harness_evolve.grader import TrajectoryGrader  # noqa: E402

MODEL = os.environ.get("MODEL", "together_ai/deepseek-ai/DeepSeek-V4.1-Flash")
PROPOSER_MODEL = os.environ.get("PROPOSER_MODEL", MODEL)
PRICES = tuple(float(x) for x in os.environ.get("PRICES_PER_MTOK", "0.30,1.20").split(","))
PROBLEMS = HERE / "data" / "problems.jsonl"

PROFILE = AgentProfile.build(
    name="math-agent", harness_root=HERE / "harness",
    agent_description=("A math word-problem solver with two tools: calculator(expression) evaluates "
                       "arithmetic, submit_answer(answer) submits the final number and ends the run."),
    safety_mechanisms=["submitting a single numeric answer with submit_answer"],
    constraints="Edits must help on any arithmetic word problem, not on particular questions or numbers.",
    gaming_examples="E.g. telling the agent to round answers so near-misses count as correct.",
    setting_bounds={"max_turns": (3, 16), "nudges": (0, 3), "remind_after": (1, 12), "stop_after": (1, 12)},
    grader=TrajectoryGrader())


def build(run_dir: str | Path, budget_usd: float = 2.0, rounds: int = 3, *,
          harness_dir: Path | None = None, problems: Path | None = None, n_evolve: int = 50,
          heldout: bool = True, proposer_llm=None, critic_llm=None, confirm: bool = False,
          prune: bool = False, max_stall: int = 3) -> Optimizer:
    """The example's optimizer. Keyword options are for the benchmark (bench.py): another starting
    harness, a larger problem set, scripted proposer/critic, no held-out phase, confirm-before-accept, pruning."""
    problems = Path(problems or PROBLEMS)
    if not problems.exists():
        raise SystemExit(f"no data at {problems}: run `python {HERE / 'fetch_data.py'}` first")
    profiles.use(PROFILE)
    ids = [json.loads(x)["id"] for x in problems.read_text().splitlines()]
    random.Random(0).shuffle(ids)
    evolve, held = ids[:n_evolve], ids[n_evolve:]
    cfg = OptimizerConfig(
        evolve_cases=evolve, guard_cases=[], final_cases=held if heldout else [], final_trials=3,
        budget_usd=budget_usd, max_rounds=rounds, max_stall=max_stall, trials=2, calibration_trials=2,
        default_case_cost_usd=0.003, cost_band=True, allow_unpriced_llm=proposer_llm is not None,
        parallel_lanes=8, case_lanes={c: c for c in ids}, confirm=confirm, prune=prune)
    llm = litellm_llm(PROPOSER_MODEL, prices=PRICES if PROPOSER_MODEL == MODEL else None)
    return Optimizer(run_dir, Path(harness_dir or HERE / "harness"), cfg,
                     evaluator=MathEvaluator(problems, MODEL, PRICES),
                     proposer_llm=proposer_llm or llm, critic_llm=critic_llm or llm, critic_patterns=[])
