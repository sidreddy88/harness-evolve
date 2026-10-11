"""
The optimizer benchmark on the math agent (tier 2). Run with:

    python examples/math_agent/fetch_data.py --n 260 --out examples/math_agent/data/problems-bench.jsonl
    harness-evolve bench examples/math_agent/bench.py --out runs/bench-1

Known-answer scenarios (scripted proposer and critic, so they test the acceptance rules alone):
  accept-good   planted flaw max_turns = 1 (most correct runs need 2 turns); the scripted fix restores
                max_turns = 8. Must be accepted: the rules can see a real gain.
  reject-bad    the scripted edit tells the agent to round its answer to the nearest hundred before
                submitting; most GSM-Hard answers aren't multiples of 100, so this must hurt. Must not be
                accepted. (A first version, "skip the calculator and estimate", turned out not to hurt
                DeepSeek-V4.1-Flash at all and was 13% cheaper, so accepting it was correct: verify a
                known-answer scenario's premise before trusting it.)
  reject-noise  a neutral rewording of the system prompt. Must not be accepted (false-accept check).
Open scenarios (real proposer and critic, reported):
  find-flaw     the planted-flaw harness: does the proposer find the problem from trajectories alone?
  standard      the normal harness on 200 problems: held-out gain, cost, what was accepted.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness_evolve.bench import ACCEPTED, NOT_ACCEPTED, Scenario, scripted_llm  # noqa: E402

import config  # noqa: E402

PROBLEMS = HERE / "data" / "problems-bench.jsonl"
FLAWED = HERE / "bench_harnesses" / "max_turns_1"

GOOD_FIX = {"component": "settings", "hypothesis": "runs end after one turn, before submitting; allow more turns",
            "edits": [{"file": "settings.json", "find": '"max_turns": 1', "replace": '"max_turns": 8'}]}
BAD_EDIT = {"component": "prompts", "hypothesis": "round answers to save effort",
            "edits": [{"file": "task.prompt",
                       "find": "When you are done, call submit_answer with the final number.",
                       "replace": "When you are done, round the final number to the nearest hundred and "
                                  "call submit_answer with the rounded value."}]}
NEUTRAL_EDIT = {"component": "prompts", "hypothesis": "neutral rewording (false-accept check)",
                "edits": [{"file": "system.prompt",
                           "find": "You are a helpful assistant that solves math word problems.",
                           "replace": "Your job is to solve math word problems."}]}


def _kat(edit: dict, harness: Path | None = None):
    return lambda run_dir: config.build(run_dir, budget_usd=3.0, rounds=1, harness_dir=harness,
                                        problems=PROBLEMS, n_evolve=100, heldout=False,
                                        proposer_llm=scripted_llm(edit), critic_llm=scripted_llm(edit))


def scenarios() -> list[Scenario]:
    return [
        Scenario("accept-good", "planted flaw (max_turns 1) + scripted real fix (max_turns 8): must be accepted",
                 _kat(GOOD_FIX, FLAWED), ACCEPTED),
        Scenario("reject-bad", "scripted harmful edit (round answers to the nearest hundred): must not be accepted",
                 _kat(BAD_EDIT), NOT_ACCEPTED),
        Scenario("reject-noise", "scripted neutral rewording of the system prompt: must not be accepted",
                 _kat(NEUTRAL_EDIT), NOT_ACCEPTED),
        Scenario("find-flaw", "planted flaw (max_turns 1) with the real proposer and critic, 2 rounds: reported",
                 lambda run_dir: config.build(run_dir, budget_usd=5.0, rounds=2, harness_dir=FLAWED,
                                              problems=PROBLEMS, n_evolve=100)),
        Scenario("standard", "normal harness, 200 evolve problems, real proposer and critic, 3 rounds: reported",
                 lambda run_dir: config.build(run_dir, budget_usd=6.0, rounds=3,
                                              problems=PROBLEMS, n_evolve=200)),
    ]
