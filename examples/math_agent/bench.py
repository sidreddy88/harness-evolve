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
  accept-threshold  planted flaw stop_after = 1 (stuck detection stops runs at their second calculator
                call; 52 of 400 measured runs make 2+ calls, 34 of them correct); the scripted fix restores
                stop_after = 8. Must be accepted.
Open scenarios (real proposer and critic, reported):
  find-flaw     the planted-flaw harness: does the proposer find the problem from trajectories alone?
  find-threshold  the stop_after = 1 harness with the real proposer: does it find a too-tight threshold?
  find-switch   starts with the structural switch plan = "on" (an extra planning call per problem, so
                more cost); real proposer, 2 rounds: does it check the switch? Reported.
  noise-many    40 scripted neutral rewordings of the system prompt in a row, on 20 evolve problems, with
                confirm on. Every candidate that passes its first measurement is a false accept confirm-off
                would have made; the fresh trials then confirm it or catch it (`unconfirmed`). Reported.
  standard      the normal harness on 200 problems: held-out gain, cost, what was accepted.

BENCH_CONFIRM=1 turns on confirm-before-accept and BENCH_PRUNE=1 turns on pruning of accepted edits in every
scenario (for A/B runs into a separate --out).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402

from harness_evolve.bench import ACCEPTED, NOT_ACCEPTED, Scenario, scripted_llm  # noqa: E402

PROBLEMS = HERE / "data" / "problems-bench.jsonl"
FLAWED = HERE / "bench_harnesses" / "max_turns_1"
TIGHT = HERE / "bench_harnesses" / "stop_after_1"
PLAN_ON = HERE / "bench_harnesses" / "plan_on"
CONFIRM = os.environ.get("BENCH_CONFIRM") == "1"
PRUNE = os.environ.get("BENCH_PRUNE") == "1"

GOOD_FIX = {"component": "settings", "hypothesis": "runs end after one turn, before submitting; allow more turns",
            "edits": [{"file": "settings.json", "find": '"max_turns": 1', "replace": '"max_turns": 8'}]}
BAD_EDIT = {"component": "prompts", "hypothesis": "round answers to save effort",
            "edits": [{"file": "task.prompt",
                       "find": "When you are done, call submit_answer with the final number.",
                       "replace": "When you are done, round the final number to the nearest hundred and "
                                  "call submit_answer with the rounded value."}]}
THRESHOLD_FIX = {"component": "settings", "hypothesis": "runs are stopped at their second calculator call; allow more",
                 "edits": [{"file": "settings.json", "find": '"stop_after": 1', "replace": '"stop_after": 8'}]}
NEUTRAL_EDIT = {"component": "prompts", "hypothesis": "neutral rewording (false-accept check)",
                "edits": [{"file": "system.prompt",
                           "find": "You are a helpful assistant that solves math word problems.",
                           "replace": "Your job is to solve math word problems."}]}

_OPENERS = ("You are a helpful assistant that", "You are an assistant that", "You are a careful helper that",
            "You are a friendly assistant that", "You are a tutor that")
NEUTRAL_VARIANTS = [f"{o} {v} math word problems{e}" for o in _OPENERS
                    for v in ("solves", "works through", "answers", "handles") for e in (".", " for users.")][:40]


def _neutral_stream(run_dir: Path):
    """A scripted proposer that rewords the current system prompt to the next neutral variant each round."""
    calls = [0]

    async def call(system: str, prompt: str):
        from harness_evolve.llm import LLMReply
        if "strict reviewer" in system:
            return LLMReply('{"verdict": "accept", "reasons": ["scripted"]}', 0.0)
        current = (Path(run_dir) / "incumbent" / "system.prompt").read_text().strip()
        options = [v for v in NEUTRAL_VARIANTS if v != current]
        new = options[calls[0] % len(options)]
        calls[0] += 1
        return LLMReply(json.dumps({"component": "prompts", "hypothesis": f"neutral rewording {calls[0]}",
                                    "edits": [{"file": "system.prompt", "find": current, "replace": new}]}), 0.0)
    return call


def _noise_many(run_dir: Path):
    llm = _neutral_stream(run_dir)
    return config.build(run_dir, budget_usd=8.0, rounds=40, max_stall=1000, problems=PROBLEMS, n_evolve=20,
                        heldout=False, proposer_llm=llm, critic_llm=llm, confirm=True)


def _kat(edit: dict, harness: Path | None = None):
    return lambda run_dir: config.build(run_dir, budget_usd=3.0, rounds=1, harness_dir=harness,
                                        problems=PROBLEMS, n_evolve=100, heldout=False,
                                        proposer_llm=scripted_llm(edit), critic_llm=scripted_llm(edit), confirm=CONFIRM)


def scenarios() -> list[Scenario]:
    return [
        Scenario("accept-good", "planted flaw (max_turns 1) + scripted real fix (max_turns 8): must be accepted",
                 _kat(GOOD_FIX, FLAWED), ACCEPTED),
        Scenario("reject-bad", "scripted harmful edit (round answers to the nearest hundred): must not be accepted",
                 _kat(BAD_EDIT), NOT_ACCEPTED),
        Scenario("reject-noise", "scripted neutral rewording of the system prompt: must not be accepted",
                 _kat(NEUTRAL_EDIT), NOT_ACCEPTED),
        Scenario("accept-threshold", "planted flaw (stop_after 1) + scripted fix (stop_after 8): must be accepted",
                 _kat(THRESHOLD_FIX, TIGHT), ACCEPTED),
        Scenario("find-flaw", "planted flaw (max_turns 1) with the real proposer and critic, 2 rounds: reported",
                 lambda run_dir: config.build(run_dir, budget_usd=5.0, rounds=2, harness_dir=FLAWED,
                                              problems=PROBLEMS, n_evolve=100, confirm=CONFIRM, prune=PRUNE)),
        Scenario("find-threshold", "planted flaw (stop_after 1) with the real proposer and critic, 2 rounds: reported",
                 lambda run_dir: config.build(run_dir, budget_usd=5.0, rounds=2, harness_dir=TIGHT,
                                              problems=PROBLEMS, n_evolve=100, confirm=CONFIRM, prune=PRUNE)),
        Scenario("find-switch", "starts with plan = on (extra call per problem), real proposer, 2 rounds: reported",
                 lambda run_dir: config.build(run_dir, budget_usd=5.0, rounds=2, harness_dir=PLAN_ON,
                                              problems=PROBLEMS, n_evolve=100, confirm=CONFIRM, prune=PRUNE)),
        Scenario("noise-many", "40 scripted neutral rewordings, 20 evolve problems, confirm on: reported",
                 _noise_many),
        Scenario("standard", "normal harness, 200 evolve problems, real proposer and critic, 3 rounds: reported",
                 lambda run_dir: config.build(run_dir, budget_usd=6.0, rounds=3,
                                              problems=PROBLEMS, n_evolve=200, confirm=CONFIRM, prune=PRUNE)),
    ]
