"""Confirm before accepting: a would-be winner is re-measured on fresh trials and accepted only if
that independent measurement passes too."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from harness_evolve import candidates
from harness_evolve.acceptance import CaseResult, EvalResult
from harness_evolve.evaluator import EvalOutcome
from tests.test_optimizer import _opt


class LuckyEvaluator:
    """The TERSE candidate looks 40% cheaper the first time each case is measured, and costs the
    same as the incumbent afterwards (a lucky first measurement). With `real=True` it is cheaper
    every time (a real winner)."""

    def __init__(self, real: bool = False):
        self.real, self.seen, self.calls = real, set(), 0

    async def evaluate(self, harness_dir, case_ids, trials):
        (case,) = case_ids
        self.calls += 1
        terse = "TERSE" in (Path(harness_dir) / "log_context_missing.prompt").read_text()
        key = (candidates.content_hash(harness_dir), case)
        cheap = terse and (self.real or key not in self.seen)
        self.seen.add(key)
        verdicts = ["PASS", "FAIL"][:trials] if case == "c1" else ["PASS"] * trials
        cost = (0.6 if cheap else 1.0) * trials
        cr = CaseResult(verdicts.count("PASS"), trials, 0, cost, verdicts)
        return EvalOutcome(EvalResult({case: cr}), cost, [{"instance_id": case, "verdict": verdicts[0],
                                                            "steps": []}])


def _history(tmp_path):
    return [json.loads(x) for x in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]


def test_a_lucky_first_measurement_is_not_confirmed(tmp_path):
    state = asyncio.run(_opt(tmp_path, LuckyEvaluator(), max_rounds=1, confirm=True).run_until_stopped())
    assert state.phase == "done" and state.accepted == []
    (h,) = _history(tmp_path)
    assert h["outcome"] == "unconfirmed" and h["reason"].startswith("not confirmed on fresh trials")


def test_a_real_winner_is_confirmed_and_accepted(tmp_path):
    ev = LuckyEvaluator(real=True)
    state = asyncio.run(_opt(tmp_path, ev, max_rounds=1, confirm=True).run_until_stopped())
    assert state.accepted == ["r1"]
    (h,) = _history(tmp_path)
    assert "confirmed on fresh trials" in h["reason"]
    # round 0 (4 cases) + candidate (4) + confirmation: incumbent (4) + candidate (4)
    assert ev.calls == 16
    assert any("k1c" in p.name or "k1c" in str(p) for p in (tmp_path / "run").rglob("*"))


def test_without_confirm_the_lucky_candidate_is_accepted(tmp_path):
    state = asyncio.run(_opt(tmp_path, LuckyEvaluator(), max_rounds=1).run_until_stopped())
    assert state.accepted == ["r1"]          # the risk confirm-before-accept removes
