"""
The evaluator: how harness-evolve measures a candidate harness for the math
agent. This is the one interface you implement for your own agent.

For each case and trial it runs the agent with the candidate harness, checks
the submitted number against the answer, and returns pass/fail, cost and a
trajectory (tool calls) the proposer reads as evidence. The right answer goes
after HIDDEN_MARKER, so the proposer never sees it.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from harness_evolve import CaseResult, EvalOutcome, EvalResult, ProviderFailure
from harness_evolve.evidence import HIDDEN_MARKER

from agent import solve


def _correct(answer: float | None, target: float) -> bool:
    return answer is not None and abs(answer - target) <= 1e-4 * max(1.0, abs(target))


def _provider_problem(exc: Exception) -> bool:
    name, text = type(exc).__name__, str(exc).lower()
    return (name in ("AuthenticationError", "RateLimitError", "ServiceUnavailableError", "APIConnectionError",
                     "BudgetExceededError", "PermissionDeniedError")
            or any(w in text for w in ("credit", "billing", "quota", "insufficient", "api key")))


class MathEvaluator:
    def __init__(self, problems_path: Path, model: str, prices: tuple[float, float], concurrency: int = 8):
        self.problems = {p["id"]: p for p in map(json.loads, Path(problems_path).read_text().splitlines())}
        self.model, self.prices = model, prices
        self.gate = asyncio.Semaphore(concurrency)

    async def _trial(self, harness_dir: Path, case: str, trial: int) -> dict:
        p = self.problems[case]
        async with self.gate:
            try:
                run = await solve(p["question"], harness_dir, self.model, self.prices)
            except Exception as exc:
                if _provider_problem(exc):
                    raise ProviderFailure(f"{type(exc).__name__}: {exc}") from exc
                return {"instance_id": case, "trial": trial, "verdict": "ERROR", "detail": f"{exc}"[:300],
                        "steps": [], "submitted": False, "cost": {"cost_usd": 0.0}}
        ok = _correct(run.answer, p["answer"])
        detail = ("no answer submitted" if run.answer is None else f"submitted {run.answer:g}")
        return {"instance_id": case, "trial": trial, "verdict": "PASS" if ok else "FAIL",
                "detail": f"{detail} {HIDDEN_MARKER} expected {p['answer']:g}",
                "steps": run.steps, "turns": run.turns, "submitted": run.answer is not None,
                "cost": {"cost_usd": run.cost_usd}}

    async def evaluate(self, harness_dir: Path, case_ids: list[str], trials: int) -> EvalOutcome:
        trajs = await asyncio.gather(*(self._trial(harness_dir, c, t + 1)
                                       for c in case_ids for t in range(trials)))
        per_case = {}
        for c in case_ids:
            mine = [t for t in trajs if t["instance_id"] == c]
            verdicts = [t["verdict"] for t in mine]
            per_case[c] = CaseResult(passes=verdicts.count("PASS"), trials=len(mine), verdicts=verdicts,
                                     escalations=sum(not t["submitted"] for t in mine),
                                     cost_usd=sum(t["cost"]["cost_usd"] for t in mine))
        return EvalOutcome(result=EvalResult(per_case), cost_usd=sum(t["cost"]["cost_usd"] for t in trajs),
                           trajectories=trajs)
