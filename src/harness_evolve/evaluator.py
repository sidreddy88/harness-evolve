"""
The evaluator interface: how harness-evolve measures a candidate harness.

You implement `Evaluator.evaluate(harness_dir, case_ids, trials)`: run your
agent with the harness in `harness_dir` on each case, `trials` times, and
return an `EvalOutcome` with per-case results (pass/fail per trial and cost)
plus, optionally, trajectories the proposer can read as evidence.

Raise `ProviderFailure` (billing, auth, rate limits, outages) instead of
scoring a trial as failed: a run that measured nothing must stop, not count.
Optional attribute `batch_size` (int > 1): the loop then hands you cases in
chunks, for evaluators that grade a whole batch at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from harness_evolve.acceptance import EvalResult


class ProviderFailure(RuntimeError):
    """Auth/billing/rate-limit/outage: the evaluation measured nothing. Stop the
    run rather than score it (the lesson of two credit outages mid-gate-run).
    cost_usd is what the trials that did run cost, so the budget stays honest."""

    def __init__(self, message: str, cost_usd: float = 0.0):
        super().__init__(message)
        self.cost_usd = cost_usd


class UnpricedModel(ProviderFailure):
    """A call came back from a model with no known price, so its cost reads as
    $0 and the budget cap can't see it. Stop (resumably) rather than keep
    spending blind. (In the first real run this happened silently: the
    evaluated model had no price, and an $80 cap recorded $0 per case.)"""


@dataclass
class EvalOutcome:
    result: EvalResult
    cost_usd: float
    trajectories: list[dict] = field(default_factory=list)


class Evaluator(Protocol):
    async def evaluate(self, harness_dir: Path, case_ids: list[str], trials: int) -> EvalOutcome: ...
