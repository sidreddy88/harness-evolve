"""Pruning accepted edits: after the rounds, an accepted edit the final harness doesn't need is removed."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from harness_evolve import candidates
from harness_evolve.acceptance import CaseResult, EvalResult
from harness_evolve.evaluator import EvalOutcome
from tests.test_optimizer import _opt


class SubsumedEvaluator:
    """TERSE (log_context_missing.prompt) makes runs 40% cheaper, LEAN (stack_trace.prompt) 60%
    cheaper, and with LEAN, TERSE adds nothing. So round 1 accepts TERSE, round 2 accepts LEAN, and
    TERSE is no longer needed."""

    def __init__(self):
        self.calls = 0

    async def evaluate(self, harness_dir, case_ids, trials):
        (case,) = case_ids
        self.calls += 1
        h = Path(harness_dir)
        terse = "TERSE" in (h / "log_context_missing.prompt").read_text()
        lean = "LEAN" in (h / "stack_trace.prompt").read_text()
        verdicts = ["PASS", "FAIL"][:trials] if case == "c1" else ["PASS"] * trials
        cost = (0.4 if lean else 0.6 if terse else 1.0) * trials
        cr = CaseResult(verdicts.count("PASS"), trials, 0, cost, verdicts)
        return EvalOutcome(EvalResult({case: cr}), cost, [{"instance_id": case, "verdict": verdicts[0],
                                                            "steps": []}])


def two_edit_proposer(second_file="stack_trace.prompt"):
    edits = [{"file": "log_context_missing.prompt", "find": "STEPS 1-3", "replace": "TERSE STEPS 1-3"}]
    if second_file == "stack_trace.prompt":
        text = (Path(__file__).parent / "fixtures" / "diagnosis_harness" / "stack_trace.prompt").read_text()
        word = text.split()[0]
        second = {"file": "stack_trace.prompt", "find": word, "replace": f"LEAN {word}"}
    else:
        second = {"file": "log_context_missing.prompt", "find": "TERSE", "replace": "TERSE LEAN"}
    calls = []

    async def llm(system, prompt):
        calls.append(1)
        e = edits[0] if len(calls) == 1 else second
        return json.dumps({"component": "prompt_fragments", "hypothesis": f"edit {len(calls)}", "edits": [e]})
    return llm


def _history(tmp_path):
    return [json.loads(x) for x in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]


def test_a_subsumed_edit_is_pruned(tmp_path):
    state = asyncio.run(_opt(tmp_path, SubsumedEvaluator(), two_edit_proposer(), prune=True).run_until_stopped())
    assert state.phase == "done" and state.accepted == ["r1", "r2"]
    assert state.prune["removed"] == ["r1"] and state.prune["kept"] == ["r2"]
    inc = tmp_path / "run" / "incumbent"
    assert "TERSE" not in (inc / "log_context_missing.prompt").read_text()
    assert "LEAN" in (inc / "stack_trace.prompt").read_text()
    assert "pruned r1" in state.stop_reason
    outcomes = [(h["candidate_id"], h["outcome"]) for h in _history(tmp_path)]
    assert outcomes[-2:] == [("r1", "pruned"), ("r2", "prune_kept")]


def test_same_file_edits_are_not_pruned_separately(tmp_path, monkeypatch):
    class SameFile(SubsumedEvaluator):
        async def evaluate(self, harness_dir, case_ids, trials):
            out = await super().evaluate(harness_dir, case_ids, trials)
            if "TERSE LEAN" in (Path(harness_dir) / "log_context_missing.prompt").read_text():
                (case,) = case_ids
                r = out.result.per_case[case]
                r.cost_usd = 0.4 * trials
            return out
    state = asyncio.run(_opt(tmp_path, SameFile(), two_edit_proposer("same"), prune=True).run_until_stopped())
    assert state.accepted == ["r1", "r2"]
    assert state.prune["kept"][0] == "r1" and state.prune["removed"] == []
    first = next(h for h in _history(tmp_path) if h["outcome"] == "prune_kept")
    assert "later accepted edit changed the same files" in first["reason"]


def test_prune_off_keeps_every_accepted_edit_and_resume_mid_prune_pays_nothing_twice(tmp_path):
    state = asyncio.run(_opt(tmp_path, SubsumedEvaluator(), two_edit_proposer()).run_until_stopped())
    assert state.prune is None
    assert "TERSE" in (tmp_path / "run" / "incumbent" / "log_context_missing.prompt").read_text()

    other = tmp_path / "b"
    other.mkdir()
    ev = SubsumedEvaluator()
    asyncio.run(_opt(other, ev, two_edit_proposer(), prune=True).run_until_stopped())
    st = json.loads((other / "run" / "state.json").read_text())
    st.update(phase="prune", stop_reason=None, prune={"next": 0, "removed": [], "kept": []})
    for name in ("log_context_missing.prompt",):     # back to the pre-prune incumbent
        (other / "run" / "incumbent" / name).write_text(
            (other / "run" / "candidates" / "r2" / name).read_text())
    st["incumbent_eval"] = json.dumps({"hash": candidates.content_hash(other / "run" / "candidates" / "r2"),
                                       "trials": 1, "cases": json.loads(st["incumbent_eval"])["cases"]})
    (other / "run" / "state.json").write_text(json.dumps(st))
    ev2 = SubsumedEvaluator()
    again = asyncio.run(_opt(other, ev2, two_edit_proposer(), prune=True).run_until_stopped())
    assert again.prune["removed"] == ["r1"] and ev2.calls == 0


def test_a_stop_during_a_prune_evaluation_retries_that_edit(tmp_path):
    from harness_evolve.evaluator import ProviderFailure

    class FailsOnce(SubsumedEvaluator):
        failed = False

        async def evaluate(self, harness_dir, case_ids, trials):
            h = Path(harness_dir)
            if h.name.startswith("prune-") and not FailsOnce.failed:
                FailsOnce.failed = True
                raise ProviderFailure("credit balance is too low")
            return await super().evaluate(harness_dir, case_ids, trials)

    first = asyncio.run(_opt(tmp_path, FailsOnce(), two_edit_proposer(), prune=True).run_until_stopped())
    assert first.phase == "prune" and first.prune["next"] == 0
    again = asyncio.run(_opt(tmp_path, FailsOnce(), two_edit_proposer(), prune=True).run_until_stopped())
    assert again.prune["removed"] == ["r1"] and again.phase == "done"
