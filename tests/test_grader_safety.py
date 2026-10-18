"""Grader safety: anchors graded before any agent run, the always-pass tripwire in round 0, and the task
contract shown to the critic. Offline: fake evaluator, scripted LLMs."""
from __future__ import annotations

import asyncio
import json
import sys

from harness_evolve import AgentProfile, anchors
from harness_evolve.acceptance import CaseResult, EvalResult
from harness_evolve.bench import scripted_llm
from harness_evolve.command import CommandEvaluator
from harness_evolve.evaluator import EvalOutcome
from harness_evolve.loop import Optimizer, OptimizerConfig

from .conftest import FIXTURE_HARNESS

CASES = [f"c{i}" for i in range(20)]


class Fake:
    """Round 0 passes `pass_share` of cases on every trial; grade() passes outputs in `passes`."""

    def __init__(self, pass_share=0.6, passes=("right",)):
        self.pass_share, self.passes, self.evaluated, self.graded = pass_share, set(passes), 0, []

    async def grade(self, case_id, output):
        self.graded.append((case_id, output))
        return {"passed": output in self.passes, "cost_usd": 0.001}

    async def evaluate(self, harness_dir, case_ids, trials):
        self.evaluated += 1
        per, trajs = {}, []
        for c in case_ids:
            ok = CASES.index(c) < self.pass_share * len(CASES)
            per[c] = CaseResult(trials if ok else 0, trials, 0, 0.1 * trials, ["PASS" if ok else "FAIL"] * trials)
            trajs += [{"instance_id": c, "trial": t + 1, "verdict": per[c].verdicts[t], "submitted": True,
                       "steps": [{"name": "grep_codebase", "input": {}, "output": "ok"}],
                       "cost": {"cost_usd": 0.1}} for t in range(trials)]
        return EvalOutcome(EvalResult(per), sum(r.cost_usd for r in per.values()), trajs)


def _run(tmp_path, evaluator, **cfg):
    llm = scripted_llm({})
    opt = Optimizer(tmp_path / "run", FIXTURE_HARNESS,
                    OptimizerConfig(**{"evolve_cases": CASES, "guard_cases": [], "budget_usd": 50.0, "max_rounds": 0,
                                       "allow_unpriced_llm": True, **cfg}),
                    evaluator=evaluator, proposer_llm=llm, critic_llm=llm, critic_patterns=[])
    return asyncio.run(opt.run_until_stopped())


GOOD = [{"case": "c0", "output": "right", "passed": True}, {"case": "c0", "output": "wrong", "passed": False}]


def test_good_anchors_let_the_run_start(tmp_path):
    fake = Fake()
    state = _run(tmp_path, fake, anchors=GOOD)
    assert state.phase == "done" and fake.evaluated > 0
    assert ("c0", "") in fake.graded                                  # the automatic blank output
    assert state.health["anchors"]["problems"] == [] and state.spent_usd > 0


def test_mislabelled_anchor_pauses_before_any_agent_run_and_resumes_once_fixed(tmp_path):
    fake = Fake(passes=("right", "wrong", ""))                        # passes everything it is shown
    state = _run(tmp_path, fake, anchors=GOOD)
    assert state.phase == "anchors" and fake.evaluated == 0
    assert "'wrong' should fail but passed" in state.stop_reason and "a blank output" in state.stop_reason
    fake.passes = {"right"}                                           # the user fixes the grader
    state = _run(tmp_path, fake, anchors=GOOD)
    assert state.phase == "done" and fake.evaluated > 0


def test_anchors_need_an_evaluator_that_can_grade(tmp_path):
    class NoGrade(Fake):
        grade = None
    state = _run(tmp_path, NoGrade(), anchors=GOOD)
    assert state.phase == "anchors" and "can't grade an output" in state.stop_reason
    state = _run(tmp_path / "b", Fake(), anchors=[{"case": "c0", "output": "x"}])
    assert "\"passed\" must be true or false" in state.stop_reason


def test_always_pass_round0_trips(tmp_path):
    state = _run(tmp_path, Fake(pass_share=1.0))
    assert state.phase == "evaluate" and "passed 100% of 40 evolve trials" in state.stop_reason
    assert _run(tmp_path / "off", Fake(pass_share=1.0), always_pass_tripwire=None).phase == "done"
    assert _run(tmp_path / "ok", Fake(pass_share=0.9)).phase == "done"     # 90% is a good agent, not a broken grader


def test_command_evaluator_grade(tmp_path):
    script = tmp_path / "grade.py"
    script.write_text("import json, sys\n"
                      "out = open(sys.argv[2]).read()\n"
                      "if out == 'boom': sys.exit('grader crashed')\n"
                      "print(json.dumps({'passed': out == sys.argv[1] + '!', 'cost_usd': 0.002}))\n")
    ev = CommandEvaluator(["x"], grade_command=[sys.executable, str(script), "{case}", "{output}"])
    probs, cost = asyncio.run(anchors.run(ev, [{"case": "q", "output": "q!", "passed": True},
                                              {"case": "q", "output": "nope", "passed": False}]))
    assert probs == [] and cost == 0.006                              # 2 anchors + the blank
    probs, _ = asyncio.run(anchors.run(ev, [{"case": "q", "output": "boom", "passed": False}]))
    assert "grader crashed" in probs[0]
    assert CommandEvaluator(["x"]).grade is None                      # no grade_command: can't grade alone


def test_anchor_file_and_problems(tmp_path):
    f = tmp_path / "anchors.jsonl"
    assert anchors.load(f) == []
    f.write_text(json.dumps(GOOD[0]) + "\n\n" + json.dumps({"case": 1, "output": "x", "passed": True}) + "\n")
    loaded = anchors.load(f)
    assert len(loaded) == 2 and anchors.problems(loaded) == ['anchor 2: needs a string "case" and a string "output"']
    blanks = anchors.with_blanks([GOOD[0], {"case": "c0", "output": " ", "passed": False}])
    assert len(blanks) == 2                                           # c0 already has a blank anchor


def test_task_contract_reaches_the_critic_only_when_set():
    with_contract = AgentProfile.build(name="t", harness_root=".", agent_description="d",
                                       task_contract="the final number equals the problem's answer")
    assert "WHAT COUNTS AS CORRECT (the task contract): the final number equals" in with_contract.critic_system
    assert "the task contract" not in with_contract.proposer_system
    assert "WHAT COUNTS AS CORRECT" not in AgentProfile.build(name="t", harness_root=".",
                                                              agent_description="d").critic_system
