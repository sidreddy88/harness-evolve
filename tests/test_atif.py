"""ATIF (Harbor) trajectories as harness-evolve evidence."""
from __future__ import annotations

import json

from harness_evolve import evidence
from harness_evolve.acceptance import CaseResult, EvalResult
from harness_evolve.atif import from_atif, harbor_job, harbor_trial
from harness_evolve.grader import TrajectoryGrader

ATIF = {
    "schema_version": "ATIF-v1.7",
    "agent": {"name": "mini-swe-agent", "version": "2.4.6", "model_name": "some/model"},
    "steps": [
        {"step_id": 1, "source": "system", "message": "You are a helpful assistant."},
        {"step_id": 2, "source": "user", "message": "Fix the bug."},
        {"step_id": 3, "source": "agent", "message": "Look around.", "llm_call_count": 1,
         "tool_calls": [{"tool_call_id": "a", "function_name": "bash", "arguments": {"command": "ls"}},
                        {"tool_call_id": "b", "function_name": "bash", "arguments": {"command": "cat x.py"}}],
         "observation": {"results": [{"source_call_id": "b", "content": "print(1)"},
                                     {"source_call_id": "a", "content": "x.py"}]},
         "metrics": {"prompt_tokens": 100, "completion_tokens": 10, "cost_usd": 0.001}},
        {"step_id": 4, "source": "agent", "message": [{"type": "text", "text": "Run it."}],
         "tool_calls": [{"tool_call_id": "c", "function_name": "bash", "arguments": {"command": "python x.py"}}],
         "observation": {"results": [{"content": [{"type": "text", "text": "Error: boom"},
                                                  {"type": "image", "source": {"path": "s.png"}}]}]},
         "metrics": {"cost_usd": 0.002}},
    ],
    "final_metrics": {"total_prompt_tokens": 300, "total_completion_tokens": 30, "total_cost_usd": 0.0035},
}


def test_tool_calls_are_matched_to_outputs_by_id_then_order():
    rec = from_atif(ATIF, instance_id="case-1", verdict="FAIL")
    assert [(s["name"], s["input"]["command"], s["output"]) for s in rec["steps"]] == [
        ("bash", "ls", "x.py"), ("bash", "cat x.py", "print(1)"), ("bash", "python x.py", "Error: boom\n[image]")]
    assert rec["turns"] == 2 and rec["cost"] == {"cost_usd": 0.0035}
    assert rec["usage"]["total_prompt_tokens"] == 300 and rec["agent"]["name"] == "mini-swe-agent"


def test_cost_falls_back_to_step_metrics_and_long_outputs_are_cut():
    d = json.loads(json.dumps(ATIF))
    d.pop("final_metrics")
    d["steps"][2]["observation"]["results"][1]["content"] = "y" * 5000
    rec = from_atif(d, instance_id="c", max_output_chars=100)
    assert rec["cost"]["cost_usd"] == 0.003
    assert rec["steps"][0]["output"].startswith("y" * 100) and "4900 more chars" in rec["steps"][0]["output"]


def test_answer_is_hidden_from_the_proposer_and_records_work_as_evidence():
    rec = from_atif(ATIF, instance_id="case-1", verdict="FAIL", detail="tests failed", answer="fix is in y.py")
    assert "y.py" not in evidence.outcome(rec)
    text = evidence.build(EvalResult({"case-1": CaseResult(passes=0, trials=1, verdicts=["FAIL"])}), [rec])
    assert "y.py" not in text and "python x.py" in text
    g = TrajectoryGrader().grade(rec)
    assert g.calls == 3 and g.tool_errors == 1


def test_harbor_trial_and_job_layout(tmp_path):
    def trial(name, reward=None, exc=None, with_traj=True):
        d = tmp_path / "job" / name
        (d / "agent").mkdir(parents=True)
        res = {"verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
               "exception_info": {"exception_type": exc} if exc else None,
               "agent_result": {"cost_usd": 0.5}}
        (d / "result.json").write_text(json.dumps(res))
        if with_traj:
            (d / "agent" / "trajectory.json").write_text(json.dumps(ATIF))
        return d

    trial("django__django-11099__AbC123", reward=1.0)
    trial("astropy__astropy-12907__XyZ789", reward=0.0)
    trial("sympy__sympy-20916__Q1w2E3", exc="NonZeroAgentExitCodeError", with_traj=False)
    (tmp_path / "job" / "config.json").write_text("{}")
    recs = {r["instance_id"]: r for r in harbor_job(tmp_path / "job")}
    assert recs["django__django-11099"]["verdict"] == "PASS"
    assert recs["astropy__astropy-12907"]["verdict"] == "FAIL" and len(recs["astropy__astropy-12907"]["steps"]) == 3
    assert recs["sympy__sympy-20916"]["verdict"] == "ERROR" and recs["sympy__sympy-20916"]["cost"]["cost_usd"] == 0.5
    one = harbor_trial(tmp_path / "job" / "django__django-11099__AbC123", trial=2)
    assert one["trial"] == 2
