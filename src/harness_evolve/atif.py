"""
Read agent trajectories in ATIF (Agent Trajectory Interchange Format), the format Harbor writes for
every agent it runs (`<trial>/agent/trajectory.json`), so an evaluator built on Harbor can hand them to
harness-evolve as evidence without writing its own parser.

    from harness_evolve.atif import from_atif, harbor_trial, harbor_job
    rec = from_atif(path_or_dict, instance_id="django__django-11099", verdict="FAIL")
    rec = harbor_trial("jobs/2026-10-10__13-03-04/django__django-11099__AbC123")   # reads the reward too
    recs = harbor_job("jobs/2026-10-10__13-03-04")

The result is a trajectory record as described in evidence.py: `steps` (tool name, input, output),
`turns`, `cost`, `verdict`, `detail`, plus `usage` (token totals). Plain JSON, no dependency on Harbor.
Tool outputs are matched to their calls by id (ATIF `observation.results[].source_call_id`), falling
back to order when ids are missing. Tested against ATIF v1.0–v1.8 field names.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness_evolve.evidence import HIDDEN_MARKER


def _text(content: Any) -> str:
    """ATIF message/observation content: a string, or a list of content parts (text, image, audio)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text") or "")
                else:
                    parts.append(f"[{p.get('type', 'content')}]")
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)


def _load(traj: dict | str | Path) -> dict:
    if isinstance(traj, dict):
        return traj
    return json.loads(Path(traj).read_text())


def from_atif(traj: dict | str | Path, *, instance_id: str, trial: int = 1, verdict: str | None = None,
              detail: str = "", answer: str | None = None, submitted: bool | None = None,
              max_output_chars: int = 4000) -> dict:
    """One ATIF trajectory as a harness-evolve trajectory record.

    verdict: "PASS"/"FAIL"/... from your grader (ATIF doesn't carry it). answer: the right answer or any
    detail the proposer must not see; it goes after HIDDEN_MARKER. submitted: whether the agent finished
    with a final answer (None lets the grader infer it from the verdict)."""
    d = _load(traj)
    steps, turns, step_cost = [], 0, 0.0
    for st in d.get("steps") or []:
        if st.get("source") != "agent":
            continue
        turns += int(st.get("llm_call_count") or 1)
        step_cost += float(((st.get("metrics") or {}).get("cost_usd")) or 0.0)
        results = ((st.get("observation") or {}).get("results")) or []
        by_id = {r.get("source_call_id"): r for r in results if r.get("source_call_id")}
        for i, call in enumerate(st.get("tool_calls") or []):
            res = by_id.get(call.get("tool_call_id"))
            if res is None and i < len(results) and not results[i].get("source_call_id"):
                res = results[i]
            out = _text((res or {}).get("content"))
            if len(out) > max_output_chars:
                out = out[:max_output_chars] + f"... [{len(out) - max_output_chars} more chars]"
            steps.append({"name": call.get("function_name"), "input": call.get("arguments") or {},
                          "output": out, "iteration": turns})
    fm = d.get("final_metrics") or {}
    cost = fm.get("total_cost_usd")
    rec = {
        "instance_id": instance_id, "trial": trial, "verdict": verdict,
        "detail": detail + (f" {HIDDEN_MARKER} {answer}" if answer else ""),
        "steps": steps, "turns": turns,
        "cost": {"cost_usd": float(cost) if cost is not None else round(step_cost, 6)},
        "usage": {k: fm.get(k) for k in ("total_prompt_tokens", "total_completion_tokens",
                                         "total_cached_tokens", "total_steps") if fm.get(k) is not None},
        "agent": {k: (d.get("agent") or {}).get(k) for k in ("name", "version", "model_name")},
        "schema_version": d.get("schema_version"),
    }
    if submitted is not None:
        rec["submitted"] = submitted
    return rec


def harbor_trial(trial_dir: str | Path, *, trial: int = 1, pass_threshold: float = 1.0) -> dict:
    """A Harbor trial directory (agent/trajectory.json + result.json) as a trajectory record. The
    verdict comes from the verifier's reward: PASS if reward >= pass_threshold, FAIL otherwise, ERROR
    if the trial raised. The case id is the directory name without Harbor's random suffix."""
    trial_dir = Path(trial_dir)
    result = json.loads((trial_dir / "result.json").read_text()) if (trial_dir / "result.json").exists() else {}
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    exc = (result.get("exception_info") or {}).get("exception_type")
    verdict = "ERROR" if exc else ("PASS" if reward is not None and reward >= pass_threshold else "FAIL")
    detail = f"exception {exc}" if exc else (f"reward {reward}" if reward is not None else "no reward")
    case = trial_dir.name.rsplit("__", 1)[0] if "__" in trial_dir.name else trial_dir.name
    traj = trial_dir / "agent" / "trajectory.json"
    if not traj.exists():
        return {"instance_id": case, "trial": trial, "verdict": verdict, "detail": detail, "steps": [],
                "turns": 0, "cost": {"cost_usd": float((result.get("agent_result") or {}).get("cost_usd") or 0.0)}}
    return from_atif(traj, instance_id=case, trial=trial, verdict=verdict, detail=detail)


def harbor_job(job_dir: str | Path, **kwargs) -> list[dict]:
    """Every trial in a Harbor job directory, as trajectory records (sorted by case id)."""
    return [harbor_trial(t, **kwargs) for t in sorted(Path(job_dir).iterdir())
            if t.is_dir() and (t / "result.json").exists()]
