"""
A generic trajectory grader: what's countable in any tool-using agent's run.

Works on the trajectory records your evaluator returns (see evidence.py):
    steps: [{"name": tool, "input": ..., "output": ...}], plus optional
    "submitted": bool (did the agent reach a final answer?).

Per trajectory it counts tool calls, repeated identical calls (same tool and
input: wasted turns), and tool errors (outputs starting with "Error"), and
records whether the run ended with a submission. `accepted` is what the loop
uses to calibrate the escalation guard: runs that never submit are the
"gave up" signal.

Pass `TrajectoryGrader()` as a profile's grader, or write your own with the
same three methods (grade, summarize, render) for domain checks.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Grade:
    instance_id: str
    accepted: bool
    calls: int = 0
    redundant_calls: int = 0
    tool_errors: int = 0
    tools: dict = field(default_factory=dict)


def _norm(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, sort_keys=True, default=str)


class TrajectoryGrader:
    __name__ = "generic trajectory checks"

    def grade(self, rec: dict) -> Grade:
        steps = rec.get("steps") or []
        seen: Counter = Counter()
        errors = 0
        for st in steps:
            seen[(st.get("name"), _norm(st.get("input")))] += 1
            if str(st.get("output") or "").lstrip().startswith("Error"):
                errors += 1
        submitted = rec.get("submitted")
        if submitted is None:
            submitted = rec.get("verdict") not in ("ERROR", "TIMEOUT", "ESCALATE", None)
        return Grade(instance_id=rec.get("instance_id", "?"), accepted=bool(submitted), calls=len(steps),
                     redundant_calls=sum(n - 1 for n in seen.values() if n > 1), tool_errors=errors,
                     tools=dict(Counter(st.get("name") for st in steps)))

    def summarize(self, grades: list[Grade]) -> dict:
        n = len(grades) or 1
        tools: Counter = Counter()
        for g in grades:
            tools.update(g.tools)
        return {"trajectories": len(grades),
                "never_submitted": sum(not g.accepted for g in grades),
                "mean_calls": round(sum(g.calls for g in grades) / n, 1),
                "redundant_calls": sum(g.redundant_calls for g in grades),
                "tool_errors": sum(g.tool_errors for g in grades),
                "calls_by_tool": dict(tools.most_common(10))}

    def render(self, s: dict) -> str:
        return (f"  {s['trajectories']} runs: {s['never_submitted']} never submitted; "
                f"{s['mean_calls']} tool calls per run; {s['redundant_calls']} repeated identical calls; "
                f"{s['tool_errors']} tool errors\n  calls by tool: {s['calls_by_tool']}")
