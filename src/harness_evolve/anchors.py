"""
Grader anchors: outputs with known verdicts that the grader must label correctly before a run spends
anything on the agent.

A grader that passes everything looks like progress: in "Who Verifies the Verifier?"
(arXiv 2610.11464), a grader evolved without anchor guards collapsed into always-pass, and the agent
still appeared to improve just as well. harness-evolve never changes the grader, but since
`init` + CommandEvaluator a user brings their own (often an LLM judge or loose matching), and a run
trusts it completely. Anchors are the cheap check that it can tell right from wrong at all.

Each anchor (one JSON object per line in anchors.jsonl, or a dict in OptimizerConfig.anchors):

    {"case": "q1", "output": "Answer: 391", "passed": true, "why": "correct"}
    {"case": "q1", "output": "Answer: 390", "passed": false, "why": "off by one"}

Every anchored case also gets a blank output that must fail, so even a one-line anchors file checks
the commonest broken grader (one that passes an empty answer). Include clearly wrong outputs, and
near-misses your grader should reject (a copy of the question, the right number in the wrong format).

The evaluator grades an output without running the agent through an optional method:

    async def grade(self, case_id: str, output: str) -> dict:   # {"passed": bool, "cost_usd": float}

CommandEvaluator implements it with `grade_command`.
"""
from __future__ import annotations

import json
from pathlib import Path


def load(path: str | Path) -> list[dict]:
    """Anchors from a JSONL file; a missing file means none."""
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def problems(anchors: list[dict]) -> list[str]:
    """What's wrong with the anchors themselves (checked for free, before any grading)."""
    out = []
    for i, a in enumerate(anchors, 1):
        if not isinstance(a, dict) or not isinstance(a.get("case"), str) or not isinstance(a.get("output"), str):
            out.append(f"anchor {i}: needs a string \"case\" and a string \"output\"")
        elif not isinstance(a.get("passed"), bool):
            out.append(f"anchor {i} ({a['case']}): \"passed\" must be true or false")
    return out


def with_blanks(anchors: list[dict]) -> list[dict]:
    """The anchors plus, for each anchored case without one, a blank output that must fail."""
    blank = {a["case"] for a in anchors if not a["output"].strip()}
    extra = [{"case": c, "output": "", "passed": False, "why": "added automatically"}
             for c in dict.fromkeys(a["case"] for a in anchors) if c not in blank]
    return list(anchors) + extra


async def run(evaluator, anchors: list[dict]) -> tuple[list[str], float]:
    """Grade every anchor (and the automatic blanks). Returns (mislabelled anchors, cost)."""
    out, cost = [], 0.0
    for a in with_blanks(anchors):
        res = await evaluator.grade(a["case"], a["output"])
        cost += float(res.get("cost_usd") or 0.0)
        if res.get("error"):
            out.append(f"{a['case']}: grading {_show(a['output'])} failed: {res['error']}")
        elif bool(res.get("passed")) != a["passed"]:
            want = "pass" if a["passed"] else "fail"
            got = "passed" if res.get("passed") else "failed"
            out.append(f"{a['case']}: {_show(a['output'])} should {want} but {got}"
                       + (f" ({a['why']})" if a.get("why") else ""))
    return out, cost


def _show(text: str) -> str:
    text = text.strip().replace("\n", " ")
    return "a blank output" if not text else repr(text[:60] + ("..." if len(text) > 60 else ""))
