"""
Evidence for the proposer: what the agent actually did under the incumbent.

Built from the trajectory records your evaluator returns (EvalOutcome.trajectories,
one dict per trial). Layers, from aggregate to specific:

1. Per case: verdict per trial, turns used, cost.
2. Trajectory checks from the profile's grader, if it has one (redundant calls,
   tool errors, runs that never finished, ...), and cost by prompt source if
   trials report it, so the proposer can see where turns and money go.
3. The agent's own words about the setup after failing runs (self_feedback).
4. A few failing trajectories, turn by turn, so an edit can be aimed at a
   concrete failure, as in GEPA.

Trajectory record keys this module reads (all optional except instance_id):
    instance_id, trial, verdict ("PASS"/"FAIL"/...), detail, turns,
    steps: [{"name", "input", "output"}], self_feedback,
    cost_by_source: {source: $}, e.g. {"system prompt": 0.01, "tool:search": 0.03}

Never let the right answer reach the proposer: put anything after
HIDDEN_MARKER in `detail` and it is cut off before the proposer sees it.
Case ids and inputs do appear (the proposer needs to understand failures);
the critic's precheck keeps them out of the harness itself.
"""
from __future__ import annotations

from harness_evolve.acceptance import EvalResult


HIDDEN_MARKER = "[[hidden]]"


def outcome(rec: dict) -> str:
    """A trajectory's verdict and detail, with anything after HIDDEN_MARKER removed."""
    verdict, detail = rec.get("verdict"), rec.get("detail") or ""
    if verdict == "PASS":
        return "PASS"
    shown, cut, _ = detail.partition(HIDDEN_MARKER)
    return f"{verdict}: {shown.rstrip()}" if (shown or not cut) else f"{verdict}"


def _cost_by_source(trajectories: list[dict]) -> list[tuple[str, float, float]]:
    totals: dict[str, float] = {}
    for r in trajectories:
        for src, c in (r.get("cost_by_source") or {}).items():
            totals[src] = totals.get(src, 0.0) + float(c or 0.0)
    whole = sum(totals.values())
    return [(k, v / whole, v) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])] if whole else []


def build(result: EvalResult, trajectories: list[dict], max_failing: int = 3,
          max_chars: int = 20_000) -> str:
    lines = [f"Incumbent: S={result.S:.3f} (mean per-case pass rate), cost/trial=${result.C:.3f}, "
             f"escalation rate={result.escalation_rate:.1%}", "", "Per case:"]
    turns: dict[str, list[int]] = {}
    for rec in trajectories:
        turns.setdefault(rec["instance_id"], []).append(rec.get("turns") or len(rec.get("llm_calls") or []))
    for cid, cr in sorted(result.per_case.items()):
        lines.append(f"  {cid}: {cr.verdicts} turns={turns.get(cid, [])} cost=${cr.cost_usd:.2f}")

    from harness_evolve import profiles
    g_mod = profiles.active().grader
    grades: list = [None] * len(trajectories)
    if g_mod is not None and trajectories:
        grades = [g_mod.grade(rec) for rec in trajectories]
        name = getattr(g_mod, "__name__", type(g_mod).__name__)
        lines += ["", f"Trajectory checks ({name}):", g_mod.render(g_mod.summarize(grades))]

    by_source = _cost_by_source(trajectories)
    if by_source:
        lines += ["", "Cost by prompt source (share of spend):"]
        lines += [f"  {src}: {share:.1%} (${cost:.2f})" for src, share, cost in by_source[:10]]

    feedback = [(r["instance_id"], r["self_feedback"]) for r in trajectories
                if r.get("self_feedback") and r.get("verdict") != "PASS"][:6]
    if feedback:
        lines += ["", "Agent self-feedback after failing runs (the agent's own words about the setup):"]
        lines += [f"  - {fb[:400]}" for _, fb in feedback]

    failing = [(r, g) for r, g in zip(trajectories, grades) if r.get("verdict") != "PASS"][:max_failing]
    for rec, g in failing:
        lines += ["", f"=== Failing trajectory: {rec['instance_id']} trial {rec.get('trial')} "
                      f"({outcome(rec)[:160]}) ==="]
        lines += [f"  flag: {f}" for f in (getattr(g, "flags", None) or [])]
        for i, step in enumerate(rec.get("steps") or [], 1):
            out = str(step.get("output", ""))
            # Short outputs and rejections/errors are shown; long ones only by size.
            shown = out.lstrip().startswith(("REJECTED", "Error")) or len(out) <= 200
            note = f" -> {out[:300]!r}" if shown else f" -> {len(out)} chars"
            turn = step.get("iteration", i)
            lines.append(f"  turn {turn}: {step.get('name')}({str(step.get('input'))[:120]}){note}")

    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "\n[evidence truncated]"
