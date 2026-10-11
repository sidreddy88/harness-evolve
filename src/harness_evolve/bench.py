"""
Benchmark the optimizer itself, so "this change makes harness-evolve better" is a measured claim.

A benchmark is a list of scenarios. Each builds an Optimizer in its own run directory and may state
what must happen:

- Known-answer scenarios: a scripted proposer (and critic) puts one known edit in front of the
  acceptance rules, so there is a right answer. A real fix to a planted flaw must be accepted (the
  rules can see a real gain); a harmful edit must be rejected; a neutral rewording must not be
  accepted (false accepts).
- Open scenarios: the real proposer, reported but not pass/fail: does it find a planted flaw, and
  what does a standard run gain on held-out cases, per dollar?

    harness-evolve bench examples/math_agent/bench.py --out runs/bench-1

Run directories persist, so a stopped benchmark resumes like any run. Results go to
<out>/bench.json and <out>/bench.md.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from harness_evolve.llm import LLM, LLMReply

ACCEPTED, NOT_ACCEPTED = "accepted", "not_accepted"


@dataclass
class Scenario:
    name: str
    description: str
    build: Callable[[Path], object]          # run_dir -> Optimizer
    expect: str | None = None                # ACCEPTED, NOT_ACCEPTED, or None (report only)


def scripted_llm(edit: dict, critic_accepts: bool = True) -> LLM:
    """A stand-in for the proposer and critic: the proposer always proposes `edit`
    ({"component", "hypothesis", "edits": [...]}); the critic always accepts (or rejects).
    With the critic scripted too, the scenario tests the acceptance rules alone."""
    async def call(system: str, prompt: str) -> LLMReply:
        if "strict reviewer" in system:
            verdict = "accept" if critic_accepts else "reject"
            return LLMReply(json.dumps({"verdict": verdict, "reasons": ["scripted"]}), 0.0)
        return LLMReply(json.dumps(edit), 0.0)
    return call


def outcome(run_dir: Path) -> dict:
    """What a finished scenario run did: rounds, acceptances, held-out numbers, spend."""
    state = json.loads((run_dir / "state.json").read_text())
    rounds = []
    hist = run_dir / "history.jsonl"
    if hist.exists():
        for line in hist.read_text().splitlines():
            e = json.loads(line)
            if not str(e.get("outcome", "")).startswith("pilot_"):
                rounds.append({k: e.get(k) for k in ("round", "component", "outcome", "delta_S", "delta_C",
                                                     "reason", "hypothesis")})
    out = {"phase": state.get("phase"), "stop_reason": state.get("stop_reason"),
           "accepted": state.get("accepted") or [], "pruned": (state.get("prune") or {}).get("removed", []),
           "spent_usd": round(state.get("spent_usd") or 0.0, 3),
           "S_star": state.get("S_star"), "delta": state.get("delta"), "rounds": rounds}
    rep = run_dir / "report" / "report.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        out["heldout"] = {
            "original_pass@1": (r.get("original") or {}).get("pass@1"),
            "evolved_pass@1": (r.get("evolved") or {}).get("pass@1"),
            "vs_original_single_run": (r.get("comparisons") or {}).get("vs_original_single_run"),
            "vs_rerun_baseline_best_of_2": (r.get("comparisons") or {}).get("vs_rerun_baseline_best_of_2"),
        }
    return out


def judge(expect: str | None, result: dict) -> bool | None:
    if result.get("phase") != "done":
        return False
    if expect == ACCEPTED:
        return bool(result["accepted"])
    if expect == NOT_ACCEPTED:
        return not result["accepted"]
    return None


async def run(scenarios: list[Scenario], out_dir: Path, only: list[str] | None = None) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for sc in scenarios:
        if only and sc.name not in only:
            continue
        run_dir = out_dir / sc.name
        opt = sc.build(run_dir)
        try:
            await opt.run_until_stopped()
        except Exception as exc:                       # one broken scenario shouldn't hide the rest
            results.append({"name": sc.name, "description": sc.description, "expect": sc.expect,
                            "passed": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        res = outcome(run_dir)
        results.append({"name": sc.name, "description": sc.description, "expect": sc.expect,
                        "passed": judge(sc.expect, res), **res})
    if only and (out_dir / "bench.json").exists():
        # A partial re-run updates its scenarios in the existing results instead of replacing them.
        previous = json.loads((out_dir / "bench.json").read_text())
        fresh = {r["name"]: r for r in results}
        names = [r["name"] for r in previous] + [n for n in fresh if n not in {r["name"] for r in previous}]
        old = {r["name"]: r for r in previous}
        results = [fresh.get(n, old.get(n)) for n in names]
    (out_dir / "bench.json").write_text(json.dumps(results, indent=1))
    (out_dir / "bench.md").write_text(render(results))
    return results


def render(results: list[dict]) -> str:
    def pct(x):
        return "-" if x is None else f"{x:.3f}"
    lines = ["# harness-evolve benchmark", "",
             "| Scenario | Expect | Result | Accepted | First round (dS, dC) "
             "| Held-out pass@1 (orig → evolved) | Spent |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        verdict = {True: "PASS", False: "FAIL", None: "report"}[r.get("passed")]
        if r.get("error"):
            lines.append(f"| {r['name']} | {r.get('expect') or '-'} | ERROR | - | {r['error'][:80]} | - | - |")
            continue
        first = next(iter(r.get("rounds") or []), None)
        fr = (f"{first['outcome']} ({first['delta_S']:+.3f}, {first['delta_C']:+.0%})"
              if first and first.get("delta_S") is not None and first.get("delta_C") is not None
              else (first or {}).get("outcome", "-"))
        h = r.get("heldout") or {}
        ho = (f"{pct(h.get('original_pass@1'))} → {pct(h.get('evolved_pass@1'))}" if h else "-")
        lines.append(f"| {r['name']} | {r.get('expect') or '-'} | {verdict} | {r.get('accepted') or 'none'} | "
                     f"{fr} | {ho} | ${r.get('spent_usd', 0):.2f} |")
    lines += ["", *[f"- **{r['name']}**: {r['description']}" for r in results]]
    return "\n".join(lines) + "\n"
