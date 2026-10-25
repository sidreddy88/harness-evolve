"""
Held-out results as agentstats attempt rows: one JSON line per trial, so the original and evolved harnesses can be
compared (or gated) with agentstats without harness-evolve depending on it.

    report/agentstats/original.jsonl     written at the end of every run (the final phase)
    report/agentstats/evolved.jsonl      only when an edit was accepted
    harness-evolve export RUN_DIR        the same for a run that finished before this existed

    agentstats compare --a report/agentstats/original.jsonl --b report/agentstats/evolved.jsonl

Row format (agentstats' input; see its README):
    {"arm": "original", "run": 1, "case": "...", "passed": true, "infra_error": null,
     "cost_usd": 0.031, "steps": 12, "tags": []}

- run is the trial number, so trial t of every case forms run t.
- cost_usd is the trial's own cost when the trajectory records it, else the case's cost split evenly.
- steps is a count (trajectories store a list of steps; agentstats wants a number).
- infra_error is always null: harness-evolve stops a run on a provider failure (ProviderFailure) instead of scoring it,
  so no cached trial is an infrastructure failure. TIMEOUT and ERROR verdicts are the agent's failures and count as
  failed attempts, as they do in harness-evolve's own report; they are tagged verdict:TIMEOUT / verdict:ERROR.
"""
from __future__ import annotations

import json
from pathlib import Path

from harness_evolve import candidates
from harness_evolve.state import RunDir

ARMS = ("original", "evolved")


def _trial_cost(traj: dict, fallback: float) -> float:
    cost = traj.get("cost")
    if isinstance(cost, dict) and cost.get("cost_usd") is not None:
        return float(cost["cost_usd"])
    if traj.get("cost_usd") is not None:
        return float(traj["cost_usd"])
    return fallback


def case_rows(arm: str, case: str, data: dict) -> list[dict]:
    """Rows for one cached case result ({"case": CaseResult fields, "trajectories": [...]})."""
    res = data["case"]
    verdicts = res.get("verdicts") or (["PASS"] * res["passes"] + ["FAIL"] * (res["trials"] - res["passes"]))
    trajs = data.get("trajectories") or []
    even = res.get("cost_usd", 0.0) / len(verdicts) if verdicts else 0.0
    rows = []
    for i, verdict in enumerate(verdicts):
        traj = trajs[i] if i < len(trajs) else {}
        row = {"arm": arm, "run": int(traj.get("trial") or i + 1), "case": case, "passed": verdict == "PASS",
               "infra_error": None, "cost_usd": _trial_cost(traj, even),
               "tags": [] if verdict in ("PASS", "FAIL") else [f"verdict:{verdict}"]}
        steps = traj.get("steps")
        if isinstance(steps, list):
            row["steps"] = len(steps)
        elif isinstance(steps, (int, float)):
            row["steps"] = steps
        rows.append(row)
    return rows


def arm_rows(run: RunDir, arm: str, harness_hash: str, cases: list[str], trials: int) -> list[dict]:
    rows, missing = [], []
    for case in cases:
        data = run.load_eval(f"{harness_hash}/k{trials}/{case}")
        if data is None:
            missing.append(case)
            continue
        rows.extend(case_rows(arm, case, data))
    if missing:
        raise FileNotFoundError(f"{arm}: no held-out result for {len(missing)} of {len(cases)} cases "
                                f"(e.g. {missing[0]}); has the run reached its final phase?")
    return rows


def write(out_dir: Path, rows_by_arm: dict[str, list[dict]]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for arm, rows in rows_by_arm.items():
        p = out_dir / f"{arm}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        paths.append(p)
    return paths


def final_rows(run: RunDir, cases: list[str], trials: int) -> dict[str, list[dict]]:
    """The held-out rows of a run: the original, and the evolved harness if it differs."""
    orig_hash = candidates.content_hash(run.original_dir)
    evolved_hash = candidates.content_hash(run.incumbent_dir)
    out = {"original": arm_rows(run, "original", orig_hash, cases, trials)}
    if evolved_hash != orig_hash:
        out["evolved"] = arm_rows(run, "evolved", evolved_hash, cases, trials)
    return out


def export(run_dir: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    """Write a run's held-out results as agentstats rows (default: RUN_DIR/report/agentstats/)."""
    run = RunDir(run_dir)
    if not run.state_path.exists():
        raise FileNotFoundError(f"{run_dir}: no state.json (not a harness-evolve run directory)")
    cfg = json.loads(run.state_path.read_text()).get("config") or {}
    cases, trials = list(cfg.get("final_cases") or []), int(cfg.get("final_trials") or 0)
    if not cases or not trials:
        raise ValueError(f"{run_dir}: the run's config has no held-out cases (final_cases) or final_trials")
    return write(Path(out_dir) if out_dir else run.root / "report" / "agentstats", final_rows(run, cases, trials))
