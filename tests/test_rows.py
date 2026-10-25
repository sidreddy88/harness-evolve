"""Held-out results as agentstats rows (rows.py): the format, and export of a finished run."""
from __future__ import annotations

import json

import pytest

from harness_evolve import candidates, cli, rows
from harness_evolve.state import RunDir


def _run(tmp_path, *, same=False):
    """A finished run directory: original/ and incumbent/ harnesses, held-out results cached at k=2."""
    run = RunDir(tmp_path / "run")
    for d, text in ((run.original_dir, "v1"), (run.incumbent_dir, "v1" if same else "v2")):
        d.mkdir(parents=True)
        (d / "system.prompt").write_text(text)
    run.state_path.write_text(json.dumps({"config": {"final_cases": ["c1", "c2"], "final_trials": 2}}))
    for d, verdicts in ((run.original_dir, ["PASS", "FAIL"]), (run.incumbent_dir, ["PASS", "TIMEOUT"])):
        for case in ("c1", "c2"):
            data = {"case": {"passes": verdicts.count("PASS"), "trials": 2, "escalations": 0, "cost_usd": 0.3,
                             "verdicts": verdicts},
                    "trajectories": [{"trial": 1, "steps": [{}, {}, {}], "cost": {"cost_usd": 0.1}},
                                     {"trial": 2, "steps": [{}], "cost": {"cost_usd": 0.2}}]}
            p = run.eval_path(f"{candidates.content_hash(d)}/k2/{case}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data))
    return run


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_case_rows_one_row_per_trial_with_trial_cost_and_step_count():
    data = {"case": {"passes": 1, "trials": 2, "cost_usd": 0.3, "verdicts": ["PASS", "ERROR"]},
            "trajectories": [{"trial": 1, "steps": [{}, {}], "cost": {"cost_usd": 0.1}},
                             {"trial": 2, "steps": [], "cost_usd": 0.2}]}
    r = rows.case_rows("evolved", "c1", data)
    assert r == [
        {"arm": "evolved", "run": 1, "case": "c1", "passed": True, "infra_error": None, "cost_usd": 0.1,
         "tags": [], "steps": 2},
        {"arm": "evolved", "run": 2, "case": "c1", "passed": False, "infra_error": None, "cost_usd": 0.2,
         "tags": ["verdict:ERROR"], "steps": 0},
    ]


def test_case_rows_without_trajectories_split_the_case_cost_evenly():
    r = rows.case_rows("original", "c1", {"case": {"passes": 1, "trials": 2, "cost_usd": 0.5}})
    assert [x["passed"] for x in r] == [True, False]
    assert [x["cost_usd"] for x in r] == [0.25, 0.25] and [x["run"] for x in r] == [1, 2]
    assert "steps" not in r[0]


def test_export_writes_both_arms(tmp_path):
    run = _run(tmp_path)
    paths = rows.export(run.root)
    assert [p.name for p in paths] == ["original.jsonl", "evolved.jsonl"]
    orig, evolved = _read(paths[0]), _read(paths[1])
    assert len(orig) == len(evolved) == 4
    assert [r["passed"] for r in orig] == [True, False, True, False]
    assert [r["tags"] for r in evolved][:2] == [[], ["verdict:TIMEOUT"]]
    assert [r["steps"] for r in orig][:2] == [3, 1] and [r["cost_usd"] for r in orig][:2] == [0.1, 0.2]


def test_export_with_nothing_accepted_writes_only_the_original(tmp_path):
    run = _run(tmp_path, same=True)
    assert [p.name for p in rows.export(run.root, tmp_path / "out")] == ["original.jsonl"]


def test_export_refuses_a_run_without_heldout_results(tmp_path):
    run = _run(tmp_path)
    next((run.root / "evals").rglob("c2.json")).unlink()
    with pytest.raises(FileNotFoundError, match="no held-out result for 1 of 2 cases"):
        rows.export(run.root)
    run.state_path.write_text(json.dumps({"config": {}}))
    with pytest.raises(ValueError, match="no held-out cases"):
        rows.export(run.root)


def test_cli_export(tmp_path, capsys):
    run = _run(tmp_path)
    assert cli.main(["export", str(run.root), "--out", str(tmp_path / "rows")]) == 0
    assert "evolved.jsonl (4 rows)" in capsys.readouterr().out
    assert cli.main(["export", str(tmp_path / "nope")]) == 1
