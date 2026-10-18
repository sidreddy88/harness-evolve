"""GitHubPublisher with real git (a bare remote and a clone) and a fake `gh` that keeps PRs in a JSON file:
one draft PR per run, one commit per accepted edit, idempotent re-publishing, finalize (pruning sync,
report, ready), refusing a stale base harness, and failures that never stop the run."""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import textwrap

import pytest

from harness_evolve.github import GitHubPublisher, description
from harness_evolve.state import RunDir

from .conftest import FIXTURE_HARNESS

FAKE_GH = textwrap.dedent('''
    import json, os, sys
    db = os.environ["FAKE_GH_DB"]
    prs = json.load(open(db)) if os.path.exists(db) else []
    a = sys.argv[1:]
    def opt(name):
        return a[a.index(name) + 1] if name in a else None
    if a[:2] == ["pr", "list"]:
        print(json.dumps([{"number": p["number"], "url": p["url"], "isDraft": p["draft"]} for p in prs
                          if p["head"] == opt("--head") and p["state"] == "open"]))
    elif a[:2] == ["pr", "create"]:
        n = len(prs) + 1
        prs.append({"number": n, "url": f"https://github.com/o/r/pull/{n}", "head": opt("--head"),
                    "base": opt("--base"), "title": opt("--title"), "body": open(opt("--body-file")).read(),
                    "draft": "--draft" in a, "state": "open"})
        print("Creating pull request...\\n" + prs[-1]["url"])
    elif a[:2] == ["pr", "edit"]:
        prs[int(a[2]) - 1]["body"] = open(opt("--body-file")).read()
    elif a[:2] == ["pr", "ready"]:
        prs[int(a[2]) - 1]["draft"] = False
    else:
        sys.exit("unexpected gh call: " + " ".join(a))
    json.dump(prs, open(db, "w"))
''')


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    remote, repo = tmp_path / "remote.git", tmp_path / "repo"
    git(tmp_path, "init", "--quiet", "--bare", "-b", "main", str(remote))
    git(tmp_path, "clone", "--quiet", str(remote), str(repo))
    for k, v in (("user.email", "t@example.com"), ("user.name", "T")):
        git(repo, "config", k, v)
    shutil.copytree(FIXTURE_HARNESS, repo / "agent" / "harness")
    (repo / "app.py").write_text("print('agent')\n")
    git(repo, "add", "-A")
    git(repo, "commit", "--quiet", "-m", "init")
    git(repo, "push", "--quiet", "origin", "HEAD:main")
    gh = tmp_path / "gh.py"
    gh.write_text(FAKE_GH)
    gh_cmd = tmp_path / "gh"
    gh_cmd.write_text(f"#!/bin/sh\nexec {sys.executable} {gh} \"$@\"\n")
    gh_cmd.chmod(0o755)
    db = tmp_path / "prs.json"
    monkeypatch.setenv("FAKE_GH_DB", str(db))
    for k, v in (("GIT_AUTHOR_NAME", "T"), ("GIT_AUTHOR_EMAIL", "t@example.com"),
                 ("GIT_COMMITTER_NAME", "T"), ("GIT_COMMITTER_EMAIL", "t@example.com")):
        monkeypatch.setenv(k, v)
    run = RunDir(tmp_path / "runs" / "r-1")
    shutil.copytree(FIXTURE_HARNESS, run.original_dir)
    shutil.copytree(FIXTURE_HARNESS, run.incumbent_dir)
    run.state_path.write_text(json.dumps({"phase": "propose", "S_star": 0.6, "delta": 0.05}))
    pub = GitHubPublisher(repo, "agent/harness", gh=str(gh_cmd))
    return {"repo": repo, "remote": remote, "run": run, "pub": pub, "db": db, "tmp": tmp_path}


def prs(s):
    return json.loads(s["db"].read_text()) if s["db"].exists() else []


def candidate(s, name, **settings):
    d = s["tmp"] / "cands" / name
    shutil.copytree(s["run"].incumbent_dir, d)
    cur = json.loads((d / "settings.json").read_text())
    (d / "settings.json").write_text(json.dumps({**cur, **settings}, indent=1))
    return d


def accept(s, r, cand, hypothesis):
    hist = s["run"].root / "history.jsonl"
    with hist.open("a") as f:
        f.write(json.dumps({"round": r, "candidate_id": f"r{r}", "component": "settings", "outcome": "accepted",
                            "reason": "clear gain", "hypothesis": hypothesis, "delta_S": 0.2, "delta_C": 0.1,
                            "improved": ["a", "b"], "regressed": []}) + "\n")
    out = s["pub"].publish(s["run"], r, cand, "diff", {"reason": "clear gain", "improved": ["a"], "regressed": []},
                           hypothesis)
    shutil.rmtree(s["run"].incumbent_dir)
    shutil.copytree(cand, s["run"].incumbent_dir)
    return json.loads((out / "github.json").read_text())


def remote_settings(s, ref):
    return json.loads(git(s["remote"], "show", f"{ref}:agent/harness/settings.json"))


def test_one_draft_pr_per_run_one_commit_per_edit(setup):
    s = setup
    git(s["repo"], "switch", "--quiet", "-c", "my-work")
    (s["repo"] / "scratch.txt").write_text("uncommitted\n")
    rec = accept(s, 1, candidate(s, "c1", max_iterations=20), "more turns")
    assert rec["committed"] and rec["pull_request"] == "https://github.com/o/r/pull/1"
    assert remote_settings(s, "harness-evolve/r-1")["max_iterations"] == 20
    assert remote_settings(s, "main")["max_iterations"] == 15                     # base untouched
    assert prs(s)[0]["draft"] and prs(s)[0]["base"] == "main" and "more turns" in prs(s)[0]["body"]

    rec2 = accept(s, 2, candidate(s, "c2", grep_max_matches=80), "more grep matches")
    assert rec2["committed"] and len(prs(s)) == 1                                 # same PR, updated
    log = git(s["remote"], "log", "--format=%s", "harness-evolve/r-1").splitlines()
    assert log[:2] == ["harness-evolve r2: more grep matches", "harness-evolve r1: more turns"]
    assert "more grep matches" in prs(s)[0]["body"]

    again = s["pub"].sync(s["run"], s["run"].incumbent_dir, "same", "")          # a resume re-publishing
    assert not again["committed"] and len(prs(s)) == 1
    assert git(s["repo"], "branch", "--show-current") == "my-work"               # your checkout untouched
    assert (s["repo"] / "scratch.txt").read_text() == "uncommitted\n"
    assert "harness-evolve/r-1" not in git(s["repo"], "branch", "--list")


def test_finalize_syncs_pruning_adds_the_report_and_marks_ready(setup):
    s = setup
    accept(s, 1, candidate(s, "c1", max_iterations=20), "more turns")
    pruned = candidate(s, "c-pruned", max_iterations=15)                       # pruning dropped r1's change
    shutil.rmtree(s["run"].incumbent_dir)
    shutil.copytree(pruned, s["run"].incumbent_dir)
    (s["run"].root / "report").mkdir()
    (s["run"].root / "report" / "report.md").write_text("# Held-out report\n\n+0.4 better\n```\n{json}\n```\n")
    s["run"].state_path.write_text(json.dumps({"phase": "done", "prune": {"removed": ["r1"]}}))
    s["pub"].finalize(s["run"])
    pr = prs(s)[0]
    assert not pr["draft"] and "+0.4 better" in pr["body"] and "{json}" not in pr["body"]
    assert "r1 (removed by pruning)" in pr["body"]
    assert remote_settings(s, "harness-evolve/r-1")["max_iterations"] == 15
    assert git(s["remote"], "log", "-1", "--format=%s", "harness-evolve/r-1").startswith("harness-evolve: final")


def test_refuses_when_the_repo_harness_changed_since_the_run_started(setup):
    s = setup
    (s["repo"] / "agent" / "harness" / "task_prompt.prompt").write_text("edited on main meanwhile {x}\n")
    git(s["repo"], "commit", "--quiet", "-am", "edit")
    git(s["repo"], "push", "--quiet", "origin", "HEAD:main")
    rec = accept(s, 1, candidate(s, "c1", max_iterations=20), "more turns")
    assert "differs from the run's starting harness" in rec["error"]
    assert not git(s["remote"], "branch", "--list", "harness-evolve/*") and prs(s) == []
    assert (s["run"].root / "proposals" / "r1" / "harness.diff").exists()          # the local proposal still is


def test_failures_are_recorded_not_raised(setup):
    s = setup
    s["pub"].gh = "/nonexistent/gh"
    with pytest.raises(FileNotFoundError):
        s["pub"]._gh("pr", "list")
    s["pub"].gh = "false"
    rec = accept(s, 1, candidate(s, "c1", max_iterations=20), "more turns")
    assert "failed" in rec["error"]


def test_description_lists_edits_and_noise_band(setup):
    s = setup
    accept(s, 1, candidate(s, "c1", max_iterations=20), "more turns | pipes")
    text = description(s["run"])
    assert "noise band ±0.050" in text and "| r1 | settings | +0.200 | +10% |" in text
    assert "Improved 2 cases, regressed 0." in text and "still going" in text


def test_end_to_end_through_the_optimizer(setup, monkeypatch):
    """A real run (scripted proposer, command agent) accepts an edit; the publisher opens the PR and the
    finished run marks it ready."""
    from harness_evolve.bench import scripted_llm
    from harness_evolve.command import CommandEvaluator
    from harness_evolve.loop import Optimizer, OptimizerConfig

    s = setup
    agent = s["tmp"] / "agent.py"
    agent.write_text("import json, sys\nfrom pathlib import Path\n"
                     "s = json.loads((Path(sys.argv[1]) / 'settings.json').read_text())\n"
                     "ok = s['max_iterations'] >= 20 or int(sys.argv[2][1:]) < 8\n"     # round 0: 8 of 12
                     "print(json.dumps({'passed': ok, 'cost_usd': 0.001,"
                     " 'steps': [{'name': 'grep_codebase', 'input': {}, 'output': 'ok'}]}))\n")
    ids = [f"c{i}" for i in range(12)]
    cfg = OptimizerConfig(evolve_cases=ids, guard_cases=[], budget_usd=1.0, max_rounds=1, trials=1,
                          allow_unpriced_llm=True)
    edit = {"component": "settings", "hypothesis": "more turns",
            "edits": [{"file": "settings.json", "find": '"max_iterations": 15', "replace": '"max_iterations": 20'}]}
    llm = scripted_llm(edit)
    opt = Optimizer(s["tmp"] / "runs" / "e2e", FIXTURE_HARNESS, cfg,
                    evaluator=CommandEvaluator([sys.executable, str(agent), "{harness}", "{case}"]),
                    proposer_llm=llm, critic_llm=llm, critic_patterns=[], publisher=s["pub"])
    state = asyncio.run(opt.run_until_stopped())
    assert state.accepted == ["r1"], state.stop_reason
    pr = prs(s)[0]
    assert pr["head"] == "harness-evolve/e2e" and not pr["draft"] and "more turns" in pr["body"]
    assert remote_settings(s, "harness-evolve/e2e")["max_iterations"] == 20
