"""`harness-evolve serve`: pages and API over run directories, reviews, start/resume as separate
processes, check in a subprocess, and the guards (token, host, JSON-only POSTs, paths, escaping)."""
from __future__ import annotations

import json
import os
import textwrap
import time
from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from harness_evolve import server  # noqa: E402

from .conftest import FIXTURE_HARNESS  # noqa: E402

XSS = "<script>alert(1)</script>"


def _run(root, name, phase="done", **state):
    d = root / name
    (d / "proposals" / "r1").mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"phase": phase, "round": 2, "spent_usd": 1.25, "accepted": ["r1"],
                                              "S_star": 0.76, "delta": 0.056, "stop_reason": "completed 1 rounds",
                                              "config": {"budget_usd": 5.0}, **state}))
    (d / "history.jsonl").write_text(
        json.dumps({"round": 0, "outcome": "pilot_x"}) + "\n"
        + json.dumps({"round": 1, "candidate_id": "r1", "component": "settings", "outcome": "accepted",
                      "reason": "clear gain", "hypothesis": f"raise max_turns {XSS}", "delta_S": 0.5,
                      "delta_C": 0.2}) + "\n")
    (d / "proposals" / "r1" / "harness.diff").write_text(f"--- a/s\n+++ b/s\n-old\n+new {XSS}\n")
    (d / "proposals" / "r1" / "decision.json").write_text(json.dumps({"reason": "clear gain"}))
    (d / "proposals" / "r1" / "README.md").write_text("# Proposed edit\n")
    (d / "report").mkdir()
    (d / "report" / "report.md").write_text("# Held-out report\n+0.4 better\n")
    return d


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "runs"
    _run(r, "r1")
    _run(r, "bench/find-flaw", phase="evaluate", stop_reason="budget: needs ~$1")
    return r


def _client(root, token=None):
    return TestClient(server.create_app(root, token))


def test_lists_runs_including_nested_ones(root):
    runs = {r["id"]: r for r in _client(root).get("/api/runs").json()}
    assert set(runs) == {"r1", "bench/find-flaw"}
    assert runs["r1"]["status"] == "done" and runs["bench/find-flaw"]["status"] == "stopped"
    assert runs["r1"]["budget_usd"] == 5.0 and runs["r1"]["has_report"]


def test_run_detail_and_proposal(root):
    c = _client(root)
    x = c.get("/api/runs/bench/find-flaw").json()
    assert [e["candidate_id"] for e in x["rounds"]] == ["r1"]          # pilot entries hidden
    assert x["proposals"] == [{"round": "r1", "reason": "clear gain", "review": "pending"}]
    assert "Held-out report" in x["report_md"]
    assert "+new" in c.get("/api/runs/bench/find-flaw/proposals/r1").json()["diff"]


def test_pages_escape_everything(root):
    c = _client(root)
    for url in ("/", "/runs/r1", "/runs/r1/proposals/r1"):
        page = c.get(url)
        assert page.status_code == 200 and XSS not in page.text
    assert "&lt;script&gt;" in c.get("/runs/r1/proposals/r1").text


def test_paths_outside_the_root_are_not_found(root, tmp_path):
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "state.json").write_text("{}")
    c = _client(root)
    for url in ("/api/runs/../outside", "/api/runs/nope", "/api/runs/r1/proposals/r9",
                "/api/runs/r1/proposals/..%2F..", "/runs/r1/proposals/x"):
        assert c.get(url).status_code == 404, url


def test_review(root):
    c = _client(root)
    r = c.post("/api/runs/r1/proposals/r1/review", json={"decision": "approved", "note": "looks right"})
    assert r.status_code == 200
    assert json.loads((root / "r1" / "proposals" / "r1" / "review.json").read_text())["decision"] == "approved"
    assert c.get("/api/runs/r1").json()["proposals"][0]["review"] == "approved"
    assert c.post("/api/runs/r1/proposals/r1/review", json={"decision": "maybe"}).status_code == 400


def test_token(root):
    c = _client(root, token="s3cret")
    assert c.get("/api/runs").status_code == 401
    assert c.get("/api/runs", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/api/runs", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    c.get("/login?token=s3cret")                                     # sets the cookie
    assert c.get("/").status_code == 200


def test_local_only_without_a_token_and_json_only_posts(root):
    c = _client(root)
    assert c.get("/api/runs", headers={"Host": "evil.example"}).status_code == 403     # DNS rebinding
    assert c.get("/api/runs", headers={"Host": "127.0.0.1:8765"}).status_code == 200
    r = c.post("/api/runs/r1/proposals/r1/review", content='{"decision": "approved"}',
               headers={"Content-Type": "text/plain"})                 # a cross-site "simple" request
    assert r.status_code == 415 and not (root / "r1" / "proposals" / "r1" / "review.json").exists()
    assert _client(root, token="t").get("/api/runs", headers={"Host": "evil.example",
                                                             "Authorization": "Bearer t"}).status_code == 200


def _wait(pred, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def test_start_runs_a_separate_process_and_resume_needs_a_config(root, tmp_path):
    cfg = tmp_path / "config.py"
    cfg.write_text("def build(run_dir, budget_usd=1.0, rounds=3):\n"
                   "    raise SystemExit(f'built with budget {budget_usd} rounds {rounds}')\n")
    c = _client(root)
    r = c.post("/api/runs", json={"config": str(cfg), "name": "new-run", "budget": 2.5, "rounds": 1})
    assert r.status_code == 201
    d = root / "new-run"
    assert json.loads((d / "serve.json").read_text())["pid"] == r.json()["pid"]
    assert _wait(lambda: "built with budget 2.5 rounds 1" in (d / "serve.log").read_text())
    assert c.post("/api/runs", json={"config": str(cfg), "name": "new-run"}).status_code == 409
    assert c.post("/api/runs", json={"config": str(cfg), "name": "../escape"}).status_code == 400
    assert c.post("/api/runs", json={"config": str(tmp_path / "missing.py"), "name": "x"}).status_code == 400
    assert c.post("/api/runs/r1/resume", json={}).status_code == 409                        # done
    assert c.post("/api/runs/bench/find-flaw/resume", json={}).status_code == 400            # no config known
    r = c.post("/api/runs/bench/find-flaw/resume", json={"config": str(cfg), "budget": 9})
    assert r.status_code == 200 and "--budget" in r.json()["cmd"]


def test_check_runs_in_a_subprocess(root, tmp_path):
    cfg = tmp_path / "config.py"
    cfg.write_text(textwrap.dedent(f'''
        from pathlib import Path
        from harness_evolve import AgentProfile, Optimizer, OptimizerConfig, profiles
        from harness_evolve.bench import scripted_llm

        def build(run_dir, budget_usd=1.0, rounds=3):
            profiles.use(AgentProfile.build(name="t", harness_root=Path({str(FIXTURE_HARNESS)!r}),
                                            agent_description="t"))
            ids = [f"c{{i}}" for i in range(60)]
            cfg = OptimizerConfig(evolve_cases=ids[:40], guard_cases=[], final_cases=ids[40:],
                                  budget_usd=budget_usd, max_rounds=rounds, default_case_cost_usd=0.01)
            llm = scripted_llm({{}})
            return Optimizer(run_dir, Path({str(FIXTURE_HARNESS)!r}), cfg, evaluator=None, proposer_llm=llm,
                             critic_llm=llm, critic_patterns=[])
    '''))
    r = _client(root).post("/api/check", json={"config": str(cfg), "budget": 0.01}).json()
    assert r["exit_code"] == 1 and "FAIL  budget" in r["output"]


def test_running_from_heartbeat(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    now = datetime.now(UTC)
    (d / "heartbeat.json").write_text(json.dumps({"time": now.isoformat(), "pid": os.getpid()}))
    assert server.running(d)
    (d / "heartbeat.json").write_text(json.dumps({"time": (now - timedelta(minutes=10)).isoformat(),
                                                  "pid": os.getpid()}))
    assert not server.running(d)


def test_serve_refuses_a_public_host_without_a_token(monkeypatch, capsys):
    monkeypatch.delenv("HE_SERVE_TOKEN", raising=False)
    assert server.serve("runs", host="0.0.0.0") == 1
    assert "refusing" in capsys.readouterr().out
