"""
`harness-evolve serve [RUNS_DIR]`: a web app and REST API over run directories.

    pip install "harness-evolve[serve]"
    harness-evolve serve runs              # http://127.0.0.1:8765

Run directories are the database: state.json, history.jsonl, proposals/ and report/ are what a run
already writes, so the server adds no storage of its own. Pages and /api/ routes show every run (phase,
spend, noise band, each round's decision and reason, diffs, the held-out report, tripwire pauses) and
let you review proposals, check a config, and start or resume a run.

Runs are started as separate processes in their own session, never inside the web server: a long run
must not die when the server restarts (a watcher once killed its children at its 2h limit). Each one
logs to RUN_DIR/serve.log, and RUN_DIR/serve.json records how it was started so it can be resumed.

Starting a run executes its config file, so anyone who can reach the server can run code as you. It
binds to 127.0.0.1; any other host needs a token (HE_SERVE_TOKEN, sent as "Authorization: Bearer ..." or
set as a cookie by /login?token=...), and the server refuses to start without one (fail closed).
"""
# No `from __future__ import annotations` here: FastAPI resolves the route functions' annotations
# (Request) at runtime, and fastapi is imported inside create_app so the module loads without it.
import hmac
import html
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

RUN_FILES = {"candidates", "evals", "incumbent", "original", "proposals", "report"}   # not runs themselves
NAME = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")
ROUND = re.compile(r"^r\d+$")
HEARTBEAT_STALE_S = 180          # the watchdog writes heartbeat.json every 60s
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class NotFound(LookupError):
    pass


class Conflict(RuntimeError):
    pass


# ---- reading runs -------------------------------------------------------------------------------

def find_runs(root: Path, depth: int = 3) -> list[Path]:
    """Directories under `root` holding a state.json (bench runs nest scenarios one level down)."""
    out = []

    def walk(d: Path, level: int) -> None:
        if (d / "state.json").is_file():
            out.append(d)
        if level >= depth:
            return
        for child in sorted(p for p in d.iterdir() if p.is_dir() and p.name not in RUN_FILES
                            and not p.name.startswith(".")):
            walk(child, level + 1)

    if root.is_dir():
        walk(root, 0)
    return [d for d in out if d != root]


def resolve(root: Path, run_id: str) -> Path:
    """The run directory for `run_id`, refusing anything outside `root`."""
    if not NAME.match(run_id) or ".." in run_id.split("/"):
        raise NotFound(run_id)
    d = (root / run_id).resolve()
    if root.resolve() not in d.parents or not (d / "state.json").is_file():
        raise NotFound(run_id)
    return d


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def running(d: Path) -> bool:
    """A process started by serve is still alive, or the run's heartbeat is fresh and its pid alive."""
    started = _read_json(d / "serve.json") or {}
    if started.get("pid") and _alive(started["pid"]):
        return True
    hb = _read_json(d / "heartbeat.json") or {}
    try:
        age = (datetime.now(UTC) - datetime.fromisoformat(hb["time"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return False
    return age < HEARTBEAT_STALE_S and _alive(hb.get("pid"))


def summary(root: Path, d: Path) -> dict:
    s = _read_json(d / "state.json") or {}
    cfg = s.get("config") if isinstance(s.get("config"), dict) else {}
    is_running = running(d)
    status = "running" if is_running else "done" if s.get("phase") == "done" else "stopped"
    return {"id": d.resolve().relative_to(root.resolve()).as_posix(), "phase": s.get("phase"), "round": s.get("round"),
            "status": status, "spent_usd": round(s.get("spent_usd") or 0.0, 4), "budget_usd": cfg.get("budget_usd"),
            "accepted": s.get("accepted") or [], "stop_reason": s.get("stop_reason"),
            "S_star": s.get("S_star"), "delta": s.get("delta"), "delta_cost": s.get("delta_cost"),
            "updated": datetime.fromtimestamp((d / "state.json").stat().st_mtime, UTC).isoformat(timespec="seconds"),
            "has_report": (d / "report" / "report.md").is_file()}


def rounds(d: Path) -> list[dict]:
    p = d / "history.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if str(e.get("outcome", "")).startswith("pilot_"):
            continue
        out.append({k: e.get(k) for k in ("round", "candidate_id", "component", "outcome", "reason", "hypothesis",
                                          "delta_S", "delta_C", "cost_usd")})
    return out


def proposals(d: Path) -> list[dict]:
    root = d / "proposals"
    if not root.is_dir():
        return []
    out = []
    for p in sorted((x for x in root.iterdir() if x.is_dir() and ROUND.match(x.name)), key=lambda x: int(x.name[1:])):
        review = _read_json(p / "review.json") or {}
        decision = _read_json(p / "decision.json") or {}
        out.append({"round": p.name, "reason": decision.get("reason"), "review": review.get("decision", "pending")})
    return out


def proposal(d: Path, r: str) -> dict:
    p = d / "proposals" / r
    if not ROUND.match(r) or not p.is_dir():
        raise NotFound(r)

    def read(name: str) -> str:
        return (p / name).read_text() if (p / name).is_file() else ""

    return {"round": r, "diff": read("harness.diff"), "readme": read("README.md"),
            "decision": _read_json(p / "decision.json") or {}, "review": _read_json(p / "review.json")}


def detail(root: Path, d: Path) -> dict:
    s = _read_json(d / "state.json") or {}
    report = d / "report" / "report.md"
    log = d / "serve.log"
    return {**summary(root, d), "rounds": rounds(d), "proposals": proposals(d),
            "report_md": report.read_text() if report.is_file() else None,
            "trips": (s.get("health") or {}).get("trips", []),
            "heartbeat": _read_json(d / "heartbeat.json"),
            "started": _read_json(d / "serve.json"),
            "log_tail": "\n".join(log.read_text(errors="replace").splitlines()[-40:]) if log.is_file() else None}


# ---- actions ------------------------------------------------------------------------------------

def review(d: Path, r: str, decision: str, note: str = "") -> dict:
    """Record a person's verdict on a proposal (proposals/rN/review.json). Applying it stays with you."""
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be 'approved' or 'rejected'")
    proposal(d, r)                                   # 404 if missing
    rec = {"decision": decision, "note": note, "time": datetime.now(UTC).isoformat(timespec="seconds")}
    (d / "proposals" / r / "review.json").write_text(json.dumps(rec, indent=1))
    return rec


def _config(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p.suffix != ".py" or not p.is_file():
        raise ValueError(f"config must be an existing .py file: {path}")
    return p


def _spawn(d: Path, config: Path, budget: float | None, rounds_: int | None) -> dict:
    cmd = [sys.executable, "-m", "harness_evolve.cli", "run", str(config), "--run-dir", str(d)]
    if budget is not None:
        cmd += ["--budget", str(budget)]
    if rounds_ is not None:
        cmd += ["--rounds", str(rounds_)]
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "serve.log", "a") as log:
        log.write(f"\n--- {datetime.now(UTC).isoformat(timespec='seconds')} {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=config.parent, start_new_session=True)
    rec = {"pid": proc.pid, "config": str(config), "cmd": cmd,
           "started": datetime.now(UTC).isoformat(timespec="seconds")}
    (d / "serve.json").write_text(json.dumps(rec, indent=1))
    return rec


def start(root: Path, config: str, name: str, budget: float | None = None, rounds_: int | None = None) -> dict:
    """Start a new run in RUNS_DIR/name as its own process."""
    if not NAME.match(name) or ".." in name.split("/"):
        raise ValueError("run name: letters, digits, '.', '_' and '-' (and '/' between parts)")
    d = (root / name).resolve()
    if d.exists():
        raise Conflict(f"{name} already exists; resume it instead")
    return _spawn(d, _config(config), budget, rounds_)


def resume(d: Path, config: str | None = None, budget: float | None = None) -> dict:
    """Resume a stopped run with the config it was started with (or the one given)."""
    if running(d):
        raise Conflict("the run is still running")
    if (_read_json(d / "state.json") or {}).get("phase") == "done":
        raise Conflict("the run is done")
    config = config or (_read_json(d / "serve.json") or {}).get("config")
    if not config:
        raise ValueError("this run wasn't started by serve: give the config file it was started with")
    return _spawn(d, _config(config), budget, None)


def check(config: str, budget: float | None = None, rounds_: int | None = None, timeout_s: float = 120) -> dict:
    """`harness-evolve check` in a subprocess, so the user's config never loads into the server."""
    cfg = _config(config)
    cmd = [sys.executable, "-m", "harness_evolve.cli", "check", str(cfg)]
    if budget is not None:
        cmd += ["--budget", str(budget)]
    if rounds_ is not None:
        cmd += ["--rounds", str(rounds_)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=cfg.parent)
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "output": f"check did not finish in {timeout_s:g}s"}
    return {"exit_code": res.returncode, "output": (res.stdout + res.stderr).strip()}


# ---- pages ----------------------------------------------------------------------------------------

E = html.escape

CSS = """body{font:14px/1.45 -apple-system,system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;
color:#1d1d1f}
a{color:#0b57d0;text-decoration:none}
a:hover{text-decoration:underline}
h1{font-size:20px}
h2{font-size:16px;margin-top:28px}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #e5e5e5;vertical-align:top}
th{font-weight:600;color:#555}
pre{background:#f6f8fa;padding:10px;overflow:auto;white-space:pre-wrap;font-size:12.5px}
.add{color:#116329}
.del{color:#a40e26}
.muted{color:#777}
.pill{padding:1px 7px;border-radius:9px;font-size:12px;background:#eee}
.running{background:#d6f5dd}
.stopped{background:#fde7c8}
.done{background:#e3e9f7}
.approved{background:#d6f5dd}
.rejected{background:#f9d7d7}
input{padding:4px 6px;margin-right:6px}
button{padding:4px 10px;cursor:pointer}
form{margin:8px 0}
.kv td:first-child{color:#555;width:170px}"""

JS = """async function post(url, body){const r=await fetch(url,{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
const j=await r.json();if(!r.ok){alert(j.detail||r.status);return null}return j}
function val(id){const v=document.getElementById(id).value.trim();return v===''?null:v}
function num(id){const v=val(id);return v===null?null:Number(v)}"""


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset=utf-8><title>{E(title)} - harness-evolve</title>"
            f"<style>{CSS}</style><script>{JS}</script></head><body>"
            f"<p class=muted><a href='/'>harness-evolve runs</a></p><h1>{E(title)}</h1>{body}</body></html>")


def _pct(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def _signed(x, pct=False) -> str:
    if x is None:
        return "-"
    return f"{x:+.0%}" if pct else f"{x:+.3f}"


def _pill(text: str) -> str:
    return f"<span class='pill {E(text)}'>{E(text)}</span>"


def _diff_html(diff: str) -> str:
    lines = []
    for line in diff.splitlines():
        cls = "add" if line.startswith("+") and not line.startswith("+++") else \
              "del" if line.startswith("-") and not line.startswith("---") else ""
        lines.append(f"<span class={cls}>{E(line)}</span>" if cls else E(line))
    return "<pre>" + "\n".join(lines) + "</pre>"


def runs_page(root: Path) -> str:
    rows = []
    for s in sorted((summary(root, d) for d in find_runs(root)), key=lambda x: x["updated"], reverse=True):
        rows.append(f"<tr><td><a href='/runs/{E(s['id'])}'>{E(s['id'])}</a></td><td>{_pill(s['status'])}</td>"
                    f"<td>{E(str(s['phase']))}</td><td>{E(str(s['round']))}</td>"
                    f"<td>{E(', '.join(s['accepted']) or '-')}</td><td>${s['spent_usd']:.2f}</td>"
                    f"<td class=muted>{E((s['stop_reason'] or '')[:110])}</td>"
                    f"<td class=muted>{E(s['updated'][:16].replace('T', ' '))}</td></tr>")
    table = ("<table><tr><th>run</th><th>status</th><th>phase</th><th>round</th><th>accepted</th><th>spent</th>"
             "<th>stopped because</th><th>updated (UTC)</th></tr>" + "".join(rows) + "</table>") if rows else \
        f"<p class=muted>No runs under {E(str(root))} yet.</p>"
    forms = """<h2>Check a config (free, no model calls)</h2>
<form onsubmit="event.preventDefault();post('/api/check',{config:val('cc'),budget:num('cb'),rounds:num('cr')})
.then(j=>{if(j)document.getElementById('co').textContent=j.output+'\\n\\nexit '+j.exit_code})">
<input id=cc size=50 placeholder="path/to/config.py" required><input id=cb size=8 placeholder="budget $">
<input id=cr size=6 placeholder="rounds"><button>check</button></form><pre id=co class=muted>-</pre>
<h2>Start a run</h2>
<form onsubmit="event.preventDefault();post('/api/runs',{config:val('sc'),name:val('sn'),budget:num('sb'),
rounds:num('sr')}).then(j=>{if(j)location.href='/runs/'+val('sn')})">
<input id=sc size=40 placeholder="path/to/config.py" required><input id=sn size=16 placeholder="run name" required>
<input id=sb size=8 placeholder="budget $"><input id=sr size=6 placeholder="rounds"><button>start</button></form>
<p class=muted>A run starts as its own process and keeps going if this server stops.
Run <code>check</code> first.</p>"""
    return _page("Runs", table + forms)


def run_page(root: Path, d: Path) -> str:
    x = detail(root, d)
    rid = x["id"]
    kv = [("status", _pill(x["status"])), ("phase / round", E(f"{x['phase']} / {x['round']}")),
          ("starting pass rate", _pct(x["S_star"])), ("noise band", _pct(x["delta"])),
          ("cost band", "-" if x["delta_cost"] is None else f"{x['delta_cost']:.1%}"),
          ("spent", f"${x['spent_usd']:.2f}" + (f" of ${x['budget_usd']:.2f}" if x["budget_usd"] else "")),
          ("accepted", E(", ".join(x["accepted"]) or "none")), ("stopped because", E(x["stop_reason"] or "-"))]
    body = "<table class=kv>" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in kv) + "</table>"
    if x["status"] == "stopped":
        cfg = (x["started"] or {}).get("config", "")
        body += (f"""<form onsubmit="event.preventDefault();post('/api/runs/{E(rid)}/resume',{{config:val('rc'),
budget:num('rb')}}).then(j=>{{if(j)location.reload()}})"><input id=rc size=50 value="{E(cfg)}"
placeholder="config.py it was started with"><input id=rb size=8 placeholder="new budget $">
<button>resume</button></form>""")
    body += "<h2>Rounds</h2>"
    if x["rounds"]:
        body += ("<table><tr><th>round</th><th>component</th><th>outcome</th><th>pass</th><th>cost</th>"
                 "<th>reason / hypothesis</th></tr>")
        for e in x["rounds"]:
            body += (f"<tr><td>r{E(str(e['round']))}</td><td>{E(str(e['component']))}</td>"
                     f"<td>{E(str(e['outcome']))}</td><td>{_signed(e['delta_S'])}</td>"
                     f"<td>{_signed(e['delta_C'], pct=True)}</td><td>{E(e['reason'] or '')}<br>"
                     f"<span class=muted>{E((e['hypothesis'] or '')[:300])}</span></td></tr>")
        body += "</table>"
    else:
        body += "<p class=muted>No rounds judged yet.</p>"
    body += "<h2>Proposals</h2>"
    if x["proposals"]:
        body += "<table><tr><th>round</th><th>review</th><th>decision</th></tr>" + "".join(
            f"<tr><td><a href='/runs/{E(rid)}/proposals/{E(p['round'])}'>{E(p['round'])}</a></td>"
            f"<td>{_pill(p['review'])}</td><td>{E(p['reason'] or '')}</td></tr>" for p in x["proposals"]) + "</table>"
    else:
        body += "<p class=muted>No accepted edits.</p>"
    if x["report_md"]:
        body += f"<h2>Held-out report</h2><pre>{E(x['report_md'])}</pre>"
    if x["trips"]:
        body += "<h2>Tripwire pauses</h2><pre>" + E(json.dumps(x["trips"], indent=1)) + "</pre>"
    if x["log_tail"]:
        body += f"<h2>Log (last 40 lines of serve.log)</h2><pre>{E(x['log_tail'])}</pre>"
    return _page(rid, body)


def proposal_page(root: Path, d: Path, r: str) -> str:
    p = proposal(d, r)
    rid = d.resolve().relative_to(root.resolve()).as_posix()
    rv = p["review"]
    status = (_pill(rv["decision"]) + f" <span class=muted>{E(rv.get('time', ''))} {E(rv.get('note', ''))}</span>"
              if rv else _pill("pending"))
    body = (f"<p><a href='/runs/{E(rid)}'>back to {E(rid)}</a></p><p>Review: {status}</p>"
            f"""<form onsubmit="event.preventDefault()"><input id=note size=50 placeholder="note (optional)">
<button onclick="post('/api/runs/{E(rid)}/proposals/{E(r)}/review',{{decision:'approved',note:val('note')||''}})
.then(j=>{{if(j)location.reload()}})">approve</button>
<button onclick="post('/api/runs/{E(rid)}/proposals/{E(r)}/review',{{decision:'rejected',note:val('note')||''}})
.then(j=>{{if(j)location.reload()}})">reject</button></form>
<p class=muted>Approving records your verdict; apply <code>harness.diff</code> through your normal review.</p>"""
            f"<h2>Diff</h2>{_diff_html(p['diff'])}<h2>Proposal</h2><pre>{E(p['readme'])}</pre>")
    return _page(f"{rid} {r}", body)


# ---- the app --------------------------------------------------------------------------------------

def create_app(root: str | Path, token: str | None = None):
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, RedirectResponse

    root = Path(root)
    app = FastAPI(title="harness-evolve", docs_url="/api/docs", openapi_url="/api/openapi.json")

    def auth(request: Request) -> None:
        # Without a token the server is local-only: refuse requests addressed to any other host name, so a
        # web page can't reach it by DNS rebinding.
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if not token and host not in LOCAL_HOSTS | {"testserver"}:
            raise HTTPException(403, f"host {host!r} not allowed without a token")
        # POSTs must be JSON: a cross-site page can't send that without a CORS preflight, which this
        # server never grants, so it can't start runs from your browser.
        if request.method == "POST" and not request.headers.get("content-type", "").startswith("application/json"):
            raise HTTPException(415, "send JSON (Content-Type: application/json)")
        if not token:
            return
        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or \
            request.cookies.get("he_token", "")
        if not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(401, "missing or wrong token: send Authorization: Bearer <token>, "
                                     "or open /login?token=<token>")

    def run_dir(run_id: str) -> Path:
        try:
            return resolve(root, run_id)
        except NotFound:
            raise HTTPException(404, f"no run {run_id}") from None

    def act(fn, *args):
        try:
            return fn(*args)
        except NotFound as exc:
            raise HTTPException(404, f"not found: {exc}") from None
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/login")
    def login(request: Request):
        given = request.query_params.get("token", "")
        resp = RedirectResponse("/", status_code=303)
        if token and hmac.compare_digest(given.encode(), token.encode()):
            resp.set_cookie("he_token", given, httponly=True, samesite="strict")
        return resp

    guard = [Depends(auth)]

    @app.get("/api/runs", dependencies=guard)
    def api_runs():
        return [summary(root, d) for d in find_runs(root)]

    @app.post("/api/runs", dependencies=guard, status_code=201)
    async def api_start(request: Request):
        b = await request.json()
        return act(start, root, str(b.get("config") or ""), str(b.get("name") or ""), b.get("budget"),
                   b.get("rounds"))

    @app.post("/api/check", dependencies=guard)
    async def api_check(request: Request):
        b = await request.json()
        return act(check, str(b.get("config") or ""), b.get("budget"), b.get("rounds"))

    @app.get("/api/runs/{run_id:path}/proposals/{r}", dependencies=guard)
    def api_proposal(run_id: str, r: str):
        return act(proposal, run_dir(run_id), r)

    @app.post("/api/runs/{run_id:path}/proposals/{r}/review", dependencies=guard)
    async def api_review(run_id: str, r: str, request: Request):
        b = await request.json()
        return act(review, run_dir(run_id), r, str(b.get("decision") or ""), str(b.get("note") or ""))

    @app.post("/api/runs/{run_id:path}/resume", dependencies=guard)
    async def api_resume(run_id: str, request: Request):
        b = await request.json()
        return act(resume, run_dir(run_id), b.get("config"), b.get("budget"))

    @app.get("/api/runs/{run_id:path}", dependencies=guard)
    def api_run(run_id: str):
        return detail(root, run_dir(run_id))

    @app.get("/", response_class=HTMLResponse, dependencies=guard)
    def page_runs():
        return runs_page(root)

    @app.get("/runs/{run_id:path}/proposals/{r}", response_class=HTMLResponse, dependencies=guard)
    def page_proposal(run_id: str, r: str):
        return act(proposal_page, root, run_dir(run_id), r)

    @app.get("/runs/{run_id:path}", response_class=HTMLResponse, dependencies=guard)
    def page_run(run_id: str):
        return run_page(root, run_dir(run_id))

    return app


def serve(root: str | Path, host: str = "127.0.0.1", port: int = 8765) -> int:
    token = os.environ.get("HE_SERVE_TOKEN") or None
    if host not in LOCAL_HOSTS and not token:
        print(f"refusing to serve on {host} without a token: anyone who reaches it can start runs (run code as "
              "you). Set HE_SERVE_TOKEN, or serve on 127.0.0.1")
        return 1
    try:
        import uvicorn
    except ImportError:
        print('serve needs its extras: pip install "harness-evolve[serve]"')
        return 1
    print(f"harness-evolve serving {Path(root).resolve()} on http://{host}:{port}"
          + (" (token required)" if token else ""))
    uvicorn.run(create_app(root, token), host=host, port=port, log_level="warning")
    return 0
