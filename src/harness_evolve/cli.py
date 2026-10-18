"""
harness-evolve command line.

    harness-evolve init my-agent          # a starter project to fill in
    harness-evolve check CONFIG.py [--budget 5] [--rounds 3] [--strict]   # free pre-run checks
    harness-evolve run CONFIG.py --run-dir runs/r1 [--budget 5] [--rounds 3]
    harness-evolve status runs/r1
    harness-evolve serve runs              # web app + REST API over run directories
    harness-evolve diff runs/r1
    harness-evolve bench BENCH.py --out runs/bench-1 [--only accept-good,reject-bad]

CONFIG.py defines `build(run_dir, budget_usd=..., rounds=...) -> Optimizer`
(see examples/math_agent/config.py); BENCH.py defines `scenarios() -> list[Scenario]`
(see examples/math_agent/bench.py and harness_evolve/bench.py). `run` on an existing run directory
resumes it where it stopped; nothing already paid for is paid again.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import sys
import tempfile
from pathlib import Path


def _load_module(path: Path, required: str):
    spec = importlib.util.spec_from_file_location("harness_evolve_config", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"can't load config {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    if not hasattr(mod, required):
        raise SystemExit(f"{path} must define {required}()")
    return mod


def _status(run_dir: Path) -> str:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return f"no run at {run_dir}"
    s = json.loads(state_path.read_text())
    lines = [f"run {run_dir}: phase {s.get('phase')}, round {s.get('round')}, "
             f"spent ${s.get('spent_usd', 0):.2f}"]
    if s.get("S_star") is not None:
        lines.append(f"  starting pass rate {s['S_star']:.3f}; noise band {s.get('delta') or 0:.3f}"
                     + (f"; cost band {s['delta_cost']:.1%}" if s.get("delta_cost") is not None else ""))
    lines.append(f"  accepted: {s.get('accepted') or 'none'}")
    if s.get("stop_reason"):
        lines.append(f"  stopped: {s['stop_reason']}")
    hist = run_dir / "history.jsonl"
    if hist.exists():
        for line in hist.read_text().splitlines():
            e = json.loads(line)
            if e.get("outcome", "").startswith("pilot_"):
                continue
            meas = (f" (pass {e['delta_S']:+.3f}, cost {e['delta_C']:+.0%})"
                    if e.get("delta_S") is not None and e.get("delta_C") is not None else "")
            lines.append(f"  r{e.get('round')} [{e.get('component')}] {e.get('outcome')}{meas}: "
                         f"{(e.get('hypothesis') or '')[:100]}")
    report = run_dir / "report" / "report.md"
    if report.exists():
        lines.append(f"  held-out report: {report}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness-evolve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init", help="write a starter project for your agent")
    i.add_argument("directory", type=Path)
    i.add_argument("--force", action="store_true", help="overwrite existing starter files")
    c = sub.add_parser("check", help="free pre-run checks: no model calls; exit 1 on a FAIL")
    c.add_argument("config", type=Path)
    c.add_argument("--budget", type=float, help="the cap you will pass to run")
    c.add_argument("--rounds", type=int, help="the rounds you will pass to run")
    c.add_argument("--strict", action="store_true", help="exit 1 on a WARN too")
    r = sub.add_parser("run", help="start or resume a run")
    r.add_argument("config", type=Path)
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--budget", type=float, help="hard cap in USD")
    r.add_argument("--rounds", type=int, help="maximum rounds")
    v = sub.add_parser("serve", help="web app and REST API over run directories")
    v.add_argument("runs_dir", type=Path, nargs="?", default=Path("runs"))
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8765)
    s = sub.add_parser("status", help="summarize a run")
    s.add_argument("run_dir", type=Path)
    d = sub.add_parser("diff", help="show accepted edits")
    d.add_argument("run_dir", type=Path)
    b = sub.add_parser("bench", help="benchmark the optimizer: known-answer and open scenarios")
    b.add_argument("spec", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--only", help="comma-separated scenario names")
    args = ap.parse_args(argv)

    if args.cmd == "init":
        from harness_evolve import scaffold
        try:
            written = scaffold.write(args.directory, force=args.force)
        except FileExistsError as exc:
            print(exc)
            return 1
        print(f"wrote {len(written)} files to {args.directory}/; next steps are in {args.directory}/README.md")
        return 0
    if args.cmd == "serve":
        from harness_evolve import server
        return server.serve(args.runs_dir, args.host, args.port)
    if args.cmd == "status":
        print(_status(args.run_dir))
        return 0
    if args.cmd == "diff":
        found = sorted((args.run_dir / "proposals").glob("r*/harness.diff"))
        if not found:
            print("no accepted edits")
        for f in found:
            print(f"### {f.parent.name}\n{f.read_text()}")
        return 0

    if args.cmd == "bench":
        from harness_evolve import bench
        mod = _load_module(args.spec.resolve(), "scenarios")
        only = args.only.split(",") if args.only else None
        results = asyncio.run(bench.run(mod.scenarios(), args.out, only))
        print(bench.render(results))
        return 0 if all(r.get("passed") is not False for r in results) else 1

    mod = _load_module(args.config.resolve(), "build")
    kwargs = {}
    params = inspect.signature(mod.build).parameters
    if args.budget is not None and "budget_usd" in params:
        kwargs["budget_usd"] = args.budget
    if args.rounds is not None and "rounds" in params:
        kwargs["rounds"] = args.rounds
    if args.cmd == "check":
        from harness_evolve import check
        with tempfile.TemporaryDirectory() as tmp:          # build() may create files; keep them out of runs/
            findings = check.run(mod.build(Path(tmp) / "check", **kwargs))
        print(check.render(findings))
        return check.exit_code(findings, args.strict)
    opt = mod.build(args.run_dir, **kwargs)
    try:
        asyncio.run(opt.run_until_stopped())
    except KeyboardInterrupt:
        print("\ninterrupted; `harness-evolve run` with the same --run-dir resumes it")
    print(_status(args.run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
