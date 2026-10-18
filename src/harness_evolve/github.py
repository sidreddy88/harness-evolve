"""
GitHubPublisher: a run's accepted edits as one draft pull request on the agent's repo.

    from harness_evolve.github import GitHubPublisher
    publisher = GitHubPublisher(repo="~/code/my-agent", harness_path="agent/harness")
    Optimizer(..., publisher=publisher)

One PR per run, not per edit: accepted edits build on each other (round 2's harness already contains
round 1's edit), so separate PRs would conflict. Each accepted edit is one commit on the branch
`harness-evolve/<run id>`, so a reviewer can read them one at a time; the description lists every edit
with its hypothesis, decision and measured change. When the run finishes, the branch is synced to the
final harness (an edit removed by pruning gets its own commit), the held-out report goes into the
description, and the PR is marked ready for review. Merging it adopts the evolved harness; closing it
discards it. Nothing reaches your code without a person merging.

Safe by construction:
- It works in a temporary git worktree, never in your checkout, so your working tree and current
  branch are untouched.
- Before the first commit it compares the repo's harness at `base` with the run's starting harness and
  refuses if they differ: a PR built from a stale copy would silently revert changes made since.
- It only ever pushes the run's own branch, as a fast-forward, and never to `base`.
- Publishing failures (no network, gh not logged in) never stop the run: the edit is still written to
  run_dir/proposals/ as LocalPublisher does, and the error is saved next to it (github.json); the next
  accepted edit or the end of the run retries the sync.

Needs `git` and the GitHub CLI (`gh`, logged in, allowed to push branches and open PRs on the repo).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from harness_evolve.candidates import harness_files
from harness_evolve.state import RunDir

logger = logging.getLogger(__name__)


class PublishError(RuntimeError):
    pass


class GitHubPublisher:
    def __init__(self, repo: str | Path, harness_path: str, base: str = "main", remote: str = "origin",
                 branch_prefix: str = "harness-evolve/", git: str = "git", gh: str = "gh"):
        self.repo = Path(repo).expanduser().resolve()
        self.harness_path = harness_path.strip("/")
        self.base, self.remote, self.prefix = base, remote, branch_prefix
        self.git, self.gh = git, gh

    # ---- the Publisher interface --------------------------------------------------------------

    def publish(self, run: RunDir, round_no: int, candidate_dir: Path, diff: str,
                decision: dict, hypothesis: str) -> Path:
        from harness_evolve.loop import LocalPublisher
        out = LocalPublisher().publish(run, round_no, candidate_dir, diff, decision, hypothesis)
        first = hypothesis.strip().splitlines()[0] if hypothesis.strip() else ""
        self._record(out, lambda: self.sync(run, candidate_dir, f"harness-evolve r{round_no}: {first[:60]}",
                                            f"Hypothesis: {hypothesis}\n\nDecision: {decision.get('reason', '')}"))
        return out

    def finalize(self, run: RunDir) -> None:
        """At the end of a run: the final harness (after pruning), the held-out report, ready for review."""
        if not (run.root / "proposals").is_dir():
            return                                    # nothing was accepted: no PR
        out = run.root / "proposals"
        self._record(out, lambda: self.sync(run, run.incumbent_dir, "harness-evolve: final harness after pruning",
                                            "", ready=True), name="github-final.json")

    # ---- the work -------------------------------------------------------------------------------

    def branch(self, run: RunDir) -> str:
        return self.prefix + re.sub(r"[^A-Za-z0-9._-]+", "-", run.root.resolve().name).strip("-")

    def sync(self, run: RunDir, files_dir: Path, subject: str, body: str, ready: bool = False) -> dict:
        """Make the run's branch hold `files_dir` as the harness (one commit if it changed), push it, and
        create or update the PR. Idempotent: syncing the same files twice changes nothing."""
        branch = self.branch(run)
        self._git("fetch", "--quiet", self.remote, self.base)
        exists = bool(self._git("ls-remote", "--heads", self.remote, branch).strip())
        if exists:
            self._git("fetch", "--quiet", self.remote, branch)
        start = f"{self.remote}/{branch}" if exists else f"{self.remote}/{self.base}"
        tmp = Path(tempfile.mkdtemp(prefix="he-pr-"))
        wt = tmp / "wt"
        self._git("worktree", "add", "--quiet", "--detach", str(wt), start)
        try:
            target = wt / self.harness_path
            if not target.is_dir():
                raise PublishError(f"{self.harness_path} is not a directory in {self.repo} at {start}")
            if not exists and harness_files(target) != harness_files(run.original_dir):
                raise PublishError(f"the harness in {self.remote}/{self.base}:{self.harness_path} differs from the "
                                   "run's starting harness; a PR from this run would revert those changes. Start "
                                   "a run from the current harness, or publish this one by hand")
            for name, text in harness_files(files_dir).items():
                (target / name).write_text(text)
            self._git("add", "--all", "--", self.harness_path, cwd=wt)
            changed = subprocess.run([self.git, "diff", "--cached", "--quiet"], cwd=wt).returncode == 1
            if changed:
                self._git("commit", "--quiet", "-m", subject + ("\n\n" + body if body else ""), cwd=wt)
            commit = self._git("rev-parse", "HEAD", cwd=wt).strip()
            if changed or not exists:
                self._git("push", "--quiet", self.remote, f"HEAD:refs/heads/{branch}", cwd=wt)
        finally:
            self._git("worktree", "remove", "--force", str(wt))
            shutil.rmtree(tmp, ignore_errors=True)
        url = self._pull_request(run, branch, ready)
        return {"branch": branch, "commit": commit, "committed": changed, "pull_request": url}

    def _pull_request(self, run: RunDir, branch: str, ready: bool) -> str:
        found = json.loads(self._gh("pr", "list", "--head", branch, "--state", "open", "--json", "number,url,isDraft")
                           or "[]")
        title = f"harness-evolve: evolved harness from run {run.root.resolve().name}"
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(description(run))
        try:
            if found:
                pr = found[0]
                self._gh("pr", "edit", str(pr["number"]), "--body-file", f.name)
                if ready and pr.get("isDraft"):
                    self._gh("pr", "ready", str(pr["number"]))
                return pr["url"]
            args = ["pr", "create", "--base", self.base, "--head", branch, "--title", title, "--body-file", f.name]
            return self._gh(*args, *([] if ready else ["--draft"])).strip().splitlines()[-1]
        finally:
            Path(f.name).unlink(missing_ok=True)

    def _record(self, out: Path, fn, name: str = "github.json") -> None:
        try:
            rec = fn()
        except Exception as exc:             # never stop the run over publishing
            logger.warning("GitHub publish failed: %s", exc)
            rec = {"error": str(exc)}
        (out / name).write_text(json.dumps(rec, indent=1))

    def _git(self, *args: str, cwd: Path | None = None) -> str:
        return _run([self.git, *args], cwd or self.repo)

    def _gh(self, *args: str) -> str:
        return _run([self.gh, *args], self.repo)


def _run(cmd: list[str], cwd: Path) -> str:
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise PublishError(f"{' '.join(cmd[:3])}... failed: {(res.stderr or res.stdout).strip()[-400:]}")
    return res.stdout


def description(run: RunDir) -> str:
    """The PR description: every accepted edit with its evidence, the noise band, the held-out report."""
    state = json.loads(run.state_path.read_text()) if run.state_path.exists() else {}
    entries = []
    hist = run.root / "history.jsonl"
    if hist.exists():
        for line in hist.read_text().splitlines():
            e = json.loads(line)
            if e.get("outcome") == "accepted":
                entries.append(e)
    pruned = set((state.get("prune") or {}).get("removed", []))
    lines = ["Proposed by [harness-evolve](https://github.com/sidreddy88/harness-evolve) from run "
             f"`{run.root.resolve().name}`. **Merge to adopt the evolved harness; close to discard it.** "
             "Each accepted edit is one commit.", ""]
    if state.get("delta") is not None:
        lines.append(f"Starting pass rate {state.get('S_star', 0):.3f} on the evolve cases; noise band "
                     f"±{state['delta']:.3f}" + (f", cost band {state['delta_cost']:.1%}"
                                                 if state.get("delta_cost") is not None else "") + ".")
        lines.append("")
    lines.append("| round | component | pass rate | cost | decision |")
    lines.append("|---|---|---|---|---|")
    for e in entries:
        ds = "-" if e.get("delta_S") is None else f"{e['delta_S']:+.3f}"
        dc = "-" if e.get("delta_C") is None else f"{e['delta_C']:+.0%}"
        tag = " (removed by pruning)" if e.get("candidate_id") in pruned else ""
        lines.append(f"| {e.get('candidate_id')}{tag} | {e.get('component')} | {ds} | {dc} | "
                     f"{(e.get('reason') or '').replace('|', '/')} |")
    for e in entries:
        lines += ["", f"**{e.get('candidate_id')}:** {e.get('hypothesis') or ''}",
                  f"Improved {len(e.get('improved') or [])} cases, regressed {len(e.get('regressed') or [])}."]
    report = run.root / "report" / "report.md"
    if report.exists():
        lines += ["", "## Held-out report", "", report.read_text().split("```")[0].strip()]
    elif state.get("phase") != "done":
        lines += ["", "_The run is still going: more edits or the held-out report may follow._"]
    return "\n".join(lines) + "\n"
