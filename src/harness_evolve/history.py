"""
Edit history: the optimizer's memory across rounds.

One JSONL line per judged candidate, in the shape of RRSI's evidence-aware
credit assignment (rrsi/history.py): which component was edited, the
hypothesis behind it, the diff, the measured score and cost change, which
cases moved, and the accept/reject decision with its reason. Candidates the
critic rejected are recorded too, without a measurement, so the proposer
doesn't keep rediscovering the same leaky idea.

The proposer can't be handed the full history forever: diffs are large and a
long run accumulates many. summary() is the compaction step. It keeps the
most recent entries in full, collapses older ones to one line each, and
always includes per-component tallies, bounded to a character budget. The
full record stays on disk (history.jsonl), so nothing is lost; only what the
proposer sees each round is compacted.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class HistoryEntry:
    round: int
    candidate_id: str
    component: str                  # e.g. "task_prompt", "tool_descriptions", "settings"
    hypothesis: str
    diff: str
    outcome: str                    # "accepted" | "rejected" | "critic_rejected" | "invalid"
    reason: str
    delta_S: float | None = None
    delta_C: float | None = None    # relative
    improved: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)
    cost_usd: float = 0.0


class EditHistory:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, entry: HistoryEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
            f.flush()

    def entries(self) -> list[HistoryEntry]:
        if not self.path.exists():
            return []
        return [HistoryEntry(**json.loads(line)) for line in self.path.read_text().splitlines() if line.strip()]

    def summary(self, recent: int = 3, max_chars: int = 6000, diff_chars: int = 1500) -> str:
        """Compacted view for the proposer: tallies, then older entries as
        one-liners, then the most recent `recent` in full, then truncated to
        `max_chars` from the old end, so the newest evidence always survives."""
        es = self.entries()
        if not es:
            return "No candidates judged yet."
        tally = Counter((e.component, e.outcome) for e in es)
        lines = ["Per component (outcome: count):"]
        for comp in sorted({e.component for e in es}):
            parts = ", ".join(f"{o}: {n}" for (c, o), n in sorted(tally.items()) if c == comp)
            lines.append(f"  {comp}: {parts}")

        def fmt_delta(e: HistoryEntry) -> str:
            if e.delta_S is None:
                return "not measured"
            return f"dS {e.delta_S:+.3f}, dC {e.delta_C:+.1%}"

        old, new = es[:-recent] if len(es) > recent else [], es[-recent:]
        if old:
            lines.append("\nEarlier candidates:")
            lines += [f"  r{e.round} [{e.component}] {e.outcome} ({fmt_delta(e)}): {e.hypothesis[:140]}"
                      for e in old]
        lines.append("\nMost recent candidates, in full:")
        for e in new:
            lines += [
                f"--- r{e.round} {e.candidate_id} [{e.component}] {e.outcome}: {e.reason}",
                f"hypothesis: {e.hypothesis}",
                f"result: {fmt_delta(e)}; improved {e.improved or '-'}; regressed {e.regressed or '-'}",
                f"diff:\n{e.diff[:diff_chars]}" + ("\n[diff truncated]" if len(e.diff) > diff_chars else ""),
            ]
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = "[older history truncated]\n" + text[-(max_chars - 27):]
        return text


def snapshot_prior(sources: list[str | Path], out: str | Path) -> int:
    """Copy earlier runs' history entries (a history.jsonl, or a run directory holding one) into one file,
    each tagged with the run it came from. Pilot notes are skipped. Taken once at run start, so a resumed run
    sees the same prior history even if those runs change. Returns the number of entries copied."""
    rows = []
    for src in sources:
        p = Path(src)
        p = p / "history.jsonl" if p.is_dir() else p
        if not p.exists():
            raise FileNotFoundError(f"prior history not found: {p}")
        for line in p.read_text().splitlines():
            if line.strip() and not str((e := json.loads(line)).get("outcome", "")).startswith("pilot_"):
                rows.append({"source_run": p.parent.name, **e})
    Path(out).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return len(rows)


def prior_summary(path: str | Path, max_chars: int = 4000) -> str:
    """One line per earlier-run candidate for the proposer: what was tried, and how it came out. Diffs are
    left out: they were made against those runs' harnesses. Newest kept if it doesn't fit."""
    p = Path(path)
    if not p.exists() or not p.read_text().strip():
        return ""
    lines = []
    for line in p.read_text().splitlines():
        e = json.loads(line)
        meas = f", dS {e['delta_S']:+.3f}" if e.get("delta_S") is not None else ""
        moved = (f"; improved {e['improved']}" if e.get("improved") else "") + \
                (f"; regressed {e['regressed']}" if e.get("regressed") else "")
        lines.append(f"  {e.get('source_run')} r{e.get('round')} [{e.get('component')}] {e.get('outcome')}"
                     f"{meas}: {(e.get('hypothesis') or '')[:200]}{moved}")
    head = ("Candidates from earlier runs (possibly a different starting harness or model; measured against "
            "their own incumbents). Don't repeat a rejected idea without a reason it would differ now:")
    body = "\n".join(lines)
    if len(head) + len(body) + 1 > max_chars:
        body = "  [older entries truncated]\n" + body[-(max_chars - len(head) - 30):]
    return head + "\n" + body
