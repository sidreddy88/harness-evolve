"""
`harness-evolve init DIR`: a starter project for optimizing your own agent.

It writes a working question-answering agent (one model call per case, graded by matching the
expected answer) wired to the optimizer through CommandEvaluator, so the shape of every piece is
visible. Replace the parts marked TODO with your agent; the optimizer side needs no changes.
"""
from __future__ import annotations

from pathlib import Path

FILES: dict[str, str] = {}

FILES["README.md"] = """# {name}: a harness-evolve project

Four steps to optimize your agent:

1. **Harness** (`harness/`): move the text and knobs that shape your agent's behaviour here.
   `*.prompt` files are templates (`{question}` is filled by the agent), `tool_descriptions.json` is
   what the model is told each tool does, `settings.json` holds numbers or named options, each
   documented under `"_doc"` (the proposer reads those descriptions).
2. **Agent** (`run_case.py`): runs ONE case with the harness directory it is given and prints one JSON
   line: `{"passed": ..., "cost_usd": ..., ...}` (see the docstring). The starter is a one-call
   question-answering agent; replace `run_agent` with yours. Any language works: point `COMMAND` in
   config.py at your own program.
3. **Cases** (`cases.jsonl`): one JSON object per line with an `"id"` and whatever your agent needs.
   Use real tasks from your workload; 30+ for the noise band to be useful, 50+ to also keep a
   held-out set.
4. **Config** (`config.py`): describe your agent (`AGENT_DESCRIPTION`), list what an edit must never
   weaken (`SAFETY`), set bounds for numeric settings. Then:

```bash
export MODEL=together_ai/deepseek-ai/DeepSeek-V4.1-Flash     # any litellm model id
export PRICES_PER_MTOK=0.30,1.20                             # $ per million input,output tokens (optional)
python run_case.py harness q1                                # one case by hand: check the JSON line
harness-evolve run config.py --run-dir runs/r1 --budget 2
harness-evolve status runs/r1
harness-evolve diff runs/r1
```

`--budget` is a hard cap in dollars; a stopped run resumes with the same command.
"""

FILES["harness/system.prompt"] = "You are a helpful assistant that answers questions accurately.\n"

FILES["harness/task.prompt"] = """Question: {question}

Give the final answer on the last line, as: Answer: <answer>
"""

FILES["harness/tool_descriptions.json"] = "{}\n"

FILES["harness/settings.json"] = """{
 "_doc": {
  "max_tokens": "maximum length of the model's reply, in tokens"
 },
 "max_tokens": 1000
}
"""

FILES["cases.jsonl"] = """{"id": "q1", "question": "What is 17 * 23?", "expected": "391"}
{"id": "q2", "question": "What is the capital of Australia?", "expected": "Canberra"}
{"id": "q3", "question": "How many days are in a leap year?", "expected": "366"}
"""

FILES["run_case.py"] = '''"""
Run the agent on ONE case with a given harness, grade it, and print one JSON line (the last line of
stdout is what harness-evolve reads):

    python run_case.py HARNESS_DIR CASE_ID

    {"passed": bool, "cost_usd": float,              required
     "submitted": bool, "detail": str, "answer": str,  optional ("answer" is hidden from the proposer)
     "steps": [{"name", "input", "output"}], "turns": int,
     "provider_failure": str}                           set on billing/auth/rate-limit errors

TODO: replace run_agent (and grade, if exact matching doesn't fit your task).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from harness_evolve import load_harness

HERE = Path(__file__).resolve().parent


def run_agent(harness, case: dict) -> dict:
    """The starter agent: one model call. Returns the reply text, cost and tool steps (none here)."""
    import litellm

    resp = litellm.completion(
        model=os.environ["MODEL"], max_tokens=int(harness.setting("max_tokens")),
        messages=[{"role": "system", "content": harness.render("system")},
                  {"role": "user", "content": harness.render("task", question=case["question"])}])
    text = resp.choices[0].message.content or ""
    prices = os.environ.get("PRICES_PER_MTOK")
    if prices:
        p_in, p_out = (float(x) for x in prices.split(","))
        cost = (resp.usage.prompt_tokens * p_in + resp.usage.completion_tokens * p_out) / 1e6
    else:
        cost = litellm.completion_cost(completion_response=resp)
    return {"text": text, "cost_usd": cost, "steps": [], "turns": 1}


def grade(case: dict, text: str) -> tuple[bool, bool, str]:
    """(passed, submitted, what the agent answered). Exact match on the "Answer:" line."""
    lines = [x for x in text.strip().splitlines() if x.lower().startswith("answer:")]
    if not lines:
        return False, False, "no Answer: line"
    given = lines[-1].split(":", 1)[1].strip()
    return given.lower().rstrip(".") == str(case["expected"]).lower(), True, f"answered {given[:80]}"


def _provider_problem(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(w in text for w in ("authentication", "ratelimit", "rate limit", "credit", "billing", "quota",
                                   "insufficient", "api key", "serviceunavailable"))


def main(harness_dir: str, case_id: str) -> dict:
    cases = {c["id"]: c for c in map(json.loads, (HERE / "cases.jsonl").read_text().splitlines()) if c}
    case = cases[case_id]
    try:
        run = run_agent(load_harness(harness_dir), case)
    except Exception as exc:
        if _provider_problem(exc):
            return {"passed": False, "cost_usd": 0.0, "provider_failure": f"{type(exc).__name__}: {exc}"}
        raise
    passed, submitted, detail = grade(case, run["text"])
    return {"passed": passed, "submitted": submitted, "cost_usd": run["cost_usd"], "detail": detail,
            "answer": f"expected {case['expected']}", "steps": run["steps"], "turns": run["turns"]}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2])))
'''

FILES["config.py"] = '''"""
Run configuration: the agent profile, the evaluator and the case split. The CLI loads this file:

    harness-evolve run config.py --run-dir runs/r1 --budget 2

Environment: MODEL (litellm model id for the agent), PROPOSER_MODEL (for the proposer and critic;
defaults to MODEL), PRICES_PER_MTOK="0.30,1.20" (the agent model's price, if litellm doesn't know it).
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from harness_evolve import AgentProfile, Optimizer, OptimizerConfig, litellm_llm, profiles
from harness_evolve.command import CommandEvaluator
from harness_evolve.grader import TrajectoryGrader

HERE = Path(__file__).resolve().parent

# TODO: what your agent does, its tools, and how it finishes a task. The proposer and critic read this.
AGENT_DESCRIPTION = "TODO: describe your agent"
# TODO: behaviours an edit must never weaken (the critic rejects edits that do).
SAFETY = ["giving a final answer on the last line"]
# TODO: allowed ranges for numeric settings, or {name: {option: what it does}} as SWITCHES.
SETTING_BOUNDS = {"max_tokens": (200, 4000)}
SWITCHES: dict = {}
# How a candidate harness is measured: one run of this command per case and trial.
COMMAND = [sys.executable, str(HERE / "run_case.py"), "{harness}", "{case}"]

PROFILE = AgentProfile.build(
    name=HERE.name, harness_root=HERE / "harness", agent_description=AGENT_DESCRIPTION,
    safety_mechanisms=SAFETY, constraints="Edits must help on any task of this kind, not on particular cases.",
    setting_bounds=SETTING_BOUNDS, switches=SWITCHES, grader=TrajectoryGrader())


def build(run_dir, budget_usd: float = 2.0, rounds: int = 3) -> Optimizer:
    if AGENT_DESCRIPTION.startswith("TODO"):
        raise SystemExit(f"describe your agent first: AGENT_DESCRIPTION in {__file__}")
    profiles.use(PROFILE)
    ids = [json.loads(x)["id"] for x in (HERE / "cases.jsonl").read_text().splitlines() if x.strip()]
    random.Random(0).shuffle(ids)
    n_held = len(ids) * 3 // 10 if len(ids) >= 50 else 0      # keep ~30% unseen for the final report
    evolve, held = ids[n_held:], ids[:n_held]
    cfg = OptimizerConfig(evolve_cases=evolve, guard_cases=[], final_cases=held, final_trials=3,
                          budget_usd=budget_usd, max_rounds=rounds, trials=2, calibration_trials=2,
                          default_case_cost_usd=0.01, cost_band=True, parallel_lanes=4,
                          case_lanes={c: c for c in ids})
    model = os.environ.get("PROPOSER_MODEL") or os.environ.get("MODEL")
    if not model:
        raise SystemExit("set MODEL (and optionally PROPOSER_MODEL) to a litellm model id")
    prices = os.environ.get("PRICES_PER_MTOK")
    same_model = model == os.environ.get("MODEL")          # PRICES_PER_MTOK is the agent model's price
    llm = litellm_llm(model, prices=tuple(float(x) for x in prices.split(",")) if prices and same_model else None)
    return Optimizer(run_dir, HERE / "harness", cfg, evaluator=CommandEvaluator(COMMAND, concurrency=4),
                     proposer_llm=llm, critic_llm=llm, critic_patterns=[])
'''


def write(directory: str | Path, force: bool = False) -> list[Path]:
    """Write the starter project into `directory`. Refuses to overwrite files unless force."""
    root = Path(directory)
    existing = [root / rel for rel in FILES if (root / rel).exists()]
    if existing and not force:
        raise FileExistsError(f"{len(existing)} starter files already exist in {root} (e.g. {existing[0]}); "
                              "use --force to overwrite")
    written = []
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.replace("{name}", root.resolve().name) if rel == "README.md" else text)
        written.append(path)
    return written
