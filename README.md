# harness-evolve

[![tests](https://github.com/sidreddy88/harness-evolve/actions/workflows/tests.yml/badge.svg)](https://github.com/sidreddy88/harness-evolve/actions/workflows/tests.yml) [![PyPI](https://img.shields.io/pypi/v/harness-evolve)](https://pypi.org/project/harness-evolve/)

Improve an AI agent's harness (its prompts, tool descriptions and settings) from the agent's own failure
traces, and keep a change only when it beats measured run-to-run noise.

Agents are noisy: run the same agent on the same task twice and the result can flip. Tune a prompt,
rerun, see +3 points, and you may have measured nothing. harness-evolve measures that noise first,
then accepts an edit only if it clears it, checks it doesn't break the cases that already work, and
reports the result on cases it never saw, against the cheapest alternative: just running the original
agent twice.

## Results

Used on a production code-diagnosis agent ([agent-platform](https://github.com/sidreddy88/agent-platform)), on SWE-bench Verified with
DeepSeek-V4.1-Flash:

![Held-out accuracy and cost per task: original one run 78.8% at $0.033, original best of 2 runs 89.9% at $0.066, evolved one run 89.9% at $0.040](https://raw.githubusercontent.com/sidreddy88/harness-evolve/main/docs/results.png)

| Held-out: 66 cases from repositories the optimizer never saw | Original harness | Evolved harness |
|---|---|---|
| Finds the file the fix must change (single run) | 78.8% | **89.9%** (+11.1, 95% CI +6.1 to +16.2) |
| Gives up with no answer | 19.7% | 5.6% |
| Solves it on all 3 trials (reliability) | 60.6% | 78.8% |

The evolved harness matched running the original twice (best-of-2) at about 60% of the cost.
Two edits were accepted in a 7.4-hour run, and the evolved harness went to production.

It also says no. An earlier run of the same optimizer on Claude Sonnet 5 accepted nothing, and so
did a three-round run on the same project's fix agent: every candidate edit either stayed inside the
noise band or broke a guard case. Halfway through that run, one edit led 7 to 4, and a single
measurement would have reported it as a gain.

## Quickstart: the math example

The example (`examples/math_agent/`) is a small math agent (a calculator tool and a submit tool) on
GSM-Hard, GSM8K questions with large, awkward numbers that leave room to improve. Everything that shapes
its behaviour is in `examples/math_agent/harness/`, the directory the optimizer edits:

- prompts: `system.prompt`, `task.prompt`, `nudge.prompt` (sent when a reply has no tool call),
  `remind.prompt` (stuck reminder), `plan.prompt` (planning step)
- `tool_descriptions.json`: what the model is told each tool does
- `settings.json`: `max_turns`, `nudges`, stuck detection (`remind_after`, `stop_after`), and two
  structural switches, `plan` (off/on) and `calculator` (on/off)

`config.py` declares the profile (setting bounds and switches), the evaluator and the case split;
`bench.py` holds the benchmark scenarios built on it (see Benchmarking the optimizer below).

```bash
pip install "harness-evolve[litellm]"        # the library and the harness-evolve command
git clone https://github.com/sidreddy88/harness-evolve && cd harness-evolve   # for the example
python examples/math_agent/fetch_data.py      # downloads 80 problems
export TOGETHER_API_KEY=...                   # or any litellm provider; set MODEL to match
harness-evolve check examples/math_agent/config.py --budget 2   # free: no model calls
harness-evolve run examples/math_agent/config.py --run-dir runs/math-1 --budget 2
harness-evolve status runs/math-1
harness-evolve diff runs/math-1               # accepted edits, as diffs to review
```

What to expect (our run, DeepSeek-V4.1-Flash, $1.06, before the stuck-detection settings and switches
were added): the starting harness solved 76% of the 50 evolve
problems, with a measured noise band of ±5.6 points. The proposer read the trajectories and aimed three
edits at real failures (runs that never submit an answer, then wrong arithmetic); each gained +2 to +3
points, inside the noise, so none was accepted. With 50 cases only bigger gains are detectable; more
cases narrow the band.

`--budget` is a hard cap in dollars. A run that stops (budget, Ctrl-C, a provider outage) resumes with
the same command, and nothing already paid for is paid again.

## How a run works

```
round 0   run the starting harness on the evolve cases, 2 trials each
          -> the noise band (how much the pass rate moves between identical runs)
round r   propose one edit, aimed at failures in the trajectories (GEPA-style)
          -> structural checks (free)
          -> a critic screens for leakage, gaming and weakened safeguards (cheap)
          -> evaluate the candidate (the only expensive step)
          -> accept only if it beats the noise band, or is cheaper beyond cost noise
             at the same pass rate, and no guard case regresses
final     original vs evolved on held-out cases, k trials each, including the
          original's best-of-2 (the matched-budget rerun baseline)
```

Accepted edits are written as diffs under `runs/<id>/proposals/` for you to review; your code is never
changed automatically.

Built for long unattended runs: every phase is checkpointed; every case's result is cached by the
candidate's content hash; a hard budget is checked before each case; provider failures (billing, auth,
rate limits) stop the run instead of being scored as failures; tripwires pause it when measurements
look broken; a watchdog exits cleanly if it stops making progress.

## Using it on your agent

The quickest start is a generated project:

```bash
harness-evolve init my-agent     # harness/, cases.jsonl, run_case.py, config.py, README.md
```

It is a working one-call question-answering agent wired to the optimizer. Fill in four things,
following `my-agent/README.md`:

1. `harness/`: your agent's prompts, tool descriptions and settings.
2. `run_case.py`: runs your agent on one case with the harness directory it is given, and prints one
   JSON line, `{"passed": ..., "cost_usd": ..., "steps": [...]}`. It can call an agent in any language
   or behind an API; or point `COMMAND` in `config.py` at your own program and skip Python entirely
   (`harness_evolve.command.CommandEvaluator` runs it once per case and trial).
3. `cases.jsonl`: real tasks, 30+ (50+ to also keep a held-out set).
4. `config.py`: a description of your agent, what an edit must never weaken, setting bounds.

Then `harness-evolve check config.py --budget 2`, then `harness-evolve run config.py --run-dir runs/r1
--budget 2`. The config refuses to run until the agent is described.

`check` builds the optimizer from your config and reads it without calling a model or running the agent,
so it is free; it exits 1 on a FAIL (`--strict`: on a WARN too), so CI can run it before a scheduled run.
It checks that the starting harness passes the validation every candidate must pass and that every
numeric setting is bounded; that held-out cases don't overlap the evolve and guard cases; the smallest
gain your case and trial counts can accept; whether the budget finishes round 0 and how many rounds it
covers; that the proposer and critic models have prices; and that the tripwire baselines and the smoke
stage are set. Each is a mistake one of our runs paid for. What only a real run shows (a tool failing on
every call, $0 replays) is the smoke stage's job (`smoke_cases`, ~$1).

The rest of this section is what that project contains, if you'd rather wire it up yourself.
You need five things.

**1. A harness directory your agent reads at run time**

```
my_harness/
  system.prompt              # templates, rendered with str.format
  task.prompt
  tool_descriptions.json     # {"search": "Search the docs ..."}
  settings.json              # {"max_turns": 12, "_doc": {"max_turns": "turn budget"}}
```

Load it with `harness_evolve.load_harness(path)`. The optimizer edits copies of these files, never
adds or deletes files, and checks every edit still loads.

**2. An evaluator**

```python
class MyEvaluator:
    async def evaluate(self, harness_dir, case_ids, trials) -> EvalOutcome:
        ...  # run your agent with harness_dir on each case, `trials` times
```

Return pass/fail and cost per trial, and trajectories if you can: the proposer reads them to find
out why runs fail. Raise `ProviderFailure` for billing, auth or outage errors. Never let the right
answer reach the proposer: put it after `evidence.HIDDEN_MARKER` in a trajectory's `detail`.
See `examples/math_agent/evaluator.py`.

If your agent runs in [Harbor](https://github.com/harbor-framework/harbor), you don't need to build
trajectory records yourself: `harness_evolve.atif.harbor_job(job_dir)` turns a Harbor job's ATIF
trajectories and rewards into records the proposer can read (`from_atif` does one trajectory).

**3. A profile**

```python
from harness_evolve import AgentProfile, profiles
profiles.use(AgentProfile.build(
    name="support-bot", harness_root="my_harness/",
    agent_description="Answers support tickets with search and read tools; finishes by calling reply.",
    safety_mechanisms=["citing a source document for every claim"],
    setting_bounds={"max_turns": (3, 20)}))
```

The description and safety mechanisms become the proposer's and critic's instructions; setting
bounds keep an edit from "saving cost" by starving the agent.

**4. Cases**: an evolve set (what it improves on, ideally 50+), optional guard cases (must keep
passing) and held-out cases (never seen during evolution, for the final report).

**5. A model for the proposer and critic, and a budget**: `litellm_llm("model-id", prices=(in, out))`
or any async `(system, prompt) -> LLMReply(text, cost_usd)`.

Then:

```python
opt = Optimizer("runs/r1", "my_harness/", OptimizerConfig(evolve_cases=..., budget_usd=50),
                evaluator=MyEvaluator(), proposer_llm=llm, critic_llm=llm, critic_patterns=[])
asyncio.run(opt.run_until_stopped())
```

Or put a `build(run_dir, budget_usd, rounds)` function in a config file and use the CLI, as the
example does.

## Benchmarking the optimizer

`harness-evolve bench` measures the optimizer itself, so a change to its method is judged by
evidence, not opinion. Known-answer scenarios have a right outcome: a scripted proposer puts one known
edit in front of the acceptance rules. A real fix to a planted flaw must be accepted, a harmful edit
must be rejected, and a neutral rewording must not be accepted (a false accept). Open scenarios run the
real proposer and report what it finds and what it gains on held-out cases, per dollar.

```bash
python examples/math_agent/fetch_data.py --n 260 --out examples/math_agent/data/problems-bench.jsonl
harness-evolve bench examples/math_agent/bench.py --out runs/bench-1
```

Results go to `runs/bench-1/bench.md`. Our first run (DeepSeek-V4.1-Flash, $9.37 in total):

| Scenario | Expected | Result |
|---|---|---|
| accept-good: planted flaw (`max_turns` 1) + scripted fix | accepted | PASS: +49 points |
| reject-bad: scripted "round answers to the nearest hundred" | not accepted | PASS: −21.5 points, rejected |
| reject-noise: scripted neutral rewording | not accepted | PASS: rejected |
| find-flaw: planted flaw, real proposer | (reported) | found the flaw from trajectories: held-out 28.3% → 79.0% |
| standard: normal harness, 200 problems, real proposer | (reported) | one prompt edit accepted: held-out accuracy unchanged (83.3% → 83.9%), ~11% cheaper |

The first version of reject-bad ("skip the calculator and estimate") was accepted: it didn't hurt this
model at all and was 13% cheaper, so the optimizer was right and the test was wrong. A known-answer
scenario's premise has to be measured, not assumed.

Later scenarios test stuck-detection thresholds (`remind_after`, `stop_after` in the example's settings),
starting from a harness that stops every run at its second calculator call ($3.89):

| Scenario | Expected | Result |
|---|---|---|
| accept-threshold: planted flaw (`stop_after` 1) + scripted fix (8) | accepted | PASS: +9.5 points |
| find-threshold: same flaw, real proposer | (reported) | found it from the stop messages in the trajectories and set `stop_after` to 4: held-out 71.7% → 78.5% (+6.9, 95% CI +4.0 to +10.2); within noise of rerunning the original twice |

A last scenario measures what confirm-before-accept (`confirm=True`) catches: 40 scripted neutral rewordings
in a row on a small evolve set (20 problems), so measurement noise is large ($3.83). One of the 40 passed
its first measurement (18% cheaper, beyond the 15.4% cost band), a false accept with confirm off; on fresh
trials it was 10% more expensive, and confirm rejected it. The other 39 were rejected outright. No pass-rate
change came near the noise band (largest +0.05 against 0.12): on a small evolve set, false accepts come from
cost noise.

Structural switches (`switches` in `AgentProfile.build`: settings with named options, such as a planning
step on or off, which the proposer is told to check before rewording) were tested with the example's
`plan` and `calculator` switches ($7.50):

| Scenario | Result |
|---|---|
| find-switch: starts with `plan` on (an extra model call per problem), real proposer | round 1 switched `plan` off: 22% cheaper on the evolve set, pass rate within noise; held-out 80.4% → 79.0% (within noise) at 34% lower cost per run, though runs that never submit rose from 2.3% to 5.6% |
| standard with switches, real proposer, 3 rounds | round 1 tried a switch first (`plan` on: +0.75 points at +41% cost, rejected); a tool-description edit and more nudges were rejected too; nothing accepted | Write your own scenarios with `harness_evolve.bench.Scenario`
and `scripted_llm` (see `examples/math_agent/bench.py`).

## Limits

- It needs an automatic pass/fail check per case. No grader, no optimizer.
- A noisy agent needs more cases or trials before any gain is detectable; the noise band will tell
  you, and "nothing accepted" is a real answer.
- Every candidate costs a full evaluation. Slow or expensive agents make slow, expensive runs.

## Credits

Each component, its source, and what we changed: [docs/methods.md](docs/methods.md).

- **Acceptance rules and noise band:** RRSI, Xia et al., "RRSI: Regularized Recursive Self-Improvement of Agent
  Harnesses", [arXiv 2609.24972](https://arxiv.org/abs/2609.24972). The critic's reject classes are adapted
  from its prompt ([google-research/rrsi](https://github.com/google-research/rrsi), Apache-2.0; see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
- **Reflective, trace-driven proposals:** GEPA, Agrawal et al., "GEPA: Reflective Prompt Evolution Can
  Outperform Reinforcement Learning", ICLR 2026.
- **Comparing against reruns:** Wang et al. (AI2), "Rethinking the Evaluation of Harness Evolution for Agents",
  [arXiv 2607.12227](https://arxiv.org/abs/2607.12227).
- **Agent self-feedback as evidence:** HarnessCompass, [arXiv 2608.01918](https://arxiv.org/abs/2608.01918).
- **Near-miss archive:** Darwin Gödel Machine ([arXiv 2505.22954](https://arxiv.org/abs/2505.22954)) and
  Huxley-Gödel Machine ([arXiv 2510.21614](https://arxiv.org/abs/2510.21614)).
- **Cost only on shared successes:** Hugging Face / FineEnvs, "Multi-harness RL".
- **Structural switches and confirm-before-accept:** motivated by Fan et al., "An Empirical Study of Harness
  Design for Coding Agents", [arXiv 2609.20804](https://arxiv.org/abs/2609.20804).
- **pass@k and pass^k:** Chen et al., "Evaluating Large Language Models Trained on Code" (HumanEval, 2021);
  Yao et al., τ-bench (2024).

MIT license.
