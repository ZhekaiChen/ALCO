# AGENTS.md

## Project identity

This repository studies action-level reasoning and reinforcement learning for large language models on combinatorial optimization, starting with the Traveling Salesman Problem (TSP).

The central task is not one-shot full-tour prediction.
Instead, the model must solve TSP step by step:
- reason globally,
- output exactly one next-node action at each rollout step,
- receive validation and reward at the action level.

The project has four tightly coupled layers:
1. online TSP data / instance handling,
2. LKH3 reference solving and constrained completion,
3. rollout prompting / parsing / logging,
4. SFT and RL training.

---

## Project invariants

The following rules are mandatory and should not be changed unless the user explicitly updates them.

### Node indexing
- All TSP node ids follow TSPLIB-style indexing.
- For an instance with `n` nodes, node ids are exactly `1..n`.
- Do not silently convert the public-facing representation to 0-based numbering in prompts, logs, or saved data.
- If internal tensors use a different indexing convention, the conversion must be explicit and documented.

### Partial route semantics
- `partial_route` is an ordered, directed, fixed prefix.
- It is not a bag/set of visited nodes.
- It must not be reordered during constrained completion or reward generation.

### Reward interface
- Reward logic must remain configurable.
- Do not hardcode a single final reward formula.
- Different reward functions will be compared experimentally.

### Reasoning logs
- Full reasoning text from rollout, SFT, and RL-related evaluation outputs is an important research artifact.
- Do not discard or truncate reasoning traces unless explicitly configured.
- Parsing of the final action must rely only on the final action tag, but the full reasoning text must always be stored in logs.

---

## Core task protocol

At each rollout step, the model is given:
- number of nodes,
- node ids and 2D coordinates,
- current partial route,
- optionally the current position,
- explicit instruction to think globally about the full tour,
- instruction to output one final next node.

The model response must contain:
1. detailed reasoning text,
2. exactly one final tag:

`<FINAL_NEXT_NODE>INTEGER</FINAL_NEXT_NODE>`

The parser must rely only on the final tag.
The full reasoning text must still be logged and saved for analysis and later SFT extraction.

If the final tag is missing, malformed, duplicated, or invalid, this must be recorded explicitly.

---

## Solver policy

LKH3 is used in two distinct modes:

1. Full reference solve from scratch
   - used to obtain the reference tour and tour length for the instance.

2. Online constrained completion / re-solving
   - used after the model predicts the next node,
   - must preserve the already chosen partial route,
   - must use the user-specified fixed-edge / fixed-sequence mechanism in LKH3,
   - must be implemented as an explicit solver mode, not hidden inside unrelated logic.

Never blur these two solver modes together.

If the exact constrained behavior is not fully implemented yet:
- do not fake it,
- do not silently swap in a different method,
- raise a clear TODO or NotImplementedError,
- document the limitation in code and README.

---

## Online reward policy

Reward generation is online.

For each rollout step:
1. receive current partial route,
2. obtain the model's next-node prediction,
3. validate the action,
4. append it to the route if valid,
5. call LKH3 constrained completion,
6. compare the resulting completed tour with the LKH3 full reference solution,
7. compute reward.

Reward design must remain configurable.
Do not hardcode one fixed reward formula unless explicitly instructed.

Keep placeholders for:
- normalized tour gap reward,
- exact-match style reward,
- invalid-action penalty,
- repeated-node penalty,
- formatting penalty,
- optional step shaping.

---

## Repository structure conventions

Prefer the following structure unless the repository already has a better equivalent:

- `src/` for project code
- `configs/` for YAML/JSON/TOML configuration
- `scripts/` for runnable entrypoints
- `tests/` for smoke tests and parsers
- `third_party/` or `external/` for SLIME and LKH3
- `data/` for generated instances and caches
- `outputs/` for logs, rollout traces, training artifacts, and reports

Do not mix third-party code into project-specific modules.

The recommended package path in this repository is `src/tsp_action_rl/`.
If the package name changes later, update all specs consistently.

---

## Dependency placement policy

### SLIME
- Do not copy SLIME source files into project modules.
- Keep SLIME isolated under `third_party/slime/`.
- Prefer a git submodule or a clearly version-pinned vendored directory.
- Project-specific integration code must live outside SLIME, for example in `src/tsp_action_rl/rl/slime_adapter.py`.

### LKH3
- Keep LKH3 source files isolated under `third_party/LKH3/`.
- Provide scripts to download and build it reproducibly.
- Prefer a transparent Python subprocess wrapper around the LKH3 executable.
- A thin Python `lkh` library may be used as an optional helper, but it must not hide solver file generation, parameter files, stdout/stderr capture, or constrained-completion behavior.

---

## Engineering bias: avoid overengineering

- Prefer one clear implementation path over many speculative branches.
- Do not add generic strategy layers, compatibility wrappers, or backend selectors unless they are required by the current task.
- Validate strictly at system boundaries:
  - config loading,
  - prompt parsing,
  - action validity,
  - solver invocation,
  - file I/O.
- Inside the core pipeline, prefer simple explicit logic.
- Fail fast with clear errors instead of adding broad fallback logic.
- If a future extension is plausible but not needed now, leave a TODO rather than building the abstraction early.

## No fake completeness

- Do not create placeholder abstraction layers that pretend a feature is implemented when the core logic is not.
- If constrained LKH completion is incomplete, raise an explicit error or mark the code path clearly as TODO.
- Prefer a small honest implementation over a large but partially fake architecture.

---

## Working style

When implementing:
- inspect the existing code first,
- summarize the current state before major changes,
- propose the file tree if changing architecture,
- then implement in small reviewable steps.

Do not silently invent missing requirements.
If a decision is unresolved, add a visible TODO in code and docs.

Prefer:
- modular interfaces,
- explicit configs,
- typed Python,
- docstrings,
- small smoke tests,
- reproducible scripts.

Avoid:
- giant monolithic scripts,
- hardcoded absolute paths,
- hidden assumptions,
- parser logic that depends on free-form reasoning text.

---

## Prompting and logging rules

All prompt templates must consistently enforce:
- global reasoning,
- detailed thought process,
- strict final action tag.

All rollout logs should store:
- rendered prompt,
- raw model output,
- extracted reasoning text,
- extracted final action,
- parsing status,
- action validity,
- solver completion result,
- reward-relevant stats,
- episode status.

Reasoning traces are important research artifacts.
Do not discard them.

---

## Training pipeline rules

The intended training order is:
1. zero-shot rollout collection,
2. select best traces,
3. small-scale SFT warm start,
4. RL fine-tuning,
5. held-out evaluation.

Do not skip the logging and data export pieces just because training code exists.

The RL environment must expose:
- state,
- action,
- validity checks,
- transition,
- reward,
- done condition.

Keep rollout logic and training logic separated where possible.

---

## LKH3 debugging rules

Whenever solver behavior is implemented or changed:
- keep generated problem files inspectable,
- keep parameter files inspectable,
- keep solver stdout/stderr capturable,
- save temporary artifacts in debug mode,
- make node indexing conventions explicit,
- document node ids as 1-based externally.

Before hardcoding LKH3 constraint-field usage, verify the local LKH3 version and a minimal runnable example.

---

## Testing expectations

Whenever possible, add or update:
- parser tests,
- prompt rendering tests,
- TSP instance generation tests,
- LKH3 wrapper smoke tests,
- constrained completion smoke tests,
- rollout environment smoke tests.

---

## Planning rule

If a task is likely to:
- touch more than 5 files,
- affect more than one subsystem,
- or require architecture changes,

first create or update `PLANS.md` before making large edits.

The plan should include:
- goal,
- affected files,
- assumptions,
- unresolved decisions,
- implementation steps,
- validation steps.

---

## Review checklist

Before considering a task complete, verify:
- the final action tag contract is preserved,
- reasoning text is logged,
- solver assumptions are documented,
- no third-party code was mixed into project modules,
- config values are not hardcoded,
- TODOs remain explicit where design is unresolved,
- the relevant script or smoke test can run.
