# ALCO

Action-level reasoning and reinforcement learning research codebase for TSP with LLMs.

## Phase 2 LKH Integration

- Primary Python interface: `lkh.solve(...)` via [`src/tsp_action_rl/solvers/lkh_integration.py`](src/tsp_action_rl/solvers/lkh_integration.py)
- Canonical pinned local LKH source archive: `third_party/LKH3/LKH-3.0.14.tar`
- Solver settings config: `configs/lkh.yaml`
- Solver executable path is explicit and configurable with `solver_executable`.

### Implemented solver modes

- Full reference solve from scratch:
  - `LKHIntegration.solve_reference(instance)`
- Constrained completion with fixed prefix:
  - `LKHIntegration.solve_with_fixed_prefix(instance, partial_route)`
  - Uses `FIXED_EDGES_SECTION` on consecutive prefix edges.
  - Validates post-solve that the completed cycle can be canonicalized to the exact ordered `partial_route`.
  - Raises `LKHConstraintError` if ordered prefix semantics are not preserved.

### Debug artifacts

When `debug.enabled: true` in `configs/lkh.yaml`, each solve writes inspectable artifacts under `debug.output_root`, including:
- `problem.tsp`
- `params.par` (effective parameter preview)
- `solution.tour` (LKH output target)
- `result.json` (normalized route + length + params snapshot)

### CLI smoke usage

```bash
PYTHONPATH=src python scripts/solve_tsp_with_lkh.py \
  --instance-path tests/fixtures/tsp_instance_minimal.json \
  --mode reference \
  --lkh-config configs/lkh.yaml
```

```bash
PYTHONPATH=src python scripts/solve_tsp_with_lkh.py \
  --instance-path tests/fixtures/tsp_instance_minimal.json \
  --mode constrained \
  --partial-route 1,2,3 \
  --lkh-config configs/lkh.yaml
```

## Phase 2.5 Local Build + Real Validation

Pinned archive (kept in place):
- `third_party/LKH3/LKH-3.0.14.tar`

Reproducible local build from pinned archive:

```bash
bash scripts/setup_lkh.sh
```

Expected executable path:
- `third_party/LKH3/LKH-3.0.14/LKH`

`configs/lkh.yaml` now pins:
- `solver_executable: third_party/LKH3/LKH-3.0.14/LKH`
- `source_archive: third_party/LKH3/LKH-3.0.14.tar`

Run real integration tests (built solver + python `lkh`):

```bash
pytest -q tests/test_lkh_real_integration.py
```

## Phase 3 Zero-Shot Rollout Harness

Implemented modules:
- Prompt rendering: `src/tsp_action_rl/prompts/tsp_prompt.py`
- Strict final-tag parser: `src/tsp_action_rl/parsing/final_tag_parser.py`
- Model interface/backends: `src/tsp_action_rl/inference/`
- Episode runner/logging: `src/tsp_action_rl/rollout/zeroshot_runner.py`

Backends:
- `local_deterministic` (implemented, runnable)
- `local_static` (implemented, fixture/test replay)
- `dmx_openai_compatible` (implemented, OpenAI-compatible `/chat/completions`)
- `api_todo` (explicit TODO fallback; raises clear `NotImplementedError`)

Default config:
- `configs/zeroshot_eval.yaml`

Run one local zero-shot rollout run:

```bash
PYTHONPATH=src python scripts/run_zeroshot_rollout.py --backend local_deterministic
```

## Phase 3.5 Real API Zero-Shot Traces (DMXAPI)

This round uses stateless per-step API calls:
- each rollout step is a standalone request,
- no multi-turn chat memory is used,
- only structured TSP state (especially `partial_route`) is carried across steps.

Required environment variables (no secrets in code/config):
- `DMXAPI_BASE_URL` (unless `api.base_url` is set directly in config/CLI)
- `DMXAPI_API_KEY`

First real model defaults in `configs/zeroshot_eval.yaml`:
- backend: `dmx_openai_compatible`
- model: `claude-opus-4-6-thinking`
- thinking effort: `high`

Default real-run test setup:
- random integer-coordinate TSP instances with coordinates in `[0, 10000]`
- node counts: `10, 25, 50`
- random start node
- rollout step policy: predict `node_count - 2` steps, then auto-complete last node

Run the default real zero-shot sweep:

```bash
PYTHONPATH=src python scripts/run_zeroshot_rollout.py
```

Override with a smaller quick smoke run:

```bash
PYTHONPATH=src python scripts/run_zeroshot_rollout.py \
  --generate-random \
  --node-counts 10 \
  --instances-per-node-count 1 \
  --episodes-per-instance 1 \
  --backend dmx_openai_compatible \
  --enable-solver-completion
```

Outputs are written under `outputs/zeroshot/<run_name>/` and include:
- per-episode structured logs under `episodes/`
- generated instances under `instances/` (if enabled)
- `run_summary.json` with config snapshot and aggregate metrics
