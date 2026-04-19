# ALCO

Action-level reasoning and reinforcement learning research codebase for TSP with LLMs.

## Phase 3.6a UV Environment Workflow

This repository now uses a uv-managed dependency workflow with `pyproject.toml` as the source of truth and `uv.lock` as the reproducible lockfile.

### Dependency groups

- `core`: base runtime for current implemented pipeline (data/config, LKH integration via `lkh`, rollout/parsing/logging, SFT trace extraction).
- `api`: API rollout stage dependencies (kept explicit for staged installs; currently no extra packages beyond stdlib).
- `local-infer`: local open-source inference preparation utilities (`huggingface_hub` download tooling and local-serve helpers).
- `sft`: reserved for future SFT training stack (not implemented in this round).
- `rl`: reserved for future RL training/integration stack (not implemented in this round).
- `dev`: developer/test tools (`pytest`).

Note: `lkh==2.0.0` and `vllm` currently have incompatible `click` version requirements, so the repo keeps `lkh` in `core` and starts vLLM via `uvx` (isolated tool env) instead of installing vLLM into this project environment.

### Recommended uv commands

If `uv` is not on your `PATH`, use `python -m uv ...` for the same commands.

Create/sync base runtime environment:

```bash
uv sync --group core
```

Sync for API rollout work (current production path):

```bash
uv sync --group core --group api
```

Sync for development/tests:

```bash
uv sync --group core --group api --group dev
```

Prepare staged environments:

```bash
uv sync --group core --group local-infer
uv sync --group core --group sft
uv sync --group core --group rl
```

Regenerate/update lockfile after dependency changes:

```bash
uv lock
```

Use locked versions only (CI/repro runs):

```bash
uv sync --frozen --group core --group api
```

### Run scripts with uv

```bash
uv run --group core python scripts/solve_tsp_with_lkh.py --help
uv run --group core --group api python scripts/run_zeroshot_rollout.py --help
uv run --group core --group local-infer python scripts/download_hf_model.py --help
uv run --group core --group local-infer python scripts/serve_vllm_openai.py --help
uv run --group core python scripts/extract_sft_data.py --help
uv run --group core --group api --group dev pytest -q
```

### API environment variables

For DMXAPI rollout runs, set:
- `DMXAPI_BASE_URL`
- `DMXAPI_API_KEY`

For local vLLM runs (Phase 3.6b), optional/common env vars are:
- `HF_TOKEN` (for gated HF model downloads)
- `LOCAL_VLLM_BASE_URL` (if overriding local server URL)
- `LOCAL_VLLM_API_KEY` (optional local server auth)

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

## Phase 3.7 Step Retry + DMX Profiles

Step-level retry is configurable in `configs/zeroshot_eval.yaml`:
- `max_step_retries` (bounded retries per step state, not per episode)
- `retry_on_parse_failure` (`missing_tag`, `multiple_tags`, `malformed_tag`)
- `retry_on_provider_error` (transient timeout / HTTP 5xx style failures)
- `retry_backoff_seconds`

Retry details are additive and backward-compatible:
- per-step retry attempt details are stored in `step_logs[*].metadata.retry`
- run/episode-level retry counters are stored in episode `metadata` and `run_summary.json`

DMX model profiles are now explicit via `api.model_profile`:
- `claude-opus-4-6-thinking`
- `glm-5.1`
- `gpt-5.4`

Each profile can control request-shaping defaults (model id, reasoning field usage, timeout, omit list, extra body).
Manual overrides still work via `api.model_id`, `api.thinking_effort*`, and `api.omit_request_fields`.

Minimal profile smoke commands:

```bash
PYTHONPATH=src python scripts/run_zeroshot_rollout.py \
  --backend dmx_openai_compatible \
  --api-model-profile claude-opus-4-6-thinking \
  --node-counts 10 --instances-per-node-count 1 --episodes-per-instance 1 \
  --max-step-retries 1 --retry-on-parse-failure --retry-on-provider-error
```

```bash
PYTHONPATH=src python scripts/run_zeroshot_rollout.py \
  --backend dmx_openai_compatible \
  --api-model-profile glm-5.1 \
  --node-counts 10 --instances-per-node-count 1 --episodes-per-instance 1
```

```bash
PYTHONPATH=src python scripts/run_zeroshot_rollout.py \
  --backend dmx_openai_compatible \
  --api-model-profile gpt-5.4 \
  --node-counts 10 --instances-per-node-count 1 --episodes-per-instance 1
```

## Phase 3.6b-3.6d Local OSS Inference (vLLM Serve Path + TP=8 Ops)

First local model target:
- `Qwen/Qwen3-30B-A3B-Thinking-2507`

External model storage default:
- `/mnt/zc/models`

Phase-3.6b config files:
- `configs/local_vllm.yaml`
- `configs/zeroshot_local_vllm.yaml`

### 1) Sync the local-infer tooling env

```bash
uv sync --group core --group local-infer
```

### 2) Download model weights to external storage

```bash
uv run --group core --group local-infer python scripts/download_hf_model.py \
  --config configs/local_vllm.yaml
```

You can override destination/model explicitly:

```bash
uv run --group core --group local-infer python scripts/download_hf_model.py \
  --model-id Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --models-root /mnt/zc/models
```

### 3) Launch local OpenAI-compatible vLLM server (single-node TP=8)

Recommended helper (uses `uvx --from vllm ...` by default):

```bash
uv run --group core --group local-infer python scripts/serve_vllm_openai.py \
  --config configs/local_vllm.yaml \
  --launcher uvx
```

Optional one-time tool bootstrap (can reduce first-launch wait):

```bash
python -m uv tool install vllm
```

Explicit TP=8 override:

```bash
uv run --group core --group local-infer python scripts/serve_vllm_openai.py \
  --config configs/local_vllm.yaml \
  --launcher uvx \
  --tensor-parallel-size 8 \
  --host 127.0.0.1 \
  --port 8000
```

Equivalent direct command pattern:

```bash
uvx --from vllm vllm serve /mnt/zc/models/Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --tensor-parallel-size 8 \
  --max-model-len 32768
```

Readiness check:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | python -m json.tool
```

### 4) Run minimal local zero-shot smoke (with per-step progress)

```bash
uv run --group core --group api python scripts/run_zeroshot_rollout.py \
  --config configs/zeroshot_local_vllm.yaml \
  --backend local_vllm_openai_compatible \
  --node-counts 10 \
  --instances-per-node-count 1 \
  --episodes-per-instance 1 \
  --show-progress
```

This reuses existing rollout/parsing/logging protocol and writes structured logs under `outputs/zeroshot/<run_name>/`.

### 5) Operator sweep command for `[20, 30, 40]`

```bash
uv run --group core --group api python scripts/run_zeroshot_rollout.py \
  --config configs/zeroshot_local_vllm.yaml \
  --backend local_vllm_openai_compatible \
  --node-counts 20,30,40 \
  --instances-per-node-count 1 \
  --episodes-per-instance 1 \
  --show-progress \
  --run-name local_vllm_tp8_n20_30_40
```

### 6) Phase-4 dry-run extraction on produced local logs

```bash
uv run --group core python scripts/extract_sft_data.py \
  --input-root outputs/zeroshot/local_vllm_tp8_n20_30_40 \
  --dry-run
```

## Phase 4 SFT Trace Mining + Export

Implemented modules:
- `src/tsp_action_rl/sft/models.py`
- `src/tsp_action_rl/sft/trace_mining.py`
- CLI: `scripts/extract_sft_data.py`
- Default config: `configs/sft.yaml`

Supported Phase-4 flow:
- load/index one or more zero-shot run directories,
- filter step traces with explicit quality constraints,
- export per-step SFT examples (internal JSONL),
- optionally export chat-style JSONL,
- emit summary statistics JSON.

Run with default config:

```bash
PYTHONPATH=src python scripts/extract_sft_data.py
```

Dry-run summary only (no example files written):

```bash
PYTHONPATH=src python scripts/extract_sft_data.py --dry-run
```

Example custom extraction run:

```bash
PYTHONPATH=src python scripts/extract_sft_data.py \
  --input-root outputs/zeroshot \
  --require-episode-success \
  --require-solver-completion \
  --node-counts 10,25,50 \
  --model-ids claude-opus-4-6-thinking \
  --output-jsonl outputs/sft/phase4_examples.jsonl \
  --output-chat-jsonl outputs/sft/phase4_chat_examples.jsonl \
  --summary-path outputs/sft/phase4_summary.json
```

Output fields preserve:
- full `prompt_text`,
- full `reasoning_text`,
- canonical final tag (`<FINAL_NEXT_NODE>...`),
- machine-readable `next_node_label`,
- source mapping (`run_name`, `episode_id`, `step_index`, `instance_id`, `node_count`, `model_id`, file paths).

## Step-Level RL Environment + Reward Interface

Implemented modules:
- `src/tsp_action_rl/rl/environment.py`
- `src/tsp_action_rl/rl/reward.py`
- Config loader: `src/tsp_action_rl/config/rl.py`
- Default config: `configs/rl.yaml`

Design points:
- Step-level action semantics only: one action = one next node id.
- Uses online solver-based reward inputs:
  - `solve_reference(instance)` at reset
  - `solve_with_fixed_prefix(instance, partial_route)` after each valid action
- Preserves ordered fixed-prefix route semantics and 1-based node ids.
- Reward is modular/configurable (`gap_to_reference_absolute`, `gap_to_reference_delta`, `sparse_terminal`, `gap_action_inverse`) with explicit invalid/parse penalty hooks.
- Environment emits additive step diagnostics (`action_validation`, `solver_completion`, `reward_signal`, `done_reason`) for later RL debugging and SLIME adapter integration.

Validation test:

```bash
pytest -q tests/test_rl_env.py
```

Load config snapshot in Python:

```python
from tsp_action_rl.config import load_lkh_settings, load_rl_env_settings
from tsp_action_rl.rl import TSPRLStepEnvironment
from tsp_action_rl.solvers import LKHIntegration

lkh_settings = load_lkh_settings("configs/lkh.yaml")
rl_settings = load_rl_env_settings("configs/rl.yaml")
env = TSPRLStepEnvironment(solver=LKHIntegration(lkh_settings), settings=rl_settings)
```

## SLIME Adapter + RL Validation Entrypoints

Real SLIME source is pinned locally under:
- `third_party/slime/` (GitHub `THUDM/slime`, tag `v0.2.4`)

This integration keeps glue code project-owned and non-invasive:
- `src/tsp_action_rl/rl/slime_adapter.py`
- training entrypoint: `scripts/run_rl.py`
- evaluation entrypoint: `scripts/evaluate_model.py`

Adapter behavior:
- preserves step-level action semantics (`action = one next node id`),
- forwards transition/reward/done through `TSPRLStepEnvironment`,
- exposes reset/step in a lightweight adapter format,
- supports optional future real SLIME hooks via config:
  - `slime_adapter.train_entrypoint: module:function`
  - `slime_adapter.eval_entrypoint: module:function`
- supports real SLIME rollout-contract validation mode via config:
  - `slime_adapter.use_real_slime_rollout_contract: true`
  - `slime_adapter.slime_repo_path: third_party/slime`
  - per-mode rollout callable path:
    - `train.slime_rollout_function_path`
    - `eval.slime_rollout_function_path`
- falls back to built-in train/eval validation loops when both entrypoints are null and contract mode is disabled.

Clone/pin SLIME locally (if needed):

```bash
cd /root/ALCO
rm -rf third_party/slime
git init third_party/slime
cd third_party/slime
git remote add origin https://github.com/THUDM/slime.git
git fetch --depth 1 origin bd217a63f113d74488170914e4d39e68c12e8cf0
git checkout --detach FETCH_HEAD
```

Verify pinned revision:

```bash
cd /root/ALCO/third_party/slime
git rev-parse HEAD
```

Quick RL train validation run:

```bash
PYTHONPATH=src python scripts/run_rl.py \
  --config configs/rl.yaml \
  --lkh-config configs/lkh.yaml \
  --pipeline adapter_validation \
  --node-count 10 \
  --train-episodes 1 \
  --run-name rl_train_validation
```

Quick RL eval validation run:

```bash
PYTHONPATH=src python scripts/evaluate_model.py \
  --config configs/rl.yaml \
  --lkh-config configs/lkh.yaml \
  --pipeline adapter_validation \
  --node-count 10 \
  --eval-episodes 1 \
  --run-name rl_eval_validation
```

Quick validation using real SLIME rollout-function contract path:

```bash
PYTHONPATH=src python scripts/run_rl.py \
  --config configs/rl.yaml \
  --lkh-config configs/lkh.yaml \
  --pipeline adapter_validation \
  --node-count 10 \
  --train-episodes 1 \
  --use-real-slime-rollout-contract \
  --slime-repo-path third_party/slime \
  --slime-rollout-function-path tsp_action_rl.rl.slime_adapter.tsp_slime_rollout \
  --slime-rollout-batch-size 1 \
  --slime-n-samples-per-prompt 1 \
  --run-name rl_train_contract_validation
```

```bash
PYTHONPATH=src python scripts/evaluate_model.py \
  --config configs/rl.yaml \
  --lkh-config configs/lkh.yaml \
  --pipeline adapter_validation \
  --node-count 10 \
  --eval-episodes 1 \
  --use-real-slime-rollout-contract \
  --slime-repo-path third_party/slime \
  --slime-rollout-function-path tsp_action_rl.rl.slime_adapter.tsp_slime_rollout \
  --slime-rollout-batch-size 1 \
  --slime-n-samples-per-prompt 1 \
  --run-name rl_eval_contract_validation
```

Outputs:
- `outputs/rl/<run_name>/train_summary.json`
- `outputs/rl/<run_name>/eval_summary.json`

These include per-episode return/length and step-level traces (action, validity, reward components, done reason) for validation diagnostics.

## Real SLIME RL Pipeline (Step-Level TSP)

The SLIME training pipeline adds a real step-level RL path:
- one rollout sample = one TSP state + ordered fixed prefix,
- one model action = exactly one next node,
- reward mode:
  - `reward = 1 / (1 + gap_action)`
  - `gap_action = (ConstrainedTourLength - ReferenceTourLength) / (ReferenceTourLength - PrefixPartialTourLength)`.

The training path is wired through:
- `src/tsp_action_rl/rl/slime_training.py`
- `scripts/run_rl.py --pipeline slime_training`
- `scripts/evaluate_model.py --pipeline slime_training`

Default checkpoint/output root:
- `/opt/aeon/container/`

### Dependency note

Tracking is implemented through a wandb-style API using SwanLab import semantics:

```python
import swanlab as wandb
```

Install RL extras (including SwanLab):

```bash
uv sync --group core --group rl --group dev
```

### Minimal Plan-Only Run (No Training Launch)

```bash
PYTHONPATH=src python scripts/run_rl.py \
  --pipeline slime_training \
  --config configs/rl.yaml \
  --lkh-config configs/lkh.yaml \
  --slime-hf-checkpoint Qwen/Qwen3-4B \
  --slime-algorithm grpo \
  --plan-only \
  --run-name rl_grpo_plan
```

### Actor-Critic (PPO) Training Launch

```bash
PYTHONPATH=src python scripts/run_rl.py \
  --pipeline slime_training \
  --config configs/rl.yaml \
  --lkh-config configs/lkh.yaml \
  --slime-hf-checkpoint /path/to/hf_checkpoint \
  --slime-algorithm actor_critic \
  --slime-num-rollout 2 \
  --training-rollout-batch-size 1 \
  --training-n-samples-per-prompt 1 \
  --run-name rl_actor_critic_train
```

### GRPO Training Launch

```bash
PYTHONPATH=src python scripts/run_rl.py \
  --pipeline slime_training \
  --config configs/rl.yaml \
  --lkh-config configs/lkh.yaml \
  --slime-hf-checkpoint /path/to/hf_checkpoint \
  --slime-algorithm grpo \
  --slime-num-rollout 2 \
  --training-rollout-batch-size 1 \
  --training-n-samples-per-prompt 4 \
  --run-name rl_grpo_train
```

### Evaluation Launch

```bash
PYTHONPATH=src python scripts/evaluate_model.py \
  --pipeline slime_training \
  --config configs/rl.yaml \
  --lkh-config configs/lkh.yaml \
  --slime-hf-checkpoint /path/to/hf_checkpoint \
  --slime-algorithm grpo \
  --training-rollout-batch-size 1 \
  --training-n-samples-per-prompt 4 \
  --run-name rl_eval
```

Training run artifacts are saved under:
- `/opt/aeon/container/tsp_action_rl/training/train/<run_name>/`
- `/opt/aeon/container/tsp_action_rl/training/eval/<run_name>/`

Each run writes:
- summary json (`train_summary.json` or `eval_summary.json`),
- checkpoint directory (`checkpoints/`),
- optional rollout traces (`rollout_traces/`) containing prompt/raw output/reasoning/parse/validity/reward diagnostics.
