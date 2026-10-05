# Local GGUF BFCL evaluation (DGX Spark)

Run the BFCL shared-catalog paper protocol against **five locally-served GGUF models** on a DGX Spark (~128 GB unified memory) using **llama.cpp** inside the **ToolScope devcontainer** (`.devcontainer/`).

Two tiers: **SLM** (3B–7B workhorse models) and **local agent** (32B–70B models that can run full agent loops on-device).

## Value proposition

ToolScope filters 50+ candidate tools down to the **3–5 most relevant** before the LLM
sees them. On smaller models this often takes tool selection accuracy from **~30% baseline
to 85%+ with ToolScope** — a crystal-clear signal on the library's impact. The headline
benchmark: **an 8B model with ToolScope matching 70B tool-calling reliability without it.**

## Three-tier comparative slate

| Tier | Models | Role |
|---|---|---|
| **High-Sensitivity SLM** | `qwen2.5-7b-instruct`, `llama-3.2-3b-instruct` | Primary testbed — largest delta |
| **Mid-Sized Production** | `llama-3.1-8b-instruct`, `qwen3-32b` | Edge agent sweet spot (Meta 8B + Qwen 32B) |
| **Control Ceiling** | `llama-3.3-70b-instruct` | Top-tier open weights baseline |

```bash
eval/local/scripts/run_local_matrix.sh --tier high_sensitivity --samples 5  # SLM pilot
eval/local/scripts/run_local_matrix.sh --tier mid_production                # 32B tier
eval/local/scripts/run_local_matrix.sh                                      # full matrix
```

Frozen paper v1.0 results: [`eval/paper/artifacts/`](../paper/artifacts/).

## Devcontainer (recommended)

Reopen this repo in the **ToolScope + llama.cpp** devcontainer. It includes:

- CUDA **llama-server** (aarch64, llama.cpp b6985+, **CUDA 13.0** / GB10 `121a-real`)
- Python eval harness (post-create installs ToolScope + eval deps)
- **Native inference** — `serve_model.sh` starts `llama-server` directly (no nested containers)

From the host, live runs auto-delegate to the devcontainer:

```bash
eval/local/scripts/run_local_matrix.sh --samples 5   # pilot
eval/local/scripts/run_local_matrix.sh             # full matrix
```

Or run commands explicitly inside the devcontainer:

```bash
eval/local/scripts/run_in_devcontainer.sh eval/local/scripts/run_local_matrix.sh --samples 5
```

Build the image manually (first time; compiles llama.cpp and bakes eval Python deps):

```bash
docker build -f .devcontainer/Dockerfile -t toolscope-dev:latest .
# or: eval/local/scripts/run_in_devcontainer.sh --rebuild bash -lc 'echo ok'
```

After changing `.devcontainer/Dockerfile` or `eval/requirements.txt`, rebuild with
`run_in_devcontainer.sh --rebuild …` so ephemeral runs do not re-pip on every invocation.

## Models

Registry: [`models.yaml`](models.yaml). Models run **one at a time** (sequential matrix, smallest-first).

### SLM tier (3B–7B)

| ID | HF repo | Quant | ~Size | Notes |
|---|---|---|---|---|
| `llama-3.2-3b-instruct` | `bartowski/Llama-3.2-3B-Instruct-GGUF` | Q8_0 | ~3.4 GB | Floor model |
| `qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct-GGUF` | Q8_0 | ~8.1 GB | Workhorse SLM; strong JSON, weak with 30–50+ tool distractors |

### Mid + ceiling tier (8B–70B)

| ID | HF repo | Quant | ~Size | Notes |
|---|---|---|---|---|
| `llama-3.1-8b-instruct` | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | Q4_K_M | ~5 GB | Meta mid-tier; headline 8B bridge |
| `qwen3-32b` | `Qwen/Qwen3-32B-GGUF` | Q4_K_M | ~20 GB | Upper mid dense (single-file) |
| `llama-3.3-70b-instruct` | `bartowski/Llama-3.3-70B-Instruct-GGUF` | Q4_K_M | ~43 GB | On-device ceiling |

Full slate disk budget: **~80 GB** (sequential runs; purge-after-eval default).

Avoid **358B MoE** models (e.g. GLM-4.7): Q4_K_M alone is ~216 GB and does not load on 128 GB unified memory.

## Prerequisites

### Devcontainer (recommended)

Reopen the repo in the **ToolScope + llama.cpp** devcontainer (`.devcontainer/`). It bundles:

- Python eval harness (`pip install -e ".[st,dev]"` + eval requirements)
- CUDA **llama-server** (aarch64, pinned llama.cpp tag)
- Native inference mode (`TOOLSCOPE_INFERENCE_MODE=native`) — `serve_model.sh` starts `llama-server` directly; no nested podman/docker required inside the devcontainer

Model weights persist in the `toolscope-model-cache` volume (`/workspace/eval/local/models`).

### Host (without devcontainer)

On the DGX Spark host:

- **podman** or **docker** with NVIDIA GPU support
- **CUDA** drivers for aarch64 (Grace Blackwell)
- **Python 3.10+** with ToolScope eval deps installed
- **~80 GB** free disk for remaining GGUF weights (gitignored under `eval/local/models/`)
- **huggingface-cli** or `hf` for weight download

Set `TOOLSCOPE_INFERENCE_MODE=container` (default on host) to use the standalone `toolscope-llamacpp` image.

```bash
# From repo root
pip install -e ".[st]"
pip install -r eval/requirements.txt
pip install -r eval/paper/requirements.txt
```

Copy [`.env.example`](../../.env.example) to `.env`:

```
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=local
TOOLSCOPE_MODEL_CACHE=eval/local/models
```

## Quick start

### Full automated matrix

Inside the devcontainer (or on host with deps installed):

```bash
eval/local/scripts/run_local_matrix.sh
```

On host without devcontainer, build the inference image first (skipped automatically in devcontainer native mode):

```bash
eval/local/scripts/build_image.sh   # container mode only
eval/local/scripts/run_local_matrix.sh
```

This will:

1. Build the `toolscope-llamacpp:latest` podman image (llama.cpp b4586, CUDA aarch64)
2. Download GGUF weights into `TOOLSCOPE_MODEL_CACHE`
3. For each model: serve → healthcheck → tool-call smoke → BFCL eval → stop

### Pilot (5 queries per model, no weight download if cached)

```bash
eval/local/scripts/run_local_matrix.sh --samples 5
```

### Dry-run (config wiring only, no GPU/server)

```bash
eval/local/scripts/run_local_matrix.sh --dry-run --samples 20
```

### Single model

```bash
eval/local/scripts/download_models.sh --model qwen3-32b
eval/local/scripts/serve_model.sh qwen3-32b
eval/local/scripts/healthcheck.sh
python eval/local/smoke/tool_call_probe.py --model qwen3-32b

python eval/run_eval.py \
  --config eval/paper/bfcl_multiple.yaml \
  --model qwen3-32b

eval/local/scripts/stop_server.sh
```

## Scripts

| Script | Purpose |
|---|---|
| [`scripts/build_image.sh`](scripts/build_image.sh) | `podman build` the llama.cpp image |
| [`scripts/download_models.sh`](scripts/download_models.sh) | HF download + `manifest.json` |
| [`scripts/serve_model.sh`](scripts/serve_model.sh) | Start one model in podman |
| [`scripts/stop_server.sh`](scripts/stop_server.sh) | Tear down the server container |
| [`scripts/healthcheck.sh`](scripts/healthcheck.sh) | Poll `GET /v1/models` |
| [`scripts/run_local_matrix.sh`](scripts/run_local_matrix.sh) | End-to-end orchestration |
| [`scripts/purge_model.sh`](scripts/purge_model.sh) | Remove cached weights after each model (matrix default) |
| [`scripts/prefetch_models.sh`](scripts/prefetch_models.sh) | Sequential background download queue (used during eval) |
| [`smoke/tool_call_probe.py`](smoke/tool_call_probe.py) | Minimal `bind_tools` gate |

## Outputs

Runtime (gitignored): `eval/results/paper/local/`

Frozen snapshot (git-tracked after a full run): `eval/paper/artifacts/`

| File | Contents |
|---|---|
| `summary.csv` | Per (model × condition) metrics (heuristic + usage prompt tokens) |
| `table.md` | Name/AST accuracy + compression |
| `harness_results.md` | McNemar, error taxonomy, flips, model-reported prompt lengths |
| `tool_name_collisions.json` | Catalog hazard report |

Runtime traces (gitignored) under `eval/results/paper/local/`:

| Path | Contents |
|---|---|
| `bfcl_eval_*.json` | Aggregates + per-instance traces (`tokens` heuristic, `prompt_tokens` usage) |
| `checkpoints/*.jsonl` | Resume lines with `baseline_prompt_tokens` / retriever `prompt_tokens` |

`finalize_local_artifacts.py` merges usage tokens from checkpoints into older result JSONs that omitted them, then regenerates artifacts.

## Troubleshooting

**Container build fails on aarch64 CUDA**

- Confirm `podman run --device nvidia.com/gpu=all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi` works.
- Rebuild with `eval/local/scripts/build_image.sh --rebuild`.
- Fallback: build `llama-server` natively on the host and adjust the Containerfile to copy the binary.

**OOM / context pressure on full-catalogue baseline (443 tools)**

- Configured `context_size` in [`models.yaml`](models.yaml): **32768** (3B/7B/8B), **65536** (Qwen3 32B), **131072** (Llama 3.3 70B).
- The harness does **not** shrink the tool list to fit `n_ctx`. Heuristic catalogue size is ~60k tool-token equivalents; model-reported usage for SLM baselines is much lower (see `harness_results.md`).
- Qwen3 / 70B baselines need the extended-context llama-server build (`eval/local/patches/llama_server_extended_ctx.patch`) so slots are not capped to `n_ctx_train`.
- Instant baseline `api_fail` (~200–300 ms) usually means context overflow; use `rerun_baseline_local.sh` after raising `context_size`.
- Lower `n_gpu_layers` in `models.yaml` defaults only if VRAM/unified memory is exhausted.

**Tool-call smoke fails**

- Check `podman logs toolscope-llama` or `eval/local/.podman/llama-server.log` (`task.n_tokens`, `n_ctx_slot`, `truncated=`).
- Qwen3 requires `--jinja` (set in `models.yaml`).
- Bump `LLAMA_CPP_TAG` in [`container/Containerfile`](container/Containerfile) if chat templates are outdated.

**Eval HTTP timeouts**

- Increase `timeout_seconds` per model in [`eval/paper/bfcl_multiple.yaml`](../paper/bfcl_multiple.yaml).

## Protocol

Same as [`eval/paper/README.md`](../paper/README.md): shared catalog C=443, BM25 + ToolScope, LangGraph one-turn `bind_tools`. Config: [`eval/paper/bfcl_multiple.yaml`](../paper/bfcl_multiple.yaml).

**K-ablation** (local runs): `toolscope.k_values: [5, 10, 20]` scores nested prefixes per retriever (`BM25@5`, `ToolScope@10`, …). Retrieve once at k_max=20; anchor k=10 is used for headline deltas in `harness_results.md`. Override via CLI: `--k-values 5 10 20`.

**Failure handling (local GGUF):** condition-level `api_fail` is fail-closed; `parse_fail` means a response arrived but no valid tool call was parsed. Details and the usage-token table: [`eval/README.md`](../README.md) and [`../paper/artifacts/harness_results.md`](../paper/artifacts/harness_results.md).
