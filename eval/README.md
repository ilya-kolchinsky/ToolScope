# Evaluation

ToolScope includes a [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) harness so you can measure whether **retrieving a short tool list** helps an LLM pick the right function, compared with stuffing the whole catalog into the prompt.

Nothing in this folder talks to real APIs. Tools are static JSON definitions. Predictions are graded by name and, when ground-truth arguments exist, by BFCL-style AST matching against `possible_answer`.

There are two protocols in the same runner (`eval/run_eval.py`):

| | Default (`eval/config.yaml`) | Paper (`eval/paper/`) |
|---|---|---|
| Catalog | A sampled **distractor pool** (default 100 tools) rebuilt per instance | One **shared catalog C**: every unique function in BFCL V4 Non-Live Multiple |
| Conditions | Baseline (full pool) vs retrieval baselines vs ToolScope | Baseline (all of C) vs BM25@k vs ToolScope@k |
| Typical use | Local Hugging Face models, quick iteration | High-cardinality Tool RAG numbers for a write-up |

The paper protocol is documented in [`eval/paper/README.md`](paper/README.md). **Local GGUF evaluation** on DGX Spark is in [`eval/local/README.md`](local/README.md). This page covers install, the runner, metrics, and the default protocol.

---

## Local GGUF models (DGX Spark)

Serve five GGUF models via llama.cpp on DGX Spark and run the shared-catalog paper protocol:

```bash
eval/local/scripts/run_local_matrix.sh
```

Models: Llama 3.2 3B + Qwen2.5 7B (SLM), Llama 3.1 8B + Qwen3 32B (mid), Llama 3.3 70B (ceiling). Frozen v1.0 results: [`eval/paper/artifacts/`](paper/artifacts/). See [`eval/local/README.md`](local/README.md) for prerequisites, troubleshooting, and single-model runs.

---

## Setup

From the **repository root**:

```bash
pip install -e ".[st]"
pip install -r eval/requirements.txt
```

BFCL data is downloaded on first run and cached under `eval/.bfcl_cache/`.

API keys (if you use hosted models) live in a gitignored repo-root `.env`. Copy [`.env.example`](../.env.example):

- **OpenAI-compatible** `/v1` hosts — `OPENAI_BASE_URL` and `OPENAI_API_KEY`. Change those two lines to switch servers.
- **Gemini** — `GOOGLE_API_KEY`.

YAML may still set `base_url` or `api_key_env` per model; empty values fall back to the env vars above. `${OPENAI_BASE_URL}` placeholders are expanded.

---

## Quick start

All commands run from the repository root.

```bash
# Default YAML (local Hugging Face models — needs GPU for the listed 7B-class entries)
python eval/run_eval.py

# Full pipeline, no model and no GPU
python eval/run_eval.py --dry-run --samples 20

# Smaller local run
python eval/run_eval.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --samples 50 \
  --pool-size 100 \
  --k 10
```

`--dry-run` uses a dummy model that always picks the first bound tool. Use it to check data loading, indexing, scoring, and reports.

---

## Talking to a model

The runner has three backends. Mix them freely across `model.entries`; each entry can override `backend`, `base_url`, and `api_key_env`.

**Hugging Face (local)** — `backend: hf` (default in `eval/config.yaml`). Loads `transformers` on `cpu` / `cuda` / `mps`.

**OpenAI-compatible HTTP** — `backend: openai`. Any server that implements chat completions with `tools` (vLLM, a cloud gateway, a local proxy, …):

```bash
python eval/run_eval.py \
  --backend openai \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-72B-Instruct
```

Or set `OPENAI_BASE_URL` / `OPENAI_API_KEY` in `.env` and omit `--base-url` / `--api-key`.

**LangGraph** — `backend: langchain`. Used by the paper protocol (`eval/paper/bfcl_multiple.yaml`). `ChatOpenAI` against an OpenAI-compatible `/v1` endpoint (llama.cpp for local GGUF). See [`eval/paper/README.md`](paper/README.md).

```bash
python eval/run_eval.py --config eval/paper/bfcl_multiple.yaml --dry-run --samples 20
```

---

## What the default protocol does

1. Load BFCL entries (query, function defs, ground-truth call, `possible_answer`).
2. Build a **pool** of `pool_size` tools: functions that appear in some ground truth, plus distractors from other entries.
3. Index that pool once (ToolScope embeddings, plus BM25 / TF-IDF / etc.).
4. For each query:
   - **Baseline** — bind the full pool, generate one tool call.
   - **Each retriever** — bind only that retriever’s top-k, generate one tool call.
5. Parse the call, score it, checkpoint, move on.

Default retrievers (override with `retrievers:` in YAML or `--retrievers`):

| Retriever | Role |
|---|---|
| Random | Lower bound |
| BM25 | Sparse lexical baseline |
| TF-IDF | Sparse lexical baseline |
| Oracle* | Cheats with the ground-truth name (upper bound on retrieval) |
| ToolScope | Dense retrieval over the same pool (`k` from config) |

---

## Configuration

`eval/config.yaml` sets defaults. CLI flags override one value at a time.

| Section | Key | Meaning |
|---|---|---|
| `model.defaults` | `backend` | `hf` \| `openai` \| `langchain` |
| `model.defaults` | `base_url` | OpenAI-compatible `/v1` URL; else `OPENAI_BASE_URL`; else localhost |
| `model.defaults` | `api_key_env` | Env var for the Bearer token (default `OPENAI_API_KEY`) |
| `model.defaults` | `max_new_tokens` | Cap on the generated tool call |
| `model.defaults` | `timeout_seconds` | HTTP timeout for OpenAI-compatible / LangGraph backends |
| `model` | `entries` | List of `{name, ...}`. `--model NAME` runs one entry |
| `dataset` | `protocol` | Omit for the distractor-pool protocol; `shared_catalog` for the paper |
| `dataset` | `categories` | BFCL splits (`simple`, `multiple`, …) |
| `dataset` | `samples` | Cap instances (`null` = all) |
| `dataset` | `pool_size` | Distractor-pool size (`null` = entire catalog in paper mode) |
| `dataset` | `seed` | Sampling and shuffling |
| `retrievers` | — | Optional allow-list of retriever names |
| `toolscope` | `k` | Tools each retriever binds |
| `toolscope.embedding` | `model` | Sentence-Transformers (or other) embedder |
| `output` | `results_dir` | Where JSON / CSV / checkpoints go |

Useful flags: `--samples`, `--k`, `--pool-size`, `--category`, `--retrievers`, `--verbose`, `--no-resume` (ignore checkpoints).

---

## Metrics

Every condition (baseline and each retriever) is scored the same way:

| Metric | What it answers |
|---|---|
| **Tool name accuracy** | Did the model call a ground-truth function name? (selection) |
| **AST accuracy** | Did name **and** arguments match `possible_answer`? (optional args may be omitted) |
| **Exact match** | Stricter name + required-arg equality used by the older pool protocol |
| **Recall@k / NDCG@k** | Did retrieval surface the ground-truth tool, and how high? |
| **Heuristic tool tokens** | `⌊len(json.dumps(openai_tools)) / 4⌋` — catalogue-size proxy used for compression |
| **Usage prompt tokens** | Model-reported `usage.prompt_tokens` / LangChain `input_tokens` after templating (per model) |
| **Compression** | `1 − heuristic_tokens(top-k) / heuristic_tokens(full catalog)` |
| **Latency** | One-turn generate; tools are never executed |

Errors are labelled `retrieval_miss`, `wrong_tool`, `bad_args`, `no_call`, `parse_fail`, or `api_fail`. A failed API call on **one** condition (baseline, BM25, or ToolScope) is fail-closed for that condition only; the others still run.

### Oversized prompts, truncation, and failures

- The harness **always attempts** to bind the tools selected for the condition (full catalogue for baseline; top-k for retrievers). It does **not** truncate the tool list client-side to fit a context window.
- **Heuristic tokens** measure the serialized tool JSON offered to the model and are identical across models for the same bound set (~60,051 for C=443).
- **Usage prompt tokens** are whatever the OpenAI-compatible server reports after chat+tools templating. They are model- and `n_ctx`-specific and are written to checkpoints (`baseline_prompt_tokens` / `prompt_tokens`) and to result JSON (`baseline.prompt_tokens`, `retrievers.*.prompt_tokens`).
- If the HTTP / LangGraph invoke raises (timeout, connection, server reject), that condition is scored **`api_fail`** (fail-closed). Other conditions on the same query still run.
- If the request succeeds but no valid tool call can be parsed, the condition is **`parse_fail`**. On local GGUF baselines with `n_ctx` smaller than the catalogue heuristic size, high `parse_fail` rates should be read together with the usage prompt-token table (effective exposure may be far below the heuristic catalogue size).
- Large-model baselines that need the full catalogue (Qwen3 32B, Llama 3.3 70B) use extended llama-server slot contexts in [`eval/local/models.yaml`](local/models.yaml) so actual usage can fit.

Deltas are retriever − baseline, in percentage points.

---

## Outputs

Under `eval/results/` (or `eval/results/paper/` for the paper YAML):

- `bfcl_eval_{model}_{timestamp}.json` — config, aggregates, per-instance traces (includes heuristic `tokens` and model-reported `prompt_tokens` when the server returned usage metadata)
- `checkpoints/` — JSONL resume files (protocol and catalog size are part of the key, so a smoke run cannot resume into a full run); each line stores `baseline_prompt_tokens` and per-retriever `prompt_tokens`
- Paper YAML (`eval/paper/bfcl_multiple.yaml`) also writes `summary.csv`, `table.md`, `harness_results.md`, and `tool_name_collisions.json`. A full (non-dry, unsampled) run copies the first three into [`eval/paper/artifacts/`](paper/artifacts/) as the v1.0 snapshot. `harness_results.md` includes a **model-reported prompt lengths** table when usage tokens are present.

---

## Notes

- **multiple** is the informative BFCL split: each item already has several candidate functions; pooling them makes selection a high-cardinality problem.
- Models that expose native tool calling (`tools=` on Hugging Face chat templates, or the OpenAI `tools` parameter) use it. The paper / LangGraph path **requires** native `bind_tools` and will not fall back to a JSON system prompt.
- Checkpoints resume by default. `--no-resume` starts that model over.
- Local Hugging Face entries and remote OpenAI-compatible entries can sit in the same `model.entries` list.
