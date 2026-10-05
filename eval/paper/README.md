# High-cardinality paper eval (BFCL V4 Multiple) — v1.0

This protocol asks:

> Can **locally-runnable open-weight models** (3B–70B) handle a **443-tool catalog**,
> and does **selecting k≈10 with ToolScope** beat binding the full catalog?

**Headline claim (v1.0 evidence):** For 3B–8B models, full-catalog binding fails (2–40% tool
name accuracy). Retrieving k = 10 tools lifts selection to **~85–92%** with ~98% less tool
JSON in the prompt. Gains shrink as baseline selection improves; 70B already reaches **79%**
on the full catalog.

Scores are **BFCL-derived**. They are not official Gorilla leaderboard numbers: the catalog,
agent, grader, and model serving stack all live in this repo.

Frozen results: [`artifacts/`](artifacts/) (v1.0 snapshot). Full analysis:
[`artifacts/harness_results.md`](artifacts/harness_results.md).

The shared runner is documented in [`eval/README.md`](../README.md). **Local GGUF serving**
on DGX Spark is in [`eval/local/README.md`](../local/README.md).

---

## Design

Every query uses the **same catalog C**: the unique function definitions in the BFCL Multiple
split (first-seen schema wins if a name appears with two signatures). Baseline, BM25, and
ToolScope all draw from C.

| Condition | What the model can call |
|---|---|
| **Baseline** | All of C (~60,051 **heuristic** tool-schema tokens) |
| **BM25@k** | Top-k from C (sparse lexical retrieval) |
| **ToolScope@k** | Top-k from C via `ToolSelector` (LangChain adapter, MiniLM-L6-v2) |

**k** defaults to **10** for the paper; the v1.0 matrix also records k ∈ {5, 10, 20}.
Retrieval metrics (Recall@k, NDCG@k) are identical across models for a given retriever.

**Agent.** One LangGraph turn: retrieve → `bind_tools` → read `AIMessage.tool_calls`. Tools are
**never executed**. Native tool calling is required.

**Grading.**

- **Name accuracy** — was a ground-truth function named? (selection headline)
- **AST accuracy** — do arguments match BFCL `possible_answer`?
- **Compression** — heuristic tool-token reduction vs baseline (`JSON chars ÷ 4`)
- **Usage prompt tokens** — model-reported lengths after templating (see `harness_results.md`)

If the provider rejects a call for one condition, that condition is scored fail-closed
(`api_fail`). Other conditions on the same query still run. The harness does not truncate
the bound tool list client-side; effective full-catalogue exposure still depends on each
model’s configured llama.cpp `n_ctx` (32,768 for 3B/7B/8B; 65,536 for Qwen3 32B; 131,072
for Llama 3.3 70B). See [`eval/README.md`](../README.md) for the full oversized /
`parse_fail` / `api_fail` handling notes.

---

## Models (v1.0 matrix)

Five GGUF models served via llama.cpp on DGX Spark (Q8_0 for 3B/7B; Q4_K_M for 8B–70B):

| Tier | Models | Role |
|---|---|---|
| SLM | `llama-3.2-3b-instruct`, `qwen2.5-7b-instruct` | Largest ToolScope delta |
| Mid | `llama-3.1-8b-instruct`, `qwen3-32b` | Production-scale edge agents |
| Ceiling | `llama-3.3-70b-instruct` | Full-catalog selection baseline |

---

## Setup

From the ToolScope repository root:

```bash
pip install -e ".[st]"
pip install -r eval/requirements.txt
pip install -r eval/paper/requirements.txt
```

Create a gitignored `.env` in the repo root (see [`.env.example`](../../.env.example)):

```
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=local
TOOLSCOPE_MODEL_CACHE=eval/local/models
```

Run the full matrix:

```bash
eval/local/scripts/run_local_matrix.sh
```

---

## Configuration

Canonical protocol config: [`bfcl_multiple.yaml`](bfcl_multiple.yaml)

```yaml
model:
  defaults:
    backend: langchain
    provider: openai
    base_url: http://127.0.0.1:8000/v1
  entries:
    - name: llama-3.2-3b-instruct
    # ... five models total

dataset:
  protocol: shared_catalog
  categories: [multiple]
  samples: null          # all 200 items
  pool_size: null        # entire catalog C

retrievers: [BM25, ToolScope]
toolscope:
  k: 10
  k_values: [5, 10, 20]

output:
  results_dir: eval/results/paper/local
  versioned_dir: eval/paper/artifacts
```

Model weights and llama.cpp settings: [`eval/local/models.yaml`](../local/models.yaml).

---

## Run

```bash
# Full matrix (llama.cpp on DGX Spark)
eval/local/scripts/run_local_matrix.sh

# Pilot: five queries per model
eval/local/scripts/run_local_matrix.sh --samples 5

# Single model (server must already be running)
python eval/run_eval.py --config eval/paper/bfcl_multiple.yaml --model qwen3-32b

# Config wiring only
eval/local/scripts/run_local_matrix.sh --dry-run --samples 20
```

Checkpoints resume automatically. `--no-resume` starts that model from scratch.

---

## Outputs

| Location | Contents |
|---|---|
| `eval/results/paper/local/` (gitignored) | Per-model JSON traces, checkpoints, logs |
| `eval/paper/artifacts/` (git-tracked) | Frozen v1.0: `table.md`, `summary.csv`, `harness_results.md` |

A full run copies the three markdown/CSV artifacts into `eval/paper/artifacts/`. Per-model
JSON traces stay gitignored.

---

## Reading the results

- **Name accuracy** — did filtering help the model *pick* the right tool?
- **AST accuracy** — did it call the tool *correctly*? Retrieval does not fix `bad_args`.
- **Recall@k ≈ 1** with **name acc ≪ 1** — sibling confusion (GT was bound, model picked a near-duplicate).
- **McNemar p** — paired significance for ToolScope@10 vs baseline on the same 200 queries.

Do not mix these numbers with the default distractor-pool protocol (`eval/config.yaml`) or
with the official BFCL generate/eval pipeline.

---

## Catalog quirks

- **Same name, different schema** — `artifacts/tool_name_collisions.json`
- **Dotted vs underscored aliases** — sanitized for OpenAI tool names; grader uses BFCL names

Follow-up experiments: [`next-experiments.md`](next-experiments.md).
