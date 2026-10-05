# Frozen local GGUF k=10 matrix



BFCL V4 Non-Live Multiple, shared catalog **C = 443**, **k = 10** (anchor; k-ablation
{k ∈ 5, 10, 20} in [`harness_results.md`](harness_results.md)), MiniLM-L6-v2.
**n = 200** per completed model. Scores are BFCL-derived, not official Gorilla numbers.

Runtime writes go to gitignored `eval/results/paper/local/`. This directory is the
git-tracked **paper v1.0** snapshot.

| Model | Source file (gitignored) | SHA-256 |
|---|---|---|
| llama-3.2-3b-instruct | `bfcl_eval_llama-3.2-3b-instruct_1788642468.json` | `8ca48c218af824ee6369a1c55471621e7b8492b65dff3121a5374fd20ecce2d6` |
| qwen2.5-7b-instruct | `bfcl_eval_qwen2.5-7b-instruct_1788656165.json` | `1ce620d6bacc9bd0d77ad2d97fd1403c16bd5339bf22025bbefd51b726bdb45d` |
| llama-3.1-8b-instruct | `bfcl_eval_llama-3.1-8b-instruct_1788679598.json` | `24cd33f6ae875b71df8e22fbb4d3838660b447a99558a12cc5340e2bd411fff1` |
| qwen3-32b | `bfcl_eval_qwen3-32b_1789298779.json` | `98046c2b58f775918bfd3277399f6ffe7ec992889fda16f9e8e340c4b59db17a` |
| llama-3.3-70b-instruct | `bfcl_eval_llama-3.3-70b-instruct_1789309778.json` | `d4c8546f8ab5e073e189fd9a57b78e35c0cfeb3c155739b1686f68d63acf476d` |

## Tool name accuracy

| Model | Baseline | BM25 | ToolScope | Δ ToolScope vs baseline |
|---|---|---|---|---|
| llama-3.2-3b-instruct | 2.5% | 85.5% | **84.5%** | **+82.0 pp** (McNemar exact p = < 0.001; +165 / −1) |
| llama-3.1-8b-instruct | 6.0% | 91.5% | **92.0%** | **+86.0 pp** (McNemar exact p = < 0.001; +173 / −1) |
| qwen2.5-7b-instruct | 40.0% | 86.0% | **87.0%** | **+47.0 pp** (McNemar exact p = < 0.001; +104 / −10) |
| qwen3-32b | 72.5% | 89.0% | **88.0%** | **+15.5 pp** (McNemar exact p = < 0.001; +41 / −10) |
| llama-3.3-70b-instruct | 79.0% | 90.5% | **91.0%** | **+12.0 pp** (McNemar exact p = < 0.001; +28 / −4) |

BM25 / ToolScope columns are **@k=10** (`BM25@10`, `ToolScope@10` in the full matrix).

## AST accuracy

| Model | Baseline | BM25 | ToolScope |
|---|---|---|---|
| llama-3.2-3b-instruct | 2.0% | 47.0% | 46.5% |
| llama-3.1-8b-instruct | 3.5% | 50.5% | 52.0% |
| qwen2.5-7b-instruct | 23.5% | 53.5% | 53.5% |
| qwen3-32b | 45.0% | 55.5% | 55.0% |
| llama-3.3-70b-instruct | 46.5% | 60.0% | 61.0% |

Retrieval (identical across models): BM25 Recall@10 **97.0%** / NDCG **0.881**; ToolScope Recall@10 **98.5%** / NDCG **0.885**. Compression **97.7%** uses the heuristic catalogue measure (~60,051 → ~1,362 tool-schema token-equivalents at k=10). Model-reported usage prompt tokens differ by model/`n_ctx` (see [harness_results.md](harness_results.md)).

See [harness_results.md](harness_results.md) for the analysis (name/AST, McNemar, error taxonomy, flips, usage prompt lengths). [table.md](table.md) and [summary.csv](summary.csv) are the compact matrix (includes k-ablation columns). Historical API-model results: [`../README.md`](../README.md). Follow-up experiments: [../../next-experiments.md](../../next-experiments.md).
