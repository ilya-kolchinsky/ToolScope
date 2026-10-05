"""
Results formatting: console table and JSON file output.
"""

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evaluate import AggregateMetrics, InstanceResult


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _delta(d: float) -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.1f}pp"


def _f3(v: float) -> str:
    return f"{v:.3f}"


def print_report(
    metrics: AggregateMetrics,
    model_name: str,
    categories: List[str],
    pool_size: int,
    k: int,
    embedding_model: str,
) -> None:
    retriever_names = list(metrics.retrievers.keys())
    # Column layout: label + Baseline + one column per retriever
    label_w = 26
    col_w = 11
    n_cols = 1 + len(retriever_names)   # Baseline + retrievers
    total_w = label_w + col_w * n_cols

    print()
    print("=" * total_w)
    print("  BFCL Evaluation — Retrieval Baselines vs. ToolScope")
    print("=" * total_w)
    print(f"  Model      : {model_name}")
    print(f"  Categories : {', '.join(categories)}")
    print(f"  Instances  : {metrics.n}  (skipped: {metrics.n_skipped})")
    print(f"  Pool size  : {pool_size} tools  (≈ {metrics.mean_baseline_tokens:,.0f} tokens/query)")
    print(f"  k          : {k}")
    print(f"  Embedder   : {embedding_model}")
    print()

    sep = "-" * total_w

    # ── Header row ──────────────────────────────────────────────────────────
    header = f"  {'Metric':<{label_w - 2}}" + f"{'Baseline':>{col_w}}"
    for rname in retriever_names:
        header += f"{rname:>{col_w}}"
    print(header)
    print(sep)

    def row(label: str, baseline_val: str, retriever_vals: List[str]) -> None:
        line = f"  {label:<{label_w - 2}}{baseline_val:>{col_w}}"
        for v in retriever_vals:
            line += f"{v:>{col_w}}"
        print(line)

    def retriever_only_row(label: str, retriever_vals: List[str]) -> None:
        row(label, "—", retriever_vals)

    # ── Model accuracy ───────────────────────────────────────────────────────
    row(
        "Tool name accuracy",
        _pct(metrics.baseline_name_acc),
        [_pct(metrics.retrievers[n].name_acc) for n in retriever_names],
    )
    row(
        "Exact match (name + args)",
        _pct(metrics.baseline_exact_match),
        [_pct(metrics.retrievers[n].exact_match) for n in retriever_names],
    )
    row(
        "AST accuracy",
        _pct(getattr(metrics, "baseline_ast_acc", metrics.baseline_exact_match)),
        [_pct(getattr(metrics.retrievers[n], "ast_acc", metrics.retrievers[n].exact_match))
         for n in retriever_names],
    )
    row(
        f"Δ name acc vs baseline",
        "—",
        [_delta(metrics.retrievers[n].delta_name_acc) for n in retriever_names],
    )
    row(
        f"Δ exact match vs baseline",
        "—",
        [_delta(metrics.retrievers[n].delta_exact_match) for n in retriever_names],
    )
    row(
        f"Δ AST acc vs baseline",
        "—",
        [_delta(getattr(metrics.retrievers[n], "delta_ast_acc", 0.0))
         for n in retriever_names],
    )

    print(sep)

    # ── Retrieval quality ────────────────────────────────────────────────────
    retriever_only_row(
        f"Recall@{k}",
        [_pct(metrics.retrievers[n].recall) for n in retriever_names],
    )
    retriever_only_row(
        f"DCG@{k}",
        [_f3(metrics.retrievers[n].dcg) for n in retriever_names],
    )
    retriever_only_row(
        f"NDCG@{k}",
        [_f3(metrics.retrievers[n].ndcg) for n in retriever_names],
    )
    retriever_only_row(
        "Context compression",
        [_pct(metrics.retrievers[n].mean_compression_rate) for n in retriever_names],
    )
    retriever_only_row(
        "Avg prompt tokens",
        [f"{metrics.retrievers[n].mean_tokens:,.0f}" for n in retriever_names],
    )

    print(sep)
    print()

    if "Oracle*" in metrics.retrievers:
        print(
            "  * Oracle always includes the ground-truth tool — not a real retriever.\n"
            "    Its accuracy shows the ceiling if retrieval were perfect."
        )
        print()

    low_recall = {
        n: metrics.retrievers[n].recall
        for n in retriever_names
        if metrics.retrievers[n].recall < 0.9 and n != "Oracle*"
    }
    if low_recall:
        worst = min(low_recall, key=low_recall.get)
        missed = round((1 - low_recall[worst]) * 100, 1)
        print(
            f"  Note: {worst} missed the correct tool in {missed}% of cases.\n"
            f"  Consider increasing k or using a stronger embedding model."
        )
        print()


def print_cross_model_summary(
    all_metrics: Dict[str, AggregateMetrics],
    k: int,
) -> None:
    """
    Print a compact model × retriever summary table.

    Shows tool-name accuracy for every (model, retriever) pair so the full
    interaction effect is visible at a glance.
    """
    if not all_metrics:
        return

    model_names    = list(all_metrics.keys())
    retriever_names: List[str] = []
    seen: set[str] = set()
    for m in all_metrics.values():
        for rn in m.retrievers:
            if rn not in seen:
                seen.add(rn)
                retriever_names.append(rn)
    retriever_names.sort()

    # Column widths — widen for k-ablation keys like ToolScope@10
    short_names = [n.split("/")[-1] for n in model_names]
    label_w = max(len(s) for s in short_names) + 2
    col_w = max(10, max((len(rn) for rn in retriever_names), default=0) + 2)
    n_cols = 1 + len(retriever_names)  # Baseline + retrievers
    total_w = label_w + col_w * n_cols

    sep = "─" * total_w

    def _print_table(title: str, value_fn, baseline_fn=None) -> None:
        """baseline_fn(m) → str for the Baseline column; None means show '—'."""
        print(title)
        header = f"  {'Model':<{label_w - 2}}" + f"{'Baseline':>{col_w}}"
        for rn in retriever_names:
            header += f"{rn:>{col_w}}"
        print(header)
        print(sep)
        for mname, short in zip(model_names, short_names):
            m = all_metrics[mname]
            b_str = baseline_fn(m) if baseline_fn else "—"
            line = f"  {short:<{label_w - 2}}{b_str:>{col_w}}"
            for rn in retriever_names:
                rm = m.retrievers.get(rn)
                line += f"{(value_fn(rm) if rm is not None else '—'):>{col_w}}"
            print(line)
        print(sep)
        print()

    print()
    print("═" * total_w)
    print("  Cross-model summary")
    print("═" * total_w)
    print()

    _print_table(
        f"  Tool name accuracy",
        lambda rm: _pct(rm.name_acc),
        baseline_fn=lambda m: _pct(m.baseline_name_acc),
    )
    _print_table(
        f"  Exact match (name + args)",
        lambda rm: _pct(rm.exact_match),
        baseline_fn=lambda m: _pct(m.baseline_exact_match),
    )
    _print_table(
        f"  NDCG@{k}  (retrieval quality — same across models)",
        lambda rm: _f3(rm.ndcg),
        baseline_fn=None,
    )
    _print_table(
        f"  Δ name acc vs baseline",
        lambda rm: _delta(rm.delta_name_acc),
        baseline_fn=None,
    )


def save_results(
    results: List[InstanceResult],
    metrics: AggregateMetrics,
    config: Dict[str, Any],
    output_dir: Path,
    model_name: str = "",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    slug = model_name.split("/")[-1].replace(" ", "_") if model_name else ""
    filename = f"bfcl_eval_{slug}_{ts}.json" if slug else f"bfcl_eval_{ts}.json"
    path = output_dir / filename

    def _pred(pred: Any) -> Optional[Dict]:
        return dataclasses.asdict(pred) if pred is not None else None

    instance_data = [
        {
            "id": r.id,
            "query": r.query,
            "ground_truth_names": r.ground_truth_names,
            "baseline": {
                "name_acc": r.baseline_name_acc,
                "exact_match": r.baseline_exact_match,
                "ast_acc": getattr(r, "baseline_ast_acc", r.baseline_exact_match),
                "tokens": r.baseline_tokens,
                "prompt_tokens": getattr(r, "baseline_prompt_tokens", None),
                "predicted": _pred(r.baseline_pred),
                "error": getattr(r, "baseline_error", None),
                "latency_ms": getattr(r, "baseline_latency_ms", 0.0),
            },
            "retrievers": {
                rname: {
                    "name_acc": rr.name_acc,
                    "exact_match": rr.exact_match,
                    "ast_acc": getattr(rr, "ast_acc", rr.exact_match),
                    "recall": rr.recall,
                    "dcg": rr.dcg,
                    "ndcg": rr.ndcg,
                    "gt_rank": rr.gt_rank,
                    "tokens": rr.tokens,
                    "prompt_tokens": getattr(rr, "prompt_tokens", None),
                    "compression_rate": rr.compression_rate,
                    "tool_names": rr.tool_names,
                    "predicted": _pred(rr.predicted),
                    "error": getattr(rr, "error", None),
                    "latency_ms": getattr(rr, "latency_ms", 0.0),
                }
                for rname, rr in r.retrievers.items()
            },
        }
        for r in results
    ]

    # AggregateMetrics contains nested dataclasses; asdict handles them recursively
    data = {
        "config": config,
        "metrics": dataclasses.asdict(metrics),
        "instances": instance_data,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"  Results saved → {path}")
    return path


def print_verbose_instance(r: InstanceResult, k: int) -> None:
    """Compact per-instance summary for --verbose mode."""
    gt = ", ".join(r.ground_truth_names)
    b_name = r.baseline_pred.name if r.baseline_pred else "<none>"
    b_mark = "✓" if r.baseline_name_acc else "✗"
    retriever_parts = []
    for rname, rr in r.retrievers.items():
        t_name = rr.predicted.name if rr.predicted else "<none>"
        t_mark = "✓" if rr.name_acc else "✗"
        rank_str = f"rank={rr.gt_rank}" if rr.gt_rank is not None else "rank=—"
        retriever_parts.append(
            f"{rname}={t_name} {t_mark} (ndcg={rr.ndcg:.2f} {rank_str})"
        )
    print(
        f"  [{r.id}] {r.query[:55]!r}\n"
        f"    GT={gt}  base={b_name} {b_mark}  |  "
        + "  |  ".join(retriever_parts)
    )


def write_paper_artifacts(
    all_metrics: Dict[str, AggregateMetrics],
    output_dir: Path,
    k: int,
    catalog_size: int,
    protocol: str,
    *,
    all_instances: Optional[Dict[str, List[dict]]] = None,
    collisions: Optional[List[dict]] = None,
    catalog_names: Optional[List[str]] = None,
    embedder: str = "",
    versioned_dir: Optional[Path] = None,
) -> None:
    """Write summary.csv, table.md, and harness_results.md for the paper protocol."""
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "table.md"

    rows = []
    for model_name, m in all_metrics.items():
        short = model_name.split("/")[-1]
        rows.append({
            "model": short,
            "condition": "Baseline",
            "n": m.n,
            "name_acc": m.baseline_name_acc,
            "ast_acc": getattr(m, "baseline_ast_acc", m.baseline_exact_match),
            "recall_at_k": "",
            "ndcg_at_k": "",
            # Heuristic catalogue size (JSON chars // 4); compression uses this.
            "mean_prompt_tokens": m.mean_baseline_tokens,
            # Model-reported usage.prompt_tokens when available.
            "mean_usage_prompt_tokens": getattr(
                m, "mean_baseline_usage_prompt_tokens", None
            ),
            "n_usage_prompt_tokens": getattr(
                m, "n_baseline_usage_prompt_tokens", 0
            ),
            "compression": 0.0,
            "mean_latency_ms": getattr(m, "mean_baseline_latency_ms", 0.0),
        })
        for rname, rm in m.retrievers.items():
            rows.append({
                "model": short,
                "condition": rname,
                "n": m.n,
                "name_acc": rm.name_acc,
                "ast_acc": getattr(rm, "ast_acc", rm.exact_match),
                "recall_at_k": rm.recall,
                "ndcg_at_k": rm.ndcg,
                "mean_prompt_tokens": rm.mean_tokens,
                "mean_usage_prompt_tokens": getattr(
                    rm, "mean_usage_prompt_tokens", None
                ),
                "n_usage_prompt_tokens": getattr(rm, "n_usage_prompt_tokens", 0),
                "compression": rm.mean_compression_rate,
                "mean_latency_ms": getattr(rm, "mean_latency_ms", 0.0),
            })

    fieldnames = [
        "model", "condition", "n", "name_acc", "ast_acc",
        "recall_at_k", "ndcg_at_k", "mean_prompt_tokens",
        "mean_usage_prompt_tokens", "n_usage_prompt_tokens",
        "compression", "mean_latency_ms",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# BFCL V4 Multiple — high-cardinality Tool RAG",
        "",
        f"Protocol: `{protocol}` (BFCL-derived; **not** an official Gorilla leaderboard score).",
        f"Catalog size: {catalog_size} tools. k={k}.",
        "",
        "## Tool name accuracy",
        "",
    ]

    model_names = list(all_metrics.keys())
    if model_names:
        retriever_names: list = []
        for m in all_metrics.values():
            if m.retrievers:
                retriever_names = list(m.retrievers.keys())
                break
        header = "| Model | Baseline |" + "".join(f" {rn} |" for rn in retriever_names)
        sep = "|---|---|" + "".join("---|" for _ in retriever_names)
        lines.append(header)
        lines.append(sep)
        for mname in model_names:
            m = all_metrics[mname]
            short = mname.split("/")[-1]
            cells = [_pct(m.baseline_name_acc)]
            cells += [
                _pct(m.retrievers[rn].name_acc) if rn in m.retrievers else "—"
                for rn in retriever_names
            ]
            lines.append("| " + short + " | " + " | ".join(cells) + " |")
        lines.extend(["", "## AST accuracy", "", header, sep])
        for mname in model_names:
            m = all_metrics[mname]
            short = mname.split("/")[-1]
            cells = [_pct(getattr(m, "baseline_ast_acc", m.baseline_exact_match))]
            cells += [
                _pct(getattr(m.retrievers[rn], "ast_acc", m.retrievers[rn].exact_match))
                if rn in m.retrievers else "—"
                for rn in retriever_names
            ]
            lines.append("| " + short + " | " + " | ".join(cells) + " |")
        lines.extend(["", "## Context compression", "", header, sep])
        for mname in model_names:
            m = all_metrics[mname]
            short = mname.split("/")[-1]
            cells = ["0.0%"]
            cells += [
                _pct(m.retrievers[rn].mean_compression_rate) if rn in m.retrievers else "—"
                for rn in retriever_names
            ]
            lines.append("| " + short + " | " + " | ".join(cells) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Paper table → {md_path}")
    print(f"  Paper CSV   → {csv_path}")

    from .harness_report import freeze_paper_artifacts, write_harness_results

    harness_path = output_dir / "harness_results.md"
    write_harness_results(
        harness_path,
        all_metrics,
        all_instances or {},
        k=k,
        catalog_size=catalog_size,
        protocol=protocol,
        embedder=embedder,
        collisions=collisions,
        catalog_names=catalog_names,
    )
    print(f"  Harness results → {harness_path}")

    if versioned_dir is not None:
        copied = freeze_paper_artifacts(output_dir, Path(versioned_dir))
        for dest in copied:
            print(f"  Versioned     → {dest}")

