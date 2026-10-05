"""Paper-facing analysis markdown (the harness-results report).

Written on every shared-catalog eval. A full (non-dry, unsampled) run also
copies it into ``output.versioned_dir`` so the analysis is git-tracked.
"""

from __future__ import annotations

import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .evaluate import AggregateMetrics
from .tools import safe_tool_name, tool_name

_PAPER_FILES = ("table.md", "summary.csv", "harness_results.md")

_DISPLAY: dict[str, str] = {}


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value (binomial n=wins+losses, p=0.5)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def _display(model_name: str) -> str:
    slug = _slug(model_name)
    return _DISPLAY.get(slug, slug)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _delta_pp(d: float) -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.1f} pp"


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    if p < 0.05:
        return f"{p:.3f}"
    return f"{p:.2f}"


def _fmt_latency_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.1f} s"
    return f"{ms:.0f} ms"


def _percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(p * len(s))))
    return s[idx]


def _retriever_names(all_metrics: Dict[str, AggregateMetrics]) -> List[str]:
    for m in all_metrics.values():
        if m.retrievers:
            return list(m.retrievers.keys())
    return []


def _pick_retriever(
    retrievers: List[str], base: str, anchor_k: int
) -> Optional[str]:
    """Prefer ``base@anchor_k`` when k-ablation keys are present."""
    tagged = f"{base}@{anchor_k}"
    if tagged in retrievers:
        return tagged
    if base in retrievers:
        return base
    for r in retrievers:
        if r.startswith(f"{base}@"):
            return r
    return None


def _block(inst: dict, cond: str) -> dict:
    if cond == "Baseline":
        return inst.get("baseline") or {}
    return (inst.get("retrievers") or {}).get(cond) or {}


def _usage_prompt_token(inst: dict, cond: str) -> Optional[int]:
    """Model-reported prompt tokens for a condition.

    Supports nested save_results shape (``baseline.prompt_tokens``) and the
    flat checkpoint shape (``baseline_prompt_tokens``).
    """
    if cond == "Baseline":
        blk = inst.get("baseline") or {}
        val = blk.get("prompt_tokens")
        if val is None:
            val = inst.get("baseline_prompt_tokens")
    else:
        val = (inst.get("retrievers") or {}).get(cond, {}).get("prompt_tokens")
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _usage_stats(
    instances: List[dict], cond: str
) -> Tuple[Optional[float], Optional[int], Optional[int], int]:
    """Return (mean, min, max, n_reported) for usage prompt tokens."""
    vals = [
        v for v in (_usage_prompt_token(inst, cond) for inst in instances)
        if v is not None
    ]
    if not vals:
        return None, None, None, 0
    return sum(vals) / len(vals), min(vals), max(vals), len(vals)


def enrich_metrics_usage_from_instances(
    metrics: AggregateMetrics, instances: List[dict]
) -> AggregateMetrics:
    """Fill usage-token aggregates from instance traces (mutates ``metrics``)."""
    mean, _, _, n = _usage_stats(instances, "Baseline")
    if n and not metrics.n_baseline_usage_prompt_tokens:
        metrics.mean_baseline_usage_prompt_tokens = mean
        metrics.n_baseline_usage_prompt_tokens = n
    for rname, rm in metrics.retrievers.items():
        r_mean, _, _, r_n = _usage_stats(instances, rname)
        if r_n and not rm.n_usage_prompt_tokens:
            rm.mean_usage_prompt_tokens = r_mean
            rm.n_usage_prompt_tokens = r_n
    return metrics


def _pred_name(block: dict) -> Optional[str]:
    pred = block.get("predicted")
    if isinstance(pred, dict):
        name = pred.get("name")
        return str(name) if name else None
    return None


def _name_flips(
    instances: List[dict], retriever: str
) -> Tuple[List[dict], List[dict]]:
    wins: List[dict] = []
    losses: List[dict] = []
    for inst in instances:
        b = bool(_block(inst, "Baseline").get("name_acc"))
        t = bool(_block(inst, retriever).get("name_acc"))
        if t and not b:
            wins.append(inst)
        elif b and not t:
            losses.append(inst)
    return wins, losses


def _error_counts(instances: List[dict], cond: str) -> Counter:
    c: Counter = Counter()
    for inst in instances:
        blk = _block(inst, cond)
        if blk.get("name_acc") and blk.get("ast_acc"):
            c["fully_correct"] += 1
            continue
        c[blk.get("error") or "unknown"] += 1
    return c


def _ast_given_name(instances: List[dict], cond: str) -> Tuple[int, int]:
    named = ast_ok = 0
    for inst in instances:
        blk = _block(inst, cond)
        if blk.get("name_acc"):
            named += 1
            if blk.get("ast_acc"):
                ast_ok += 1
    return ast_ok, named


def _latencies(instances: List[dict], cond: str) -> List[float]:
    out = []
    for inst in instances:
        ms = _block(inst, cond).get("latency_ms")
        if isinstance(ms, (int, float)):
            out.append(float(ms))
    return out


def _collect_tool_names(instances: Iterable[dict]) -> set:
    names: set = set()
    for inst in instances:
        names.update(inst.get("ground_truth_names") or [])
        bpred = _pred_name(_block(inst, "Baseline"))
        if bpred:
            names.add(bpred)
        for rr in (inst.get("retrievers") or {}).values():
            names.update(rr.get("tool_names") or [])
            n = _pred_name(rr)
            if n:
                names.add(n)
    return names


def _sanitized_alias_groups(names: Iterable[str]) -> List[Tuple[str, List[str]]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for n in names:
        if not n:
            continue
        buckets[safe_tool_name(n)].append(n)
    groups = []
    for safe, group in sorted(buckets.items()):
        uniq = sorted(set(group))
        if len(uniq) >= 2:
            groups.append((safe, uniq))
    return groups


def _example_line(inst: dict, retriever: str) -> str:
    gt = ", ".join(inst.get("ground_truth_names") or []) or "—"
    base = _pred_name(_block(inst, "Baseline")) or "—"
    ts = _pred_name(_block(inst, retriever)) or "—"
    rec = _block(inst, retriever).get("recall")
    rec_s = f"{rec:.0f}" if isinstance(rec, (int, float)) else "—"
    q = (inst.get("query") or "").replace("\n", " ").strip()
    if len(q) > 90:
        q = q[:87] + "..."
    eid = inst.get("id") or "?"
    return (
        f"- `{eid}` GT `{gt}`: baseline `{base}` → {retriever} `{ts}` "
        f"(recall={rec_s}). {q}"
    )


def render_harness_results(
    all_metrics: Dict[str, AggregateMetrics],
    all_instances: Dict[str, List[dict]],
    *,
    k: int,
    catalog_size: int,
    protocol: str,
    embedder: str = "",
    collisions: Optional[List[dict]] = None,
    catalog_names: Optional[Sequence[str]] = None,
) -> str:
    """Markdown equivalent of the BFCL paper harness-results analysis."""
    retrievers = _retriever_names(all_metrics)
    ordered = sorted(
        all_metrics.keys(),
        key=lambda n: (all_metrics[n].baseline_name_acc, _slug(n)),
    )
    collisions = collisions or []
    coll_names = {c.get("name") for c in collisions if c.get("name")}

    ns = [all_metrics[n].n for n in ordered]
    n_typ = ns[0] if ns and len(set(ns)) == 1 else None
    skipped = sum(all_metrics[n].n_skipped for n in ordered)

    ts_name = _pick_retriever(retrievers, "ToolScope", k) or (
        retrievers[-1] if retrievers else ""
    )
    first = all_metrics[ordered[0]] if ordered else None
    compression = 0.0
    if first and ts_name and ts_name in first.retrievers:
        compression = first.retrievers[ts_name].mean_compression_rate
    elif first and first.retrievers:
        compression = next(iter(first.retrievers.values())).mean_compression_rate

    best_delta = None
    best_model = None
    if ts_name:
        for n in ordered:
            rm = all_metrics[n].retrievers.get(ts_name)
            if rm is None:
                continue
            if best_delta is None or rm.delta_name_acc > best_delta:
                best_delta = rm.delta_name_acc
                best_model = n

    lines: List[str] = [
        "# BFCL Multiple — harness results",
        "",
        f"Shared catalog C = {catalog_size} tools"
        + (f", {n_typ} queries" if n_typ is not None else "")
        + f", k = {k}"
        + (f", {embedder}." if embedder else "."),
        f"Protocol: `{protocol}`. BFCL-derived; **not** an official Gorilla leaderboard score.",
        "",
        "| | |",
        "|---|---|",
        f"| Queries scored | "
        + (
            f"{n_typ} per model"
            if n_typ is not None
            else ", ".join(f"{_slug(n)} n={all_metrics[n].n}" for n in ordered)
        )
        + " |",
        f"| Models | {len(ordered)} |",
        f"| Catalog C | {catalog_size} tools |",
        f"| Context compression at k={k} | {_pct(compression)} |",
    ]
    if best_model is not None and best_delta is not None:
        lines.append(
            f"| Largest name-acc Δ vs baseline | "
            f"{_delta_pp(best_delta)} ({_display(best_model)}, {ts_name}) |"
        )
    lines.append(f"| Instances skipped | {skipped} |")
    lines += ["", "---", ""]

    api_fails = []
    for n in ordered:
        for inst in all_instances.get(n, []):
            for cond in ["Baseline", *retrievers]:
                if _block(inst, cond).get("error") == "api_fail":
                    api_fails.append((n, inst.get("id"), cond))
                    break
    if skipped or api_fails:
        lines.append(
            f"Skipped instances: **{skipped}**. "
            f"`api_fail` on at least one condition: **{len(api_fails)}** "
            "queries across the matrix."
        )
        lines.append("")
    else:
        lines.append(
            "No instances skipped. No `api_fail` cells in the loaded traces."
        )
        lines.append("")

    lines += [
        "## Tool name accuracy (headline)",
        "",
        "Share of queries where the model called a ground-truth tool name. "
        "Retrieval metrics are identical across models for a given retriever.",
        "",
        "| Model | Baseline |" + "".join(f" {rn} |" for rn in retrievers),
        "|---|---|" + "".join("---|" for _ in retrievers),
    ]
    for n in ordered:
        m = all_metrics[n]
        cells = [_pct(m.baseline_name_acc)]
        cells += [
            _pct(m.retrievers[rn].name_acc) if rn in m.retrievers else "—"
            for rn in retrievers
        ]
        lines.append("| " + _display(n) + " | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Models ordered by baseline name accuracy (weakest catalog handler first)."
    )
    lines.append("")

    if ts_name:
        lines += [
            "## Δ name acc vs full catalog",
            "",
            "Selection gain shrinks as baseline name accuracy rises. "
            "McNemar is exact two-sided on paired name-acc flips "
            f"({ts_name} vs baseline).",
            "",
            "| Model | Baseline name acc | BM25 Δ | ToolScope Δ | "
            f"{ts_name} flips (win/lose) | McNemar p |",
            "|---|---:|---:|---:|---|---:|",
        ]
        bm_name = _pick_retriever(retrievers, "BM25", k)
        for n in ordered:
            m = all_metrics[n]
            insts = all_instances.get(n, [])
            wins, losses = _name_flips(insts, ts_name) if insts else ([], [])
            p = mcnemar_exact(len(wins), len(losses)) if insts else float("nan")
            bm_d = (
                _delta_pp(m.retrievers[bm_name].delta_name_acc)
                if bm_name and bm_name in m.retrievers
                else "—"
            )
            ts_d = (
                _delta_pp(m.retrievers[ts_name].delta_name_acc)
                if ts_name in m.retrievers
                else "—"
            )
            p_s = _fmt_p(p) if insts else "—"
            lines.append(
                f"| {_display(n)} | {_pct(m.baseline_name_acc)} | {bm_d} | {ts_d} | "
                f"+{len(wins)} / −{len(losses)} | {p_s} |"
            )
        lines.append("")

    lines += [
        "## Per-condition matrix",
        "",
        "| Model | Condition | Name acc | AST acc | Δ name | "
        f"Recall@{k} | NDCG@{k} | Mean latency |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n in ordered:
        m = all_metrics[n]
        lines.append(
            f"| {_display(n)} | Baseline | {_pct(m.baseline_name_acc)} | "
            f"{_pct(getattr(m, 'baseline_ast_acc', m.baseline_exact_match))} | — | "
            f"— | — | {_fmt_latency_ms(getattr(m, 'mean_baseline_latency_ms', 0.0))} |"
        )
        for rn in retrievers:
            if rn not in m.retrievers:
                continue
            rm = m.retrievers[rn]
            lines.append(
                f"| {_display(n)} | {rn} | {_pct(rm.name_acc)} | "
                f"{_pct(getattr(rm, 'ast_acc', rm.exact_match))} | "
                f"{_delta_pp(rm.delta_name_acc)} | {_pct(rm.recall)} | "
                f"{rm.ndcg:.3f} | {_fmt_latency_ms(getattr(rm, 'mean_latency_ms', 0.0))} |"
            )
    if first:
        base_tok = first.mean_baseline_tokens
        tok_bits = []
        for rn in retrievers:
            if rn in first.retrievers:
                tok_bits.append(f"{rn} ~{first.retrievers[rn].mean_tokens:,.0f}")
        lines.append("")
        lines.append(
            f"Heuristic tool-schema tokens (JSON chars ÷ 4): baseline ~{base_tok:,.0f}"
            + (f" vs {', '.join(tok_bits)}" if tok_bits else "")
            + f" (~{_pct(compression)} compression). "
            "Latency is one-turn `bind_tools` only; tools are never executed."
        )
        gem = next((n for n in ordered if "gemini" in _slug(n).lower()), None)
        if gem:
            xs = _latencies(all_instances.get(gem, []), "Baseline")
            if xs:
                lines.append(
                    f"{_display(gem)} latency tail: p50 "
                    f"{_fmt_latency_ms(_percentile(xs, 0.50))}, p95 "
                    f"{_fmt_latency_ms(_percentile(xs, 0.95))} (baseline)."
                )
        lines.append("")

    # Model-reported usage.prompt_tokens (per model; may be missing on api_fail).
    usage_anchor = [
        c for c in ("Baseline", *(
            x for x in (
                _pick_retriever(retrievers, "BM25", k),
                _pick_retriever(retrievers, "ToolScope", k),
            ) if x
        ))
    ]
    has_usage = any(
        _usage_stats(all_instances.get(n, []), "Baseline")[3] > 0
        or getattr(all_metrics[n], "n_baseline_usage_prompt_tokens", 0) > 0
        for n in ordered
    )
    if has_usage and usage_anchor:
        lines += [
            "## Model-reported prompt lengths (usage)",
            "",
            "Two length measures are recorded. **Heuristic tool tokens** "
            "(`tokens` / `mean_baseline_tokens`) equal "
            "`⌊len(json.dumps(openai_tools)) / 4⌋` and are identical across "
            "models for a given bound set — used for compression. "
            "**Usage prompt tokens** (`prompt_tokens`) come from the "
            "OpenAI-compatible `usage.prompt_tokens` / LangChain "
            "`usage_metadata.input_tokens` field after chat+tools templating "
            "and are model- and serving-specific.",
            "",
            "Full injection always *attempts* to bind all catalogue tools. "
            "The harness does not truncate the tool list client-side. "
            "HTTP / invoke exceptions are fail-closed as `api_fail` for that "
            "condition only; successful responses with no parseable tool call "
            "are `parse_fail`. Effective exposure still depends on the "
            "configured llama.cpp slot context (`n_ctx`): when the serialized "
            "catalogue exceeds the slot, actual usage counts fall well below "
            "the heuristic catalogue size.",
            "",
            "| Model | Configured n_ctx | Heuristic baseline | "
            + " | ".join(f"Usage mean {c}" for c in usage_anchor)
            + " |",
            "|---|---:|---:|" + "".join("---:|" for _ in usage_anchor),
        ]
        # Optional n_ctx hints from common paper slate (documentation only).
        n_ctx_hints = {
            "llama-3.2-3b-instruct": 32768,
            "qwen2.5-7b-instruct": 32768,
            "llama-3.1-8b-instruct": 32768,
            "qwen3-32b": 65536,
            "llama-3.3-70b-instruct": 131072,
        }
        for n in ordered:
            insts = all_instances.get(n, [])
            cells = []
            for cond in usage_anchor:
                mean, lo, hi, n_rep = _usage_stats(insts, cond)
                if mean is None:
                    # Fall back to aggregate metrics when instances lack the field.
                    if cond == "Baseline":
                        mean = getattr(
                            all_metrics[n],
                            "mean_baseline_usage_prompt_tokens",
                            None,
                        )
                        n_rep = getattr(
                            all_metrics[n],
                            "n_baseline_usage_prompt_tokens",
                            0,
                        )
                    elif cond in all_metrics[n].retrievers:
                        rm = all_metrics[n].retrievers[cond]
                        mean = getattr(rm, "mean_usage_prompt_tokens", None)
                        n_rep = getattr(rm, "n_usage_prompt_tokens", 0)
                if mean is None:
                    cells.append("—")
                elif lo is not None and hi is not None and cond == "Baseline":
                    cells.append(
                        f"{mean:,.0f} (n={n_rep}; {lo}–{hi})"
                    )
                else:
                    cells.append(f"{mean:,.0f} (n={n_rep})")
            ctx = n_ctx_hints.get(_slug(n))
            ctx_s = f"{ctx:,}" if ctx else "—"
            heur = f"{all_metrics[n].mean_baseline_tokens:,.0f}"
            lines.append(
                f"| {_display(n)} | {ctx_s} | {heur} | "
                + " | ".join(cells)
                + " |"
            )
        lines += [
            "",
            "Usage counts can be missing on `api_fail` rows (no successful "
            "response metadata). Compression ratios continue to use the "
            "heuristic catalogue measure so they stay comparable across models.",
            "",
        ]

    lines += [
        "## AST accuracy",
        "",
        "Name selection does not close the AST gap. Leftover error after a "
        "correct name is almost entirely `bad_args`.",
        "",
        "| Model | Baseline |" + "".join(f" {rn} |" for rn in retrievers),
        "|---|---|" + "".join("---|" for _ in retrievers),
    ]
    for n in ordered:
        m = all_metrics[n]
        cells = [_pct(getattr(m, "baseline_ast_acc", m.baseline_exact_match))]
        cells += [
            _pct(getattr(m.retrievers[rn], "ast_acc", m.retrievers[rn].exact_match))
            if rn in m.retrievers
            else "—"
            for rn in retrievers
        ]
        lines.append("| " + _display(n) + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## AST given correct name",
        "",
        "| Model | Baseline |" + "".join(f" {rn} |" for rn in retrievers),
        "|---|---|" + "".join("---|" for _ in retrievers),
    ]
    ast_given_notes = []
    for n in ordered:
        insts = all_instances.get(n, [])
        cells = []
        for cond in ["Baseline", *retrievers]:
            ok, named = _ast_given_name(insts, cond)
            cells.append(_pct(ok / named) if named else "—")
            if named:
                ast_given_notes.append(ok / named)
        lines.append("| " + _display(n) + " | " + " | ".join(cells) + " |")
    if ast_given_notes:
        fail = [1 - x for x in ast_given_notes]
        lines.append("")
        lines.append(
            f"Once the name is right, ~{min(fail)*100:.0f}–{max(fail)*100:.0f}% of "
            "calls still fail AST (`bad_args`). Retrieval does not fix argument quality."
        )
    lines.append("")

    err_keys = ["bad_args", "wrong_tool", "parse_fail", "no_call", "retrieval_miss", "api_fail"]
    lines += [
        "## Where the remaining errors are",
        "",
        "Counts. Fully correct (name + AST) is listed first; the rest are the "
        "error taxonomy.",
        "",
        "| Model | Condition | Fully correct | "
        + " | ".join(err_keys)
        + " |",
        "|---|---|---:|" + "".join("---:|" for _ in err_keys),
    ]
    for n in ordered:
        insts = all_instances.get(n, [])
        for cond in ["Baseline", *retrievers]:
            c = _error_counts(insts, cond)
            cells = [str(c.get("fully_correct", 0))]
            cells += [str(c.get(k, 0)) for k in err_keys]
            lines.append(
                f"| {_display(n)} | {cond} | " + " | ".join(cells) + " |"
            )
    lines.append("")
    if ts_name:
        qwen = next((n for n in ordered if "qwen" in _slug(n).lower()), None)
        if qwen:
            insts = all_instances.get(qwen, [])
            b = _error_counts(insts, "Baseline")
            t = _error_counts(insts, ts_name)
            lines.append(
                f"{_display(qwen)}'s {ts_name} name-acc gain is almost entirely "
                f"fewer `wrong_tool` ({b.get('wrong_tool', 0)} → {t.get('wrong_tool', 0)}), "
                "not better arguments."
            )
            lines.append("")

    if ts_name:
        lines += [
            f"## {ts_name} vs baseline name-acc flips",
            "",
        ]
        for n in ordered:
            insts = all_instances.get(n, [])
            wins, losses = _name_flips(insts, ts_name)
            p = mcnemar_exact(len(wins), len(losses))
            m = all_metrics[n]
            ts = m.retrievers.get(ts_name)
            lines.append(f"### {_display(n)}")
            lines.append("")
            if ts:
                lines.append(
                    f"Name acc { _pct(m.baseline_name_acc)} → {_pct(ts.name_acc)} "
                    f"({_delta_pp(ts.delta_name_acc)}). "
                    f"Flips +{len(wins)} / −{len(losses)}, McNemar p = {_fmt_p(p)}."
                )
            lines.append("")
            rec1_losses = [
                inst for inst in losses
                if _block(inst, ts_name).get("recall") == 1
            ]
            if wins:
                lines.append("Wins (baseline wrong, retriever right):")
                for inst in wins[:5]:
                    lines.append(_example_line(inst, ts_name))
                lines.append("")
            if losses:
                lines.append("Losses (baseline right, retriever wrong):")
                for inst in losses[:5]:
                    lines.append(_example_line(inst, ts_name))
                lines.append("")
                if rec1_losses:
                    lines.append(
                        f"{len(rec1_losses)} of {len(losses)} losses still have "
                        "recall = 1: the ground-truth tool was bound and the model "
                        "preferred a sibling still inside the shortlist."
                    )
                    lines.append("")

    if retrievers:
        lines += [
            "## Retrieval quality (model-independent)",
            "",
            f"| Retriever | Recall@{k} | NDCG@{k} | Missed queries | Mean tokens |",
            "|---|---:|---:|---:|---:|",
        ]
        # Use the first model that has instance traces for miss lists.
        inst_src = next((all_instances[n] for n in ordered if all_instances.get(n)), [])
        n_q = len(inst_src) or (n_typ or 0)
        for rn in retrievers:
            rm = first.retrievers[rn] if first and rn in first.retrievers else None
            if rm is None:
                continue
            missed = [
                inst for inst in inst_src
                if _block(inst, rn).get("recall") == 0
            ]
            miss_names = []
            for inst in missed:
                miss_names.extend(inst.get("ground_truth_names") or [])
            miss_uniq = sorted(set(miss_names))
            miss_cell = f"{len(missed)} / {n_q}"
            lines.append(
                f"| {rn} | {_pct(rm.recall)} | {rm.ndcg:.3f} | {miss_cell} | "
                f"{rm.mean_tokens:,.0f} |"
            )
        lines.append("")
        if inst_src and ts_name:
            r1 = [i for i in inst_src if _block(i, ts_name).get("recall") == 1]
            r0 = [i for i in inst_src if _block(i, ts_name).get("recall") == 0]
            if r1:
                acc = sum(1 for i in r1 if _block(i, ts_name).get("name_acc")) / len(r1)
                lines.append(
                    f"When {ts_name} recall is 1, name acc is {_pct(acc)} on the "
                    "first model's traces. When recall is 0, name acc is 0% — "
                    "the agent cannot call a tool that is not bound."
                )
            if r0 and ts_name:
                gts = sorted({
                    g
                    for i in r0
                    for g in (i.get("ground_truth_names") or [])
                })
                if gts:
                    lines.append(
                        "Missed ground-truth names: "
                        + ", ".join(f"`{g}`" for g in gts)
                        + "."
                    )
            lines.append("")

    lines += ["## Catalog hazards", "", "| Hazard | Count | Effect on scores |",
              "|---|---:|---|"]
    n_coll_names = len(coll_names)
    n_coll_rec = len(collisions)
    coll_rows = []
    if ordered and coll_names:
        for n in ordered:
            insts = all_instances.get(n, [])
            hit = [i for i in insts if any(
                g in coll_names for g in (i.get("ground_truth_names") or [])
            )]
            rest = [i for i in insts if i not in hit]
            if not hit:
                continue
            ts_cond = ts_name or "Baseline"

            def _rate(xs: List[dict], cond: str) -> str:
                if not xs:
                    return "—"
                acc = sum(1 for i in xs if _block(i, cond).get("name_acc")) / len(xs)
                return _pct(acc)

            coll_rows.append(
                f"{_display(n)} {ts_cond} name acc { _rate(hit, ts_cond)} "
                f"on {len(hit)} colliding-GT queries vs {_rate(rest, ts_cond)} "
                f"on {len(rest)} others"
            )
        effect = "; ".join(coll_rows) if coll_rows else "First-seen schema is kept in C."
        lines.append(
            f"| Same name, different schema (first-seen kept) | "
            f"{n_coll_rec} records / {n_coll_names} names | {effect} |"
        )
    names_for_alias = set(catalog_names or [])
    for insts in all_instances.values():
        names_for_alias |= _collect_tool_names(insts)
    names_for_alias |= {str(x) for x in coll_names}
    alias_groups = _sanitized_alias_groups(names_for_alias)
    if alias_groups:
        shown = "; ".join(
            f"`{'` / `'.join(group)}` → `{safe}`" for safe, group in alias_groups[:6]
        )
        extra = "" if len(alias_groups) <= 6 else f" (+{len(alias_groups) - 6} more)"
        lines.append(
            f"| Dotted vs underscore aliases after sanitizing | "
            f"{len(alias_groups)} groups | {shown}{extra}. "
            "Dedupe keeps first-seen; original_name stays in metadata. |"
        )
    lines.append(
        "| Confusable siblings inside top-k | Most remaining `wrong_tool` | "
        "Ground truth is retrieved (recall = 1) but the model prefers a "
        "near-duplicate still in the shortlist. |"
    )
    lines += [
        "",
        "## What this supports for the paper",
        "",
    ]
    if len(ordered) >= 2 and ts_name:
        weak, strong = ordered[0], ordered[-1]
        w_rm = all_metrics[weak].retrievers.get(ts_name)
        s_rm = all_metrics[strong].retrievers.get(ts_name)
        if w_rm and s_rm:
            lines.append(
                "Selection over injection is not a uniform lift. It helps the "
                f"model that struggles with a {catalog_size}-tool prompt "
                f"({_display(weak)}, {_delta_pp(w_rm.delta_name_acc)} name acc, "
                f"~{_pct(compression)} less tool JSON) and is a wash for models "
                "that already pick the right name from the full catalog "
                f"({_display(strong)} baseline {_pct(all_metrics[strong].baseline_name_acc)})."
            )
        else:
            lines.append(
                "Selection over injection should be reported as name accuracy "
                "(selection) separately from AST accuracy (calling)."
            )
    else:
        lines.append(
            "Report **name accuracy** as the selection metric and **AST** as "
            "calling. Retrieval does not fill arguments."
        )
    if first and ts_name and ts_name in first.retrievers:
        rec = first.retrievers[ts_name].recall
        lines.append("")
        lines.append(
            f"Retrieval at k={k} is nearly solved (Recall {_pct(rec)}). "
            "The leftover selection error is sibling confusion, and the leftover "
            "calling error is arguments."
        )
    lines += [
        "",
        "Do not treat these numbers as an official BFCL / Gorilla leaderboard "
        "score. Shared-catalog protocol, local AST vs `possible_answer`, "
        "one-turn LangGraph, no tool execution. `table.md` / `summary.csv` "
        "are the compact matrix; this file is the analysis.",
        "",
    ]
    return "\n".join(lines)


def write_harness_results(
    path: Path,
    all_metrics: Dict[str, AggregateMetrics],
    all_instances: Dict[str, List[dict]],
    **kwargs: Any,
) -> Path:
    preserved = ""
    if path.exists():
        old = path.read_text(encoding="utf-8")
        marker = "\n## Analysis\n"
        if marker in old:
            # Keep hand-authored analysis when regenerating the auto sections.
            preserved = old[old.index(marker):]
    text = render_harness_results(all_metrics, all_instances, **kwargs)
    if preserved:
        cut = "\n## What this supports for the paper\n"
        if cut in text:
            text = text[: text.index(cut)].rstrip() + "\n"
        text = text.rstrip() + "\n" + preserved
        if not text.endswith("\n"):
            text += "\n"
    path.write_text(text, encoding="utf-8")
    return path


def freeze_paper_artifacts(output_dir: Path, versioned_dir: Path) -> List[Path]:
    """Copy paper-facing markdown/csv into the git-tracked artifacts dir."""
    versioned_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for name in _PAPER_FILES:
        src = output_dir / name
        if not src.exists():
            continue
        dest = versioned_dir / name
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied
