"""
Per-instance evaluation and metric aggregation.

InstanceResult stores a baseline result (model sees the full catalog / pool)
plus one RetrieverResult per retriever (model sees only the retriever's top-k).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .ast_score import compute_ast_valid
from .model import ParsedToolCall
from .tools import tool_name, tools_to_openai


# ── Per-instance result structures ─────────────────────────────────────────


@dataclass
class RetrieverResult:
    """All metrics for one retriever on one evaluation instance."""
    name_acc: bool
    exact_match: bool
    ast_acc: bool
    recall: float           # fraction of GT tools present in top-k
    dcg: float              # discounted cumulative gain at k
    ndcg: float             # normalized DCG (0–1); equals DCG for single-GT entries
    gt_rank: Optional[int]  # 1-indexed rank of first GT tool; None if not retrieved
    tool_names: List[str]   # names of tools the retriever returned
    tokens: int             # approx prompt tokens (json_chars // 4)
    compression_rate: float # 1 - tokens / baseline_tokens
    raw: str                # raw model output for debugging
    predicted: Optional[ParsedToolCall]
    error: Optional[str] = None
    latency_ms: float = 0.0
    prompt_tokens: Optional[int] = None


@dataclass
class InstanceResult:
    id: str
    query: str                    # first user message, truncated for display
    ground_truth_names: List[str]

    baseline_name_acc: bool
    baseline_exact_match: bool
    baseline_ast_acc: bool
    baseline_tokens: int
    baseline_raw: str
    baseline_pred: Optional[ParsedToolCall]
    baseline_error: Optional[str] = None
    baseline_latency_ms: float = 0.0
    baseline_prompt_tokens: Optional[int] = None

    retrievers: Dict[str, RetrieverResult] = field(default_factory=dict)


# ── Aggregate metric structures ─────────────────────────────────────────────


@dataclass
class RetrieverMetrics:
    """Aggregate metrics for one retriever across all evaluated instances."""
    name_acc: float
    exact_match: float
    ast_acc: float
    recall: float
    dcg: float
    ndcg: float
    mean_tokens: float
    mean_compression_rate: float
    mean_latency_ms: float
    delta_name_acc: float
    delta_exact_match: float
    delta_ast_acc: float
    error_counts: Dict[str, int]
    # Model-reported usage.prompt_tokens (None when never observed).
    mean_usage_prompt_tokens: Optional[float] = None
    n_usage_prompt_tokens: int = 0


@dataclass
class AggregateMetrics:
    n: int
    n_skipped: int
    baseline_name_acc: float
    baseline_exact_match: float
    baseline_ast_acc: float
    mean_baseline_tokens: float
    mean_baseline_latency_ms: float
    retrievers: Dict[str, RetrieverMetrics]
    # Model-reported usage.prompt_tokens for full-catalog binding.
    mean_baseline_usage_prompt_tokens: Optional[float] = None
    n_baseline_usage_prompt_tokens: int = 0


# ── Metric computation ──────────────────────────────────────────────────────


def _gt_names(ground_truth: List[Dict]) -> List[str]:
    names = []
    for call in ground_truth:
        if isinstance(call, dict):
            names.extend(call.keys())
    return names


def _normalize_value(v: Any) -> Any:
    if isinstance(v, str):
        v = v.strip()
        try:
            return int(v)
        except (ValueError, TypeError):
            pass
        try:
            return float(v)
        except (ValueError, TypeError):
            pass
        lower = v.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
    return v


def _args_match(pred_args: Dict, gt_args: Dict) -> bool:
    for key, expected in gt_args.items():
        if key not in pred_args:
            return False
        if _normalize_value(pred_args[key]) != _normalize_value(expected):
            return False
    return True


def compute_name_acc(pred: Optional[ParsedToolCall], gt_names: List[str]) -> bool:
    return pred is not None and pred.name in gt_names


def compute_exact_match(
    pred: Optional[ParsedToolCall],
    ground_truth: List[Dict],
    gt_names: List[str],
) -> bool:
    if pred is None or pred.name not in gt_names:
        return False
    for call in ground_truth:
        if isinstance(call, dict) and pred.name in call:
            gt_args = call[pred.name]
            if isinstance(gt_args, dict):
                return _args_match(pred.arguments, gt_args)
            return True
    return False


def compute_recall(gt_names: List[str], retrieved_names: List[str]) -> float:
    if not gt_names:
        return 0.0
    ts_set = set(retrieved_names)
    return sum(1 for n in gt_names if n in ts_set) / len(gt_names)


def compute_dcg(
    gt_names: List[str], retrieved_names: List[str]
) -> Tuple[float, float, Optional[int]]:
    """
    Returns (dcg, ndcg, first_rank).

    DCG@k  = sum_i  relevance_i / log2(rank_i + 1)   for each GT tool found
    IDCG@k = ideal DCG (all n_gt GT tools at ranks 1..n_gt)
    NDCG@k = DCG / IDCG  → [0, 1]
    """
    if not gt_names:
        return 0.0, 0.0, None

    gt_set = set(gt_names)
    first_rank: Optional[int] = None
    dcg = 0.0

    for rank_0, name in enumerate(retrieved_names):
        if name in gt_set:
            rank = rank_0 + 1
            dcg += 1.0 / math.log2(rank + 1)
            if first_rank is None:
                first_rank = rank

    n_gt = min(len(gt_names), len(retrieved_names))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, n_gt + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return dcg, ndcg, first_rank


def classify_error(
    *,
    pred: Optional[ParsedToolCall],
    raw: str,
    name_acc: bool,
    ast_acc: bool,
    gt_names: List[str],
    retrieved_names: Optional[List[str]],
) -> Optional[str]:
    if name_acc and ast_acc:
        return None
    if pred is None:
        if raw and raw.strip():
            return "parse_fail"
        return "no_call"
    if retrieved_names is not None and gt_names:
        if not any(n in set(retrieved_names) for n in gt_names):
            return "retrieval_miss"
    if pred.name not in gt_names:
        return "wrong_tool"
    if not ast_acc:
        return "bad_args"
    return None


def _approx_tokens(tools: List[Any]) -> int:
    openai = tools_to_openai(tools) if tools else []
    return max(1, len(json.dumps(openai, ensure_ascii=False)) // 4)


def _normalize_predict(model: Any, messages: List[Dict], tools: List[Any]):
    from .agent import PredictResult

    t0 = time.monotonic()
    try:
        out = model.predict(messages, tools)
    except Exception:
        return PredictResult(
            raw="",
            predicted=None,
            latency_ms=(time.monotonic() - t0) * 1000,
            native_tools=True,
            error="api_fail",
        )
    latency_ms = (time.monotonic() - t0) * 1000
    if isinstance(out, PredictResult):
        return out
    raw = out if isinstance(out, str) else str(out or "")
    return PredictResult(
        raw=raw,
        predicted=model.parse_tool_call(raw),
        latency_ms=latency_ms,
        native_tools=True,
    )


# ── Core evaluation ─────────────────────────────────────────────────────────


def _run_retriever(
    retriever: Any,
    messages: List[Dict],
    ground_truth: List[Dict],
    gt_names: List[str],
    baseline_tokens: int,
    model: Any,
    k: int,
    possible_answer: Any = None,
    functions_bfcl: Optional[List[Dict]] = None,
    ranked_tools: Optional[List[Any]] = None,
) -> RetrieverResult:
    """Run one retriever against one instance and compute all metrics."""
    if hasattr(retriever, "set_ground_truth"):
        retriever.set_ground_truth(gt_names)

    if ranked_tools is not None:
        r_tools = ranked_tools[:k]
    else:
        r_tools = retriever.filter(messages, k=k)
    r_names = [tool_name(t) for t in r_tools]
    r_tokens = _approx_tokens(r_tools) if r_tools else 0
    pred_res = (
        _normalize_predict(model, messages, r_tools)
        if r_tools
        else None
    )
    r_raw = pred_res.raw if pred_res else ""
    r_pred = pred_res.predicted if pred_res else None
    r_dcg, r_ndcg, r_gt_rank = compute_dcg(gt_names, r_names)
    compression = 1.0 - (r_tokens / baseline_tokens) if baseline_tokens > 0 else 0.0
    name_acc = compute_name_acc(r_pred, gt_names)
    exact = compute_exact_match(r_pred, ground_truth, gt_names)
    ast_ok = compute_ast_valid(
        r_pred, possible_answer, func_descriptions=functions_bfcl
    ) if possible_answer is not None else exact
    error = (pred_res.error if pred_res and pred_res.error else classify_error(
        pred=r_pred,
        raw=r_raw,
        name_acc=name_acc,
        ast_acc=ast_ok,
        gt_names=gt_names,
        retrieved_names=r_names,
    ))

    return RetrieverResult(
        name_acc=name_acc,
        exact_match=exact,
        ast_acc=ast_ok,
        recall=compute_recall(gt_names, r_names),
        dcg=r_dcg,
        ndcg=r_ndcg,
        gt_rank=r_gt_rank,
        tool_names=r_names,
        tokens=r_tokens,
        compression_rate=compression,
        raw=r_raw,
        predicted=r_pred,
        error=error,
        latency_ms=pred_res.latency_ms if pred_res else 0.0,
        prompt_tokens=pred_res.prompt_tokens if pred_res else None,
    )


def evaluate_instance(
    entry_id: str,
    messages: List[Dict],
    ground_truth: List[Dict],
    tool_pool: List[Any],
    model: Any,
    retrievers: Dict[str, Any],
    k: int,
    possible_answer: Any = None,
    functions_bfcl: Optional[List[Dict]] = None,
    k_values: Optional[List[int]] = None,
) -> Optional[InstanceResult]:
    """
    Run one evaluation instance.

    Baseline: model sees the full tool_pool (shared catalog C, or per-instance pool).
    Each retriever: model sees only its top-k tools from the same catalog.

    When ``k_values`` is set (e.g. [5, 10, 20]), retrieve once at k_max per
    retriever and score nested prefixes. Result keys become ``BM25@5``, etc.
    """
    gt_names = _gt_names(ground_truth)
    if not gt_names:
        return None

    query = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"), ""
    )[:200]

    baseline_tokens = _approx_tokens(tool_pool)
    base_res = _normalize_predict(model, messages, tool_pool)
    baseline_pred = base_res.predicted
    name_acc = compute_name_acc(baseline_pred, gt_names)
    exact = compute_exact_match(baseline_pred, ground_truth, gt_names)
    ast_ok = compute_ast_valid(
        baseline_pred, possible_answer, func_descriptions=functions_bfcl
    ) if possible_answer is not None else exact

    retriever_results: Dict[str, RetrieverResult] = {}
    ablation_ks = sorted(set(int(x) for x in (k_values or []))) if k_values else None
    k_max = max(ablation_ks) if ablation_ks else k

    for rname, retriever in retrievers.items():
        try:
            if ablation_ks:
                ranked = retriever.filter(messages, k=k_max)
                for kv in ablation_ks:
                    key = f"{rname}@{kv}"
                    retriever_results[key] = _run_retriever(
                        retriever, messages, ground_truth, gt_names,
                        baseline_tokens, model, kv,
                        possible_answer=possible_answer,
                        functions_bfcl=functions_bfcl,
                        ranked_tools=ranked,
                    )
            else:
                retriever_results[rname] = _run_retriever(
                    retriever, messages, ground_truth, gt_names,
                    baseline_tokens, model, k,
                    possible_answer=possible_answer,
                    functions_bfcl=functions_bfcl,
                )
        except Exception:
            targets = [f"{rname}@{kv}" for kv in ablation_ks] if ablation_ks else [rname]
            for key in targets:
                retriever_results[key] = RetrieverResult(
                    name_acc=False,
                    exact_match=False,
                    ast_acc=False,
                    recall=0.0,
                    dcg=0.0,
                    ndcg=0.0,
                    gt_rank=None,
                    tool_names=[],
                    tokens=0,
                    compression_rate=0.0,
                    raw="",
                    predicted=None,
                    error="api_fail",
                )

    baseline_error = (
        base_res.error
        if base_res.error
        else classify_error(
            pred=baseline_pred,
            raw=base_res.raw,
            name_acc=name_acc,
            ast_acc=ast_ok,
            gt_names=gt_names,
            retrieved_names=[tool_name(t) for t in tool_pool],
        )
    )

    return InstanceResult(
        id=entry_id,
        query=query,
        ground_truth_names=gt_names,
        baseline_name_acc=name_acc,
        baseline_exact_match=exact,
        baseline_ast_acc=ast_ok,
        baseline_tokens=baseline_tokens,
        baseline_raw=base_res.raw,
        baseline_pred=baseline_pred,
        baseline_error=baseline_error,
        baseline_latency_ms=base_res.latency_ms,
        baseline_prompt_tokens=base_res.prompt_tokens,
        retrievers=retriever_results,
    )


def _error_counts(errors: List[Optional[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in errors:
        if e:
            counts[e] = counts.get(e, 0) + 1
    return counts


def _mean_optional_ints(vals: List[Optional[int]]) -> Tuple[Optional[float], int]:
    xs = [int(v) for v in vals if v is not None]
    if not xs:
        return None, 0
    return sum(xs) / len(xs), len(xs)


def aggregate(results: List[InstanceResult], n_skipped: int) -> AggregateMetrics:
    n = len(results)
    empty_retrievers: Dict[str, RetrieverMetrics] = {}
    if n == 0:
        return AggregateMetrics(
            n=0, n_skipped=n_skipped,
            baseline_name_acc=0.0, baseline_exact_match=0.0, baseline_ast_acc=0.0,
            mean_baseline_tokens=0.0, mean_baseline_latency_ms=0.0,
            retrievers=empty_retrievers,
        )

    b_name   = sum(r.baseline_name_acc   for r in results) / n
    b_exact  = sum(r.baseline_exact_match for r in results) / n
    b_ast    = sum(getattr(r, "baseline_ast_acc", r.baseline_exact_match) for r in results) / n
    b_tokens = sum(r.baseline_tokens      for r in results) / n
    b_lat    = sum(getattr(r, "baseline_latency_ms", 0.0) for r in results) / n
    b_usage_mean, b_usage_n = _mean_optional_ints(
        [getattr(r, "baseline_prompt_tokens", None) for r in results]
    )

    rnames = list((results[0].retrievers or {}).keys())
    retriever_metrics: Dict[str, RetrieverMetrics] = {}

    for rname in rnames:
        rr = [r.retrievers[rname] for r in results if r.retrievers and rname in r.retrievers]
        nr = len(rr)
        if nr == 0:
            continue
        r_name_acc = sum(r.name_acc    for r in rr) / nr
        r_exact    = sum(r.exact_match for r in rr) / nr
        r_ast      = sum(getattr(r, "ast_acc", r.exact_match) for r in rr) / nr
        r_usage_mean, r_usage_n = _mean_optional_ints(
            [getattr(r, "prompt_tokens", None) for r in rr]
        )
        retriever_metrics[rname] = RetrieverMetrics(
            name_acc=r_name_acc,
            exact_match=r_exact,
            ast_acc=r_ast,
            recall=sum(r.recall            for r in rr) / nr,
            dcg=sum(r.dcg                  for r in rr) / nr,
            ndcg=sum(r.ndcg                for r in rr) / nr,
            mean_tokens=sum(r.tokens       for r in rr) / nr,
            mean_compression_rate=sum(r.compression_rate for r in rr) / nr,
            mean_latency_ms=sum(getattr(r, "latency_ms", 0.0) for r in rr) / nr,
            delta_name_acc=r_name_acc - b_name,
            delta_exact_match=r_exact - b_exact,
            delta_ast_acc=r_ast - b_ast,
            error_counts=_error_counts([getattr(r, "error", None) for r in rr]),
            mean_usage_prompt_tokens=r_usage_mean,
            n_usage_prompt_tokens=r_usage_n,
        )

    return AggregateMetrics(
        n=n,
        n_skipped=n_skipped,
        baseline_name_acc=b_name,
        baseline_exact_match=b_exact,
        baseline_ast_acc=b_ast,
        mean_baseline_tokens=b_tokens,
        mean_baseline_latency_ms=b_lat,
        retrievers=retriever_metrics,
        mean_baseline_usage_prompt_tokens=b_usage_mean,
        n_baseline_usage_prompt_tokens=b_usage_n,
    )
