#!/usr/bin/env python3
"""Regenerate eval/paper/artifacts from eval/results/paper/local JSONs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PAPER_CONFIG = REPO / "eval/paper/bfcl_multiple.yaml"
RESULTS_DIR = REPO / "eval/results/paper/local"
VERSIONED_DIR = REPO / "eval/paper/artifacts"


def _best_json(output_dir: Path, model_id: str) -> Path | None:
    slug = model_id.split("/")[-1]
    candidates = sorted(
        output_dir.glob(f"bfcl_eval_{slug}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    best: Path | None = None
    best_n = -1
    for path in candidates:
        if "inprogress" in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            n = int(payload.get("metrics", {}).get("n", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if n > best_n:
            best_n = n
            best = path
    return best


def _latest_checkpoint(model_id: str) -> Path | None:
    slug = model_id.split("/")[-1]
    ckpt_dir = RESULTS_DIR / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    candidates = sorted(
        ckpt_dir.glob(f"{slug}_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _enrich_instances_from_checkpoint(
    model_id: str, instances: list[dict]
) -> list[dict]:
    """Copy usage prompt_tokens from checkpoint JSONL into nested instance dicts.

    Older ``bfcl_eval_*.json`` exports omitted ``prompt_tokens`` even though
    checkpoints recorded ``baseline_prompt_tokens`` / retriever ``prompt_tokens``.
    """
    ckpt = _latest_checkpoint(model_id)
    if ckpt is None or not instances:
        return instances
    by_id: dict[str, dict] = {}
    with ckpt.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = row.get("id")
            if rid:
                by_id[str(rid)] = row
    if not by_id:
        return instances

    enriched = 0
    for inst in instances:
        row = by_id.get(str(inst.get("id") or ""))
        if not row:
            continue
        base = inst.setdefault("baseline", {})
        if base.get("prompt_tokens") is None and row.get("baseline_prompt_tokens") is not None:
            base["prompt_tokens"] = row["baseline_prompt_tokens"]
            enriched += 1
        rr_out = inst.setdefault("retrievers", {})
        for rname, rr in (row.get("retrievers") or {}).items():
            dst = rr_out.setdefault(rname, {})
            if dst.get("prompt_tokens") is None and rr.get("prompt_tokens") is not None:
                dst["prompt_tokens"] = rr["prompt_tokens"]
    if enriched:
        print(f"  {model_id}: enriched usage prompt_tokens from {ckpt.name} ({enriched} baselines)")
    return instances


def _load_all(cfg_path: Path) -> tuple[dict, dict, list]:
    from eval.bfcl_eval.harness_report import enrich_metrics_usage_from_instances
    from eval.run_eval import _load_metrics_from_json

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    model_ids = [e["name"] for e in cfg["model"]["entries"]]
    all_metrics: dict = {}
    all_instances: dict = {}
    manifest_rows: list[tuple[str, str, str, int]] = []

    for mid in model_ids:
        path = _best_json(RESULTS_DIR, mid)
        if path is None:
            continue
        metrics = _load_metrics_from_json(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        instances = _enrich_instances_from_checkpoint(
            mid, payload.get("instances") or []
        )
        enrich_metrics_usage_from_instances(metrics, instances)
        all_metrics[mid] = metrics
        all_instances[mid] = instances
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_rows.append((mid, path.name, sha, metrics.n))

    return all_metrics, all_instances, manifest_rows


def _render_readme(
    all_metrics: dict,
    all_instances: dict,
    manifest_rows: list[tuple[str, str, str, int]],
    *,
    frozen: bool,
) -> str:
    from eval.bfcl_eval.harness_report import (
        _delta_pp,
        _fmt_p,
        _name_flips,
        _pick_retriever,
        _pct,
        mcnemar_exact,
    )

    cfg = yaml.safe_load(PAPER_CONFIG.read_text(encoding="utf-8")) or {}
    expected = [e["name"] for e in cfg["model"]["entries"]]
    by_id = {mid: (fname, sha, n) for mid, fname, sha, n in manifest_rows}
    complete = [mid for mid in expected if mid in by_id and by_id[mid][2] >= 200]

    title = (
        "# Frozen local GGUF k=10 matrix"
        if frozen and len(complete) == len(expected)
        else "# Local GGUF matrix — progress snapshot"
    )
    banner = ""
    if not frozen or len(complete) < len(expected):
        banner = (
            "> **In progress:** "
            f"{len(complete)} of {len(expected)} models at n=200. "
            "This snapshot updates until the matrix finishes.\n\n"
        )

    lines = [
        title,
        "",
        banner.rstrip(),
        "",
        "BFCL V4 Non-Live Multiple, shared catalog **C = 443**, **k = 10** (anchor; k-ablation",
        "{k ∈ 5, 10, 20} in [`harness_results.md`](harness_results.md)), MiniLM-L6-v2.",
        "**n = 200** per completed model. Scores are BFCL-derived, not official Gorilla numbers.",
        "",
        "Runtime writes go to gitignored `eval/results/paper/local/`. This directory is the",
        "git-tracked **paper v1.0** snapshot.",
        "",
        "| Model | Source file (gitignored) | SHA-256 |",
        "|---|---|---|",
    ]
    for mid in expected:
        if mid in by_id:
            fname, sha, n = by_id[mid]
            note = f" (n={n})" if n < 200 else ""
            lines.append(f"| {mid} | `{fname}` | `{sha}`{note} |")
        else:
            lines.append(f"| {mid} | — | *pending* |")

    if not all_metrics:
        lines.append("")
        lines.append("_No completed result JSONs yet._")
        return "\n".join(lines) + "\n"

    first = next(iter(all_metrics.values()))
    retrievers = list(first.retrievers.keys())
    bm = _pick_retriever(retrievers, "BM25", 10) or "BM25@10"
    ts = _pick_retriever(retrievers, "ToolScope", 10) or "ToolScope@10"

    lines += [
        "",
        "## Tool name accuracy",
        "",
        "| Model | Baseline | BM25 | ToolScope | Δ ToolScope vs baseline |",
        "|---|---|---|---|---|",
    ]
    for mid in sorted(all_metrics.keys(), key=lambda m: all_metrics[m].baseline_name_acc):
        m = all_metrics[mid]
        insts = all_instances.get(mid, [])
        wins, losses = _name_flips(insts, ts) if insts else ([], [])
        p = mcnemar_exact(len(wins), len(losses)) if insts else float("nan")
        bm_acc = m.retrievers[bm].name_acc if bm in m.retrievers else 0.0
        ts_acc = m.retrievers[ts].name_acc if ts in m.retrievers else 0.0
        delta = m.retrievers[ts].delta_name_acc if ts in m.retrievers else 0.0
        p_s = _fmt_p(p) if insts else "—"
        lines.append(
            f"| {mid} | {_pct(m.baseline_name_acc)} | {_pct(bm_acc)} | "
            f"**{_pct(ts_acc)}** | **{_delta_pp(delta)}** "
            f"(McNemar exact p = {p_s}; +{len(wins)} / −{len(losses)}) |"
        )

    lines += [
        "",
        "BM25 / ToolScope columns are **@k=10** (`BM25@10`, `ToolScope@10` in the full matrix).",
        "",
        "## AST accuracy",
        "",
        "| Model | Baseline | BM25 | ToolScope |",
        "|---|---|---|---|",
    ]
    for mid in sorted(all_metrics.keys(), key=lambda m: all_metrics[m].baseline_name_acc):
        m = all_metrics[mid]
        bm_ast = getattr(m.retrievers.get(bm), "ast_acc", m.retrievers[bm].exact_match)
        ts_ast = getattr(m.retrievers.get(ts), "ast_acc", m.retrievers[ts].exact_match)
        lines.append(
            f"| {mid} | {_pct(getattr(m, 'baseline_ast_acc', m.baseline_exact_match))} | "
            f"{_pct(bm_ast)} | {_pct(ts_ast)} |"
        )

    if bm in first.retrievers and ts in first.retrievers:
        lines += [
            "",
            "Retrieval (identical across models): "
            f"BM25 Recall@10 **{_pct(first.retrievers[bm].recall)}** / "
            f"NDCG **{first.retrievers[bm].ndcg:.3f}**; "
            f"ToolScope Recall@10 **{_pct(first.retrievers[ts].recall)}** / "
            f"NDCG **{first.retrievers[ts].ndcg:.3f}**. "
            f"Compression **{_pct(first.retrievers[ts].mean_compression_rate)}** "
            f"(~{first.mean_baseline_tokens:,.0f} → ~{first.retrievers[ts].mean_tokens:,.0f} "
            "prompt tokens at k=10).",
        ]

    lines += [
        "",
        "See [harness_results.md](harness_results.md) for the analysis (name/AST, McNemar, "
        "error taxonomy, flips). [table.md](table.md) and [summary.csv](summary.csv) are the "
        "compact matrix (includes k-ablation columns). Historical API-model results: "
        "[`../README.md`](../README.md). Follow-up experiments: "
        "[../../next-experiments.md](../../next-experiments.md).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit",
        action="store_true",
        help="git commit artifacts when content changes",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="git push after commit (implies meaningful changes)",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=200,
        help="Require this many instances per model for 'frozen' README title",
    )
    args = parser.parse_args()

    all_metrics, all_instances, manifest_rows = _load_all(PAPER_CONFIG)
    if not all_metrics:
        print("No result JSONs found; nothing to freeze.", file=sys.stderr)
        return 1

    collisions_path = RESULTS_DIR / "tool_name_collisions.json"
    collisions = (
        json.loads(collisions_path.read_text(encoding="utf-8"))
        if collisions_path.exists()
        else []
    )

    from eval.bfcl_eval.report import write_paper_artifacts

    # Seed live harness_results with the versioned copy so hand-authored
    # ``## Analysis`` is preserved across regenerate (see write_harness_results).
    ver_hr = VERSIONED_DIR / "harness_results.md"
    live_hr = RESULTS_DIR / "harness_results.md"
    if ver_hr.is_file():
        live_hr.write_text(ver_hr.read_text(encoding="utf-8"), encoding="utf-8")

    write_paper_artifacts(
        all_metrics=all_metrics,
        output_dir=RESULTS_DIR,
        k=10,
        catalog_size=443,
        protocol="shared_catalog",
        all_instances=all_instances,
        collisions=collisions,
        embedder="sentence-transformers/all-MiniLM-L6-v2",
        versioned_dir=None,
    )

    for name in ("table.md", "summary.csv", "harness_results.md"):
        src = RESULTS_DIR / name
        (VERSIONED_DIR / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = yaml.safe_load(PAPER_CONFIG.read_text(encoding="utf-8")) or {}
    expected_n = len(cfg["model"]["entries"])
    complete_n = sum(1 for _, _, _, n in manifest_rows if n >= args.min_n)
    frozen = complete_n >= expected_n

    readme = _render_readme(
        all_metrics, all_instances, manifest_rows, frozen=frozen
    )
    (VERSIONED_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote artifacts → {VERSIONED_DIR} ({len(all_metrics)} models, frozen={frozen})")

    if not args.commit:
        return 0

    status = subprocess.run(
        ["git", "status", "--porcelain", str(VERSIONED_DIR.relative_to(REPO))],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    if not status.stdout.strip():
        print("No artifact changes to commit.")
        return 0

    msg = (
        "Freeze local GGUF matrix artifacts (5/5 models, n=200)."
        if frozen
        else f"Update local GGUF matrix progress artifacts ({complete_n}/{expected_n} models)."
    )
    paths = [str(VERSIONED_DIR.relative_to(REPO))]
    if frozen:
        paths += [
            "eval/local/scripts/finalize_local_artifacts.py",
            "eval/local/scripts/watch_matrix_and_finalize.sh",
            "eval/local/scripts/run_local_matrix.sh",
        ]
    subprocess.run(
        ["git", "add", *paths],
        cwd=REPO,
        check=True,
    )
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO, check=True)
    print(f"Committed: {msg}")

    if args.push:
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=REPO, check=True)
        print("Pushed to origin.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
