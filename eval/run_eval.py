#!/usr/bin/env python3
"""
BFCL evaluation: LLM tool-calling accuracy across a model × retriever matrix.

Usage (from the project root):
  python eval/run_eval.py                          # full matrix, all configured models
  python eval/run_eval.py --model Qwen/Qwen2.5-7B-Instruct  # single model override
  python eval/run_eval.py --dry-run --samples 20   # no GPU required

Run `python eval/run_eval.py --help` for all options.
"""

# ── Path bootstrap (must precede all local imports) ───────────────────────────
import os
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent     # .../eval/
_SRC_DIR = _EVAL_DIR.parent / "src"             # .../src/

for _p in [str(_EVAL_DIR), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# On Apple Silicon, remove PyTorch MPS's self-imposed memory cap so it can use
# whatever physical unified memory is actually available.  Without this, MPS
# refuses new allocations even when physical memory exists but "other" (CPU-side)
# allocations have grown to crowd the default watermark.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import gc
import json
import re
import time
import traceback
from datetime import datetime
import yaml
from tqdm import tqdm

import toolscope
from bfcl_eval.checkpoint import CheckpointManager
from bfcl_eval.dataset import (
    load_entries, collect_catalog, build_instance_pool, write_collision_report,
)
from bfcl_eval.model import DummyModel, HFModel, OpenAIModel
from bfcl_eval.evaluate import evaluate_instance, aggregate, AggregateMetrics, RetrieverMetrics
from bfcl_eval.report import (
    print_report, print_cross_model_summary, save_results, print_verbose_instance,
    write_paper_artifacts,
)
from bfcl_eval.retrieval import (
    RandomRetriever, BM25Retriever, TFIDFRetriever, OracleRetriever, ToolScopeRetriever,
)
from bfcl_eval.agent import require_llm_credentials


# ── Config helpers ──────────────────────────────────────────────────────────


_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _env_nonempty(name: str) -> str:
    return (os.environ.get(name) or "").strip().strip('"').strip("'")


def _expand_env(value: object) -> str:
    """Expand ``${VAR}`` placeholders; strip quotes. Non-strings become ''."""
    if not isinstance(value, str):
        return ""
    expanded = _ENV_PLACEHOLDER.sub(lambda m: _env_nonempty(m.group(1)), value)
    return expanded.strip().strip('"').strip("'")


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _deep_set(d: dict, dotted_key: str, value: object) -> None:
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _resolve_model_entries(model_cfg: dict, cli_model: str | None) -> list[dict]:
    """Return the ordered list of per-model config dicts.

    Each dict has at least: name, backend, device, dtype, max_new_tokens,
    base_url, api_key.

    Supports three config shapes (newest first):
      1. model.entries  — list of dicts, each with at least ``name``
      2. model.names    — list of plain model-name strings
      3. model.name     — single legacy string

    Per-entry values override ``model.defaults``, which in turn override
    hardcoded defaults.  CLI flags (--device, --backend, …) are written to
    top-level model keys by _apply_overrides and take precedence over
    ``model.defaults``.
    """
    # 1. Hardcoded defaults (OpenAI-compatible URL/key come from .env when set)
    defaults = {
        "backend": "hf",
        "provider": "openai",
        "device": "auto",
        "dtype": "auto",
        "max_new_tokens": 256,
        "timeout_seconds": 120,
        "base_url": _env_nonempty("OPENAI_BASE_URL") or "http://localhost:8000/v1",
        "api_key": _env_nonempty("OPENAI_API_KEY") or "EMPTY",
        "api_key_env": "OPENAI_API_KEY",
    }

    # 2. Config-level defaults (model.defaults section)
    defaults.update(model_cfg.get("defaults", {}))

    # 3. Top-level model keys (backward compat + CLI overrides)
    for key in list(defaults):
        if key in model_cfg and not isinstance(model_cfg[key], (dict, list)):
            defaults[key] = model_cfg[key]

    # Resolve entry list
    if cli_model:
        configured = list(model_cfg.get("entries") or [])
        matched = []
        for raw in configured:
            name = raw if isinstance(raw, str) else (raw or {}).get("name")
            if name == cli_model:
                matched.append(raw)
        raw_entries = matched if matched else [{"name": cli_model}]
    elif "entries" in model_cfg:
        raw_entries = list(model_cfg["entries"])
    elif "names" in model_cfg:
        raw_entries = [{"name": n} for n in model_cfg["names"]]
    elif "name" in model_cfg:
        raw_entries = [{"name": model_cfg["name"]}]
    else:
        raw_entries = [{"name": "Qwen/Qwen2.5-7B-Instruct"}]

    entries = []
    for raw in raw_entries:
        if isinstance(raw, str):
            raw = {"name": raw}
        entry = dict(defaults)
        entry.update(raw)
        provider = (entry.get("provider") or "openai").lower()
        entry["provider"] = provider

        env_name = entry.get("api_key_env")
        if env_name:
            entry["api_key"] = (
                _env_nonempty(str(env_name)) or str(entry.get("api_key") or "")
            )
        key = str(entry.get("api_key") or "").strip().strip('"').strip("'")
        if provider in ("google", "gemini") and not key:
            key = _env_nonempty("GOOGLE_API_KEY") or _env_nonempty("GEMINI_API_KEY")
        entry["api_key"] = key

        raw_url = entry.get("base_url")
        if isinstance(raw_url, str) and raw_url.strip():
            entry["base_url"] = _expand_env(raw_url)
        elif provider in ("google", "gemini"):
            entry["base_url"] = ""
        else:
            entry["base_url"] = (
                _env_nonempty("OPENAI_BASE_URL")
                or _expand_env(raw_url or "")
                or "http://localhost:8000/v1"
            )
        entries.append(entry)

    return entries


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.device:
        _deep_set(cfg, "model.device", args.device)
    if args.dtype:
        _deep_set(cfg, "model.dtype", args.dtype)
    if args.samples is not None:
        _deep_set(cfg, "dataset.samples", args.samples)
    if args.pool_size is not None:
        _deep_set(cfg, "dataset.pool_size", args.pool_size)
    if args.k is not None:
        _deep_set(cfg, "toolscope.k", args.k)
    if getattr(args, "k_values", None):
        _deep_set(cfg, "toolscope.k_values", list(args.k_values))
    if args.category:
        _deep_set(cfg, "dataset.categories", args.category)
    if args.seed is not None:
        _deep_set(cfg, "dataset.seed", args.seed)
    if args.output_dir:
        _deep_set(cfg, "output.results_dir", args.output_dir)
    if args.verbose:
        _deep_set(cfg, "output.verbose", True)
    if args.backend:
        _deep_set(cfg, "model.backend", args.backend)
    if args.base_url:
        _deep_set(cfg, "model.base_url", args.base_url)
    if args.api_key:
        _deep_set(cfg, "model.api_key", args.api_key)
    if getattr(args, "retrievers", None):
        cfg["retrievers"] = list(args.retrievers)
    return cfg


_GATED_MODELS = {
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def _preflight_check(model_entries: list[dict]) -> None:
    """Warn early about prerequisites that would cause the run to fail later."""
    import os

    hf_names = [e["name"] for e in model_entries if e["backend"] == "hf"]
    gated = [m for m in hf_names if m in _GATED_MODELS]
    if not gated:
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return

    print("━" * 60)
    print("  PRE-FLIGHT WARNING")
    print("━" * 60)
    print("  The following model(s) are gated on HuggingFace and")
    print("  require authentication:")
    for m in gated:
        print(f"    • {m}")
    print()
    print("  To fix this before starting the run:")
    print("    1. Accept the model licence at huggingface.co/<model>")
    print("    2. Set your HF token:  export HF_TOKEN=hf_...")
    print()
    print("  Without a token, these models will fail at load time.")
    print("  The run will continue with the remaining models, but")
    print("  you will need to re-run those models separately.")
    print("━" * 60)
    print()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BFCL evaluation: LLM tool calling across a model × retriever matrix"
    )
    p.add_argument("--config", default="eval/config.yaml")
    p.add_argument("--model",
                   help="Single model name (overrides model.entries in config)")
    p.add_argument("--device", help="Inference device: auto | cpu | cuda | mps")
    p.add_argument("--dtype", help="Model dtype: auto | float16 | bfloat16 | float32")
    p.add_argument("--samples", type=int, help="Max evaluation instances (per category)")
    p.add_argument("--pool-size", dest="pool_size", type=int)
    p.add_argument("--k", type=int, help="Tools each retriever returns per query")
    p.add_argument(
        "--k-values", nargs="+", type=int, metavar="K",
        help="K-ablation: score multiple k values per retriever (nested prefixes at k_max)",
    )
    p.add_argument("--category", nargs="+",
                   help="BFCL categories: simple multiple parallel parallel_multiple")
    p.add_argument("--seed", type=int)
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--backend", choices=["hf", "openai", "langchain"],
                   help="Model backend: hf | openai | langchain")
    p.add_argument("--base-url", dest="base_url",
                   help="Base URL for the OpenAI-compatible endpoint "
                        "(e.g. http://localhost:8000/v1)")
    p.add_argument("--api-key", dest="api_key",
                   help="API key for the OpenAI-compatible endpoint "
                        "(default: OPENAI_API_KEY env var, or 'EMPTY')")
    p.add_argument("--retrievers", nargs="+",
                   help="Retrievers to run: Random BM25 TF-IDF Oracle* ToolScope")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel instance workers for API models (default 1)")
    p.add_argument("--dry-run", action="store_true",
                   help="Use a dummy model (no GPU needed) to test the full pipeline")
    p.add_argument("--no-resume", dest="no_resume", action="store_true",
                   help="Ignore any existing checkpoint and start from scratch")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-instance details during evaluation")
    return p


def _write_error_log(
    output_dir: Path,
    model_name: str,
    instance_id: str,
    exc_type: str,
    exc_msg: str,
    tb: str,
) -> None:
    """Append a structured error record to a per-model error log file."""
    try:
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        slug = model_name.split("/")[-1]
        log_path = log_dir / f"{slug}_errors.log"
        with open(log_path, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n[{ts}] {instance_id}  {exc_type}: {exc_msg}\n")
            f.write(tb)
            f.write("\n" + "-" * 60 + "\n")
    except Exception:
        pass  # log failure must never interrupt the eval loop


def _flush_accelerator_memory() -> None:
    """Return cached GPU/MPS buffers to the system allocator.

    On MPS (Apple Silicon), PyTorch keeps Metal buffers in a free-list after
    tensors go out of scope.  Without explicit flushing the free-list grows
    across generate() calls, fragmenting memory and making each new allocation
    progressively slower.  Call this once per instance (not per predict call)
    to balance cleanup overhead vs. allocation pressure.
    """
    gc.collect()
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _free_model(model: object) -> None:
    """Delete a model and try to release accelerator memory."""
    del model
    _flush_accelerator_memory()


def _short_name(model_name: str) -> str:
    """Last path component, e.g. 'Qwen/Qwen2.5-7B-Instruct' → 'Qwen2.5-7B-Instruct'."""
    return model_name.split("/")[-1]


def _load_metrics_from_json(path: Path) -> AggregateMetrics:
    """Reconstruct AggregateMetrics from a saved result JSON written by save_results()."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    m = data["metrics"]
    retrievers = {}
    for name, rm in m.get("retrievers", {}).items():
        retrievers[name] = RetrieverMetrics(
            name_acc=rm["name_acc"],
            exact_match=rm["exact_match"],
            ast_acc=rm.get("ast_acc", rm.get("exact_match", 0.0)),
            recall=rm["recall"],
            dcg=rm["dcg"],
            ndcg=rm["ndcg"],
            mean_tokens=rm["mean_tokens"],
            mean_compression_rate=rm["mean_compression_rate"],
            mean_latency_ms=rm.get("mean_latency_ms", 0.0),
            delta_name_acc=rm["delta_name_acc"],
            delta_exact_match=rm["delta_exact_match"],
            delta_ast_acc=rm.get("delta_ast_acc", 0.0),
            error_counts=rm.get("error_counts", {}),
            mean_usage_prompt_tokens=rm.get("mean_usage_prompt_tokens"),
            n_usage_prompt_tokens=int(rm.get("n_usage_prompt_tokens") or 0),
        )
    metrics = AggregateMetrics(
        n=m["n"],
        n_skipped=m["n_skipped"],
        baseline_name_acc=m["baseline_name_acc"],
        baseline_exact_match=m["baseline_exact_match"],
        baseline_ast_acc=m.get("baseline_ast_acc", m.get("baseline_exact_match", 0.0)),
        mean_baseline_tokens=m["mean_baseline_tokens"],
        mean_baseline_latency_ms=m.get("mean_baseline_latency_ms", 0.0),
        retrievers=retrievers,
        mean_baseline_usage_prompt_tokens=m.get("mean_baseline_usage_prompt_tokens"),
        n_baseline_usage_prompt_tokens=int(
            m.get("n_baseline_usage_prompt_tokens") or 0
        ),
    )
    # Older result JSONs omit usage aggregates; recompute from instance traces.
    if metrics.n_baseline_usage_prompt_tokens == 0:
        from eval.bfcl_eval.harness_report import enrich_metrics_usage_from_instances

        enrich_metrics_usage_from_instances(metrics, data.get("instances") or [])
    return metrics


def _best_result_json(output_dir: Path, model_name: str) -> Path | None:
    """Largest-n result JSON for a model slug; break ties by mtime."""
    slug = model_name.split("/")[-1].replace(" ", "_")
    best = None
    best_n = -1
    best_mtime = -1.0
    for path in output_dir.glob(f"bfcl_eval_{slug}_*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                n = int((json.load(f).get("metrics") or {}).get("n") or -1)
        except Exception:
            continue
        mtime = path.stat().st_mtime
        if n > best_n or (n == best_n and mtime > best_mtime):
            best = path
            best_n = n
            best_mtime = mtime
    return best


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_SRC_DIR.parent / ".env", override=True)


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    _load_dotenv()
    args = _build_parser().parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = _EVAL_DIR / "config.yaml"
    if not cfg_path.exists():
        print(f"Config not found at {args.config}. Run from the project root.")
        sys.exit(1)

    cfg = _load_config(cfg_path)
    cfg = _apply_overrides(cfg, args)

    model_cfg   = cfg.get("model", {})
    ds_cfg      = cfg.get("dataset", {})
    ts_cfg      = cfg.get("toolscope", {})
    out_cfg     = cfg.get("output", {})
    embed_cfg   = ts_cfg.get("embedding", {})

    model_entries     = _resolve_model_entries(model_cfg, args.model)
    model_names       = [e["name"] for e in model_entries]
    artifact_model_names = [
        e["name"] for e in _resolve_model_entries(model_cfg, None)
    ]
    categories        = ds_cfg.get("categories", ["multiple"])
    samples           = ds_cfg.get("samples")
    pool_size         = ds_cfg.get("pool_size", 100)
    seed              = ds_cfg.get("seed", 42)
    k                 = ts_cfg.get("k", 5)
    k_values_raw      = ts_cfg.get("k_values")
    k_values          = sorted(set(int(x) for x in k_values_raw)) if k_values_raw else None
    embed_model       = embed_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    embed_provider    = embed_cfg.get("provider", "sentence-transformers")
    embed_allow_dl    = embed_cfg.get("allow_download", True)
    output_dir        = Path(out_cfg.get("results_dir", "eval/results"))
    versioned_dir_cfg = out_cfg.get("versioned_dir")
    verbose           = out_cfg.get("verbose", False)
    cache_dir         = Path(ds_cfg.get("cache_dir", "eval/.bfcl_cache"))
    protocol          = ds_cfg.get("protocol") or (
        "shared_catalog" if pool_size is None else "legacy"
    )
    requested_retrievers = cfg.get("retrievers") or None

    print()
    print("=== BFCL Evaluation — Model × Retriever Matrix ===")
    if args.dry_run:
        print("  Mode      : DRY RUN (dummy model, no GPU needed)")
    print(f"  Protocol  : {protocol}")
    print(f"  Models    : {len(model_entries)}")
    for entry in model_entries:
        tag = entry["backend"]
        if tag in ("openai", "langchain") and entry.get("provider") != "google":
            tag += f" → {entry.get('base_url', '')}"
        elif entry.get("provider") in ("google", "gemini"):
            tag += " → google"
        print(f"              {entry['name']}  ({tag})")
    print(f"  Categories: {categories}")
    print(f"  Samples   : {samples or 'all'}")
    k_label = f"k: {k_values}" if k_values else f"k: {k}"
    print(f"  Pool size : {pool_size if pool_size is not None else 'all (shared catalog)'}  |  {k_label}  |  Seed: {seed}")
    print(f"  Embedder  : {embed_model}")
    print()

    # ══ Shared setup ═════════════════════════════════════════════════════════

    print("Loading BFCL data...")
    entries = load_entries(
        categories=categories, cache_dir=cache_dir, samples=samples, seed=seed,
    )
    print(f"  {len(entries)} instances loaded")
    if not entries:
        print("No entries loaded. Check category names or network access.")
        sys.exit(1)

    print("Collecting tool definitions...")
    all_tools, collisions = collect_catalog(entries)
    print(f"  {len(all_tools)} unique tools across all entries")
    if collisions:
        report_path = output_dir / "tool_name_collisions.json"
        write_collision_report(collisions, report_path)
        print(f"  {len(collisions)} name/schema collisions (first-seen kept) → {report_path}")
    print()

    catalog_size = len(all_tools)
    tool_list = list(all_tools.values())
    catalog_names = []
    try:
        from bfcl_eval.tools import tool_name as _tool_name
        catalog_names = [_tool_name(t) for t in tool_list]
    except Exception:
        catalog_names = list(all_tools.keys())
    shared_pool = tool_list if protocol == "shared_catalog" else None
    ckpt_pool_size = catalog_size if protocol == "shared_catalog" else int(pool_size or 100)

    print("Building ToolScope index...")
    embedding_config = toolscope.EmbeddingConfig(
        provider=embed_provider,
        model=embed_model,
        allow_download=embed_allow_dl,
        normalize=True,
    )

    use_lc_selector = protocol == "shared_catalog"
    if use_lc_selector:
        from toolscope.adapters.langchain import ToolSelector
        from bfcl_eval.tools import catalog_to_langchain

        lc_catalog = catalog_to_langchain(tool_list)
        ts_selector = ToolSelector(embedding=embedding_config)
        ts_retriever: object = ToolScopeRetriever(ts_selector, lc_catalog)
        print("  ToolSelector (LangChain adapter) ready.")
    else:
        ts_retriever = toolscope.index(tool_list, embedding=embedding_config)
        print("  Index ready.")
    print()

    print("Building retrieval baselines...")
    retrievers = {
        "Random":    RandomRetriever(tool_list, seed=seed),
        "BM25":      BM25Retriever(tool_list),
        "TF-IDF":    TFIDFRetriever(tool_list),
        "Oracle*":   OracleRetriever(tool_list, seed=seed),
        "ToolScope": ts_retriever,
    }
    if requested_retrievers:
        wanted = set(requested_retrievers)
        retrievers = {n: r for n, r in retrievers.items() if n in wanted}
        missing = wanted - set(retrievers)
        if missing:
            print(f"  Warning: unknown retrievers ignored: {sorted(missing)}")
    print(f"  Retrievers: {', '.join(retrievers)}")
    print()

    # ══ Pre-flight checks ════════════════════════════════════════════════════

    _preflight_check(model_entries)
    if not args.dry_run:
        for entry in model_entries:
            if entry.get("backend") in ("openai", "langchain"):
                require_llm_credentials(entry)

    # ══ Model loop ═══════════════════════════════════════════════════════════

    all_metrics: dict[str, AggregateMetrics] = {}   # model_name → metrics
    failed_models: list[str] = []

    for model_idx, entry in enumerate(model_entries):

        model_name = entry["name"]
        t0 = time.monotonic()
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("─" * 60)
        print(f"  Model {model_idx + 1}/{len(model_entries)}: {model_name}")
        print(f"  Backend   : {entry['backend']}"
              + (f"  ({entry['base_url']})" if entry["backend"] == "openai" else ""))
        print(f"  Started   : {started_at}")
        print("─" * 60)
        print()

        try:
            with CheckpointManager(
                output_dir=output_dir,
                model_name=model_name,
                categories=categories,
                pool_size=ckpt_pool_size,
                seed=seed,
                k=k,
                resume=not args.no_resume,
                dry_run=args.dry_run,
                protocol=protocol,
                catalog_size=catalog_size,
                k_values=k_values,
                samples=samples,
            ) as ckpt:

                # ── Resume: load any previously evaluated instances ──────────
                done = ckpt.load()
                done_ids = set(done.keys())
                remaining = [e for e in entries if e.id not in done_ids]

                # In an exec chain, models completed in a prior process invocation
                # have nothing left to do.  Skip silently — metrics are reconstructed
                # from the saved JSON file at cross-model summary time.
                if not remaining and not args.model and len(model_entries) > 1:
                    continue

                print(f"  Checkpoint: {ckpt.path}")
                if done:
                    print(
                        f"  Resuming  : {len(done)} done, "
                        f"{len(remaining)} remaining"
                    )
                else:
                    print(f"  Resuming  : fresh start ({len(entries)} instances)")
                print()

                # ── Load model only if there is work left to do ──────────────
                model = None
                if remaining:
                    if args.dry_run:
                        model = DummyModel()
                    elif entry["backend"] == "langchain":
                        from bfcl_eval.agent import LangGraphAgent
                        model = LangGraphAgent(entry)
                    elif entry["backend"] == "openai":
                        model = OpenAIModel(
                            model_name=model_name,
                            base_url=entry["base_url"],
                            api_key=entry["api_key"],
                            max_new_tokens=entry["max_new_tokens"],
                        )
                    else:
                        model = HFModel(
                            model_name=model_name,
                            device=entry["device"],
                            dtype=entry["dtype"],
                            max_new_tokens=entry["max_new_tokens"],
                        )
                    # Eager load: any failure here is caught at model scope,
                    # not silently swallowed instance-by-instance.
                    model.load()
                else:
                    print("  All instances already evaluated — skipping model load.")

                results = list(done.values())
                n_skipped = 0

                if remaining:
                    for inst in tqdm(
                        remaining,
                        desc=_short_name(model_name),
                        unit="inst",
                        initial=len(done),
                        total=len(entries),
                    ):
                        if shared_pool is not None:
                            instance_pool = shared_pool
                        else:
                            instance_pool = build_instance_pool(
                                entry=inst, all_tools=all_tools,
                                pool_size=int(pool_size), seed=seed,
                            )
                        result = None
                        try:
                            result = evaluate_instance(
                                entry_id=inst.id,
                                messages=inst.messages,
                                ground_truth=inst.ground_truth,
                                tool_pool=instance_pool,
                                model=model,
                                retrievers=retrievers,
                                k=k,
                                possible_answer=inst.possible_answer,
                                functions_bfcl=inst.functions_bfcl,
                                k_values=k_values,
                            )
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            tb = traceback.format_exc()
                            tqdm.write(
                                f"\n  WARNING: {inst.id} failed — skipping\n"
                                f"  {type(exc).__name__}: {exc}\n"
                                f"{tb}"
                            )
                            _write_error_log(
                                output_dir, model_name, inst.id,
                                type(exc).__name__, str(exc), tb,
                            )

                        if result is None:
                            n_skipped += 1
                        else:
                            results.append(result)
                            ckpt.write(result)
                            if verbose:
                                print_verbose_instance(result, k)

                        # Flush GPU/MPS allocator free-list after every instance
                        # (success or failure).  On Apple Silicon, PyTorch keeps
                        # Metal buffers in a free-list after tensors go out of
                        # scope; without periodic flushing the free-list grows
                        # across generate() calls and inference slows to a crawl.
                        _flush_accelerator_memory()

                    if model is not None:
                        _free_model(model)

        except KeyboardInterrupt:
            print(f"\n  Interrupted during {model_name}. Checkpoint is intact.")
            raise
        except Exception as exc:
            failed_models.append(model_name)
            print(f"\n  Model {model_name} failed: {type(exc).__name__}: {exc}")
            print("  Continuing to the next model.\n")
            continue

        elapsed_s = time.monotonic() - t0
        h, rem = divmod(int(elapsed_s), 3600)
        m, s = divmod(rem, 60)
        elapsed_str = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
        finished_at = datetime.now().strftime("%H:%M:%S")
        print(f"  Completed in {elapsed_str}  (finished {finished_at})")
        print()

        metrics = aggregate(results, n_skipped=n_skipped)
        all_metrics[model_name] = metrics

        print_report(
            metrics=metrics,
            model_name=model_name,
            categories=categories,
            pool_size=ckpt_pool_size,
            k=k,
            embedding_model=embed_model,
        )

        save_results(
            results=results,
            metrics=metrics,
            config=cfg,
            output_dir=output_dir,
            model_name=model_name,
        )
        print()

        # After each model (except the last), replace this process image with a
        # fresh Python interpreter for the next model.  os.execv() reclaims ALL
        # memory — Python heap, MPS Metal buffers, macOS compressor pages — so
        # the next model always starts with the full memory budget.  Unlike a
        # subprocess, there is zero coordinator overhead: there is no second
        # process sitting in memory alongside the worker.
        # Checkpoints ensure no work is lost across the exec chain.
        # --no-resume is stripped so exec'd invocations don't delete checkpoints
        # that were built by earlier links in the chain.
        if not args.model and model_idx < len(model_entries) - 1:
            exec_argv = [str(Path(__file__).resolve())] + [
                a for a in sys.argv[1:] if a != "--no-resume"
            ]
            os.execv(sys.executable, [sys.executable] + exec_argv)
            # never reached

    # ══ Cross-model summary (only meaningful with ≥2 models) ════════════════

    if failed_models:
        print(f"  Note: {len(failed_models)} model(s) failed and were skipped:")
        for fm in failed_models:
            print(f"    • {fm}")
        print()

    # Load metrics from models that ran in a prior process invocation (exec
    # chain), and from siblings when this is a single-model rerun.
    for mname in artifact_model_names:
        if mname not in all_metrics:
            slug = mname.split("/")[-1].replace(" ", "_")
            candidates = sorted(
                output_dir.glob(f"bfcl_eval_{slug}_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            loaded = None
            loaded_n = -1
            loaded_mtime = -1.0
            for path in candidates:
                try:
                    metrics = _load_metrics_from_json(path)
                except Exception:
                    continue
                mtime = path.stat().st_mtime
                if metrics.n > loaded_n or (
                    metrics.n == loaded_n and mtime > loaded_mtime
                ):
                    loaded = metrics
                    loaded_n = metrics.n
                    loaded_mtime = mtime
            if loaded is not None:
                all_metrics[mname] = loaded
            elif candidates:
                print(f"  Warning: could not load metrics for {mname}")

    ordered: dict[str, AggregateMetrics] = {}
    for mname in artifact_model_names:
        if mname in all_metrics:
            ordered[mname] = all_metrics[mname]
    for mname, metrics in all_metrics.items():
        if mname not in ordered:
            ordered[mname] = metrics
    all_metrics = ordered

    all_instances: dict[str, list] = {}
    for mname in all_metrics:
        path = _best_result_json(output_dir, mname)
        if path is None:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            all_instances[mname] = payload.get("instances") or []
        except Exception:
            print(f"  Warning: could not load instance traces for {mname}")

    if len(all_metrics) >= 2:
        print_cross_model_summary(
            all_metrics=all_metrics,
            k=k,
        )

    if protocol == "shared_catalog" and all_metrics:
        freeze_dir = None
        if versioned_dir_cfg and not args.dry_run and not samples:
            freeze_dir = Path(versioned_dir_cfg)
        elif versioned_dir_cfg and (args.dry_run or samples):
            print(
                "  Note: skipping versioned artifacts "
                "(dry-run or --samples). Full runs copy table.md, "
                "summary.csv, and harness_results.md to "
                f"{versioned_dir_cfg}."
            )
        write_paper_artifacts(
            all_metrics=all_metrics,
            output_dir=output_dir,
            k=k,
            catalog_size=catalog_size,
            protocol=protocol,
            all_instances=all_instances,
            collisions=collisions,
            catalog_names=catalog_names,
            embedder=str(embed_model or ""),
            versioned_dir=freeze_dir,
        )


if __name__ == "__main__":
    main()
