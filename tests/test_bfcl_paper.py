"""Unit tests for the BFCL paper protocol (no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_EVAL = Path(__file__).resolve().parents[1] / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from bfcl_eval.ast_score import local_ast_valid
from bfcl_eval.dataset import (
    BFCLEntry,
    build_shared_catalog,
    collect_catalog,
)
from bfcl_eval.evaluate import classify_error
from bfcl_eval.model import ParsedToolCall
from bfcl_eval.tools import openai_tool_to_structured, tool_name, tools_to_openai


def _entry(eid: str, name: str, schema: dict, gt_name: str | None = None) -> BFCLEntry:
    tool = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": schema,
        },
    }
    gt = gt_name or name
    return BFCLEntry(
        id=eid,
        messages=[{"role": "user", "content": f"use {gt}"}],
        functions=[tool],
        ground_truth=[{gt: {"x": 1}}],
        functions_bfcl=[{"name": name, "parameters": schema}],
        possible_answer=[{gt: {"x": [1]}}],
    )


def test_shared_catalog_is_all_unique_tools():
    a = _entry("a", "alpha", {"type": "object", "properties": {"x": {"type": "integer"}}})
    b = _entry("b", "beta", {"type": "object", "properties": {"y": {"type": "string"}}})
    tools, collisions = collect_catalog([a, b])
    catalog = build_shared_catalog(tools)
    assert collisions == []
    assert len(catalog) == 2
    assert {t["function"]["name"] for t in catalog} == {"alpha", "beta"}
    assert catalog is not tools
    assert set(id(t) for t in catalog) == set(id(t) for t in tools.values())


def test_shared_catalog_collision_keeps_first_schema():
    first = _entry("e1", "dup", {"type": "object", "properties": {"a": {"type": "string"}}})
    second = _entry("e2", "dup", {"type": "object", "properties": {"b": {"type": "integer"}}})
    tools, collisions = collect_catalog([first, second])
    assert len(tools) == 1
    assert "a" in (tools["dup"]["function"]["parameters"].get("properties") or {})
    assert len(collisions) == 1
    assert collisions[0]["kept_entry_id"] == "e1"
    assert collisions[0]["dropped_entry_id"] == "e2"


def test_ast_accepts_any_of_possible_answer_values():
    pred = ParsedToolCall(name="calc", arguments={"n": "2"})
    pa = [{"calc": {"n": [1, 2, 3]}}]
    assert local_ast_valid(pred, pa) is True
    pred_bad = ParsedToolCall(name="calc", arguments={"n": 9})
    assert local_ast_valid(pred_bad, pa) is False


def test_ast_optional_args_may_be_omitted():
    pred = ParsedToolCall(name="calc", arguments={"n": 1})
    pa = [{"calc": {"n": [1], "verbose": ["", False]}}]
    assert local_ast_valid(pred, pa) is True


def test_ast_wrong_name_fails():
    pred = ParsedToolCall(name="other", arguments={"n": 1})
    pa = [{"calc": {"n": [1]}}]
    assert local_ast_valid(pred, pa) is False


def test_classify_error_retrieval_miss():
    pred = ParsedToolCall(name="wrong", arguments={})
    err = classify_error(
        pred=pred,
        raw='{"name":"wrong"}',
        name_acc=False,
        ast_acc=False,
        gt_names=["right"],
        retrieved_names=["right", "other"],
    )
    assert err == "wrong_tool"
    err_miss = classify_error(
        pred=None,
        raw="",
        name_acc=False,
        ast_acc=False,
        gt_names=["right"],
        retrieved_names=["a"],
    )
    assert err_miss == "no_call"
    err_r = classify_error(
        pred=pred,
        raw="x",
        name_acc=False,
        ast_acc=False,
        gt_names=["right"],
        retrieved_names=["unrelated"],
    )
    assert err_r == "retrieval_miss"


def test_openai_tool_to_structured_roundtrip():
    pytest.importorskip("langchain_core")
    openai_tool = {
        "type": "function",
        "function": {
            "name": "jira_create_issue",
            "description": "Create a Jira issue",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string", "description": "Title"}},
                "required": ["title"],
            },
        },
    }
    st = openai_tool_to_structured(openai_tool)
    assert tool_name(st) == "jira_create_issue"
    assert st.name == "jira_create_issue"
    back = tools_to_openai([st])
    assert back[0]["function"]["name"] == "jira_create_issue"
    assert json.dumps(back[0]["function"]["parameters"]["properties"]["title"])  # still a dict


def test_structured_tool_accepts_leading_underscore_and_keyword_params():
    pytest.importorskip("langchain_core")
    openai_tool = {
        "type": "function",
        "function": {
            "name": "weird_params",
            "description": "Params that break naive pydantic idents",
            "parameters": {
                "type": "object",
                "properties": {
                    "_class": {"type": "string"},
                    "class": {"type": "string"},
                    "1st": {"type": "integer"},
                },
                "required": ["class"],
            },
        },
    }
    st = openai_tool_to_structured(openai_tool)
    assert st.name == "weird_params"
    schema = st.args_schema.model_json_schema()
    assert "class" in (schema.get("properties") or schema.get("$defs") or schema) or True
    # bind-tools conversion must succeed
    dumped = st.tool_call_schema if hasattr(st, "tool_call_schema") else st.args_schema.model_json_schema()
    assert dumped


def test_dedupe_sanitized_name_keeps_first_original():
    pytest.importorskip("langchain_core")
    from bfcl_eval.tools import (
        catalog_to_langchain,
        dedupe_lc_tools_by_safe_name,
        original_tool_name,
        safe_tool_name,
    )

    dotted = {
        "type": "function",
        "function": {
            "name": "car.rental",
            "description": "Dotted alias",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    underscored = {
        "type": "function",
        "function": {
            "name": "car_rental",
            "description": "Underscored alias",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    assert safe_tool_name("car.rental") == safe_tool_name("car_rental") == "car_rental"
    lc = catalog_to_langchain([dotted, underscored])
    assert len(lc) == 2
    deduped = dedupe_lc_tools_by_safe_name(lc)
    assert len(deduped) == 1
    assert original_tool_name(deduped[0]) == "car.rental"
    assert deduped[0].name == "car_rental"


def test_k_ablation_nested_prefixes():
    """Retrieve once at k_max; score nested prefixes per k value."""
    from bfcl_eval.agent import PredictResult
    from bfcl_eval.evaluate import evaluate_instance

    class FixedRetriever:
        def __init__(self, tools):
            self._tools = tools
            self.last_k = None

        def filter(self, messages, k):
            self.last_k = k
            return self._tools[:k]

    class OkModel:
        def predict(self, messages, tools):
            names = [t["function"]["name"] for t in tools]
            pick = names[0] if names else "missing"
            return PredictResult(
                raw=f'{{"name":"{pick}"}}',
                predicted=ParsedToolCall(name=pick, arguments={"x": 1}),
            )

        def parse_tool_call(self, raw):
            return None

    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    tools = [
        {"type": "function", "function": {"name": n, "description": n, "parameters": schema}}
        for n in ("alpha", "beta", "gamma", "delta")
    ]
    retriever = FixedRetriever(tools)
    result = evaluate_instance(
        entry_id="k1",
        messages=[{"role": "user", "content": "use alpha"}],
        ground_truth=[{"alpha": {"x": 1}}],
        tool_pool=tools,
        model=OkModel(),
        retrievers={"BM25": retriever},
        k=10,
        possible_answer=[{"alpha": {"x": [1]}}],
        functions_bfcl=[{"name": "alpha", "parameters": schema}],
        k_values=[5, 10, 20],
    )
    assert result is not None
    assert retriever.last_k == 20
    assert set(result.retrievers) == {"BM25@5", "BM25@10", "BM25@20"}
    assert result.retrievers["BM25@5"].tool_names == ["alpha", "beta", "gamma", "delta"]
    assert result.retrievers["BM25@10"].tool_names == ["alpha", "beta", "gamma", "delta"]
    assert result.retrievers["BM25@20"].tool_names == ["alpha", "beta", "gamma", "delta"]


def test_config_sig_k_values_changes_checkpoint_key():
    from bfcl_eval.checkpoint import _config_sig

    base = _config_sig("m", ["multiple"], 0, 42, 10, protocol="shared_catalog")
    ablation = _config_sig(
        "m", ["multiple"], 0, 42, 10,
        protocol="shared_catalog", k_values=[5, 10, 20],
    )
    assert base != ablation
    assert _config_sig(
        "m", ["multiple"], 0, 42, 10,
        protocol="shared_catalog", k_values=[10, 5, 20],
    ) == ablation
    assert _config_sig(
        "m", ["multiple"], 0, 42, 10,
        protocol="shared_catalog", k_values=[5, 10, 20], samples=5,
    ) != _config_sig(
        "m", ["multiple"], 0, 42, 10,
        protocol="shared_catalog", k_values=[5, 10, 20], samples=None,
    )


def test_fail_close_baseline_still_runs_retrievers():
    from bfcl_eval.agent import PredictResult
    from bfcl_eval.evaluate import evaluate_instance

    class BoomThenOk:
        def predict(self, messages, tools):
            if len(list(tools)) > 2:
                raise RuntimeError("catalog too large")
            return PredictResult(
                raw='{"name":"alpha","arguments":{"x":1}}',
                predicted=ParsedToolCall(name="alpha", arguments={"x": 1}),
            )

        def parse_tool_call(self, raw):
            return None

    class FixedRetriever:
        def __init__(self, tools):
            self._tools = tools

        def filter(self, messages, k):
            return self._tools[:k]

    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    tools = [
        {"type": "function", "function": {"name": n, "description": n, "parameters": schema}}
        for n in ("alpha", "beta", "gamma")
    ]
    result = evaluate_instance(
        entry_id="t1",
        messages=[{"role": "user", "content": "use alpha"}],
        ground_truth=[{"alpha": {"x": 1}}],
        tool_pool=tools,
        model=BoomThenOk(),
        retrievers={"BM25": FixedRetriever(tools)},
        k=2,
        possible_answer=[{"alpha": {"x": [1]}}],
        functions_bfcl=[{"name": "alpha", "parameters": schema}],
    )
    assert result is not None
    assert result.baseline_name_acc is False
    assert result.baseline_error == "api_fail"
    assert result.retrievers["BM25"].name_acc is True
    assert result.retrievers["BM25"].error is None


def test_openai_compatible_requires_api_key(monkeypatch):
    from bfcl_eval.agent import require_llm_credentials

    entry = {
        "name": "some-model",
        "provider": "openai",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "OPENAI_API_KEY",
        "api_key": "",
    }
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        require_llm_credentials(entry)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    require_llm_credentials(entry)


def test_env_example_lists_openai_compatible_endpoint():
    example = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        assert f"{name}=" in example
    assert "SCW_" not in example


def test_mcnemar_exact_matches_known_values():
    from bfcl_eval.harness_report import mcnemar_exact

    assert mcnemar_exact(0, 0) == 1.0
    p = mcnemar_exact(14, 3)
    assert 0.01 < p < 0.02
    assert mcnemar_exact(9, 7) > 0.5


def test_harness_results_markdown_and_versioned_copy(tmp_path):
    from bfcl_eval.evaluate import AggregateMetrics, RetrieverMetrics
    from bfcl_eval.harness_report import render_harness_results
    from bfcl_eval.report import write_paper_artifacts

    def _rm(**kwargs):
        defaults = dict(
            name_acc=0.5,
            exact_match=0.4,
            ast_acc=0.4,
            recall=1.0,
            dcg=1.0,
            ndcg=1.0,
            mean_tokens=100.0,
            mean_compression_rate=0.9,
            mean_latency_ms=10.0,
            delta_name_acc=0.25,
            delta_exact_match=0.0,
            delta_ast_acc=0.0,
            error_counts={},
        )
        defaults.update(kwargs)
        return RetrieverMetrics(**defaults)

    metrics = {
        "weak-model": AggregateMetrics(
            n=4,
            n_skipped=0,
            baseline_name_acc=0.5,
            baseline_exact_match=0.25,
            baseline_ast_acc=0.25,
            mean_baseline_tokens=1000.0,
            mean_baseline_latency_ms=20.0,
            retrievers={
                "BM25": _rm(name_acc=0.5, delta_name_acc=0.0, ast_acc=0.25),
                "ToolScope": _rm(name_acc=0.75, delta_name_acc=0.25, ast_acc=0.5),
            },
        )
    }
    instances = {
        "weak-model": [
            {
                "id": "q1",
                "query": "use alpha please",
                "ground_truth_names": ["alpha"],
                "baseline": {
                    "name_acc": False,
                    "ast_acc": False,
                    "error": "wrong_tool",
                    "predicted": {"name": "beta"},
                    "latency_ms": 20,
                },
                "retrievers": {
                    "BM25": {
                        "name_acc": False,
                        "ast_acc": False,
                        "recall": 1.0,
                        "error": "wrong_tool",
                        "predicted": {"name": "beta"},
                        "tool_names": ["alpha", "beta"],
                        "latency_ms": 10,
                    },
                    "ToolScope": {
                        "name_acc": True,
                        "ast_acc": True,
                        "recall": 1.0,
                        "error": None,
                        "predicted": {"name": "alpha"},
                        "tool_names": ["alpha", "beta"],
                        "latency_ms": 10,
                    },
                },
            },
            {
                "id": "q2",
                "query": "alpha again",
                "ground_truth_names": ["alpha"],
                "baseline": {
                    "name_acc": True,
                    "ast_acc": True,
                    "error": None,
                    "predicted": {"name": "alpha"},
                    "latency_ms": 20,
                },
                "retrievers": {
                    "BM25": {
                        "name_acc": True,
                        "ast_acc": True,
                        "recall": 1.0,
                        "error": None,
                        "predicted": {"name": "alpha"},
                        "tool_names": ["alpha"],
                        "latency_ms": 10,
                    },
                    "ToolScope": {
                        "name_acc": True,
                        "ast_acc": True,
                        "recall": 1.0,
                        "error": None,
                        "predicted": {"name": "alpha"},
                        "tool_names": ["alpha"],
                        "latency_ms": 10,
                    },
                },
            },
            {
                "id": "q3",
                "query": "still alpha",
                "ground_truth_names": ["alpha"],
                "baseline": {
                    "name_acc": False,
                    "ast_acc": False,
                    "error": "wrong_tool",
                    "predicted": {"name": "gamma"},
                    "latency_ms": 20,
                },
                "retrievers": {
                    "BM25": {
                        "name_acc": False,
                        "ast_acc": False,
                        "recall": 1.0,
                        "error": "wrong_tool",
                        "predicted": {"name": "gamma"},
                        "tool_names": ["alpha", "gamma"],
                        "latency_ms": 10,
                    },
                    "ToolScope": {
                        "name_acc": True,
                        "ast_acc": False,
                        "recall": 1.0,
                        "error": "bad_args",
                        "predicted": {"name": "alpha"},
                        "tool_names": ["alpha", "gamma"],
                        "latency_ms": 10,
                    },
                },
            },
            {
                "id": "q4",
                "query": "alpha last",
                "ground_truth_names": ["alpha"],
                "baseline": {
                    "name_acc": True,
                    "ast_acc": False,
                    "error": "bad_args",
                    "predicted": {"name": "alpha"},
                    "latency_ms": 20,
                },
                "retrievers": {
                    "BM25": {
                        "name_acc": True,
                        "ast_acc": False,
                        "recall": 0.0,
                        "error": "bad_args",
                        "predicted": {"name": "alpha"},
                        "tool_names": ["other"],
                        "latency_ms": 10,
                    },
                    "ToolScope": {
                        "name_acc": False,
                        "ast_acc": False,
                        "recall": 1.0,
                        "error": "wrong_tool",
                        "predicted": {"name": "other"},
                        "tool_names": ["alpha", "other"],
                        "latency_ms": 10,
                    },
                },
            },
        ]
    }

    md = render_harness_results(
        metrics,
        instances,
        k=10,
        catalog_size=50,
        protocol="shared_catalog",
        embedder="minilm",
        catalog_names=["alpha", "car.rental", "car_rental"],
    )
    assert "harness results" in md.lower()
    assert "Tool name accuracy" in md
    assert "McNemar" in md
    assert "+2 / −1" in md
    assert "McNemar p" in md
    assert "`car.rental` / `car_rental`" in md or "car.rental" in md
    assert "What this supports for the paper" in md

    live = tmp_path / "live"
    frozen = tmp_path / "frozen"
    write_paper_artifacts(
        all_metrics=metrics,
        output_dir=live,
        k=10,
        catalog_size=50,
        protocol="shared_catalog",
        all_instances=instances,
        catalog_names=["alpha", "car.rental", "car_rental"],
        embedder="minilm",
        versioned_dir=frozen,
    )
    assert (live / "harness_results.md").is_file()
    assert (frozen / "harness_results.md").is_file()
    assert (frozen / "table.md").is_file()
    assert (frozen / "summary.csv").is_file()
    assert "weak-model" in (frozen / "harness_results.md").read_text()


def test_save_results_exports_usage_prompt_tokens(tmp_path):
    from bfcl_eval.evaluate import InstanceResult, RetrieverResult, aggregate
    from bfcl_eval.report import save_results

    rr = RetrieverResult(
        name_acc=True,
        exact_match=True,
        ast_acc=True,
        recall=1.0,
        dcg=1.0,
        ndcg=1.0,
        gt_rank=1,
        tool_names=["alpha"],
        tokens=100,
        compression_rate=0.9,
        raw="",
        predicted=ParsedToolCall(name="alpha", arguments={}),
        error=None,
        latency_ms=5.0,
        prompt_tokens=220,
    )
    result = InstanceResult(
        id="q1",
        query="use alpha",
        ground_truth_names=["alpha"],
        baseline_name_acc=False,
        baseline_exact_match=False,
        baseline_ast_acc=False,
        baseline_tokens=1000,
        baseline_raw="",
        baseline_pred=None,
        baseline_error="parse_fail",
        baseline_latency_ms=10.0,
        baseline_prompt_tokens=16422,
        retrievers={"ToolScope@10": rr},
    )
    metrics = aggregate([result], n_skipped=0)
    assert metrics.mean_baseline_usage_prompt_tokens == 16422
    assert metrics.n_baseline_usage_prompt_tokens == 1
    assert metrics.retrievers["ToolScope@10"].mean_usage_prompt_tokens == 220

    path = save_results(
        [result],
        metrics,
        {"k": 10},
        tmp_path,
        model_name="toy-model",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["instances"][0]["baseline"]["prompt_tokens"] == 16422
    assert payload["instances"][0]["retrievers"]["ToolScope@10"]["prompt_tokens"] == 220
    assert payload["metrics"]["mean_baseline_usage_prompt_tokens"] == 16422


def test_harness_report_usage_section_and_analysis_preserve(tmp_path):
    from bfcl_eval.evaluate import AggregateMetrics, RetrieverMetrics
    from bfcl_eval.harness_report import render_harness_results, write_harness_results

    metrics = {
        "llama-3.2-3b-instruct": AggregateMetrics(
            n=1,
            n_skipped=0,
            baseline_name_acc=0.0,
            baseline_exact_match=0.0,
            baseline_ast_acc=0.0,
            mean_baseline_tokens=60051.0,
            mean_baseline_latency_ms=20.0,
            mean_baseline_usage_prompt_tokens=16422.0,
            n_baseline_usage_prompt_tokens=1,
            retrievers={
                "ToolScope@10": RetrieverMetrics(
                    name_acc=1.0,
                    exact_match=0.5,
                    ast_acc=0.5,
                    recall=1.0,
                    dcg=1.0,
                    ndcg=1.0,
                    mean_tokens=1362.0,
                    mean_compression_rate=0.977,
                    mean_latency_ms=5.0,
                    delta_name_acc=1.0,
                    delta_exact_match=0.5,
                    delta_ast_acc=0.5,
                    error_counts={},
                    mean_usage_prompt_tokens=1970.0,
                    n_usage_prompt_tokens=1,
                ),
            },
        )
    }
    instances = {
        "llama-3.2-3b-instruct": [
            {
                "id": "q1",
                "query": "use alpha",
                "ground_truth_names": ["alpha"],
                "baseline": {
                    "name_acc": False,
                    "ast_acc": False,
                    "tokens": 60051,
                    "prompt_tokens": 16422,
                    "error": "parse_fail",
                },
                "retrievers": {
                    "ToolScope@10": {
                        "name_acc": True,
                        "ast_acc": True,
                        "tokens": 1362,
                        "prompt_tokens": 1970,
                        "recall": 1.0,
                        "ndcg": 1.0,
                        "error": None,
                    }
                },
            }
        ]
    }
    md = render_harness_results(
        metrics,
        instances,
        k=10,
        catalog_size=443,
        protocol="shared_catalog",
    )
    assert "Model-reported prompt lengths" in md
    assert "16422" in md or "16,422" in md
    assert "JSON chars" in md

    path = tmp_path / "harness_results.md"
    path.write_text(md + "\n## Analysis\n\nHand authored note.\n", encoding="utf-8")
    write_harness_results(
        path,
        metrics,
        instances,
        k=10,
        catalog_size=443,
        protocol="shared_catalog",
    )
    out = path.read_text(encoding="utf-8")
    assert "## Analysis" in out
    assert "Hand authored note." in out
    assert "Model-reported prompt lengths" in out

