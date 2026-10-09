from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def memory_mib(value: str | None) -> float | None:
    if not value:
        return None
    match = re.match(r"([0-9.]+)([KMG]iB)", value)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2)
    return round(amount * {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}[unit], 2)


def cpu_pct(value: str | None) -> float | None:
    if not value:
        return None
    return float(value.rstrip("%"))


def source(
    label: str,
    generated_at: str,
    *,
    component_ids: list[str],
    definitions: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "provider": "Local DevFlow AI benchmark harness",
        "files": [
            "scripts/benchmark_system.py",
            "scripts/benchmark_models.py",
            ".data/performance/2026-10-07/benchmark-results.json",
        ],
        "executedAt": generated_at,
        "metricDefinitions": [
            {**definition, "componentIds": component_ids}
            for definition in definitions
        ],
        "evidenceFlow": [
            {
                "title": "Measured on the running local stack",
                "detail": (
                    "Measurements came from loopback HTTP requests, direct PostgreSQL and Milvus reads, "
                    "local BGE model inference, process sampling, Docker stats, and the configured external LLM. "
                    "No destructive API or GitHub operation was executed."
                ),
            },
            {"title": "Method", "detail": method},
            {
                "title": "Scope limit",
                "detail": (
                    "The knowledge base contained five documents. Cold RAG, both ChatAgent scenarios, and "
                    "the model cold-load checks have one sample each; the external LLM has three samples. "
                    "These values characterize this host and run, not production capacity or a statistical SLA."
                ),
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    generated_at = data["generated_at"]

    http_rows = []
    for item in data["http_benchmarks"]:
        http_rows.append(
            {
                "scenario": item["name"],
                "mode": item["kind"],
                "concurrency": item.get("concurrency", 1),
                "samples": item["samples"],
                "p50Ms": item["p50_ms"],
                "p95Ms": item["p95_ms"],
                "p99Ms": item["p99_ms"],
                "throughputRps": item["throughput_rps"],
                "errorRatePct": item["error_rate_pct"],
            }
        )

    db = data["database"]
    milvus = data["milvus_direct"]
    storage_rows = [
        {"component": "PostgreSQL SELECT 1", "samples": db["samples"], "p50Ms": db["p50_ms"], "p95Ms": db["p95_ms"], "errorRatePct": 0},
        {"component": "Milvus vector search", "samples": milvus["samples"], "p50Ms": milvus["p50_ms"], "p95Ms": milvus["p95_ms"], "errorRatePct": 0},
    ]

    rag = data["rag"]
    embedding = data["embedding_benchmark"]
    reranker = data["reranker_benchmark"]
    rag_rows = [
        {"stage": "RAG cold end-to-end", "kind": "cold", "p50Ms": rag["cold"]["p50_ms"], "p95Ms": rag["cold"]["p95_ms"], "samples": 1},
        {"stage": "Embedding cold", "kind": "cold", "p50Ms": embedding["cold_first_request_ms"], "p95Ms": embedding["cold_first_request_ms"], "samples": 1},
        {"stage": "Reranker cold, 8 pairs", "kind": "cold", "p50Ms": reranker["cold_first_request_ms"], "p95Ms": reranker["cold_first_request_ms"], "samples": 1},
        {"stage": "RAG warm end-to-end", "kind": "warm", "p50Ms": rag["warm"]["p50_ms"], "p95Ms": rag["warm"]["p95_ms"], "samples": rag["warm"]["samples"]},
        {"stage": "RAG concurrent 2", "kind": "warm", "p50Ms": rag["concurrent"]["p50_ms"], "p95Ms": rag["concurrent"]["p95_ms"], "samples": rag["concurrent"]["samples"]},
        {"stage": "Embedding warm, 1 text", "kind": "warm", "p50Ms": embedding["warm_single"]["p50_ms"], "p95Ms": embedding["warm_single"]["p95_ms"], "samples": embedding["warm_single"]["samples"]},
        {"stage": "Embedding warm, 8 texts", "kind": "warm", "p50Ms": embedding["warm_batch_8"]["p50_ms"], "p95Ms": embedding["warm_batch_8"]["p95_ms"], "samples": embedding["warm_batch_8"]["samples"]},
        {"stage": "Reranker warm, 8 pairs", "kind": "warm", "p50Ms": reranker["warm_batch_8"]["p50_ms"], "p95Ms": reranker["warm_batch_8"]["p95_ms"], "samples": reranker["warm_batch_8"]["samples"]},
    ]

    model_rows = [
        {
            "component": "BGE-M3 embedding",
            "batchSize": 8,
            "warmLatencyMs": embedding["warm_batch_8"]["mean_ms"],
            "throughputPerSec": embedding["warm_batch_8"]["texts_per_second"],
            "residentMemoryMb": embedding["process_rss_mb"],
        },
        {
            "component": "BGE reranker v2 M3",
            "batchSize": 8,
            "warmLatencyMs": reranker["warm_batch_8"]["mean_ms"],
            "throughputPerSec": reranker["warm_batch_8"]["candidate_pairs_per_second"],
            "residentMemoryMb": reranker["process_rss_mb"],
        },
    ]

    llm = data["llm_stream"]
    direct = data["chat_agent"]
    tool = data["chat_agent_tool"]
    llm_rows = [
        {"scenario": "External LLM time to first token", "type": "external_llm", "p50Ms": llm["ttft"]["p50_ms"], "p95Ms": llm["ttft"]["p95_ms"], "samples": llm["ttft"]["samples"]},
        {"scenario": "External LLM complete response", "type": "external_llm", "p50Ms": llm["total"]["p50_ms"], "p95Ms": llm["total"]["p95_ms"], "samples": llm["total"]["samples"]},
        {"scenario": "ChatAgent direct answer", "type": "agent", "p50Ms": direct["latency_ms"], "p95Ms": direct["latency_ms"], "samples": 1},
        {"scenario": "ChatAgent forced tool loop", "type": "agent", "p50Ms": tool["latency_ms"], "p95Ms": tool["latency_ms"], "samples": 1},
    ]
    agent_rows = [
        {
            "scenario": "Direct answer",
            "latencySec": round(direct["latency_ms"] / 1000, 2),
            "modelCalls": direct["model_calls"],
            "toolCalls": direct["tool_calls"],
            "inputTokens": direct["input_tokens"],
            "outputTokens": direct["output_tokens"],
            "citations": direct["citations"],
            "route": direct["route"],
        },
        {
            "scenario": "Forced RAG tool loop",
            "latencySec": round(tool["latency_ms"] / 1000, 2),
            "modelCalls": tool["model_calls"],
            "toolCalls": tool["tool_calls"],
            "inputTokens": tool["input_tokens"],
            "outputTokens": tool["output_tokens"],
            "citations": tool["citations"],
            "route": tool["route"],
        },
    ]

    resource_summary = data["resources"]["summary"]
    hot_resource = data["rag_resource_sample"]["resources"]
    resource_rows = []
    for index, row in enumerate(data["resources"]["samples"]):
        if index % 4 != 0 and index != len(data["resources"]["samples"]) - 1:
            continue
        resource_rows.append(
            {
                "elapsedSec": row["elapsed_seconds"],
                "systemCpuPct": row["system_cpu_pct"],
                "systemMemoryPct": row["system_memory_pct"],
                "availableMemoryMb": row["system_available_mb"],
                "backendRssMb": (row.get("backend") or {}).get("rss_mb"),
            }
        )

    docker_after = data["docker_after"]
    memory_rows = [
        {"component": "Backend + local models", "memoryMb": max(resource_summary["peak_backend_rss_mb"], hot_resource["peak_backend_rss_mb"])},
        {"component": "Frontend", "memoryMb": resource_summary["peak_frontend_rss_mb"]},
    ]
    for item in docker_after:
        memory_rows.append({"component": item["Name"].removeprefix("devflow-ai-"), "memoryMb": memory_mib(item.get("MemUsage"))})

    coverage_rows = []
    for item in data["http_benchmarks"]:
        coverage_rows.append({"scenario": item["name"], "category": "HTTP", "samples": item["samples"], "concurrency": item.get("concurrency", 1), "errors": item["errors"]})
    coverage_rows.extend(
        [
            {"scenario": db["name"], "category": "Database", "samples": db["samples"], "concurrency": 1, "errors": 0},
            {"scenario": milvus["name"], "category": "Vector DB", "samples": milvus["samples"], "concurrency": 1, "errors": 0},
            {"scenario": rag["cold"]["name"], "category": "RAG", "samples": 1, "concurrency": 1, "errors": 0},
            {"scenario": rag["warm"]["name"], "category": "RAG", "samples": 5, "concurrency": 1, "errors": 0},
            {"scenario": rag["concurrent"]["name"], "category": "RAG", "samples": 4, "concurrency": 2, "errors": 0},
            {"scenario": "External LLM stream", "category": "LLM", "samples": 3, "concurrency": 1, "errors": llm["errors"]},
            {"scenario": direct["name"], "category": "Agent", "samples": 1, "concurrency": 1, "errors": 0 if direct["error"] is None else 1},
            {"scenario": tool["name"], "category": "Agent", "samples": 1, "concurrency": 1, "errors": 0 if tool["error"] is None else 1},
        ]
    )

    summary_row = {
        "appStartupSec": round(data["runtime"]["app_startup_ms"] / 1000, 2),
        "frontendSequentialP95Ms": next(row["p95Ms"] for row in http_rows if row["scenario"] == "Frontend home"),
        "backendHealthLoadRps": next(row["throughputRps"] for row in http_rows if row["scenario"] == "Backend health load"),
        "ragColdSec": round(rag["cold"]["p50_ms"] / 1000, 2),
        "ragWarmP50Sec": round(rag["warm"]["p50_ms"] / 1000, 2),
        "agentToolSec": round(tool["latency_ms"] / 1000, 2),
        "peakMemoryPct": resource_summary["peak_system_memory_pct"],
        "minAvailableMemoryMb": resource_summary["min_system_available_mb"],
        "peakBackendRssMb": max(resource_summary["peak_backend_rss_mb"], hot_resource["peak_backend_rss_mb"]),
        "requestQuerySamples": sum(row["samples"] for row in coverage_rows),
        "observedErrors": sum(row["errors"] for row in coverage_rows),
    }

    files_label = "Local benchmark results · 2026-10-07"
    snapshot = {
        "title": "DevFlow AI is fast at the web layer, but local AI saturates this 16GB host",
        "surface": "report",
        "status": "observed",
        "generatedAt": generated_at,
        "filters": [],
        "report": {
            "asOf": "2026-10-07",
            "originalQuestion": "测试当前系统的各方面性能数据。",
        },
        "queries": {
            "summary": {
                "label": "Performance summary",
                "rows": [summary_row],
                "source": source(
                    files_label,
                    generated_at,
                    component_ids=["performance-kpis", "executive-summary", "recommendations"],
                    definitions=[
                        {"field": "peakMemoryPct", "label": "Peak system memory", "definition": "Highest Windows physical-memory utilization sampled every 250ms during the main benchmark."},
                        {"field": "ragWarmP50Sec", "label": "Warm RAG p50", "definition": "Median end-to-end hybrid search latency across five sequential requests after model warm-up."},
                        {"field": "agentToolSec", "label": "Tool-loop latency", "definition": "One end-to-end ChatAgent request explicitly requiring a RAG tool call."},
                    ],
                    method="Computed directly from the benchmark result JSON without substituting missing values.",
                ),
            },
            "endpoint-latency": {
                "label": "HTTP latency and throughput",
                "rows": http_rows,
                "source": source(
                    files_label,
                    generated_at,
                    component_ids=["endpoint-latency-chart", "load-throughput-chart", "endpoint-load-table"],
                    definitions=[
                        {"field": "p50Ms", "label": "p50 latency", "definition": "Median client-observed wall-clock latency in milliseconds."},
                        {"field": "p95Ms", "label": "p95 latency", "definition": "95th percentile client-observed wall-clock latency using linear interpolation."},
                        {"field": "throughputRps", "label": "Throughput", "definition": "Successful responses divided by scenario wall-clock seconds."},
                    ],
                    method="Loopback HTTP with connection reuse for sequential tests; bounded async concurrency for load tests. Responses were validated and non-2xx responses counted as errors.",
                ),
            },
            "storage-latency": {
                "label": "Storage latency",
                "rows": storage_rows,
                "source": source(
                    files_label,
                    generated_at,
                    component_ids=["storage-latency-chart"],
                    definitions=[
                        {"field": "p50Ms", "label": "p50 latency", "definition": "Median direct query latency in milliseconds."},
                        {"field": "p95Ms", "label": "p95 latency", "definition": "95th percentile direct query latency in milliseconds."},
                    ],
                    method="100 direct PostgreSQL SELECT 1 queries on one connection and 50 direct Milvus searches after one warm-up search.",
                ),
            },
            "rag-latency": {
                "label": "RAG and local model latency",
                "rows": rag_rows,
                "source": source(
                    files_label,
                    generated_at,
                    component_ids=["rag-latency-chart", "rag-warm-latency-chart", "rag-latency-table", "recommendations"],
                    definitions=[
                        {"field": "p50Ms", "label": "p50 latency", "definition": "Median wall-clock latency; single-sample cold rows equal the observed first request."},
                        {"field": "kind", "label": "Cache state", "definition": "cold starts a fresh Python application process; warm reuses loaded model instances."},
                    ],
                    method="Full RAG used the production hybrid search endpoint. Component benchmarks ran each local model in a separate fresh process to avoid simultaneous duplicate model residency.",
                ),
            },
            "model-throughput": {
                "label": "Local model throughput",
                "rows": model_rows,
                "source": source(
                    files_label,
                    generated_at,
                    component_ids=["model-throughput-chart"],
                    definitions=[
                        {"field": "throughputPerSec", "label": "Batch throughput", "definition": "Eight texts or query-document pairs divided by mean warm batch latency."},
                        {"field": "residentMemoryMb", "label": "Resident memory", "definition": "Process RSS after the model benchmark completed."},
                    ],
                    method="BGE-M3 encoded batches of eight short texts; BGE reranker scored eight query-document pairs. PyTorch resolved to CPU.",
                ),
            },
            "llm-agent": {
                "label": "External LLM and Agent latency",
                "rows": llm_rows,
                "source": source(
                    files_label,
                    generated_at,
                    component_ids=["llm-agent-latency-chart"],
                    definitions=[
                        {"field": "p50Ms", "label": "p50 latency", "definition": "Median for the three external LLM stream samples; single Agent samples show their observed end-to-end latency."},
                        {"field": "p95Ms", "label": "p95 latency", "definition": "95th percentile for external LLM samples; single Agent rows repeat the one observation and are not an SLA estimate."},
                    ],
                    method="Three short streamed gpt-5.5 requests measured first content token and completion. Agent tests used the production /api/chat route with one direct and one forced-tool request.",
                ),
            },
            "agent-efficiency": {
                "label": "Agent calls, tokens, and evidence",
                "rows": agent_rows,
                "source": source(
                    files_label,
                    generated_at,
                    component_ids=["agent-efficiency-table", "recommendations"],
                    definitions=[
                        {"field": "inputTokens", "label": "Input tokens", "definition": "Sum of provider-reported input tokens across model calls in the ChatAgent request."},
                        {"field": "toolCalls", "label": "Tool calls", "definition": "Successful or attempted tool-call records returned by the Agent response."},
                    ],
                    method="Recorded from ChatAgent response metadata; no token price or cost was inferred.",
                ),
            },
            "resource-trend": {
                "label": "System resource samples",
                "reportingField": "elapsedSec",
                "rows": resource_rows,
                "source": source(
                    files_label,
                    generated_at,
                    component_ids=["resource-trend-chart", "resource-summary"],
                    definitions=[
                        {"field": "systemCpuPct", "label": "System CPU", "definition": "Windows system CPU utilization sampled every 250ms and downsampled to approximately one row per second for the report."},
                        {"field": "systemMemoryPct", "label": "System memory", "definition": "Percentage of physical memory in use at each sample."},
                        {"field": "backendRssMb", "label": "Backend RSS", "definition": "Combined resident memory for the managed backend process tree."},
                    ],
                    method="psutil sampled the system and managed process trees every 250ms; report rows retain every fourth sample plus the final observation.",
                ),
            },
            "memory-components": {
                "label": "Process and container memory",
                "rows": memory_rows,
                "source": source(
                    files_label,
                    generated_at,
                    component_ids=["memory-components-chart", "resource-summary"],
                    definitions=[
                        {"field": "memoryMb", "label": "Memory", "definition": "Peak process-tree RSS for frontend/backend and post-test Docker-reported resident usage for containers."},
                    ],
                    method="Process memory came from psutil; container memory came from docker stats --no-stream. These scopes are not additive to total physical memory because Docker runs through its own virtualization layer.",
                ),
            },
            "coverage": {
                "label": "Benchmark coverage and errors",
                "rows": coverage_rows,
                "source": source(
                    files_label,
                    generated_at,
                    component_ids=["coverage-table", "methods-scope"],
                    definitions=[
                        {"field": "samples", "label": "Samples", "definition": "Counted requests or direct storage queries retained in the scenario summary."},
                        {"field": "errors", "label": "Observed errors", "definition": "Exceptions, non-2xx HTTP responses, or failed response validations during the measured scenario."},
                    ],
                    method="Bounded non-destructive benchmark. It is not a sustained soak, failover, ingestion-scale, or production-capacity test.",
                ),
            },
            "hardware": {
                "label": "Benchmark host",
                "rows": [data["hardware"]],
                "source": source(
                    files_label,
                    generated_at,
                    component_ids=["methods-scope"],
                    definitions=[
                        {"field": "memory_total_mb", "label": "Physical memory", "definition": "Physical memory visible to Windows during the benchmark."},
                        {"field": "torch_device", "label": "PyTorch device", "definition": "Execution device used by the installed PyTorch build for local models."},
                    ],
                    method="Python platform and psutil host inspection plus nvidia-smi GPU inventory.",
                ),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "queries": len(snapshot["queries"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
