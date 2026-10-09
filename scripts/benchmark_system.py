from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import psutil
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings  # noqa: E402
from app.db.session import engine  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "mean_ms": round(statistics.fmean(values), 2) if values else None,
        "p50_ms": round(percentile(values, 0.50) or 0.0, 2) if values else None,
        "p95_ms": round(percentile(values, 0.95) or 0.0, 2) if values else None,
        "p99_ms": round(percentile(values, 0.99) or 0.0, 2) if values else None,
        "min_ms": round(min(values), 2) if values else None,
        "max_ms": round(max(values), 2) if values else None,
    }


def timed_request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    json_body: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> tuple[float, int, Any]:
    started = time.perf_counter()
    response = client.request(method, url, json=json_body, timeout=timeout)
    elapsed_ms = (time.perf_counter() - started) * 1000
    payload: Any = None
    if response.headers.get("content-type", "").startswith("application/json"):
        payload = response.json()
    response.raise_for_status()
    return elapsed_ms, response.status_code, payload


def benchmark_sequential(
    name: str,
    method: str,
    url: str,
    *,
    runs: int,
    warmups: int = 2,
    json_body: dict[str, Any] | None = None,
    timeout: float = 60.0,
    validate: Callable[[Any], bool] | None = None,
) -> dict[str, Any]:
    latencies: list[float] = []
    errors: list[str] = []
    with httpx.Client(trust_env=False) as client:
        for _ in range(warmups):
            try:
                timed_request(client, method, url, json_body=json_body, timeout=timeout)
            except Exception:
                pass
        started = time.perf_counter()
        for _ in range(runs):
            try:
                latency_ms, _status, payload = timed_request(
                    client,
                    method,
                    url,
                    json_body=json_body,
                    timeout=timeout,
                )
                if validate and not validate(payload):
                    raise ValueError("response validation failed")
                latencies.append(latency_ms)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        wall_seconds = time.perf_counter() - started
    return {
        "name": name,
        "kind": "sequential",
        **latency_summary(latencies),
        "wall_seconds": round(wall_seconds, 3),
        "throughput_rps": round(len(latencies) / wall_seconds, 2) if wall_seconds else None,
        "errors": len(errors),
        "error_rate_pct": round(len(errors) / runs * 100, 2) if runs else 0.0,
        "error_examples": errors[:3],
    }


async def benchmark_concurrent(
    name: str,
    method: str,
    url: str,
    *,
    total: int,
    concurrency: int,
    json_body: dict[str, Any] | None = None,
    timeout: float = 60.0,
    validate: Callable[[Any], bool] | None = None,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        async def one() -> None:
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.request(method, url, json=json_body)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    response.raise_for_status()
                    payload: Any = None
                    if response.headers.get("content-type", "").startswith("application/json"):
                        payload = response.json()
                    if validate and not validate(payload):
                        raise ValueError("response validation failed")
                    latencies.append(elapsed_ms)
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")

        started = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(total)))
        wall_seconds = time.perf_counter() - started

    return {
        "name": name,
        "kind": "concurrent",
        "concurrency": concurrency,
        **latency_summary(latencies),
        "wall_seconds": round(wall_seconds, 3),
        "throughput_rps": round(len(latencies) / wall_seconds, 2) if wall_seconds else None,
        "errors": len(errors),
        "error_rate_pct": round(len(errors) / total * 100, 2) if total else 0.0,
        "error_examples": errors[:3],
    }


def benchmark_database(runs: int = 100) -> dict[str, Any]:
    latencies: list[float] = []
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
        for _ in range(runs):
            started = time.perf_counter()
            connection.execute(text("SELECT 1")).scalar_one()
            latencies.append((time.perf_counter() - started) * 1000)
    return {
        "name": "PostgreSQL SELECT 1",
        "kind": "database",
        **latency_summary(latencies),
        "errors": 0,
        "error_rate_pct": 0.0,
    }


def benchmark_milvus(repo_id: str, runs: int = 50) -> dict[str, Any]:
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or "")
    raw_vector = [math.sin(index + 1) for index in range(settings.embedding_dimensions)]
    norm = math.sqrt(sum(value * value for value in raw_vector)) or 1.0
    query_vector = [value / norm for value in raw_vector]
    filter_expression = f'repo_id == "{repo_id}"'
    client.search(
        collection_name=settings.milvus_collection,
        data=[query_vector],
        filter=filter_expression,
        limit=3,
        output_fields=["document_id"],
    )
    latencies: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        client.search(
            collection_name=settings.milvus_collection,
            data=[query_vector],
            filter=filter_expression,
            limit=3,
            output_fields=["document_id"],
        )
        latencies.append((time.perf_counter() - started) * 1000)
    return {
        "name": "Milvus vector search",
        "kind": "vector_database",
        **latency_summary(latencies),
        "errors": 0,
        "error_rate_pct": 0.0,
    }


def process_tree_metrics(root_pid: int | None) -> dict[str, float | int | None]:
    if not root_pid:
        return {"process_count": 0, "rss_mb": None, "cpu_pct": None}
    try:
        root = psutil.Process(root_pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {"process_count": 0, "rss_mb": None, "cpu_pct": None}
    rss = 0
    cpu = 0.0
    visible = 0
    for process in processes:
        try:
            rss += process.memory_info().rss
            cpu += process.cpu_percent(interval=None)
            visible += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {
        "process_count": visible,
        "rss_mb": round(rss / 1024 / 1024, 2),
        "cpu_pct": round(cpu, 2),
    }


class ResourceSampler:
    def __init__(self, backend_pid: int | None, frontend_pid: int | None) -> None:
        self.backend_pid = backend_pid
        self.frontend_pid = frontend_pid
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process_cache: dict[int, psutil.Process] = {}

    def _tree_metrics(self, root_pid: int | None) -> dict[str, float | int | None]:
        if not root_pid:
            return {"process_count": 0, "rss_mb": None, "cpu_pct": None}
        try:
            root = self._process_cache.setdefault(root_pid, psutil.Process(root_pid))
            raw_processes = [root, *root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"process_count": 0, "rss_mb": None, "cpu_pct": None}
        rss = 0
        cpu = 0.0
        visible = 0
        for raw_process in raw_processes:
            try:
                process = self._process_cache.setdefault(raw_process.pid, raw_process)
                rss += process.memory_info().rss
                cpu += process.cpu_percent(interval=None)
                visible += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return {
            "process_count": visible,
            "rss_mb": round(rss / 1024 / 1024, 2),
            "cpu_pct": round(cpu, 2),
        }

    def start(self) -> None:
        psutil.cpu_percent(interval=None)
        self._tree_metrics(self.backend_pid)
        self._tree_metrics(self.frontend_pid)

        def sample() -> None:
            while not self._stop.wait(0.25):
                memory = psutil.virtual_memory()
                self.rows.append(
                    {
                        "elapsed_seconds": round(len(self.rows) * 0.25, 2),
                        "system_cpu_pct": psutil.cpu_percent(interval=None),
                        "system_memory_pct": memory.percent,
                        "system_available_mb": round(memory.available / 1024 / 1024, 2),
                        "backend": self._tree_metrics(self.backend_pid),
                        "frontend": self._tree_metrics(self.frontend_pid),
                    }
                )

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> dict[str, Any]:
        if not self.rows:
            return {}

        def maximum(path: tuple[str, ...]) -> float | None:
            values = []
            for row in self.rows:
                value: Any = row
                for key in path:
                    value = value.get(key) if isinstance(value, dict) else None
                if isinstance(value, (int, float)):
                    values.append(float(value))
            return round(max(values), 2) if values else None

        return {
            "samples": len(self.rows),
            "peak_system_cpu_pct": maximum(("system_cpu_pct",)),
            "peak_system_memory_pct": maximum(("system_memory_pct",)),
            "min_system_available_mb": round(
                min(float(row["system_available_mb"]) for row in self.rows), 2
            ),
            "peak_backend_rss_mb": maximum(("backend", "rss_mb")),
            "peak_backend_cpu_pct": maximum(("backend", "cpu_pct")),
            "peak_frontend_rss_mb": maximum(("frontend", "rss_mb")),
            "peak_frontend_cpu_pct": maximum(("frontend", "cpu_pct")),
        }


def docker_stats() -> list[dict[str, Any]]:
    command = [
        "docker",
        "stats",
        "--no-stream",
        "--format",
        "{{json .}}",
        "devflow-ai-postgres",
        "devflow-ai-etcd",
        "devflow-ai-minio",
        "devflow-ai-milvus",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def hardware_snapshot() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    gpu_name = None
    gpu_memory_mb = None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            gpu_name, memory_text = [part.strip() for part in completed.stdout.splitlines()[0].split(",", 1)]
            gpu_memory_mb = int(memory_text)
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "memory_total_mb": round(memory.total / 1024 / 1024, 2),
        "gpu_name": gpu_name,
        "gpu_memory_mb": gpu_memory_mb,
        "torch_device": "cpu",
    }


def process_state() -> tuple[int | None, int | None]:
    state_path = PROJECT_ROOT / ".data" / "devflow-processes.json"
    if not state_path.exists():
        return None, None
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    return state.get("backend_process_id"), state.get("frontend_process_id")


def benchmark_llm_stream(runs: int = 3) -> dict[str, Any]:
    if not settings.llm_api_key:
        return {"name": "External LLM stream", "skipped": True, "reason": "LLM API key is not configured"}
    totals: list[float] = []
    ttfts: list[float] = []
    token_counts: list[int] = []
    errors: list[str] = []
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    body = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": "Reply in one short sentence."},
            {"role": "user", "content": "Why do API latency percentiles matter?"},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 80,
    }
    with httpx.Client(timeout=120) as client:
        for _ in range(runs):
            started = time.perf_counter()
            first_content_at: float | None = None
            usage_tokens: int | None = None
            try:
                with client.stream("POST", url, headers=headers, json=body) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line.removeprefix("data: ").strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if choices and isinstance(choices[0], dict):
                            content = (choices[0].get("delta") or {}).get("content")
                            if content and first_content_at is None:
                                first_content_at = time.perf_counter()
                        usage = chunk.get("usage") or {}
                        if usage.get("completion_tokens") is not None:
                            usage_tokens = int(usage["completion_tokens"])
                ended = time.perf_counter()
                totals.append((ended - started) * 1000)
                if first_content_at is not None:
                    ttfts.append((first_content_at - started) * 1000)
                if usage_tokens is not None:
                    token_counts.append(usage_tokens)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
    total_summary = latency_summary(totals)
    ttft_summary = latency_summary(ttfts)
    return {
        "name": "External LLM stream",
        "kind": "llm_stream",
        "model": settings.llm_model,
        "runs": runs,
        "total": total_summary,
        "ttft": ttft_summary,
        "completion_tokens_mean": round(statistics.fmean(token_counts), 2) if token_counts else None,
        "errors": len(errors),
        "error_rate_pct": round(len(errors) / runs * 100, 2),
        "error_examples": errors[:3],
    }


def benchmark_chat_agent(
    backend_url: str,
    repo_id: str,
    *,
    name: str,
    message: str,
) -> dict[str, Any]:
    body = {"repo_id": repo_id, "message": message, "context": {}}
    started = time.perf_counter()
    error = None
    payload: dict[str, Any] | None = None
    try:
        with httpx.Client(timeout=300, trust_env=False) as client:
            response = client.post(backend_url + "/api/chat", json=body)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000
    usage = (payload or {}).get("model_usage") or []
    tool_calls = (payload or {}).get("tool_calls") or []
    return {
        "name": name,
        "latency_ms": round(elapsed_ms, 2),
        "route": (payload or {}).get("route"),
        "tool_calls": len(tool_calls),
        "tool_names": [str(item.get("tool_name") or "") for item in tool_calls],
        "citations": len((payload or {}).get("citations") or []),
        "model_calls": len(usage),
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in usage),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in usage),
        "error": error,
    }


def write_flat_csv(results: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for item in results.get("http_benchmarks", []):
        rows.append({"category": "http", **item})
    if results.get("database"):
        rows.append({"category": "database", **results["database"]})
    rag = results.get("rag") or {}
    for key in ("cold", "warm", "concurrent"):
        if rag.get(key):
            rows.append({"category": "rag", "scenario": key, **rag[key]})
    for key in ("chat_agent", "chat_agent_tool"):
        agent = results.get(key) or {}
        if agent:
            rows.append({"category": "chat_agent", "scenario": key, **agent})
    fieldnames = sorted({key for row in rows for key in row if not isinstance(row.get(key), (list, dict))})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if not isinstance(value, (list, dict))})


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a running DevFlow AI instance.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend-url", default="http://127.0.0.1:3000")
    parser.add_argument("--app-startup-ms", type=float)
    args = parser.parse_args()

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backend_pid, frontend_pid = process_state()

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "scope": {
            "backend_url": args.backend_url,
            "frontend_url": args.frontend_url,
            "dataset": "course-demo/devflow-sample-api",
            "destructive_operations": False,
        },
        "hardware": hardware_snapshot(),
        "runtime": {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "rerank_provider": settings.rerank_provider,
            "rerank_model": settings.local_rerank_model,
            "embedding_device": settings.local_embedding_device or "auto",
            "rerank_device": settings.local_rerank_device or "auto",
            "llm_model": settings.llm_model,
            "milvus_collection": settings.milvus_collection,
            "app_startup_ms": round(args.app_startup_ms, 2) if args.app_startup_ms is not None else None,
        },
        "docker_before": docker_stats(),
    }

    with httpx.Client(timeout=20, trust_env=False) as client:
        health = client.get(args.backend_url + "/health")
        health.raise_for_status()
        frontend = client.get(args.frontend_url + "/")
        frontend.raise_for_status()
        repos = client.get(args.backend_url + "/api/repos").json()
    if not repos:
        raise RuntimeError("No repository is available for RAG benchmarks")
    repo = repos[0]
    repo_id = str(repo["id"])
    results["scope"]["repo_id"] = repo_id
    results["scope"]["document_count"] = None

    with httpx.Client(timeout=30, trust_env=False) as client:
        rag_status = client.get(f"{args.backend_url}/api/rag/{repo_id}/status").json()
    results["scope"]["document_count"] = rag_status.get("document_count")
    results["scope"]["source_count"] = rag_status.get("source_count")
    results["milvus_direct"] = benchmark_milvus(repo_id)

    sampler = ResourceSampler(backend_pid, frontend_pid)
    sampler.start()
    try:
        results["http_benchmarks"] = [
            benchmark_sequential(
                "Frontend home",
                "GET",
                args.frontend_url + "/",
                runs=30,
                validate=lambda _payload: True,
            ),
            benchmark_sequential(
                "Backend health",
                "GET",
                args.backend_url + "/health",
                runs=50,
                validate=lambda payload: isinstance(payload, dict) and payload.get("status") == "ok",
            ),
            benchmark_sequential(
                "Repository list",
                "GET",
                args.backend_url + "/api/repos",
                runs=30,
                validate=lambda payload: isinstance(payload, list),
            ),
            benchmark_sequential(
                "RAG model catalog",
                "GET",
                args.backend_url + "/api/rag/models",
                runs=20,
                validate=lambda payload: bool((payload or {}).get("embedding_models")),
            ),
            asyncio.run(
                benchmark_concurrent(
                    "Backend health load",
                    "GET",
                    args.backend_url + "/health",
                    total=200,
                    concurrency=20,
                    validate=lambda payload: isinstance(payload, dict) and payload.get("status") == "ok",
                )
            ),
            asyncio.run(
                benchmark_concurrent(
                    "Frontend home load",
                    "GET",
                    args.frontend_url + "/",
                    total=100,
                    concurrency=10,
                )
            ),
        ]

        results["database"] = benchmark_database()

        rag_body = {
            "question": "JWT token expires 时应该返回什么状态码？",
            "top_k": 3,
            "source_types": [],
            "metadata_filters": {},
        }
        cold = benchmark_sequential(
            "RAG hybrid search cold",
            "POST",
            f"{args.backend_url}/api/rag/{repo_id}/search",
            runs=1,
            warmups=0,
            json_body=rag_body,
            timeout=240,
            validate=lambda payload: isinstance(payload, dict) and payload.get("result_count", 0) >= 1,
        )
        warm = benchmark_sequential(
            "RAG hybrid search warm",
            "POST",
            f"{args.backend_url}/api/rag/{repo_id}/search",
            runs=5,
            warmups=1,
            json_body=rag_body,
            timeout=120,
            validate=lambda payload: isinstance(payload, dict) and payload.get("result_count", 0) >= 1,
        )
        concurrent = asyncio.run(
            benchmark_concurrent(
                "RAG hybrid search load",
                "POST",
                f"{args.backend_url}/api/rag/{repo_id}/search",
                total=4,
                concurrency=2,
                json_body=rag_body,
                timeout=240,
                validate=lambda payload: isinstance(payload, dict) and payload.get("result_count", 0) >= 1,
            )
        )
        results["rag"] = {"cold": cold, "warm": warm, "concurrent": concurrent}

        results["llm_stream"] = benchmark_llm_stream(runs=3)

        results["chat_agent"] = benchmark_chat_agent(
            args.backend_url,
            repo_id,
            name="ChatAgent direct issue analysis",
            message="列出当前打开的 Issue 编号和标题，并用一句话说明最高优先级项。",
        )
        results["chat_agent_tool"] = benchmark_chat_agent(
            args.backend_url,
            repo_id,
            name="ChatAgent forced RAG tool loop",
            message=(
                "性能验证：必须先调用 rag.search_similar_documents 工具检索 JWT token expires 证据，"
                "再基于工具结果回答；不要直接回答。"
            ),
        )
    finally:
        sampler.stop()

    results["resources"] = {
        "summary": sampler.summary(),
        "samples": sampler.rows,
    }
    results["docker_after"] = docker_stats()

    json_path = output_dir / "benchmark-results.json"
    csv_path = output_dir / "benchmark-summary.csv"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_flat_csv(results, csv_path)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
