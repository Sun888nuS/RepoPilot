from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings  # noqa: E402
from app.services.rag.embeddings import embed_texts  # noqa: E402
from app.services.rag.rerank import local_rerank_documents  # noqa: E402


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "samples": len(values),
        "mean_ms": round(statistics.fmean(values), 2),
        "p50_ms": round(percentile(values, 0.50), 2),
        "p95_ms": round(percentile(values, 0.95), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


def timed(call: Callable[[], Any]) -> tuple[float, Any]:
    started = time.perf_counter()
    result = call()
    return (time.perf_counter() - started) * 1000, result


def benchmark_embedding() -> dict[str, Any]:
    single_text = "DevFlow AI 使用本地 BGE 模型完成代码、Issue、PR 和知识库的语义检索。"
    batch = [f"性能测试文档 {index}：JWT 过期时接口应该返回 401。" for index in range(8)]
    cold_ms, first = timed(lambda: embed_texts([single_text]))
    warm_single = [timed(lambda: embed_texts([single_text]))[0] for _ in range(5)]
    warm_batch = [timed(lambda: embed_texts(batch))[0] for _ in range(3)]
    process = psutil.Process()
    return {
        "component": "embedding",
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
        "device": settings.local_embedding_device or "auto (resolved to CPU in this environment)",
        "cold_first_request_ms": round(cold_ms, 2),
        "warm_single": summary(warm_single),
        "warm_batch_8": {
            **summary(warm_batch),
            "texts_per_second": round(8 / (statistics.fmean(warm_batch) / 1000), 2),
        },
        "dimensions": len(first[0]),
        "process_rss_mb": round(process.memory_info().rss / 1024 / 1024, 2),
    }


def candidates(count: int = 8) -> list[dict[str, Any]]:
    return [
        {
            "id": str(index),
            "title": "JWT token expiry" if index == 0 else f"Unrelated note {index}",
            "_raw_content": (
                "JWT 过期时后端应返回 401 Unauthorized，而不是 500。"
                if index == 0
                else f"这是第 {index} 条与前端样式或构建流程有关的记录。"
            ),
            "metadata": {},
            "retrieval": {},
        }
        for index in range(count)
    ]


def benchmark_reranker() -> dict[str, Any]:
    items = candidates()
    query = "JWT token 过期应该返回什么状态码？"
    cold_ms, first = timed(lambda: local_rerank_documents(query, items, len(items)))
    warm = [timed(lambda: local_rerank_documents(query, items, len(items)))[0] for _ in range(5)]
    process = psutil.Process()
    return {
        "component": "reranker",
        "provider": settings.rerank_provider,
        "model": settings.local_rerank_model,
        "device": settings.local_rerank_device or "auto (resolved to CPU in this environment)",
        "candidates": len(items),
        "cold_first_request_ms": round(cold_ms, 2),
        "warm_batch_8": {
            **summary(warm),
            "candidate_pairs_per_second": round(8 / (statistics.fmean(warm) / 1000), 2),
        },
        "top_result_id": first[0]["id"],
        "top_score": first[0]["rerank_score"],
        "process_rss_mb": round(process.memory_info().rss / 1024 / 1024, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["embedding", "reranker"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark_embedding() if args.component == "embedding" else benchmark_reranker()
    result["generated_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
