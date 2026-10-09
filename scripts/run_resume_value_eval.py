from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.session import SessionLocal  # noqa: E402
from app.services.evals.resume_value_runner import load_resume_value_suite, run_resume_value_eval  # noqa: E402


def flatten_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for section in ("issue", "pr", "ci", "workflow", "evidence", "memory"):
        for metric, value in (result.get(section) or {}).get("metrics", {}).items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                rows.append({"section": section, "metric": metric, "value": value})
    for metric, value in (result.get("scheduler") or {}).items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            rows.append({"section": "scheduler", "metric": metric, "value": value})
    for metric, value in (result.get("llm_calls") or {}).items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            rows.append({"section": "llm_calls", "metric": metric, "value": value})
    return rows


def markdown_summary(result: dict[str, Any], suite_sha256: str) -> str:
    issue = result["issue"]["metrics"]
    pr = result["pr"]["metrics"]
    ci = result["ci"]["metrics"]
    workflow = result["workflow"]["metrics"]
    evidence = result["evidence"]["metrics"]
    memory = result["memory"]["metrics"]
    scheduler = result["scheduler"]
    llm_calls = result.get("llm_calls") or {}
    return "\n".join(
        [
            "# DevFlow AI Resume-Value Eval",
            "",
            f"- Mode: {result['evaluation']['mode']}",
            f"- Suite SHA-256: `{suite_sha256}`",
            f"- Issue category accuracy / macro-F1: {issue['category_accuracy']:.2%} / {issue['category_macro_f1']:.2%}",
            f"- Issue priority / conclusion accuracy: {issue['priority_accuracy']:.2%} / {issue['conclusion_accuracy']:.2%}",
            f"- PR blocker recall (P1 / P2): {pr['blocker_recall']:.2%} ({pr['p1_recall']:.2%} / {pr['p2_recall']:.2%})",
            f"- PR clean-case false-positive rate: {pr['clean_case_false_positive_rate']:.2%}",
            f"- CI root-cause accuracy: {ci['root_cause_accuracy']:.2%}",
            f"- Workflow final-decision accuracy: {workflow['decision_accuracy']:.2%}",
            f"- Evidence Hit@5 / top-1 source accuracy: {evidence['hit_at_5']:.2%} / {evidence['top1_source_accuracy']:.2%}",
            f"- Unsupported negative answer-gate rate: {evidence['unsupported_answer_gate_rate']:.2%}",
            f"- Approved-memory recall: {memory['approved_memory_recall']:.2%}",
            f"- Unapproved / stale memory misuse: {memory['unapproved_memory_misuse_count']} / {memory['stale_memory_misuse_count']}",
            f"- Parallel scheduler wall-time reduction: {scheduler['wall_time_reduction_pct']:.2f}%",
            *(
                [f"- LLM structured response success: {llm_calls['nonempty_responses']}/{llm_calls['calls']} ({llm_calls['success_rate']:.2%})"]
                if llm_calls
                else []
            ),
            "",
            "Only metrics from a completed, named suite with its hash should be copied into a resume.",
        ]
    ) + "\n"


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run the DevFlow resume-value quality benchmark.")
    parser.add_argument(
        "--suite",
        type=Path,
        default=PROJECT_ROOT / "docs" / "datas" / "resume_evals" / "quality_cases.json",
    )
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--llm-concurrency", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload, suite_sha256 = load_resume_value_suite(args.suite)
    db = SessionLocal()
    try:
        result = await run_resume_value_eval(
            db,
            payload,
            mode=args.mode,
            llm_concurrency=max(1, args.llm_concurrency),
        )
        result["evaluation"]["suite_sha256"] = suite_sha256
        result["evaluation"]["suite_path"] = str(args.suite.resolve())
    finally:
        db.close()

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"resume-value-{args.mode}.json"
    csv_path = output_dir / f"resume-value-{args.mode}-metrics.csv"
    markdown_path = output_dir / f"resume-value-{args.mode}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    rows = flatten_metrics(result)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(markdown_summary(result, suite_sha256), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(async_main())
