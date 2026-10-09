from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.evals.ragas_runner import (  # noqa: E402
    REQUIRED_RAGAS_METRICS,
    _diagnostic_summary,
    _metric_details,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reaggregate saved Ragas scores without rerunning ChatAgent or Judge.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = payload.get("result") or payload
    rows = result.get("cases") or []
    metric_rows = _metric_details(rows)
    hard_passed_cases = sum(1 for row in rows if (row.get("hard_rules") or {}).get("passed"))
    hard_passed = bool(rows) and hard_passed_cases == len(rows)
    hard_pass_rate = round(hard_passed_cases / len(rows), 4) if rows else 0.0
    evaluation_complete = bool(result.get("evaluation_complete"))
    gated_names = list(REQUIRED_RAGAS_METRICS)
    gated_names.extend(
        name
        for name in ("tool_call_accuracy", "tool_call_f1")
        if metric_rows[name]["scored_cases"] > 0
    )
    semantic_failed = any(metric_rows[name]["passed"] is False for name in gated_names)
    status = "passed" if rows and hard_passed and evaluation_complete and not semantic_failed else "failed"
    summary_metrics = {
        name: details["score"]
        for name, details in metric_rows.items()
        if details["score"] is not None
    }
    retrieval = result.get("retrieval") or {}
    summary_metrics.update(
        {
            "hard_gate_pass_rate": hard_pass_rate,
            "retrieval_recall_at_k": retrieval.get("recall_at_k", 0.0),
            "retrieval_precision_at_k": retrieval.get("precision_at_k", 0.0),
        }
    )
    for key in list(result.get("summary_metrics") or {}):
        result.pop(key, None)
    result.update(summary_metrics)
    result["summary_metrics"] = summary_metrics
    result["status"] = status
    result["quality_status"] = status
    result["passed"] = status == "passed"
    result["hard_gate"] = {
        "passed": hard_passed,
        "pass_rate": hard_pass_rate,
        "passed_cases": hard_passed_cases,
        "total_cases": len(rows),
    }
    result.setdefault("ragas", {})["metrics"] = metric_rows
    result["ragas"]["passed"] = status == "passed" if evaluation_complete else None
    result["diagnostics"] = _diagnostic_summary(rows)
    result["reaggregation"] = {
        "source": str(args.input.resolve()),
        "reaggregated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "change": "answer_relevancy excludes negative abstention cases; their hard rules, faithfulness, and agent goal remain scored",
        "judge_rerun": False,
    }
    if "result" in payload:
        payload["status"] = status
        payload["result"] = result
    else:
        payload = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": status, "summary_metrics": summary_metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
