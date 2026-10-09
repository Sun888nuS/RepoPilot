from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.evals.resume_value_runner import score_pr_rows  # noqa: E402
from run_resume_value_eval import flatten_metrics, markdown_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore saved DevFlow eval outputs with revised human labels.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    suite_raw = args.suite.read_bytes()
    suite = json.loads(suite_raw.decode("utf-8"))
    expected_ids = [case["id"] for case in suite["pr_cases"]]
    result_ids = [case["id"] for case in result["pr"]["cases"]]
    if expected_ids != result_ids:
        raise ValueError("Saved PR outputs do not match the revised suite case IDs/order")

    result["pr"] = score_pr_rows(suite["pr_cases"], result["pr"]["cases"])
    old_sha = result["evaluation"].get("suite_sha256")
    new_sha = hashlib.sha256(suite_raw).hexdigest()
    result["evaluation"]["suite_sha256"] = new_sha
    result["evaluation"]["suite_path"] = str(args.suite.resolve())
    result["evaluation"]["rescored_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    result["evaluation"]["rescore_of"] = str(args.result.resolve())
    result["evaluation"]["previous_suite_sha256"] = old_sha

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = result["evaluation"]["mode"]
    json_path = output_dir / f"resume-value-{mode}-rescored.json"
    csv_path = output_dir / f"resume-value-{mode}-rescored-metrics.csv"
    markdown_path = output_dir / f"resume-value-{mode}-rescored.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(flatten_metrics(result))
    markdown_path.write_text(markdown_summary(result, new_sha), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
