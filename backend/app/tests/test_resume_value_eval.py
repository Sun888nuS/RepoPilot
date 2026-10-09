import pytest

from app.services.evals.resume_value_runner import macro_f1, score_pr_rows


def test_macro_f1_penalizes_missing_and_extra_classes() -> None:
    score, by_label = macro_f1(
        ["Bug", "Bug", "Feature", "Ops"],
        ["Bug", "Feature", "Feature", "Bug"],
    )

    assert score < 1.0
    assert by_label["Feature"] > by_label["Ops"]


def test_macro_f1_is_perfect_for_exact_predictions() -> None:
    score, by_label = macro_f1(["Bug", "Feature", "Ops"], ["Bug", "Feature", "Ops"])

    assert score == 1.0
    assert all(value == 1.0 for value in by_label.values())


def test_pr_scoring_separates_detection_from_evidence_grounding() -> None:
    cases = [
        {
            "id": "pr-1",
            "expected_findings": [
                {"id": "timezone", "severity": "P2", "match_any": ["时区"], "evidence_any": ["report.py"]}
            ],
            "expected_merge": ["修改后合入"],
        }
    ]
    rows = [
        {
            "id": "pr-1",
            "latency_ms": 10,
            "result": {
                "merge_recommendation": "修改后合入",
                "review_findings": [
                    {
                        "severity": "P2",
                        "title": "时区边界未覆盖",
                        "evidence": "缺少边界测试",
                        "required_action": "补测试",
                        "blocking": True,
                    }
                ],
            },
        }
    ]

    scored = score_pr_rows(cases, rows)

    assert scored["metrics"]["blocker_recall"] == 1.0
    assert scored["metrics"]["matched_evidence_accuracy"] == 0.0
