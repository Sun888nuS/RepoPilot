from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RAGAS_DOCS_URL = "https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/"


def source(
    label: str,
    *,
    files: list[str],
    component_ids: list[str],
    definitions: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "provider": "Local fixed evaluation suite",
        "files": files,
        "links": [{"title": "Ragas available metrics", "url": RAGAS_DOCS_URL}],
        "metricDefinitions": [
            {**definition, "componentIds": component_ids}
            for definition in definitions
        ],
        "evidenceFlow": [
            {
                "title": "Fixed gold suite",
                "detail": "Every reported task-quality metric is calculated from the saved case-level outputs and a versioned suite hash. Synthetic labels and observed model outputs remain separate.",
            },
            {"title": "Method", "detail": method},
            {
                "title": "Limits",
                "detail": "The suite is intentionally small and synthetic. Percentages characterize these exact cases, not production traffic, hiring outcomes, or unseen repositories.",
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--ragas", type=Path, required=True)
    parser.add_argument("--pre-retry", type=Path, required=True)
    parser.add_argument("--pre-policy", type=Path, required=True)
    parser.add_argument("--basic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    llm = json.loads(args.llm.read_text(encoding="utf-8"))
    ragas_payload = json.loads(args.ragas.read_text(encoding="utf-8"))
    ragas = ragas_payload.get("result") or ragas_payload
    pre_retry = json.loads(args.pre_retry.read_text(encoding="utf-8"))
    pre_policy = json.loads(args.pre_policy.read_text(encoding="utf-8"))
    basic = json.loads(args.basic.read_text(encoding="utf-8"))
    bm = {section: baseline[section]["metrics"] for section in ["issue", "pr", "ci", "workflow", "evidence", "memory"]}
    lm = {section: llm[section]["metrics"] for section in ["issue", "pr", "ci", "workflow", "evidence", "memory"]}
    rm = ragas["summary_metrics"]

    summary = {
        "issueCategoryAccuracy": lm["issue"]["category_accuracy"],
        "issueMacroF1": lm["issue"]["category_macro_f1"],
        "prBlockerRecall": lm["pr"]["blocker_recall"],
        "prCleanFalsePositiveRate": lm["pr"]["clean_case_false_positive_rate"],
        "ciRootCauseAccuracy": lm["ci"]["root_cause_accuracy"],
        "workflowDecisionAccuracy": lm["workflow"]["decision_accuracy"],
        "ragasFaithfulness": rm["faithfulness"],
        "structuredResponseSuccess": llm["llm_calls"]["success_rate"],
        "schedulerReductionPct": llm["scheduler"]["wall_time_reduction_pct"],
        "goldCaseCount": 64,
        "ragasCaseCount": 5,
        "suiteSha256": llm["evaluation"]["suite_sha256"],
        "ragasSuiteSha256": ragas["evaluation_set"]["sha256"],
    }

    quality_comparison = [
        {"metric": "Issue category accuracy", "baselineRate": bm["issue"]["category_accuracy"], "fullSystemRate": lm["issue"]["category_accuracy"], "sampleCount": lm["issue"]["case_count"]},
        {"metric": "Issue priority accuracy", "baselineRate": bm["issue"]["priority_accuracy"], "fullSystemRate": lm["issue"]["priority_accuracy"], "sampleCount": lm["issue"]["case_count"]},
        {"metric": "Issue conclusion accuracy", "baselineRate": bm["issue"]["conclusion_accuracy"], "fullSystemRate": lm["issue"]["conclusion_accuracy"], "sampleCount": lm["issue"]["case_count"]},
        {"metric": "PR blocker recall", "baselineRate": bm["pr"]["blocker_recall"], "fullSystemRate": lm["pr"]["blocker_recall"], "sampleCount": lm["pr"]["case_count"]},
        {"metric": "PR merge decision accuracy", "baselineRate": bm["pr"]["merge_recommendation_accuracy"], "fullSystemRate": lm["pr"]["merge_recommendation_accuracy"], "sampleCount": lm["pr"]["case_count"]},
        {"metric": "CI root-cause accuracy", "baselineRate": bm["ci"]["root_cause_accuracy"], "fullSystemRate": lm["ci"]["root_cause_accuracy"], "sampleCount": lm["ci"]["case_count"]},
        {"metric": "Workflow decision accuracy", "baselineRate": bm["workflow"]["decision_accuracy"], "fullSystemRate": lm["workflow"]["decision_accuracy"], "sampleCount": lm["workflow"]["case_count"]},
    ]

    pr_quality = [
        {"metric": "P1 blocker recall", "value": lm["pr"]["p1_recall"], "sampleCount": 6, "higherIsBetter": True},
        {"metric": "P2 blocker recall", "value": lm["pr"]["p2_recall"], "sampleCount": 5, "higherIsBetter": True},
        {"metric": "All blocker recall", "value": lm["pr"]["blocker_recall"], "sampleCount": 11, "higherIsBetter": True},
        {"metric": "Severity accuracy", "value": lm["pr"]["matched_severity_accuracy"], "sampleCount": 10, "higherIsBetter": True},
        {"metric": "Evidence grounding", "value": lm["pr"]["matched_evidence_accuracy"], "sampleCount": 10, "higherIsBetter": True},
        {"metric": "Clean-case specificity", "value": 1 - lm["pr"]["clean_case_false_positive_rate"], "sampleCount": 4, "higherIsBetter": True},
        {"metric": "Merge decision accuracy", "value": lm["pr"]["merge_recommendation_accuracy"], "sampleCount": 15, "higherIsBetter": True},
    ]

    thresholds = {
        "Context Precision": 0.75,
        "Context Recall": 0.80,
        "Faithfulness": 0.90,
        "Positive Answer Relevancy": 0.80,
        "Agent Goal Accuracy": 0.80,
        "Hard-rule pass rate": 1.00,
    }
    ragas_rows = [
        {"metric": "Context Precision", "score": rm["context_precision"], "threshold": thresholds["Context Precision"], "scoredCases": 5},
        {"metric": "Context Recall", "score": rm["context_recall"], "threshold": thresholds["Context Recall"], "scoredCases": 5},
        {"metric": "Faithfulness", "score": rm["faithfulness"], "threshold": thresholds["Faithfulness"], "scoredCases": 5},
        {"metric": "Positive Answer Relevancy", "score": rm["answer_relevancy"], "threshold": thresholds["Positive Answer Relevancy"], "scoredCases": 4},
        {"metric": "Agent Goal Accuracy", "score": rm["agent_goal_accuracy"], "threshold": thresholds["Agent Goal Accuracy"], "scoredCases": 5},
        {"metric": "Hard-rule pass rate", "score": rm["hard_gate_pass_rate"], "threshold": thresholds["Hard-rule pass rate"], "scoredCases": 5},
    ]

    evidence_memory = [
        {"metric": "Evidence Hit@5", "value": lm["evidence"]["hit_at_5"], "sampleCount": 10},
        {"metric": "Top-1 source accuracy", "value": lm["evidence"]["top1_source_accuracy"], "sampleCount": 10},
        {"metric": "MRR", "value": lm["evidence"]["mrr"], "sampleCount": 10},
        {"metric": "Negative abstention pass rate", "value": 1 - lm["evidence"]["unsupported_answer_gate_rate"], "sampleCount": 2},
        {"metric": "Approved-memory recall", "value": lm["memory"]["approved_memory_recall"], "sampleCount": 2},
        {"metric": "Memory-policy case accuracy", "value": lm["memory"]["policy_case_accuracy"], "sampleCount": 5},
    ]

    reliability = [
        {"metric": "Structured JSON success", "beforeRate": pre_retry["llm_calls"]["success_rate"], "afterRate": llm["llm_calls"]["success_rate"], "changePp": round((llm["llm_calls"]["success_rate"] - pre_retry["llm_calls"]["success_rate"]) * 100, 2)},
        {"metric": "Issue category macro-F1", "beforeRate": pre_policy["issue"]["metrics"]["category_macro_f1"], "afterRate": lm["issue"]["category_macro_f1"], "changePp": round((lm["issue"]["category_macro_f1"] - pre_policy["issue"]["metrics"]["category_macro_f1"]) * 100, 2)},
        {"metric": "Negative abstention pass", "beforeRate": 1 - pre_policy["evidence"]["metrics"]["unsupported_answer_gate_rate"], "afterRate": 1 - lm["evidence"]["unsupported_answer_gate_rate"], "changePp": round((pre_policy["evidence"]["metrics"]["unsupported_answer_gate_rate"] - lm["evidence"]["unsupported_answer_gate_rate"]) * 100, 2)},
        {"metric": "Memory-policy accuracy", "beforeRate": pre_policy["memory"]["metrics"]["policy_case_accuracy"], "afterRate": lm["memory"]["policy_case_accuracy"], "changePp": round((lm["memory"]["policy_case_accuracy"] - pre_policy["memory"]["metrics"]["policy_case_accuracy"]) * 100, 2)},
    ]

    latency_rows = [
        {"stage": "Issue Agent", "medianSeconds": round(lm["issue"]["median_latency_ms"] / 1000, 2), "samples": 14},
        {"stage": "PR Review Agent", "medianSeconds": round(lm["pr"]["median_latency_ms"] / 1000, 2), "samples": 15},
        {"stage": "CI Debug Agent", "medianSeconds": round(lm["ci"]["median_latency_ms"] / 1000, 2), "samples": 10},
        {"stage": "Evidence retrieval", "medianSeconds": round(lm["evidence"]["median_latency_ms"] / 1000, 2), "samples": 12},
        {"stage": "Ragas full case", "medianSeconds": round(ragas["performance"]["avg_latency_ms"] / 1000, 2), "samples": 5},
    ]

    coverage = [
        {"area": "Issue triage", "cases": 14, "notes": "7 categories; priority and conclusion labels"},
        {"area": "PR review", "cases": 15, "notes": "11 blockers and 4 clean PRs"},
        {"area": "CI diagnosis", "cases": 10, "notes": "test/build/lint/dependency/permission/environment/unknown"},
        {"area": "Cross-agent gates", "cases": 8, "notes": "conflict, evidence, confidence, safety, task errors"},
        {"area": "Evidence retrieval", "cases": 12, "notes": "10 positive and 2 negative"},
        {"area": "Memory policy", "cases": 5, "notes": "2 approved and 3 negative controls"},
        {"area": "Fixed Ragas", "cases": 5, "notes": "4 answerable and 1 abstention"},
    ]

    claims = [
        {"claim": "Issue 分类准确率/宏 F1 100%", "status": "可写，必须带 14 例平衡金标集", "basis": "14/14; seven classes"},
        {"claim": "PR P1/P2 阻塞召回 90.9%", "status": "可写，建议同时写干净样本误报 25%", "basis": "10/11 blockers; 1/4 clean cases false positive"},
        {"claim": "CI 根因定位准确率 100%", "status": "可写，必须带 10 例", "basis": "10/10 failure-type, root, and file strict pass"},
        {"claim": "跨 Agent 最终决策准确率 100%", "status": "可写为规则金标场景，不可外推生产", "basis": "8/8 deterministic conflict/gate scenarios"},
        {"claim": "处理耗时降低 66.4%", "status": "仅可写独立任务调度微基准，不可写人工效率", "basis": "3 independent 120ms tasks; quality gate equal"},
        {"claim": "可追溯 RAG", "status": "可写", "basis": "Hit@5/Top-1 100%; Ragas CP .95, CR/Faithfulness 1.0"},
        {"claim": "记忆受审批约束", "status": "可写，带 2 正例/3 负控", "basis": "approved recall 100%; misuse 0"},
        {"claim": "相较人工处理耗时降低 X%", "status": "不可写，缺少人工同题基线", "basis": "human_baseline_template.csv is empty"},
    ]

    common_files = [
        "docs/datas/resume_evals/quality_cases.json",
        ".data/resume-evals/2026-10-07-final-rescored/resume-value-llm-rescored.json",
        ".data/resume-evals/2026-10-07-final-rescored/resume-value-heuristic-rescored.json",
    ]
    snapshot = {
        "title": "质量指标已能支撑简历，但 PR 误报和人工基线仍需如实披露",
        "surface": "report",
        "status": "observed",
        "generatedAt": "2026-10-07T14:10:00+08:00",
        "filters": [],
        "report": {"asOf": "2026-10-07", "originalQuestion": "哪些测试数据最能增加 DevFlow AI 的简历价值？"},
        "queries": {
            "summary": {
                "label": "Resume-value evaluation summary",
                "rows": [summary],
                "source": source(
                    "DevFlow resume-value fixed suite",
                    files=common_files,
                    component_ids=["evaluation-summary", "headline-kpis", "resume-wording"],
                    definitions=[
                        {"field": "prBlockerRecall", "label": "PR blocker recall", "definition": "Matched gold P1/P2 blockers divided by 11 labeled blockers."},
                        {"field": "structuredResponseSuccess", "label": "Structured response success", "definition": "Non-empty JSON responses from 39 LLM calls after one configured retry."},
                    ],
                    method="Case-level outputs were scored with deterministic label matchers; suite hashes are retained in the row.",
                ),
            },
            "quality-comparison": {
                "label": "Heuristic baseline versus LLM-enhanced agents",
                "rows": quality_comparison,
                "source": source(
                    "Same-suite baseline comparison",
                    files=common_files,
                    component_ids=["quality-comparison-chart", "quality-comparison-table"],
                    definitions=[
                        {"field": "baselineRate", "label": "Heuristic baseline", "definition": "Score from the deterministic fallback using the same case IDs and labels."},
                        {"field": "fullSystemRate", "label": "LLM-enhanced system", "definition": "Score from production agents using configured LLM plus validated fallback."},
                    ],
                    method="Rates use the same fixed suite SHA-256 and preserve case-level outputs.",
                ),
            },
            "pr-quality": {
                "label": "PR blocker quality",
                "rows": pr_quality,
                "source": source(
                    "PR case-level scoring",
                    files=common_files,
                    component_ids=["pr-quality-chart", "pr-quality-table"],
                    definitions=[
                        {"field": "value", "label": "Rate", "definition": "Metric-specific success rate on the labeled PR subset."},
                        {"field": "sampleCount", "label": "Denominator", "definition": "Gold blockers, matched findings, clean PRs, or all PRs depending on the metric."},
                    ],
                    method="Blocker detection and evidence grounding are scored separately. Clean-case false positives use PRs with no gold P1/P2 findings.",
                ),
            },
            "ragas-quality": {
                "label": "Fixed Ragas quality gate",
                "rows": ragas_rows,
                "source": source(
                    "Ragas fixed baseline",
                    files=["docs/datas/evals/rag_cases.json", ".data/resume-evals/2026-10-07-production-final/ragas-result-final-reaggregated.json"],
                    component_ids=["ragas-quality-chart", "ragas-quality-table"],
                    definitions=[
                        {"field": "score", "label": "Observed score", "definition": "Saved Ragas or deterministic hard-rule aggregate."},
                        {"field": "threshold", "label": "Configured threshold", "definition": "Acceptance threshold from the DevFlow evaluation configuration."},
                    ],
                    method="Negative abstention cases are excluded only from Answer Relevancy aggregation; their hard rules, Faithfulness, and Agent Goal remain scored. No Judge rerun was used for reaggregation.",
                ),
            },
            "evidence-memory": {
                "label": "Evidence and memory controls",
                "rows": evidence_memory,
                "source": source(
                    "Evidence and memory cases",
                    files=common_files,
                    component_ids=["evidence-memory-chart", "evidence-memory-table"],
                    definitions=[
                        {"field": "value", "label": "Pass rate", "definition": "Observed success rate on the stated positive or negative-control denominator."},
                    ],
                    method="Evidence metrics use ten positive questions and two negatives. Memory metrics use two approved memories plus pending, rejected, and stale controls.",
                ),
            },
            "reliability": {
                "label": "Regression-driven improvements",
                "rows": reliability,
                "source": source(
                    "Before and after fixed regression cases",
                    files=[*common_files, ".data/resume-evals/2026-10-07-final-llm/resume-value-llm.json", ".data/resume-evals/2026-10-07/resume-value-heuristic.json"],
                    component_ids=["reliability-chart", "reliability-table"],
                    definitions=[
                        {"field": "beforeRate", "label": "Before", "definition": "Saved baseline before the corresponding policy, parsing, or routing repair."},
                        {"field": "afterRate", "label": "After", "definition": "Saved result after the repair on the applicable fixed cases."},
                        {"field": "changePp", "label": "Change", "definition": "After minus before, expressed in percentage points."},
                    ],
                    method="Only preserved before/after runs are shown; PR recall is omitted because its labels changed during human review.",
                ),
            },
            "latency": {
                "label": "Observed evaluation latency",
                "rows": latency_rows,
                "source": source(
                    "Production evaluation timing",
                    files=[common_files[1], ".data/resume-evals/2026-10-07-production-final/ragas-result-final-reaggregated.json"],
                    component_ids=["latency-chart", "efficiency-caveat"],
                    definitions=[
                        {"field": "medianSeconds", "label": "Median latency", "definition": "Median case latency for specialty agents; Ragas uses average full-case latency because only aggregate performance was saved."},
                    ],
                    method="External LLM and Judge latency are included. This is not a human-time comparison.",
                ),
            },
            "coverage": {
                "label": "Gold suite coverage",
                "rows": coverage,
                "source": source(
                    "Versioned evaluation manifests",
                    files=["docs/datas/resume_evals/quality_cases.json", "docs/datas/evals/rag_cases.json"],
                    component_ids=["coverage-table", "methods-scope"],
                    definitions=[
                        {"field": "cases", "label": "Cases", "definition": "Number of fixed labeled cases in the area."},
                    ],
                    method="Counts are read from the versioned suite files; the two suites contain 64 specialty cases and five Ragas cases.",
                ),
            },
            "resume-claims": {
                "label": "Resume claim audit",
                "rows": claims,
                "source": source(
                    "Resume claim evidence audit",
                    files=[*common_files, "docs/datas/resume_evals/resume_project_rewrite.md", "docs/datas/resume_evals/human_baseline_template.csv"],
                    component_ids=["resume-claims-table", "resume-wording", "methods-scope"],
                    definitions=[
                        {"field": "status", "label": "Use status", "definition": "Whether the current evidence supports the exact claim and which qualification must accompany it."},
                    ],
                    method="Claims follow resume-craft evidence rules: sample size, denominator, comparison baseline, unit, and ownership must remain explicit.",
                ),
            },
            "skill-runtime": {
                "label": "Skill runtime checks",
                "rows": [{
                    "registeredSkillCoverage": basic["skill_registry_coverage"],
                    "entrypointAccuracy": basic["skill_entrypoint_accuracy"],
                    "toolSchemaReady": basic["tool_call_accuracy"],
                    "jsonSchemaPassRate": basic["json_schema_pass_rate"],
                    "skillCount": basic["cases"]["skills"],
                }],
                "source": source(
                    "Built-in basic eval",
                    files=[".data/resume-evals/2026-10-07-production-final/basic-eval.json"],
                    component_ids=["skill-runtime-summary", "resume-wording"],
                    definitions=[
                        {"field": "entrypointAccuracy", "label": "Skill entrypoint accuracy", "definition": "Registered entrypoint resolved to the expected Skill for all six built-in Skills."},
                    ],
                    method="Deterministic registry and schema checks; no external LLM judge.",
                ),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "queries": len(snapshot["queries"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
