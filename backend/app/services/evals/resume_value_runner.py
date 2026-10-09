from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Document, EvidenceItem, MemoryCandidate, RecallEvent, Repository
from app.schemas.workflow import WorkflowClaim, WorkflowSpec, WorkflowTask, WorkflowTaskResult
from app.services.agents.ci_debug_agent import CIDebugAgent
from app.services.agents.issue_agent import IssueAgent
from app.services.agents.observer_agent import ObserverAgent
from app.services.agents.pr_review_agent import PRReviewAgent
from app.services.agents.synthesis_agent import SynthesisAgent
from app.services.agents.workflow_orchestrator import WorkflowOrchestrator
from app.services.chat_memory import EvidenceStore, MemoryTools
from app.services.llm.client import LLMClient
from app.services.memory_hub import approve_memory_candidate
from app.services.rag.knowledge_base import get_or_create_config, retrieve_knowledge
from app.services.rag.vector_store import DocumentPayload, add_documents, delete_milvus_documents


class NoopLLM:
    async def chat_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        return {}


class TrackingLLM:
    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.calls = 0
        self.nonempty_responses = 0
        self.empty_responses = 0
        self.errors: list[str] = []
        self.latencies_ms: list[float] = []

    async def chat_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        started = time.perf_counter()
        try:
            result = await self.client.chat_json(system_prompt, user_payload)
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            result = {}
        self.latencies_ms.append((time.perf_counter() - started) * 1000)
        if result:
            self.nonempty_responses += 1
        else:
            self.empty_responses += 1
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "nonempty_responses": self.nonempty_responses,
            "empty_responses": self.empty_responses,
            "success_rate": _rate(self.nonempty_responses, self.calls),
            "median_latency_ms": _median(self.latencies_ms),
            "errors": self.errors[:5],
        }


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 2) if values else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _text(value: Any) -> str:
    return str(value or "").lower()


def macro_f1(expected: list[str], actual: list[str]) -> tuple[float, dict[str, float]]:
    labels = sorted(set(expected) | set(actual))
    by_label: dict[str, float] = {}
    for label in labels:
        tp = sum(e == label and a == label for e, a in zip(expected, actual))
        fp = sum(e != label and a == label for e, a in zip(expected, actual))
        fn = sum(e == label and a != label for e, a in zip(expected, actual))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        by_label[label] = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    return _mean(list(by_label.values())), by_label


async def _run_with_limit(coroutines: list[Any], concurrency: int) -> list[Any]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(coroutine: Any) -> Any:
        async with semaphore:
            return await coroutine

    return await asyncio.gather(*(one(coroutine) for coroutine in coroutines))


async def _timed_agent(agent: Any, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    result = await agent.run(case["input"], case.get("context") or {})
    return {
        "id": case["id"],
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "result": result,
    }


async def evaluate_issue_cases(cases: list[dict[str, Any]], llm: Any, concurrency: int) -> dict[str, Any]:
    rows = await _run_with_limit([_timed_agent(IssueAgent(llm=llm), case) for case in cases], concurrency)
    expected_categories = [case["expected"]["category"] for case in cases]
    actual_categories = [row["result"]["category"] for row in rows]
    category_macro_f1, f1_by_category = macro_f1(expected_categories, actual_categories)
    category_hits = 0
    priority_hits = 0
    conclusion_hits = 0
    for case, row in zip(cases, rows):
        result = row["result"]
        expected = case["expected"]
        row["expected"] = expected
        row["checks"] = {
            "category": result.get("category") == expected["category"],
            "priority": result.get("priority") == expected["priority"],
            "conclusion": result.get("conclusion") == expected["conclusion"],
        }
        category_hits += int(row["checks"]["category"])
        priority_hits += int(row["checks"]["priority"])
        conclusion_hits += int(row["checks"]["conclusion"])
    latencies = [row["latency_ms"] for row in rows]
    return {
        "metrics": {
            "case_count": len(cases),
            "category_accuracy": _rate(category_hits, len(cases)),
            "category_macro_f1": category_macro_f1,
            "category_f1": f1_by_category,
            "priority_accuracy": _rate(priority_hits, len(cases)),
            "conclusion_accuracy": _rate(conclusion_hits, len(cases)),
            "strict_case_accuracy": _rate(sum(all(row["checks"].values()) for row in rows), len(cases)),
            "median_latency_ms": _median(latencies),
        },
        "cases": rows,
    }


def _finding_text(finding: dict[str, Any]) -> str:
    return " ".join(
        _text(finding.get(key))
        for key in ("title", "evidence", "required_action")
    )


def _matches_gold(finding: dict[str, Any], gold: dict[str, Any]) -> bool:
    text = _finding_text(finding)
    return any(_text(term) in text for term in gold.get("match_any") or [])


def score_pr_rows(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_total = 0
    gold_by_severity = Counter()
    matched_total = 0
    matched_by_severity = Counter()
    severity_hits = 0
    evidence_hits = 0
    predicted_total = 0
    false_positive_findings = 0
    clean_cases = 0
    clean_false_positive_cases = 0
    merge_hits = 0
    for case, row in zip(cases, rows):
        result = row["result"]
        gold = case.get("expected_findings") or []
        predictions = [
            finding
            for finding in result.get("review_findings") or []
            if finding.get("severity") in {"P1", "P2"} or finding.get("blocking") is True
        ]
        gold_total += len(gold)
        gold_by_severity.update(item["severity"] for item in gold)
        predicted_total += len(predictions)
        unmatched_predictions = set(range(len(predictions)))
        matches: list[dict[str, Any]] = []
        for expected in gold:
            matched_index = next(
                (index for index in unmatched_predictions if _matches_gold(predictions[index], expected)),
                None,
            )
            if matched_index is None:
                matches.append({"gold_id": expected["id"], "matched": False, "predicted_severity": None})
                continue
            finding = predictions[matched_index]
            unmatched_predictions.remove(matched_index)
            matched_total += 1
            matched_by_severity[expected["severity"]] += 1
            severity_hit = finding.get("severity") == expected["severity"]
            severity_hits += int(severity_hit)
            evidence_terms = expected.get("evidence_any") or []
            evidence_hit = not evidence_terms or any(_text(term) in _finding_text(finding) for term in evidence_terms)
            evidence_hits += int(evidence_hit)
            matches.append(
                {
                    "gold_id": expected["id"],
                    "matched": True,
                    "expected_severity": expected["severity"],
                    "predicted_severity": finding.get("severity"),
                    "severity_correct": severity_hit,
                    "evidence_grounded": evidence_hit,
                }
            )
        false_positive_findings += len(unmatched_predictions)
        if not gold:
            clean_cases += 1
            clean_false_positive_cases += int(bool(predictions))
        merge_ok = result.get("merge_recommendation") in set(case.get("expected_merge") or [])
        merge_hits += int(merge_ok)
        row["gold_findings"] = gold
        row["finding_matches"] = matches
        row["predicted_blocker_count"] = len(predictions)
        row["unmatched_predicted_blockers"] = [predictions[index] for index in sorted(unmatched_predictions)]
        row["merge_recommendation_correct"] = merge_ok
    latencies = [row["latency_ms"] for row in rows]
    return {
        "metrics": {
            "case_count": len(cases),
            "gold_blockers": gold_total,
            "predicted_blockers": predicted_total,
            "blocker_recall": _rate(matched_total, gold_total),
            "p1_recall": _rate(matched_by_severity["P1"], gold_by_severity["P1"]),
            "p2_recall": _rate(matched_by_severity["P2"], gold_by_severity["P2"]),
            "matched_severity_accuracy": _rate(severity_hits, matched_total),
            "matched_evidence_accuracy": _rate(evidence_hits, matched_total),
            "unmatched_finding_rate": _rate(false_positive_findings, predicted_total),
            "clean_case_false_positive_rate": _rate(clean_false_positive_cases, clean_cases),
            "merge_recommendation_accuracy": _rate(merge_hits, len(cases)),
            "median_latency_ms": _median(latencies),
        },
        "cases": rows,
    }


async def evaluate_pr_cases(cases: list[dict[str, Any]], llm: Any, concurrency: int) -> dict[str, Any]:
    rows = await _run_with_limit([_timed_agent(PRReviewAgent(llm=llm), case) for case in cases], concurrency)
    return score_pr_rows(cases, rows)


async def evaluate_ci_cases(cases: list[dict[str, Any]], llm: Any, concurrency: int) -> dict[str, Any]:
    rows = await _run_with_limit([_timed_agent(CIDebugAgent(llm=llm), case) for case in cases], concurrency)
    type_hits = 0
    root_hits = 0
    file_hits = 0
    strict_hits = 0
    for case, row in zip(cases, rows):
        result = row["result"]
        expected = case["expected"]
        result_text = " ".join(
            [
                _text(result.get("first_error")),
                _text(result.get("root_cause")),
                " ".join(_text(value) for value in result.get("possible_causes") or []),
                " ".join(_text(value) for value in result.get("fix_steps") or []),
            ]
        )
        type_ok = result.get("failure_type") == expected["failure_type"]
        root_ok = any(_text(term) in result_text for term in expected.get("root_any") or [])
        related_text = " ".join(_text(value).replace("\\", "/") for value in result.get("related_files") or [])
        file_terms = expected.get("related_file_any") or []
        file_ok = not file_terms or any(_text(term).replace("\\", "/") in related_text for term in file_terms)
        row["expected"] = expected
        row["checks"] = {"failure_type": type_ok, "root_cause": root_ok, "related_file": file_ok}
        type_hits += int(type_ok)
        root_hits += int(root_ok)
        file_hits += int(file_ok)
        strict_hits += int(type_ok and root_ok and file_ok)
    latencies = [row["latency_ms"] for row in rows]
    return {
        "metrics": {
            "case_count": len(cases),
            "failure_type_accuracy": _rate(type_hits, len(cases)),
            "root_cause_accuracy": _rate(root_hits, len(cases)),
            "related_file_accuracy": _rate(file_hits, len(cases)),
            "strict_case_accuracy": _rate(strict_hits, len(cases)),
            "median_latency_ms": _median(latencies),
        },
        "cases": rows,
    }


async def evaluate_workflow_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    observer = ObserverAgent()
    synthesizer = SynthesisAgent()
    rows = []
    strict_hits = 0
    finding_recall_values: list[float] = []
    for case in cases:
        task_results = [WorkflowTaskResult.model_validate(item) for item in case.get("task_results") or []]
        tasks = [
            WorkflowTask(
                id=result.task_id,
                agent_name=result.agent_name,
                task_type=result.task_type,
                objective=f"Evaluate {result.task_type}",
                claim=WorkflowClaim(target_type=result.task_type, target_id=result.task_id),
            )
            for result in task_results
        ]
        spec = WorkflowSpec(
            goal=f"Evaluate {case['id']}",
            trigger_message=case["id"],
            tasks=tasks,
            requires_human_confirmation=bool(case.get("requires_human_confirmation")),
        )
        started = time.perf_counter()
        observation = await observer.run(
            {"spec": spec.model_dump(mode="json"), "task_results": [item.model_dump(mode="json") for item in task_results]},
            case.get("context") or {},
        )
        synthesis = await synthesizer.run(
            {"spec": spec.model_dump(mode="json"), "task_results": [item.model_dump(mode="json") for item in task_results], "observation": observation},
            case.get("context") or {},
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        actual_findings = {item["finding_type"] for item in observation.get("findings") or []}
        expected_findings = set(case.get("expected_findings") or [])
        finding_recall = _rate(len(actual_findings & expected_findings), len(expected_findings)) if expected_findings else float(not actual_findings)
        answer = str(synthesis.get("answer") or "")
        decision_ok = any(term in answer for term in case.get("expected_decision_any") or [])
        strict = decision_ok and finding_recall == 1.0
        strict_hits += int(strict)
        finding_recall_values.append(finding_recall)
        rows.append(
            {
                "id": case["id"],
                "latency_ms": latency_ms,
                "decision_correct": decision_ok,
                "finding_recall": finding_recall,
                "strict_correct": strict,
                "expected_findings": sorted(expected_findings),
                "actual_findings": sorted(actual_findings),
                "answer": answer,
            }
        )
    return {
        "metrics": {
            "case_count": len(cases),
            "decision_accuracy": _rate(sum(row["decision_correct"] for row in rows), len(cases)),
            "finding_recall": _mean(finding_recall_values),
            "strict_case_accuracy": _rate(strict_hits, len(cases)),
            "median_latency_ms": _median([row["latency_ms"] for row in rows]),
        },
        "cases": rows,
    }


async def scheduler_parallelism_benchmark(delay_seconds: float = 0.12) -> dict[str, Any]:
    spec = WorkflowSpec(
        goal="Compare serial and parallel independent task execution",
        trigger_message="benchmark",
        max_parallel_tasks=3,
        tasks=[
            WorkflowTask(
                id=f"task-{index}",
                agent_name=f"agent-{index}",
                task_type=f"task-type-{index}",
                objective=f"Independent task {index}",
                claim=WorkflowClaim(target_type="benchmark", target_id=str(index)),
            )
            for index in range(3)
        ],
    )

    async def runner(task: WorkflowTask, _spec: WorkflowSpec, _context: dict[str, Any]) -> WorkflowTaskResult:
        await asyncio.sleep(delay_seconds)
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"{task.id} complete",
            evidence_count=1,
            confidence=0.9,
        )

    runners = {task.task_type: runner for task in spec.tasks}
    started = time.perf_counter()
    parallel_report = await WorkflowOrchestrator(runners).run_spec(spec.model_dump(mode="json"), {"disable_replan": True})
    parallel_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    serial_results = [await runner(task, spec, {}) for task in spec.tasks]
    observer = ObserverAgent()
    observation = await observer.run(
        {"spec": spec.model_dump(mode="json"), "task_results": [item.model_dump(mode="json") for item in serial_results]},
        {},
    )
    synthesis = await SynthesisAgent().run(
        {"spec": spec.model_dump(mode="json"), "task_results": [item.model_dump(mode="json") for item in serial_results], "observation": observation},
        {},
    )
    serial_ms = (time.perf_counter() - started) * 1000
    quality_equal = bool(parallel_report.get("final_answer")) and bool(synthesis.get("answer"))
    return {
        "task_count": len(spec.tasks),
        "per_task_delay_ms": round(delay_seconds * 1000, 2),
        "parallel_duration_ms": round(parallel_ms, 2),
        "serial_duration_ms": round(serial_ms, 2),
        "wall_time_reduction_pct": round((serial_ms - parallel_ms) / serial_ms * 100, 2),
        "quality_gate_equal": quality_equal,
    }


def _reset_eval_repo(db: Session) -> Repository:
    repo = db.query(Repository).filter(Repository.full_name == "local-eval/resume-quality-suite").one_or_none()
    if repo is None:
        repo = Repository(
            owner="local-eval",
            name="resume-quality-suite",
            full_name="local-eval/resume-quality-suite",
            description="Dedicated synthetic repository for resume-value evaluation",
            default_branch="main",
        )
        db.add(repo)
        db.flush()
    delete_milvus_documents(repo_id=repo.id, strict=False)
    db.query(EvidenceItem).filter(EvidenceItem.repo_id == repo.id).delete(synchronize_session=False)
    db.query(RecallEvent).filter(RecallEvent.repo_id == repo.id).delete(synchronize_session=False)
    db.query(MemoryCandidate).filter(MemoryCandidate.repo_id == repo.id).delete(synchronize_session=False)
    db.query(Document).filter(Document.repo_id == repo.id).delete(synchronize_session=False)
    db.flush()
    get_or_create_config(db, repo.id, commit=False)
    return repo


async def evaluate_evidence_and_memory(db: Session, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = _reset_eval_repo(db)
    namespace = uuid.UUID("f709b62d-16cc-4a57-b877-b4812e76220b")
    documents = payload.get("evidence_documents") or []
    add_documents(
        db,
        repo.id,
        [
            DocumentPayload(
                source_type="knowledge_file",
                source_id=uuid.uuid5(namespace, item["id"]),
                title=item["title"],
                content=item["content"],
                metadata={"eval_doc_id": item["id"], "filename": f"{item['id']}.md"},
            )
            for item in documents
        ],
        commit=False,
    )
    db.commit()

    evidence_rows = []
    positive_hits = 0
    top1_hits = 0
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    negative_answer_decisions = 0
    negative_count = 0
    for case in payload.get("evidence_cases") or []:
        started = time.perf_counter()
        results, trace, _duration_ms = await retrieve_knowledge(
            db,
            repo.id,
            case["question"],
            top_k=5,
            score_threshold_enabled=False,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        result_ids = [str((item.get("metadata") or {}).get("eval_doc_id") or "") for item in results]
        expected_ids = list(case.get("expected_doc_ids") or [])
        if case.get("kind") == "negative":
            negative_count += 1
            negative_answer_decisions += int((trace.get("answer_gate") or {}).get("decision") == "answer")
            hit = not any(result_id in expected_ids for result_id in result_ids)
            rr = 0.0
            precision = 0.0
        else:
            relevant_positions = [index for index, result_id in enumerate(result_ids, start=1) if result_id in expected_ids]
            hit = bool(relevant_positions)
            positive_hits += int(hit)
            top1_hits += int(bool(result_ids and result_ids[0] in expected_ids))
            rr = 1 / min(relevant_positions) if relevant_positions else 0.0
            precision = sum(result_id in expected_ids for result_id in result_ids) / len(result_ids) if result_ids else 0.0
            reciprocal_ranks.append(rr)
            precisions.append(precision)
        evidence_rows.append(
            {
                "id": case["id"],
                "kind": case.get("kind") or "positive",
                "latency_ms": latency_ms,
                "expected_doc_ids": expected_ids,
                "retrieved_doc_ids": result_ids,
                "hit_at_5": hit,
                "reciprocal_rank": round(rr, 4),
                "precision_at_5": round(precision, 4),
                "answer_gate": trace.get("answer_gate") or {},
            }
        )
    positive_count = len([case for case in payload.get("evidence_cases") or [] if case.get("kind") != "negative"])
    evidence_result = {
        "metrics": {
            "case_count": len(evidence_rows),
            "positive_case_count": positive_count,
            "hit_at_5": _rate(positive_hits, positive_count),
            "top1_source_accuracy": _rate(top1_hits, positive_count),
            "mrr": _mean(reciprocal_ranks),
            "retrieval_precision_at_5": _mean(precisions),
            "unsupported_answer_gate_rate": _rate(negative_answer_decisions, negative_count),
            "median_latency_ms": _median([row["latency_ms"] for row in evidence_rows]),
        },
        "cases": evidence_rows,
    }

    candidate_by_case: dict[str, MemoryCandidate] = {}
    for case in payload.get("memory_cases") or []:
        candidate = MemoryCandidate(
            repo_id=repo.id,
            kind=case["kind"],
            title=f"Eval memory {case['id']}",
            content=case["content"],
            status="rejected" if case["status"] == "rejected" else "pending",
            source="resume_value_eval",
            meta={"eval_memory_id": case["id"], "stale": case["status"] == "approved_stale"},
        )
        db.add(candidate)
        db.flush()
        candidate_by_case[case["id"]] = candidate
        if case["status"] in {"approved", "approved_stale"}:
            approve_memory_candidate(db, repo, candidate.id)
    db.commit()
    EvidenceStore(db).rebuild_repo(repo)
    db.commit()

    memory_rows = []
    approved_hits = 0
    approved_count = 0
    unapproved_misuse = 0
    stale_misuse = 0
    accuracy_hits = 0
    tools = MemoryTools(db, repo)
    for case in payload.get("memory_cases") or []:
        result = await tools.search_evidence({"query": case["query"], "limit": 5}, {})
        target_id = str(candidate_by_case[case["id"]].id)
        retrieved = any(
            str((item.get("metadata") or {}).get("candidate_id") or "") == target_id
            or str((item.get("metadata") or {}).get("eval_memory_id") or "") == case["id"]
            for item in result.get("results") or []
        )
        expected_retrieved = bool(case["expected_retrieved"])
        correct = retrieved == expected_retrieved
        accuracy_hits += int(correct)
        if expected_retrieved:
            approved_count += 1
            approved_hits += int(retrieved)
        elif case["status"] in {"pending", "rejected"}:
            unapproved_misuse += int(retrieved)
        elif case["status"] == "approved_stale":
            stale_misuse += int(retrieved)
        memory_rows.append(
            {
                "id": case["id"],
                "status": case["status"],
                "expected_retrieved": expected_retrieved,
                "retrieved": retrieved,
                "correct": correct,
                "result_source_ids": [str(item.get("source_id") or "") for item in result.get("results") or []],
            }
        )
    db.commit()
    memory_result = {
        "metrics": {
            "case_count": len(memory_rows),
            "approved_memory_recall": _rate(approved_hits, approved_count),
            "unapproved_memory_misuse_count": unapproved_misuse,
            "stale_memory_misuse_count": stale_misuse,
            "policy_case_accuracy": _rate(accuracy_hits, len(memory_rows)),
        },
        "cases": memory_rows,
    }
    return evidence_result, memory_result


async def run_resume_value_eval(
    db: Session,
    payload: dict[str, Any],
    *,
    mode: str = "heuristic",
    llm_concurrency: int = 2,
) -> dict[str, Any]:
    if mode not in {"heuristic", "llm"}:
        raise ValueError("mode must be heuristic or llm")
    tracked_llm = TrackingLLM(LLMClient()) if mode == "llm" else None
    llm: Any = tracked_llm or NoopLLM()
    started = time.perf_counter()
    issue, pr, ci = await asyncio.gather(
        evaluate_issue_cases(payload.get("issue_cases") or [], llm, llm_concurrency),
        evaluate_pr_cases(payload.get("pr_cases") or [], llm, llm_concurrency),
        evaluate_ci_cases(payload.get("ci_cases") or [], llm, llm_concurrency),
    )
    workflow = await evaluate_workflow_cases(payload.get("workflow_cases") or [])
    scheduler = await scheduler_parallelism_benchmark()
    evidence, memory = await evaluate_evidence_and_memory(db, payload)
    return {
        "evaluation": {
            "name": payload.get("name"),
            "label": payload.get("label"),
            "schema_version": payload.get("schema_version"),
            "mode": mode,
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
        "issue": issue,
        "pr": pr,
        "ci": ci,
        "workflow": workflow,
        "scheduler": scheduler,
        "evidence": evidence,
        "memory": memory,
        "llm_calls": tracked_llm.summary() if tracked_llm else None,
    }


def load_resume_value_suite(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    return payload, hashlib.sha256(raw).hexdigest()
