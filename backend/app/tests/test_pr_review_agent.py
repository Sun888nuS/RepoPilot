import pytest

from app.services.agents.pr_review_agent import PRReviewAgent


class NoopLLM:
    async def chat_json(self, system_prompt, user_payload):
        return {}


class OverconfidentLLM:
    async def chat_json(self, system_prompt, user_payload):
        return {
            "merge_recommendation": "建议合入",
            "review_findings": [
                {
                    "severity": "P2",
                    "title": "权限边界缺少验证",
                    "evidence": "diff 修改了权限判断，但没有对应测试证据。",
                    "required_action": "补充权限边界测试或人工验证记录。",
                    "blocking": True,
                }
            ],
            "blocking_issues": [],
        }


@pytest.mark.asyncio
async def test_pr_review_fallback_emits_blocking_findings() -> None:
    result = await PRReviewAgent(llm=NoopLLM()).run(
        {
            "number": 12,
            "state": "open",
            "author": "alice",
            "title": "Update auth middleware",
            "body": "Refactor permission checks.",
            "files": [
                {
                    "filename": "backend/app/auth/middleware.py",
                    "additions": 320,
                    "deletions": 12,
                    "patch": "permission change",
                }
            ],
            "comments": [{"body": "Please verify token handling is not logged."}],
        },
        {"ci_summary": {"failed_runs": 1}},
    )

    severities = {item["severity"] for item in result["review_findings"]}
    assert {"P1", "P2"} <= severities
    assert result["merge_recommendation"] == "暂缓"
    assert result["blocking_issues"]


@pytest.mark.asyncio
async def test_sensitive_change_with_tests_is_not_blocked_by_filename_alone() -> None:
    result = await PRReviewAgent(llm=NoopLLM()).run(
        {
            "number": 14,
            "state": "open",
            "title": "Validate JWT audience",
            "body": "Adds strict validation and regression tests.",
            "files": [
                {"filename": "backend/app/auth/token.py", "additions": 8, "deletions": 2, "patch": "validate audience"},
                {"filename": "backend/app/tests/test_token.py", "additions": 20, "deletions": 0, "patch": "test invalid audience"},
            ],
            "comments": [],
        },
        {"ci_summary": {"failed_runs": 0}},
    )

    assert not result["blocking_issues"]
    assert result["merge_recommendation"] == "建议合入"
    assert all(item["severity"] == "P3" for item in result["review_findings"])


@pytest.mark.asyncio
async def test_pr_review_downgrades_merge_when_llm_returns_blocking_finding() -> None:
    result = await PRReviewAgent(llm=OverconfidentLLM()).run(
        {
            "number": 13,
            "state": "open",
            "author": "alice",
            "title": "Update permission UI",
            "body": "Ready to merge.",
            "files": [
                {
                    "filename": "frontend/app/page.tsx",
                    "additions": 20,
                    "deletions": 3,
                    "patch": "permission UI change",
                }
            ],
            "comments": [],
        },
        {},
    )

    assert result["merge_recommendation"] == "修改后合入"
    assert any("权限边界缺少验证" in item for item in result["blocking_issues"])
    assert result["review_findings"][0]["severity"] == "P2"
def test_pr_merge_normalizes_structured_llm_lists() -> None:
    agent = PRReviewAgent()
    fallback = agent._fallback_review(
        {"number": 1, "title": "Auth", "body": "", "state": "open", "files": [], "comments": []},
        {},
        [],
    )
    merged = agent._merge_with_fallback(
        {
            "key_changes": [{"what": "Disable signature verification", "why": "speed"}],
            "blocking_issues": [{"title": "JWT bypass", "required_action": "Restore verification"}],
            "review_checklist": [{"item": "Verify signatures", "status": "failed"}],
            "files_need_attention": [{"filename": "auth.py", "reason": "security"}],
            "review_findings": [
                {
                    "severity": "P1",
                    "title": "JWT bypass",
                    "evidence": "auth.py disables verification",
                    "required_action": "Restore verification",
                    "blocking": True,
                }
            ],
        },
        fallback,
    )

    assert merged["key_changes"] == ["Disable signature verification；speed"]
    assert all(isinstance(item, str) for item in merged["blocking_issues"])
    assert merged["files_need_attention"] == ["auth.py；security"]
