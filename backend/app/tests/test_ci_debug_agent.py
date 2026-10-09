import pytest

from app.services.agents.ci_debug_agent import CIDebugAgent


@pytest.mark.asyncio
async def test_ci_debug_agent_includes_failed_job_context() -> None:
    result = await CIDebugAgent().run(
        {
            "name": "backend-test",
            "status": "completed",
            "conclusion": "failure",
            "jobs": [{"name": "pytest", "conclusion": "failure", "steps": [{"name": "Run tests", "conclusion": "failure"}]}],
            "logs_text": "pytest tests/test_api.py FAILED AssertionError",
        },
        {},
    )
    assert result["failure_type"] == "test"
    assert "pytest" in result["failure_summary"]
def test_ci_classifier_handles_compact_module_and_parse_errors() -> None:
    agent = CIDebugAgent()

    assert agent._classify_failure("modulenotfounderror: no module named x") == "dependency"
    assert agent._classify_failure("module parse failed: unexpected token") == "build"


def test_ci_related_file_extraction_prefers_long_extensions_and_txt() -> None:
    files = CIDebugAgent()._extract_related_files(
        "frontend/app/page.tsx failed\nrequirements.txt: bad version",
        {},
    )

    assert "frontend/app/page.tsx" in files
    assert "requirements.txt" in files
