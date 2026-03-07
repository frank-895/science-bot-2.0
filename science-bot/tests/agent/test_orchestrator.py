import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from science_bot.agent.contracts import AgentRunResult, AgentStepRecord
from science_bot.agent.orchestrator import (
    OrchestratorRequest,
    run_orchestrator,
)


def test_orchestrator_request_rejects_empty_question(tmp_path: Path) -> None:
    capsule_path = tmp_path / "capsule"
    capsule_path.mkdir()

    with pytest.raises(ValidationError, match="question must be non-empty"):
        OrchestratorRequest(question="   ", capsule_path=capsule_path)


def test_orchestrator_raises_for_missing_capsule(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-capsule"

    with pytest.raises(FileNotFoundError, match="Capsule path not found"):
        asyncio.run(
            run_orchestrator(
                OrchestratorRequest(question="What is this?", capsule_path=missing_path)
            )
        )


def test_orchestrator_returns_agent_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capsule_path = tmp_path / "capsule"
    capsule_path.mkdir()

    async def fake_run_agent(
        *,
        question: str,
        capsule_path: Path,
        max_iterations: int = 6,
    ) -> AgentRunResult:
        del max_iterations
        assert question == "How many rows?"
        assert capsule_path.name == "capsule"
        return AgentRunResult(
            status="completed",
            answer="42",
            iterations_used=2,
            steps=[
                AgentStepRecord(iteration=1, decision="run_python"),
                AgentStepRecord(iteration=2, decision="respond", answer="42"),
            ],
            failure_reason=None,
        )

    monkeypatch.setattr("science_bot.agent.orchestrator.run_agent", fake_run_agent)

    result = asyncio.run(
        run_orchestrator(
            OrchestratorRequest(
                question="How many rows?",
                capsule_path=capsule_path,
            )
        )
    )

    assert result.question == "How many rows?"
    assert result.capsule_path == capsule_path
    assert result.status == "completed"
    assert result.answer == "42"
    assert result.metadata["classification_family"] == "agent"
    assert result.metadata["resolution_iterations_used"] == 2
    assert result.metadata["resolution_selected_files"] == []
    assert result.metadata["execution_family"] == "agent"
    assert result.metadata["execution_step_count"] == 2
    assert result.metadata["terminal_reason"] is None
