"""Async orchestrator entrypoint backed by the agent runtime."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from science_bot.agent.runtime import run_agent


class OrchestratorRequest(BaseModel):
    """Validated request for running the orchestrator.

    Attributes:
        question: User question that should be answered from the capsule.
        capsule_path: Filesystem path to the extracted capsule directory.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    capsule_path: Path

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Validate question text.

        Args:
            value: Candidate question text.

        Returns:
            str: Stripped question text.

        Raises:
            ValueError: If the question is empty after stripping.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must be non-empty")
        return stripped


class OrchestratorResult(BaseModel):
    """Terminal-facing result from the orchestrator.

    Attributes:
        question: Original user question.
        capsule_path: Filesystem path used for execution.
        status: Terminal status for run completion.
        answer: Final answer string returned to CLI callers.
        metadata: Structured metadata payload for downstream formatting.
        error: Optional error message when failure details are attached.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    capsule_path: Path
    status: Literal["completed"]
    answer: str
    metadata: dict[str, object]
    error: str | None = None


async def run_orchestrator(request: OrchestratorRequest) -> OrchestratorResult:
    """Run the real-mode orchestrator.

    Args:
        request: Validated orchestrator request.

    Returns:
        OrchestratorResult: Orchestrator output derived from the agent runtime.

    Raises:
        FileNotFoundError: If the capsule path does not exist.
    """
    if not request.capsule_path.exists():
        raise FileNotFoundError(f"Capsule path not found: {request.capsule_path}")

    agent_result = await run_agent(
        question=request.question,
        capsule_path=request.capsule_path,
    )

    if agent_result.status != "completed" or agent_result.answer is None:
        terminal_reason = agent_result.failure_reason or "agent_failed"
        raise RuntimeError(f"Agent failed to produce an answer: {terminal_reason}")

    return OrchestratorResult(
        question=request.question,
        capsule_path=request.capsule_path,
        status="completed",
        answer=agent_result.answer,
        metadata={
            "classification_family": "agent",
            "resolution_iterations_used": agent_result.iterations_used,
            "resolution_selected_files": [],
            "execution_family": "agent",
            "execution_step_count": len(agent_result.steps),
            "terminal_reason": agent_result.failure_reason,
        },
        error=None,
    )
