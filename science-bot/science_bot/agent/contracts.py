"""Pydantic contracts for the agent runtime."""

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class AgentIterationResponse(BaseModel):
    """Structured model output for one runtime iteration."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    python_code: str | None = Field(default=None, alias="python")
    final_answer: str | None = None

    @model_validator(mode="after")
    def validate_choice(self) -> "AgentIterationResponse":
        """Require exactly one of python or final answer.

        Returns:
            AgentIterationResponse: Validated response.

        Raises:
            ValueError: If required fields are missing.
        """
        code = self.python_code.strip() if self.python_code is not None else ""
        has_code = bool(code)
        has_answer = self.final_answer is not None
        if has_code == has_answer:
            raise ValueError("Provide exactly one of python or final_answer.")
        return self


class AgentStepRecord(BaseModel):
    """One recorded step of the agent runtime."""

    model_config = ConfigDict(extra="forbid")

    iteration: int
    script: str
    proposed_final_answer: str | None = None
    execution_status: str | None = None
    execution_error: str | None = None
    execution_answer: str | None = None
    execution_stdout_tail: str | None = None
    execution_stderr_tail: str | None = None
    execution_duration_ms: int | None = None
    execution_worker: str | None = None


class AgentRunResult(BaseModel):
    """Terminal result returned by the agent runtime."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "failed"]
    answer: str | None = None
    iterations_used: int
    steps: list[AgentStepRecord]
    failure_reason: str | None = None
    failure_detail: str | None = None


class AgentRunRequest(BaseModel):
    """Validated request for the iterative agent runtime.

    Attributes:
        question: User question to answer.
        capsule_path: Filesystem path used by generated Python scripts.
        capsule_manifest: Optional precomputed recursive file listing.
        max_iterations: Maximum number of model decisions to execute.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    capsule_path: Path
    capsule_manifest: str | None = None
    max_iterations: int = 6

    @model_validator(mode="after")
    def validate_max_iterations(self) -> "AgentRunRequest":
        """Validate the iteration budget.

        Returns:
            AgentRunRequest: Validated request.

        Raises:
            ValueError: If the iteration budget is invalid.
        """
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero.")
        return self
