"""Public schemas for the resolution stage."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from science_bot.pipeline.contracts import SupportedQuestionClassification
from science_bot.pipeline.execution.schemas import ExecutionPayload
from science_bot.tracing import TraceWriter


class ResolutionStepSummary(BaseModel):
    """Compact summary of one resolver step."""

    model_config = ConfigDict(extra="forbid")

    step_index: int
    kind: str
    tool_name: str | None = None
    message: str
    selected_files: list[str] = Field(default_factory=list)
    resolved_field_keys: list[str] = Field(default_factory=list)
    truncated: bool = False

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        """Validate the human-readable step summary message."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must be non-empty.")
        return stripped


class ResolutionStageInput(BaseModel):
    """Input contract for the resolution stage."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    question: str
    classification: SupportedQuestionClassification
    capsule_path: Path
    trace_writer: TraceWriter | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Validate the question text."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must be non-empty.")
        return stripped

    @field_validator("capsule_path")
    @classmethod
    def validate_capsule_path(cls, value: Path) -> Path:
        """Validate the capsule path."""
        resolved = value.expanduser().resolve()
        if not resolved.exists():
            raise ValueError(f"capsule_path does not exist: {resolved}")
        return resolved


class ResolutionStageOutput(BaseModel):
    """Output contract for the resolution stage."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    payload: ExecutionPayload
    iterations_used: int
    selected_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    steps: list[ResolutionStepSummary] = Field(default_factory=list)
