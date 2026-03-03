from pathlib import Path

import pandas as pd
import pytest
from science_bot.pipeline.contracts import SupportedQuestionClassification
from science_bot.pipeline.execution.schemas import AggregateExecutionInput
from science_bot.pipeline.resolution.schemas import (
    ResolutionStageInput,
    ResolutionStageOutput,
    ResolutionStepSummary,
)


def test_resolution_stage_input_validates_question_and_path(tmp_path: Path):
    stage_input = ResolutionStageInput(
        question="  hello  ",
        classification=SupportedQuestionClassification(family="aggregate"),
        capsule_path=tmp_path,
    )

    assert stage_input.question == "hello"
    assert stage_input.capsule_path == tmp_path.resolve()


def test_resolution_stage_input_rejects_empty_question(tmp_path: Path):
    with pytest.raises(ValueError):
        ResolutionStageInput(
            question="   ",
            classification=SupportedQuestionClassification(family="aggregate"),
            capsule_path=tmp_path,
        )


def test_resolution_step_summary_requires_message():
    with pytest.raises(ValueError):
        ResolutionStepSummary(step_index=1, kind="discover", message=" ")


def test_resolution_stage_output_accepts_payload():
    output = ResolutionStageOutput(
        payload=AggregateExecutionInput(
            family="aggregate",
            operation="count",
            data=pd.DataFrame({"value": [1]}),
            value_column=None,
            numerator_mask_column=None,
            denominator_mask_column=None,
            numerator_filters=[],
            denominator_filters=[],
            filters=[],
            return_format="number",
            decimal_places=None,
            round_to=None,
        ),
        iterations_used=1,
        selected_files=["data.csv"],
        notes=["note"],
        steps=[ResolutionStepSummary(step_index=1, kind="discover", message="done")],
    )

    assert output.iterations_used == 1
