import asyncio
from pathlib import Path

import pandas as pd
from science_bot.pipeline.contracts import SupportedQuestionClassification
from science_bot.pipeline.execution.schemas import AggregateExecutionInput
from science_bot.pipeline.resolution import stage
from science_bot.pipeline.resolution.schemas import (
    ResolutionStageInput,
    ResolutionStageOutput,
    ResolutionStepSummary,
)


def test_run_resolution_stage_delegates_to_controller(monkeypatch):
    temp_dir = Path.cwd()
    expected = ResolutionStageOutput(
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
        notes=[],
        steps=[
            ResolutionStepSummary(
                step_index=1,
                kind="discover",
                message="ok",
            )
        ],
    )

    async def fake_run_resolution_controller(stage_input: ResolutionStageInput):
        assert stage_input.question == "question"
        return expected

    monkeypatch.setattr(
        stage,
        "run_resolution_controller",
        fake_run_resolution_controller,
    )

    result = asyncio.run(
        stage.run_resolution_stage(
            ResolutionStageInput(
                question="question",
                classification=SupportedQuestionClassification(family="aggregate"),
                capsule_path=temp_dir,
            )
        )
    )
    assert result == expected
