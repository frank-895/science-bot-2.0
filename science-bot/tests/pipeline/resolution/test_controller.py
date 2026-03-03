import asyncio
from pathlib import Path

import pandas as pd
import pytest
from science_bot.pipeline.contracts import SupportedQuestionClassification
from science_bot.pipeline.execution.schemas import AggregateExecutionInput
from science_bot.pipeline.resolution import controller
from science_bot.pipeline.resolution.schemas import ResolutionStageInput
from science_bot.pipeline.resolution.tools.schemas import FullCapsuleManifest


def test_run_resolution_controller_detects_repeated_tool_call(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        controller,
        "list_all_capsule_files",
        lambda _path: FullCapsuleManifest(
            capsule_path=str(tmp_path),
            files=[],
            total_size_bytes=0,
        ),
    )

    class Decision:
        action = "use_find_files_with_column"
        reason = "search"
        zip_filename = None
        filename = None
        query = "gene"
        column = None
        columns = []
        n = 10
        random_sample = False
        max_values = 50
        max_matches = 50

    decisions = [Decision(), Decision()]

    async def fake_parse_structured(**_kwargs):
        return decisions.pop(0)

    monkeypatch.setattr(controller, "parse_structured", fake_parse_structured)
    monkeypatch.setattr(
        controller,
        "find_files_with_column",
        lambda *_args, **_kwargs: [],
    )

    with pytest.raises(controller.ResolutionIterationLimitError):
        asyncio.run(
            controller.run_resolution_controller(
                ResolutionStageInput(
                    question="question",
                    classification=SupportedQuestionClassification(family="aggregate"),
                    capsule_path=tmp_path,
                )
            )
        )


def test_run_resolution_controller_finalizes(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        controller,
        "list_all_capsule_files",
        lambda _path: FullCapsuleManifest(
            capsule_path=str(tmp_path),
            files=[],
            total_size_bytes=0,
        ),
    )

    class Decision:
        action = "finalize"
        reason = "done"
        zip_filename = None
        filename = "data.csv"
        query = None
        column = None
        columns = []
        n = 10
        random_sample = False
        max_values = 50
        max_matches = 50
        operation = "count"
        value_column = None
        numerator_mask_column = None
        denominator_mask_column = None
        numerator_filters = []
        denominator_filters = []
        filters = []
        return_format = "number"
        decimal_places = None
        round_to = None

    async def fake_parse_structured(**_kwargs):
        return Decision()

    monkeypatch.setattr(controller, "parse_structured", fake_parse_structured)
    monkeypatch.setattr(
        controller,
        "assemble_payload",
        lambda **_kwargs: (
            AggregateExecutionInput(
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
            ["data.csv"],
            [],
        ),
    )

    output = asyncio.run(
        controller.run_resolution_controller(
            ResolutionStageInput(
                question="question",
                classification=SupportedQuestionClassification(family="aggregate"),
                capsule_path=tmp_path,
            )
        )
    )
    assert output.selected_files == ["data.csv"]
    assert output.steps[-1].kind == "finalize"
