import asyncio
from pathlib import Path

import openpyxl
import pandas as pd
import pytest
from science_bot.pipeline.contracts import SupportedQuestionClassification
from science_bot.pipeline.execution.schemas import (
    AggregateExecutionInput,
    VariantFilteringExecutionInput,
)
from science_bot.pipeline.resolution import controller
from science_bot.pipeline.resolution.schemas import ResolutionStageInput
from science_bot.pipeline.resolution.tools.schemas import (
    AllFileInfo,
    FullCapsuleManifest,
    ZipEntry,
    ZipManifest,
)


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


def test_run_resolution_controller_finalizes_merge_plan(monkeypatch, tmp_path: Path):
    manifest = FullCapsuleManifest(
        capsule_path=str(tmp_path),
        files=[
            AllFileInfo(
                path="metadata.xlsx",
                filename="metadata.xlsx",
                extension=".xlsx",
                size_bytes=1,
                size_human="1 B",
                category="excel",
                is_supported_for_deeper_inspection=True,
            ),
            AllFileInfo(
                path="sample_a.xlsx",
                filename="sample_a.xlsx",
                extension=".xlsx",
                size_bytes=1,
                size_human="1 B",
                category="excel",
                is_supported_for_deeper_inspection=True,
            ),
        ],
        total_size_bytes=2,
    )
    monkeypatch.setattr(controller, "list_all_capsule_files", lambda _path: manifest)

    class Decision:
        action = "finalize"
        reason = "done"
        zip_filename = None
        filename = None
        query = None
        column = None
        columns = []
        n = 10
        random_sample = False
        max_values = 50
        max_matches = 50
        use_merge = True
        data_source_files = ["sample_a.xlsx"]
        data_source_sample_ids = ["A"]
        data_source_selected_columns = ["effect", "vaf"]
        metadata_file = "metadata.xlsx"
        metadata_sample_id_column = "Sample ID"
        metadata_columns = ["Status"]
        output_sample_id_column = "sample_id"
        operation = "filtered_variant_count"
        sample_column = "sample_id"
        sample_value = None
        gene_column = None
        effect_column = "effect"
        vaf_column = "vaf"
        vaf_min = None
        vaf_max = None
        filters = []
        return_format = "number"
        decimal_places = None
        round_to = None

    async def fake_parse_structured(**_kwargs):
        return Decision()

    monkeypatch.setattr(controller, "parse_structured", fake_parse_structured)
    monkeypatch.setattr(
        controller,
        "_observed_columns_by_filename",
        lambda _scratchpad: {
            "sample_a.xlsx": {"effect", "vaf"},
            "metadata.xlsx": {"Sample ID", "Status"},
        },
    )
    monkeypatch.setattr(
        controller,
        "assemble_payload",
        lambda **_kwargs: (
            VariantFilteringExecutionInput(
                family="variant_filtering",
                operation="filtered_variant_count",
                data=pd.DataFrame(
                    {"sample_id": ["A"], "effect": ["missense"], "vaf": [0.1]}
                ),
                sample_column="sample_id",
                sample_value=None,
                gene_column=None,
                effect_column="effect",
                vaf_column="vaf",
                vaf_min=None,
                vaf_max=None,
                filters=[],
                return_format="number",
                decimal_places=None,
                round_to=None,
            ),
            ["sample_a.xlsx", "metadata.xlsx"],
            [],
        ),
    )

    output = asyncio.run(
        controller.run_resolution_controller(
            ResolutionStageInput(
                question="question",
                classification=SupportedQuestionClassification(
                    family="variant_filtering"
                ),
                capsule_path=tmp_path,
            )
        )
    )

    assert output.selected_files == ["sample_a.xlsx", "metadata.xlsx"]
    assert output.steps[-1].kind == "finalize"


def test_initial_scratchpad_enriches_excel_candidates(tmp_path: Path):
    workbook = openpyxl.Workbook()
    workbook.active.title = "Tumor vs Normal"
    workbook.active.append(["protein", "gene", "log2FC"])
    workbook.save(tmp_path / "Proteomic_data.xlsx")
    workbook.close()

    scratchpad = controller._initial_scratchpad(
        ResolutionStageInput(
            question="question",
            classification=SupportedQuestionClassification(
                family="differential_expression"
            ),
            capsule_path=tmp_path,
        )
    )

    assert scratchpad.candidate_files[0].sheet_names == ["Tumor vs Normal"]
    assert scratchpad.candidate_files[0].first_sheet_columns == [
        "protein",
        "gene",
        "log2FC",
    ]
    assert scratchpad.known_sheets == {}
    assert scratchpad.known_columns == {}


def test_initial_scratchpad_keeps_candidate_when_excel_enrichment_fails(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        controller,
        "list_all_capsule_files",
        lambda _path: FullCapsuleManifest(
            capsule_path=str(tmp_path),
            files=[
                AllFileInfo(
                    path="Missing.xlsx",
                    filename="Missing.xlsx",
                    extension=".xlsx",
                    size_bytes=1,
                    size_human="1 B",
                    category="excel",
                    is_supported_for_deeper_inspection=True,
                )
            ],
            total_size_bytes=1,
        ),
    )

    scratchpad = controller._initial_scratchpad(
        ResolutionStageInput(
            question="question",
            classification=SupportedQuestionClassification(family="regression"),
            capsule_path=tmp_path,
        )
    )

    assert scratchpad.candidate_files[0].filename == "Missing.xlsx"
    assert scratchpad.candidate_files[0].sheet_names == []


def test_run_resolution_controller_bounces_back_after_invalid_finalize(
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

    class FinalizeDecision:
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
        operation = "mean"
        value_column = None
        numerator_mask_column = None
        denominator_mask_column = None
        numerator_filters = []
        denominator_filters = []
        filters = []
        return_format = "number"
        decimal_places = None
        round_to = None

    class FailDecision:
        action = "fail"
        reason = "stop"
        zip_filename = None
        filename = None
        query = None
        column = None
        columns = []
        n = 10
        random_sample = False
        max_values = 50
        max_matches = 50

    decisions = [FinalizeDecision(), FailDecision()]

    async def fake_parse_structured(**_kwargs):
        return decisions.pop(0)

    monkeypatch.setattr(controller, "parse_structured", fake_parse_structured)

    with pytest.raises(controller.ResolutionError, match="stop"):
        asyncio.run(
            controller.run_resolution_controller(
                ResolutionStageInput(
                    question="question",
                    classification=SupportedQuestionClassification(family="aggregate"),
                    capsule_path=tmp_path,
                )
            )
        )


def test_run_resolution_controller_rejects_unobserved_columns_before_assembly(
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

    class FinalizeDecision:
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
        operation = "mean"
        value_column = "made_up"
        numerator_mask_column = None
        denominator_mask_column = None
        numerator_filters = []
        denominator_filters = []
        filters = []
        return_format = "number"
        decimal_places = None
        round_to = None

    class FailDecision:
        action = "fail"
        reason = "stop"
        zip_filename = None
        filename = None
        query = None
        column = None
        columns = []
        n = 10
        random_sample = False
        max_values = 50
        max_matches = 50

    decisions = [FinalizeDecision(), FailDecision()]

    async def fake_parse_structured(**_kwargs):
        return decisions.pop(0)

    monkeypatch.setattr(controller, "parse_structured", fake_parse_structured)

    with pytest.raises(controller.ResolutionError, match="stop"):
        asyncio.run(
            controller.run_resolution_controller(
                ResolutionStageInput(
                    question="question",
                    classification=SupportedQuestionClassification(family="aggregate"),
                    capsule_path=tmp_path,
                )
            )
        )


def test_run_resolution_controller_accepts_excel_preview_columns(
    monkeypatch,
    tmp_path: Path,
):
    workbook = openpyxl.Workbook()
    workbook.active.title = "Sheet1"
    workbook.active.append(["value"])
    workbook.active.append([1])
    workbook.save(tmp_path / "results.xlsx")
    workbook.close()

    manifest = FullCapsuleManifest(
        capsule_path=str(tmp_path),
        files=[
            AllFileInfo(
                path="results.xlsx",
                filename="results.xlsx",
                extension=".xlsx",
                size_bytes=1,
                size_human="1 B",
                category="excel",
                is_supported_for_deeper_inspection=True,
            )
        ],
        total_size_bytes=1,
    )
    monkeypatch.setattr(controller, "list_all_capsule_files", lambda _path: manifest)

    class Decision:
        action = "finalize"
        reason = "done"
        zip_filename = None
        filename = "results.xlsx"
        query = None
        column = None
        columns = []
        n = 10
        random_sample = False
        max_values = 50
        max_matches = 50
        operation = "mean"
        value_column = "value"
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
                operation="mean",
                data=pd.DataFrame({"value": [1]}),
                value_column="value",
                numerator_mask_column=None,
                denominator_mask_column=None,
                numerator_filters=[],
                denominator_filters=[],
                filters=[],
                return_format="number",
                decimal_places=None,
                round_to=None,
            ),
            ["results.xlsx"],
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

    assert output.selected_files == ["results.xlsx"]


def test_run_resolution_controller_normalizes_zip_entry_filename(
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

    class ZipDecision:
        action = "use_list_zip_contents"
        reason = "inspect zip"
        zip_filename = "Animals_Cele.busco.zip"
        filename = None
        query = None
        column = None
        columns = []
        n = 10
        random_sample = False
        max_values = 50
        max_matches = 50

    class FinalizeDecision:
        action = "finalize"
        reason = "done"
        zip_filename = None
        filename = "run_eukaryota_odb10/full_table.tsv"
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

    decisions = [ZipDecision(), FinalizeDecision()]

    async def fake_parse_structured(**_kwargs):
        return decisions.pop(0)

    monkeypatch.setattr(controller, "parse_structured", fake_parse_structured)
    monkeypatch.setattr(
        controller,
        "list_zip_contents",
        lambda *_args, **_kwargs: ZipManifest(
            zip_filename="Animals_Cele.busco.zip",
            entries=[
                ZipEntry(
                    inner_path="run_eukaryota_odb10/full_table.tsv",
                    size_bytes=1,
                    file_type="tsv",
                    is_readable=True,
                )
            ],
        ),
    )

    captured = {}

    def fake_assemble_payload(**kwargs):
        captured["filename"] = kwargs["plan"].filename
        return (
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
            [kwargs["plan"].filename],
            [],
        )

    monkeypatch.setattr(controller, "assemble_payload", fake_assemble_payload)

    output = asyncio.run(
        controller.run_resolution_controller(
            ResolutionStageInput(
                question="question",
                classification=SupportedQuestionClassification(family="aggregate"),
                capsule_path=tmp_path,
            )
        )
    )

    assert (
        captured["filename"]
        == "Animals_Cele.busco.zip/run_eukaryota_odb10/full_table.tsv"
    )
    assert output.selected_files == [
        "Animals_Cele.busco.zip/run_eukaryota_odb10/full_table.tsv"
    ]
