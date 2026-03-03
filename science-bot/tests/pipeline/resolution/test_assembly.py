from pathlib import Path

import pandas as pd
from science_bot.pipeline.resolution import assembly
from science_bot.pipeline.resolution.families import (
    AggregateResolvedPlan,
    DifferentialExpressionResolvedPlan,
    VariantFilteringResolvedPlan,
)
from science_bot.pipeline.resolution.planning import (
    MetadataJoinPlan,
    MultiFileMergePlan,
    MultiFileSourceEntry,
)


def test_assemble_payload_loads_required_columns_for_aggregate(monkeypatch):
    captured = {}

    def fake_load_dataframe(capsule_path: Path, filename: str, columns: list[str]):
        captured["capsule_path"] = capsule_path
        captured["filename"] = filename
        captured["columns"] = columns
        return pd.DataFrame({"group": ["A"], "value": [1]})

    monkeypatch.setattr(assembly, "load_dataframe", fake_load_dataframe)

    payload, selected_files, notes = assembly.assemble_payload(
        capsule_path=Path("/tmp/capsule"),
        plan=AggregateResolvedPlan(
            filename="data.csv",
            operation="mean",
            value_column="value",
            filters=[],
            numerator_filters=[],
            denominator_filters=[],
            return_format="number",
        ),
    )

    assert captured["columns"] == ["value"]
    assert selected_files == ["data.csv"]
    assert notes == ["Loaded data.csv with explicit column subset of 1 columns."]
    assert payload.family == "aggregate"


def test_assemble_payload_rejects_non_precomputed_de_mode():
    plan = DifferentialExpressionResolvedPlan(
        mode="precomputed_results",
        result_table_files={"a": "table.csv"},
        operation="significant_gene_count",
        comparison_labels=[],
    )

    assert plan.mode == "precomputed_results"


def test_assemble_payload_builds_raw_count_de_payload(monkeypatch):
    captured_calls = []

    def fake_load_dataframe(capsule_path: Path, filename: str, columns):
        captured_calls.append((filename, columns))
        if filename == "counts.csv":
            return pd.DataFrame({"gene": ["g1"], "s1": [1], "s2": [2]})
        return pd.DataFrame({"sample": ["s1", "s2"], "condition": ["A", "B"]})

    monkeypatch.setattr(assembly, "load_dataframe", fake_load_dataframe)

    payload, selected_files, notes = assembly.assemble_payload(
        capsule_path=Path("/tmp/capsule"),
        plan=DifferentialExpressionResolvedPlan(
            mode="raw_counts",
            count_matrix_file="counts.csv",
            sample_metadata_file="metadata.csv",
            sample_metadata_sample_id_column="sample",
            design_factor_column="condition",
            tested_level="B",
            reference_level="A",
            count_matrix_orientation="genes_by_samples",
            count_matrix_gene_id_column="gene",
            operation="significant_gene_count",
            comparison_labels=["B vs A"],
        ),
    )

    assert captured_calls == [
        ("counts.csv", None),
        ("metadata.csv", ["sample", "condition"]),
    ]
    assert selected_files == ["counts.csv", "metadata.csv"]
    assert payload.mode == "raw_counts"
    assert payload.count_matrix is not None
    assert payload.sample_metadata is not None
    assert payload.sample_metadata_sample_id_column == "sample"
    assert payload.design_factor_column == "condition"
    assert payload.reference_level == "A"
    assert notes == [
        "Loaded counts.csv with all columns for raw-count differential expression.",
        "Loaded metadata.csv with explicit column subset of 2 columns.",
    ]


def test_assemble_payload_preserves_zip_contained_filename(monkeypatch):
    captured = {}

    def fake_load_dataframe(capsule_path: Path, filename: str, columns: list[str]):
        captured["filename"] = filename
        captured["columns"] = columns
        return pd.DataFrame({"value": [1]})

    monkeypatch.setattr(assembly, "load_dataframe", fake_load_dataframe)

    payload, selected_files, _ = assembly.assemble_payload(
        capsule_path=Path("/tmp/capsule"),
        plan=AggregateResolvedPlan(
            filename="Animals_Cele.busco.zip/run_eukaryota_odb10/full_table.tsv",
            operation="mean",
            value_column="value",
            filters=[],
            numerator_filters=[],
            denominator_filters=[],
            return_format="number",
        ),
    )

    assert (
        captured["filename"]
        == "Animals_Cele.busco.zip/run_eukaryota_odb10/full_table.tsv"
    )
    assert selected_files == [
        "Animals_Cele.busco.zip/run_eukaryota_odb10/full_table.tsv"
    ]
    assert payload.family == "aggregate"


def test_assemble_payload_builds_merged_dataframe_for_aggregate(monkeypatch):
    captured = []

    def fake_load_dataframe(capsule_path: Path, filename: str, columns):
        captured.append((filename, columns))
        if filename == "s1.csv":
            return pd.DataFrame({"vaf": [0.1], "effect": ["synonymous"]})
        if filename == "s2.csv":
            return pd.DataFrame({"vaf": [0.4], "effect": ["missense"]})
        return pd.DataFrame(
            {"Sample ID": ["S1", "S2"], "BLM Mutation Status": ["Carrier", "Control"]}
        )

    monkeypatch.setattr(assembly, "load_dataframe", fake_load_dataframe)

    payload, selected_files, notes = assembly.assemble_payload(
        capsule_path=Path("/tmp/capsule"),
        plan=AggregateResolvedPlan(
            filename=None,
            merge_plan=MultiFileMergePlan(
                data_sources=[
                    MultiFileSourceEntry(
                        filename="s1.csv",
                        sample_id="S1",
                        selected_columns=["vaf", "effect"],
                    ),
                    MultiFileSourceEntry(
                        filename="s2.csv",
                        sample_id="S2",
                        selected_columns=["vaf", "effect"],
                    ),
                ],
                join=MetadataJoinPlan(
                    metadata_file="metadata.csv",
                    metadata_sample_id_column="Sample ID",
                    metadata_columns=["BLM Mutation Status"],
                ),
            ),
            operation="count",
            filters=[],
            numerator_filters=[],
            denominator_filters=[],
            return_format="number",
        ),
    )

    assert captured == [
        ("s1.csv", ["vaf", "effect"]),
        ("s2.csv", ["vaf", "effect"]),
        ("metadata.csv", ["Sample ID", "BLM Mutation Status"]),
    ]
    assert selected_files == ["s1.csv", "s2.csv", "metadata.csv"]
    assert list(payload.data["sample_id"]) == ["S1", "S2"]
    assert list(payload.data["BLM Mutation Status"]) == ["Carrier", "Control"]
    assert any("Stamped s1.csv" in note for note in notes)


def test_assemble_payload_builds_merged_dataframe_for_variant_filtering(monkeypatch):
    def fake_load_dataframe(capsule_path: Path, filename: str, columns):
        if filename == "s1.csv":
            return pd.DataFrame(
                {"gene": ["NOTCH1"], "effect": ["missense"], "vaf": [0.2]}
            )
        return pd.DataFrame({"gene": ["TET2"], "effect": ["synonymous"], "vaf": [0.1]})

    monkeypatch.setattr(assembly, "load_dataframe", fake_load_dataframe)

    payload, selected_files, _ = assembly.assemble_payload(
        capsule_path=Path("/tmp/capsule"),
        plan=VariantFilteringResolvedPlan(
            filename=None,
            merge_plan=MultiFileMergePlan(
                data_sources=[
                    MultiFileSourceEntry(
                        filename="s1.csv",
                        sample_id="S1",
                        selected_columns=["gene", "effect", "vaf"],
                    ),
                    MultiFileSourceEntry(
                        filename="s2.csv",
                        sample_id="S2",
                        selected_columns=["gene", "effect", "vaf"],
                    ),
                ]
            ),
            operation="filtered_variant_count",
            gene_column="gene",
            effect_column="effect",
            vaf_column="vaf",
            filters=[],
            return_format="number",
        ),
    )

    assert selected_files == ["s1.csv", "s2.csv"]
    assert list(payload.data["sample_id"]) == ["S1", "S2"]
    assert payload.family == "variant_filtering"
