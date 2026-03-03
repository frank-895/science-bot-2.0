from pathlib import Path

import pandas as pd
from science_bot.pipeline.resolution import assembly
from science_bot.pipeline.resolution.families import (
    AggregateResolvedPlan,
    DifferentialExpressionResolvedPlan,
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
