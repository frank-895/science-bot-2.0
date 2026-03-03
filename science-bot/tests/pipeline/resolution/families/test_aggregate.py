import pytest
from science_bot.pipeline.resolution.families.aggregate import (
    AggregateResolutionDecision,
    AggregateResolvedPlan,
    build_aggregate_plan_from_decision,
)


def test_build_aggregate_plan_from_decision():
    plan = build_aggregate_plan_from_decision(
        AggregateResolutionDecision(
            action="finalize",
            reason="done",
            filename="data.csv",
            operation="mean",
            value_column="value",
            return_format="number",
        ),
        require_text=lambda value, _name: value,
        require_value=lambda value, _name: value,
    )

    assert plan.filename == "data.csv"


def test_aggregate_resolved_plan_requires_value_column():
    with pytest.raises(ValueError):
        AggregateResolvedPlan(
            filename="data.csv",
            operation="mean",
            value_column=None,
            filters=[],
            numerator_filters=[],
            denominator_filters=[],
            return_format="number",
        )


def test_build_aggregate_merge_plan_from_decision():
    plan = build_aggregate_plan_from_decision(
        AggregateResolutionDecision(
            action="finalize",
            reason="done",
            use_merge=True,
            data_source_files=["a.csv", "b.csv"],
            data_source_sample_ids=["A", "B"],
            data_source_selected_columns=["vaf", "effect"],
            metadata_file="metadata.csv",
            metadata_sample_id_column="Sample ID",
            metadata_columns=["BLM Mutation Status"],
            operation="count",
            numerator_filters=[],
            denominator_filters=[],
            filters=[],
            return_format="number",
        ),
        require_text=lambda value, _name: value,
        require_value=lambda value, _name: value,
    )

    assert plan.filename is None
    assert plan.merge_plan is not None
    assert [source.filename for source in plan.merge_plan.data_sources] == [
        "a.csv",
        "b.csv",
    ]
    assert plan.merge_plan.join is not None
    assert plan.merge_plan.join.metadata_file == "metadata.csv"


def test_aggregate_resolved_plan_requires_exactly_one_filename_or_merge_plan():
    with pytest.raises(ValueError):
        AggregateResolvedPlan(
            filename=None,
            merge_plan=None,
            operation="count",
            filters=[],
            numerator_filters=[],
            denominator_filters=[],
            return_format="number",
        )
