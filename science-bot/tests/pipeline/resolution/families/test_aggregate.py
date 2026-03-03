import pytest
from science_bot.pipeline.resolution.families.aggregate import (
    AggregateResolvedPlan,
    build_aggregate_plan_from_decision,
)


class Decision:
    filename = "data.csv"
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


def test_build_aggregate_plan_from_decision():
    plan = build_aggregate_plan_from_decision(
        Decision(),
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
