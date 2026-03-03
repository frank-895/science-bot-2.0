import pandas as pd
from science_bot.pipeline.execution.schemas import ResolvedFilter
from science_bot.pipeline.execution.utils import (
    apply_resolved_filters,
    format_scalar_answer,
)


def test_apply_resolved_filters_equality() -> None:
    frame = pd.DataFrame({"group": ["a", "b", "a"]})

    result = apply_resolved_filters(
        frame,
        [ResolvedFilter(column="group", operator="==", value="a")],
    )

    assert result["group"].tolist() == ["a", "a"]


def test_apply_resolved_filters_inequality() -> None:
    frame = pd.DataFrame({"group": ["a", "b", "a"]})

    result = apply_resolved_filters(
        frame,
        [ResolvedFilter(column="group", operator="!=", value="a")],
    )

    assert result["group"].tolist() == ["b"]


def test_apply_resolved_filters_numeric_comparison() -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]})

    result = apply_resolved_filters(
        frame,
        [ResolvedFilter(column="value", operator=">=", value=2)],
    )

    assert result["value"].tolist() == [2, 3]


def test_apply_resolved_filters_in_operator() -> None:
    frame = pd.DataFrame({"group": ["a", "b", "c"]})

    result = apply_resolved_filters(
        frame,
        [ResolvedFilter(column="group", operator="in", value=["a", "c"])],
    )

    assert result["group"].tolist() == ["a", "c"]


def test_apply_resolved_filters_contains_operator() -> None:
    frame = pd.DataFrame({"label": ["fungal_gene", "animal_gene", "fungal_tree"]})

    result = apply_resolved_filters(
        frame,
        [ResolvedFilter(column="label", operator="contains", value="fungal")],
    )

    assert result["label"].tolist() == ["fungal_gene", "fungal_tree"]


def test_apply_resolved_filters_multiple_filters() -> None:
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "b"],
            "value": [1, 2, 3],
        }
    )

    result = apply_resolved_filters(
        frame,
        [
            ResolvedFilter(column="group", operator="==", value="a"),
            ResolvedFilter(column="value", operator=">", value=1),
        ],
    )

    assert result["value"].tolist() == [2]


def test_format_scalar_answer_integer_like_float() -> None:
    assert format_scalar_answer(2.0, None) == "2"


def test_format_scalar_answer_explicit_decimal_places() -> None:
    assert format_scalar_answer(2.0, 2) == "2.00"


def test_format_scalar_answer_general_float_formatting() -> None:
    assert format_scalar_answer(0.123456789, None) == format(0.123456789, ".15g")


def test_format_scalar_answer_rounding() -> None:
    assert format_scalar_answer(0.666666, 2) == "0.67"


def test_format_scalar_answer_round_to_nearest_thousand() -> None:
    assert format_scalar_answer(82442.3333333333, None, 1000) == "82000"


def test_format_scalar_answer_ignores_zero_round_to() -> None:
    assert format_scalar_answer(0.216, 3, 0) == "0.216"
