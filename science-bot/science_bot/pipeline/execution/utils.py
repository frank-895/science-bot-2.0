"""Shared helper functions for execution implementations."""

import pandas as pd

from science_bot.pipeline.execution.schemas import ResolvedFilter


def apply_resolved_filters(
    data: pd.DataFrame,
    filters: list[ResolvedFilter],
) -> pd.DataFrame:
    """Apply resolved filters to a dataframe.

    Args:
        data: Input dataframe.
        filters: Resolved filter conditions.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    filtered = data.copy()
    for item in filters:
        series = filtered[item.column]
        if item.operator == "==":
            filtered = filtered[series == item.value]
        elif item.operator == "!=":
            filtered = filtered[series != item.value]
        elif item.operator == ">":
            filtered = filtered[series > item.value]
        elif item.operator == ">=":
            filtered = filtered[series >= item.value]
        elif item.operator == "<":
            filtered = filtered[series < item.value]
        elif item.operator == "<=":
            filtered = filtered[series <= item.value]
        elif item.operator == "in":
            filtered = filtered[series.isin(item.value)]
        elif item.operator == "contains":
            filtered = filtered[
                series.astype(str).str.contains(str(item.value), na=False)
            ]
    return filtered


def format_scalar_answer(
    value: float,
    decimal_places: int | None,
    round_to: int | None = None,
) -> str:
    """Format a scalar answer deterministically.

    Args:
        value: Numeric value to format.
        decimal_places: Optional rounding precision.
        round_to: Optional nearest-unit rounding to apply before formatting.

    Returns:
        str: Deterministic scalar formatting.
    """
    if round_to == 0:
        round_to = None
    if round_to is not None:
        value = round(value / round_to) * round_to
    if float(value).is_integer() and decimal_places is None:
        return str(int(value))
    if decimal_places is not None:
        return f"{value:.{decimal_places}f}"
    return format(value, ".15g")
