"""Deterministic hypothesis test execution implementation."""

import math
from typing import Final

import pandas as pd
from scipy import stats

from science_bot.pipeline.contracts import HypothesisTestType
from science_bot.pipeline.execution.schemas import (
    ExecutionStageOutput,
    HypothesisTestExecutionInput,
)
from science_bot.pipeline.execution.utils import (
    apply_resolved_filters,
    format_scalar_answer,
)

IMPLEMENTED_HYPOTHESIS_TESTS: Final[frozenset[HypothesisTestType]] = frozenset(
    {
        "t_test",
        "mann_whitney_u",
        "chi_square",
        "shapiro_wilk",
        "pearson_correlation",
        "cohens_d",
    }
)


def run_hypothesis_test_execution(
    payload: HypothesisTestExecutionInput,
) -> ExecutionStageOutput:
    """Execute a resolved hypothesis test question.

    Args:
        payload: Resolved hypothesis test execution payload.

    Returns:
        ExecutionStageOutput: Deterministic hypothesis test result.
    """
    data = apply_resolved_filters(payload.data, payload.filters)

    if payload.test == "shapiro_wilk":
        shapiro_data = data
        if payload.group_column is not None and payload.group_a_value is not None:
            shapiro_data = data.loc[data[payload.group_column] == payload.group_a_value]
        statistic, p_value = stats.shapiro(shapiro_data[payload.value_column])
        return _build_output(
            payload,
            statistic=float(statistic),
            p_value=float(p_value),
            raw_key="w_statistic",
        )

    if payload.test == "pearson_correlation":
        statistic, p_value = stats.pearsonr(
            data[payload.value_column], data[payload.second_value_column]
        )
        return _build_output(
            payload,
            statistic=float(statistic),
            p_value=float(p_value),
            raw_key="correlation",
        )

    if payload.test == "chi_square":
        contingency = pd.crosstab(
            data[payload.group_column],
            data[payload.value_column],
        )
        statistic, p_value, _, _ = stats.chi2_contingency(contingency)
        return _build_output(
            payload,
            statistic=float(statistic),
            p_value=float(p_value),
            raw_key="statistic",
        )

    left = data.loc[
        data[payload.group_column] == payload.group_a_value, payload.value_column
    ]
    right = data.loc[
        data[payload.group_column] == payload.group_b_value, payload.value_column
    ]

    if payload.test == "cohens_d":
        n_a, n_b = len(left), len(right)
        pooled_std = math.sqrt(
            ((n_a - 1) * float(left.var(ddof=1)) + (n_b - 1) * float(right.var(ddof=1)))
            / (n_a + n_b - 2)
        )
        effect_size = float((left.mean() - right.mean()) / pooled_std)
        return ExecutionStageOutput(
            family=payload.family,
            answer=format_scalar_answer(
                effect_size, payload.decimal_places, payload.round_to
            ),
            raw_result={"effect_size": effect_size},
        )

    if payload.test == "t_test":
        statistic, p_value = stats.ttest_ind(left, right, equal_var=False)
        return _build_output(
            payload,
            statistic=float(statistic),
            p_value=float(p_value),
            raw_key="statistic",
        )

    statistic, p_value = stats.mannwhitneyu(left, right, alternative="two-sided")
    return _build_output(
        payload,
        statistic=float(statistic),
        p_value=float(p_value),
        raw_key="u_statistic",
    )


def _build_output(
    payload: HypothesisTestExecutionInput,
    *,
    statistic: float,
    p_value: float,
    raw_key: str,
) -> ExecutionStageOutput:
    """Build a normalized hypothesis test output.

    Args:
        payload: Test execution payload.
        statistic: Main statistic value.
        p_value: Test p-value.
        raw_key: Key used for the statistic in raw output.

    Returns:
        ExecutionStageOutput: Formatted execution output.
    """
    if payload.return_field == "p_value":
        answer = format_scalar_answer(p_value, payload.decimal_places, payload.round_to)
    else:
        answer = format_scalar_answer(
            statistic, payload.decimal_places, payload.round_to
        )

    raw_result = {
        raw_key: statistic,
        "p_value": p_value,
    }
    return ExecutionStageOutput(
        family=payload.family,
        answer=answer,
        raw_result=raw_result,
    )
