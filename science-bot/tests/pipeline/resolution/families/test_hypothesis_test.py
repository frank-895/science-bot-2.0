import pytest
from science_bot.pipeline.resolution.families.hypothesis_test import (
    HypothesisTestResolvedPlan,
)


def test_hypothesis_test_resolved_plan_requires_value_column_for_shapiro():
    with pytest.raises(ValueError):
        HypothesisTestResolvedPlan(
            filename="data.csv",
            test="shapiro_wilk",
            value_column=None,
            filters=[],
            return_field="w_statistic",
        )
