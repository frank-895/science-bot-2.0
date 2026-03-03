import pytest
from science_bot.pipeline.resolution.families.regression import RegressionResolvedPlan


def test_regression_resolved_plan_requires_degree_for_polynomial():
    with pytest.raises(ValueError):
        RegressionResolvedPlan(
            filename="data.csv",
            model_type="polynomial",
            outcome_column="y",
            predictor_column="x",
            covariate_columns=[],
            degree=None,
            prediction_inputs={},
            filters=[],
            return_field="r_squared",
        )
