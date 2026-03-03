"""Regression-family resolution models and plan builders."""

from collections.abc import Callable
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from science_bot.pipeline.contracts import (
    RegressionModelType,
    RegressionReturnField,
    ScalarValue,
)
from science_bot.pipeline.resolution.planning import (
    BaseResolutionDecision,
    ResolvedFilterPlan,
)


class PredictionInputEntry(BaseModel):
    """One regression prediction input value."""

    model_config = ConfigDict(extra="forbid")

    column: str
    value: ScalarValue


class RegressionResolutionDecision(BaseResolutionDecision):
    """Flat resolver response for regression questions."""

    model_type: RegressionModelType | None = None
    outcome_column: str | None = None
    predictor_column: str | None = None
    covariate_columns: list[str] = Field(default_factory=list)
    degree: int | None = None
    prediction_inputs: list[PredictionInputEntry] = Field(default_factory=list)
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_field: RegressionReturnField | None = None
    decimal_places: int | None = None
    round_to: int | None = None


class RegressionResolvedPlan(BaseModel):
    """Resolved regression plan referencing concrete file and columns."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["regression"] = "regression"
    filename: str
    model_type: RegressionModelType
    outcome_column: str
    predictor_column: str
    covariate_columns: list[str] = Field(default_factory=list)
    degree: int | None = None
    prediction_inputs: dict[str, ScalarValue] = Field(default_factory=dict)
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_field: RegressionReturnField
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "RegressionResolvedPlan":
        """Validate regression-specific resolved fields."""
        if self.model_type == "polynomial":
            if self.degree is None:
                raise ValueError("polynomial regression requires degree.")
        elif self.degree is not None:
            raise ValueError("Only polynomial regression may set degree.")
        if self.return_field == "predicted_probability" and not self.prediction_inputs:
            raise ValueError("predicted_probability requires prediction_inputs.")
        return self


def prediction_inputs_to_dict(
    entries: list[PredictionInputEntry],
) -> dict[str, ScalarValue]:
    """Convert structured prediction input entries into a mapping."""
    return {entry.column: entry.value for entry in entries}


def build_regression_plan_from_decision(
    decision: RegressionResolutionDecision,
    *,
    require_text: Callable[[str | None, str], str],
    require_value: Callable[[object | None, str], object],
) -> RegressionResolvedPlan:
    """Convert a flat finalize decision into a regression plan."""
    return RegressionResolvedPlan(
        filename=require_text(decision.filename, "filename"),
        model_type=cast(
            RegressionModelType,
            require_value(decision.model_type, "model_type"),
        ),
        outcome_column=require_text(decision.outcome_column, "outcome_column"),
        predictor_column=require_text(decision.predictor_column, "predictor_column"),
        covariate_columns=decision.covariate_columns,
        degree=decision.degree,
        prediction_inputs=prediction_inputs_to_dict(decision.prediction_inputs),
        filters=decision.filters,
        return_field=cast(
            RegressionReturnField,
            require_value(decision.return_field, "return_field"),
        ),
        decimal_places=decision.decimal_places,
        round_to=decision.round_to,
    )
