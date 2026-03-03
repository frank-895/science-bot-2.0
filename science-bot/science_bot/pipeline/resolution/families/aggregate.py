"""Aggregate-family resolution models and plan builders."""

from collections.abc import Callable
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from science_bot.pipeline.contracts import AggregateOperation, AggregateReturnFormat
from science_bot.pipeline.resolution.planning import (
    BaseResolutionDecision,
    ResolvedFilterPlan,
)


class AggregateResolutionDecision(BaseResolutionDecision):
    """Flat resolver response for aggregate questions."""

    operation: AggregateOperation | None = None
    value_column: str | None = None
    numerator_mask_column: str | None = None
    denominator_mask_column: str | None = None
    numerator_filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    denominator_filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_format: AggregateReturnFormat | None = None
    decimal_places: int | None = None
    round_to: int | None = None


class AggregateResolvedPlan(BaseModel):
    """Resolved aggregate plan referencing concrete file and column names."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["aggregate"] = "aggregate"
    filename: str
    operation: AggregateOperation
    value_column: str | None = None
    numerator_mask_column: str | None = None
    denominator_mask_column: str | None = None
    numerator_filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    denominator_filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_format: AggregateReturnFormat
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "AggregateResolvedPlan":
        """Validate aggregate-specific resolved fields."""
        if (
            self.operation in {"mean", "median", "variance", "skewness"}
            and not self.value_column
        ):
            raise ValueError(
                "mean, median, variance, and skewness require value_column."
            )
        if self.operation in {"percentage", "proportion", "ratio"}:
            if not self.numerator_mask_column and not self.numerator_filters:
                raise ValueError(
                    "percentage, proportion, and ratio require numerator mask "
                    "or numerator filters."
                )
        if self.operation == "ratio":
            if not self.denominator_mask_column and not self.denominator_filters:
                raise ValueError(
                    "ratio requires denominator mask or denominator filters."
                )
        return self


def build_aggregate_plan_from_decision(
    decision: AggregateResolutionDecision,
    *,
    require_text: Callable[[str | None, str], str],
    require_value: Callable[[object | None, str], object],
) -> AggregateResolvedPlan:
    """Convert a flat finalize decision into an aggregate plan."""
    return AggregateResolvedPlan(
        filename=require_text(decision.filename, "filename"),
        operation=cast(
            AggregateOperation,
            require_value(decision.operation, "operation"),
        ),
        value_column=decision.value_column,
        numerator_mask_column=decision.numerator_mask_column,
        denominator_mask_column=decision.denominator_mask_column,
        numerator_filters=decision.numerator_filters,
        denominator_filters=decision.denominator_filters,
        filters=decision.filters,
        return_format=cast(
            AggregateReturnFormat,
            require_value(decision.return_format, "return_format"),
        ),
        decimal_places=decision.decimal_places,
        round_to=decision.round_to,
    )
