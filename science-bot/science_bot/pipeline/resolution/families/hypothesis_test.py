"""Hypothesis-test-family resolution models and plan builders."""

from collections.abc import Callable
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from science_bot.pipeline.contracts import (
    HypothesisTestReturnField,
    HypothesisTestType,
    ScalarValue,
)
from science_bot.pipeline.resolution.planning import (
    BaseResolutionDecision,
    ResolvedFilterPlan,
)


class HypothesisTestResolutionDecision(BaseResolutionDecision):
    """Flat resolver response for hypothesis-test questions."""

    test: HypothesisTestType | None = None
    value_column: str | None = None
    second_value_column: str | None = None
    group_column: str | None = None
    group_a_value: ScalarValue | None = None
    group_b_value: ScalarValue | None = None
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_field: HypothesisTestReturnField | None = None
    decimal_places: int | None = None
    round_to: int | None = None


class HypothesisTestResolvedPlan(BaseModel):
    """Resolved hypothesis-test plan referencing concrete file and columns."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["hypothesis_test"] = "hypothesis_test"
    filename: str
    test: HypothesisTestType
    value_column: str | None = None
    second_value_column: str | None = None
    group_column: str | None = None
    group_a_value: ScalarValue | None = None
    group_b_value: ScalarValue | None = None
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_field: HypothesisTestReturnField
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "HypothesisTestResolvedPlan":
        """Validate hypothesis-test-specific resolved fields."""
        if self.test == "pearson_correlation":
            if not self.value_column or not self.second_value_column:
                raise ValueError(
                    "pearson_correlation requires value_column and second_value_column."
                )
        elif self.test == "shapiro_wilk":
            if not self.value_column:
                raise ValueError("shapiro_wilk requires value_column.")
        elif self.test in {"t_test", "mann_whitney_u", "cohens_d"}:
            if not self.value_column or not self.group_column:
                raise ValueError(f"{self.test} requires value_column and group_column.")
            if self.group_a_value is None or self.group_b_value is None:
                raise ValueError(
                    f"{self.test} requires group_a_value and group_b_value."
                )
        elif self.test == "chi_square":
            if not self.value_column or not self.group_column:
                raise ValueError("chi_square requires value_column and group_column.")
        return self


def build_hypothesis_test_plan_from_decision(
    decision: HypothesisTestResolutionDecision,
    *,
    require_text: Callable[[str | None, str], str],
    require_value: Callable[[object | None, str], object],
) -> HypothesisTestResolvedPlan:
    """Convert a flat finalize decision into a hypothesis-test plan."""
    return HypothesisTestResolvedPlan(
        filename=require_text(decision.filename, "filename"),
        test=cast(HypothesisTestType, require_value(decision.test, "test")),
        value_column=decision.value_column,
        second_value_column=decision.second_value_column,
        group_column=decision.group_column,
        group_a_value=decision.group_a_value,
        group_b_value=decision.group_b_value,
        filters=decision.filters,
        return_field=cast(
            HypothesisTestReturnField,
            require_value(decision.return_field, "return_field"),
        ),
        decimal_places=decision.decimal_places,
        round_to=decision.round_to,
    )
