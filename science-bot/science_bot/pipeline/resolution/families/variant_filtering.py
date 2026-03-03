"""Variant-filtering-family resolution models and plan builders."""

from collections.abc import Callable
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from science_bot.pipeline.contracts import (
    ScalarValue,
    VariantFilteringOperation,
    VariantFilteringReturnFormat,
)
from science_bot.pipeline.resolution.planning import (
    BaseResolutionDecision,
    ResolvedFilterPlan,
)


class VariantFilteringResolutionDecision(BaseResolutionDecision):
    """Flat resolver response for variant-filtering questions."""

    operation: VariantFilteringOperation | None = None
    sample_column: str | None = None
    sample_value: ScalarValue | None = None
    gene_column: str | None = None
    effect_column: str | None = None
    vaf_column: str | None = None
    vaf_min: float | None = None
    vaf_max: float | None = None
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_format: VariantFilteringReturnFormat | None = None
    decimal_places: int | None = None
    round_to: int | None = None


class VariantFilteringResolvedPlan(BaseModel):
    """Resolved variant-filtering plan referencing concrete file and columns."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["variant_filtering"] = "variant_filtering"
    filename: str
    operation: VariantFilteringOperation
    sample_column: str | None = None
    sample_value: ScalarValue | None = None
    gene_column: str | None = None
    effect_column: str | None = None
    vaf_column: str | None = None
    vaf_min: float | None = None
    vaf_max: float | None = None
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_format: VariantFilteringReturnFormat
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "VariantFilteringResolvedPlan":
        """Validate variant-filtering-specific resolved fields."""
        if (
            self.vaf_min is not None
            and self.vaf_max is not None
            and self.vaf_min > self.vaf_max
        ):
            raise ValueError("vaf_min must be less than or equal to vaf_max.")
        if (
            self.vaf_min is not None or self.vaf_max is not None
        ) and not self.vaf_column:
            raise ValueError("VAF filtering requires vaf_column.")
        if self.operation == "gene_with_max_variants" and not self.gene_column:
            raise ValueError("gene_with_max_variants requires gene_column.")
        if self.operation == "sample_variant_count":
            if not self.sample_column or self.sample_value is None:
                raise ValueError(
                    "sample_variant_count requires sample_column and sample_value."
                )
        return self


def build_variant_filtering_plan_from_decision(
    decision: VariantFilteringResolutionDecision,
    *,
    require_text: Callable[[str | None, str], str],
    require_value: Callable[[object | None, str], object],
) -> VariantFilteringResolvedPlan:
    """Convert a flat finalize decision into a variant-filtering plan."""
    return VariantFilteringResolvedPlan(
        filename=require_text(decision.filename, "filename"),
        operation=cast(
            VariantFilteringOperation,
            require_value(decision.operation, "operation"),
        ),
        sample_column=decision.sample_column,
        sample_value=decision.sample_value,
        gene_column=decision.gene_column,
        effect_column=decision.effect_column,
        vaf_column=decision.vaf_column,
        vaf_min=decision.vaf_min,
        vaf_max=decision.vaf_max,
        filters=decision.filters,
        return_format=cast(
            VariantFilteringReturnFormat,
            require_value(decision.return_format, "return_format"),
        ),
        decimal_places=decision.decimal_places,
        round_to=decision.round_to,
    )
