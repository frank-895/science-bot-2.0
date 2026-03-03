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
    MetadataJoinPlan,
    MultiFileMergePlan,
    MultiFileSourceEntry,
    ResolvedFilterPlan,
)


class VariantFilteringResolutionDecision(BaseResolutionDecision):
    """Flat resolver response for variant-filtering questions."""

    use_merge: bool = False
    data_source_files: list[str] = Field(default_factory=list)
    data_source_sample_ids: list[str] = Field(default_factory=list)
    data_source_selected_columns: list[str] = Field(default_factory=list)
    metadata_file: str | None = None
    metadata_sample_id_column: str | None = None
    metadata_columns: list[str] = Field(default_factory=list)
    output_sample_id_column: str = "sample_id"
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
    filename: str | None = None
    merge_plan: MultiFileMergePlan | None = None
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
        if (self.filename is None) == (self.merge_plan is None):
            raise ValueError(
                "Variant filtering plan requires exactly one of filename or merge_plan."
            )
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
    merge_plan: MultiFileMergePlan | None = None
    filename: str | None = None
    if getattr(decision, "use_merge", False):
        data_source_files = decision.data_source_files
        data_source_sample_ids = decision.data_source_sample_ids
        if len(data_source_files) != len(data_source_sample_ids):
            raise ValueError(
                "use_merge requires data_source_files and data_source_sample_ids to "
                "have the same length."
            )
        if not data_source_files:
            raise ValueError("use_merge requires at least one data source file.")
        join: MetadataJoinPlan | None = None
        if decision.metadata_file is not None:
            join = MetadataJoinPlan(
                metadata_file=require_text(decision.metadata_file, "metadata_file"),
                metadata_sample_id_column=require_text(
                    decision.metadata_sample_id_column,
                    "metadata_sample_id_column",
                ),
                metadata_columns=decision.metadata_columns,
            )
        merge_plan = MultiFileMergePlan(
            data_sources=[
                MultiFileSourceEntry(
                    filename=data_filename,
                    sample_id=sample_id,
                    selected_columns=decision.data_source_selected_columns,
                )
                for data_filename, sample_id in zip(
                    data_source_files,
                    data_source_sample_ids,
                    strict=True,
                )
            ],
            join=join,
            output_sample_id_column=decision.output_sample_id_column,
        )
    else:
        filename = require_text(decision.filename, "filename")

    return VariantFilteringResolvedPlan(
        filename=filename,
        merge_plan=merge_plan,
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
