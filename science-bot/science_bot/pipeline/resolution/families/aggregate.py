"""Aggregate-family resolution models and plan builders."""

from collections.abc import Callable
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from science_bot.pipeline.contracts import AggregateOperation, AggregateReturnFormat
from science_bot.pipeline.resolution.planning import (
    BaseResolutionDecision,
    MetadataJoinPlan,
    MultiFileMergePlan,
    MultiFileSourceEntry,
    ResolvedFilterPlan,
)


class AggregateResolutionDecision(BaseResolutionDecision):
    """Flat resolver response for aggregate questions."""

    use_merge: bool = False
    data_source_files: list[str] = Field(default_factory=list)
    data_source_sample_ids: list[str] = Field(default_factory=list)
    data_source_selected_columns: list[str] = Field(default_factory=list)
    metadata_file: str | None = None
    metadata_sample_id_column: str | None = None
    metadata_columns: list[str] = Field(default_factory=list)
    output_sample_id_column: str = "sample_id"
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
    filename: str | None = None
    merge_plan: MultiFileMergePlan | None = None
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
        if (self.filename is None) == (self.merge_plan is None):
            raise ValueError(
                "Aggregate plan requires exactly one of filename or merge_plan."
            )
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

    return AggregateResolvedPlan(
        filename=filename,
        merge_plan=merge_plan,
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
