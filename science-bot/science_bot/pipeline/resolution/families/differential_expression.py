"""Differential-expression-family resolution models and plan builders."""

from collections.abc import Callable
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from science_bot.pipeline.contracts import DifferentialExpressionOperation
from science_bot.pipeline.resolution.planning import BaseResolutionDecision


class ResultTableFileEntry(BaseModel):
    """One differential-expression comparison label to file mapping."""

    model_config = ConfigDict(extra="forbid")

    label: str
    filename: str


class DifferentialExpressionResolutionDecision(BaseResolutionDecision):
    """Flat resolver response for differential-expression questions."""

    mode: Literal["precomputed_results"] | None = None
    result_table_files: list[ResultTableFileEntry] = Field(default_factory=list)
    operation: DifferentialExpressionOperation | None = None
    comparison_labels: list[str] = Field(default_factory=list)
    reference_label: str | None = None
    target_gene: str | None = None
    gene_column: str | None = None
    log_fold_change_column: str | None = None
    adjusted_p_value_column: str | None = None
    base_mean_column: str | None = None
    significance_threshold: float | None = None
    fold_change_threshold: float | None = None
    basemean_threshold: float | None = None
    use_lfc_shrinkage: bool = False
    correction_methods: list[str] = Field(default_factory=list)
    decimal_places: int | None = None
    round_to: int | None = None


class DifferentialExpressionResolvedPlan(BaseModel):
    """Resolved differential-expression plan for precomputed results."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["differential_expression"] = "differential_expression"
    mode: Literal["precomputed_results"]
    result_table_files: dict[str, str] = Field(default_factory=dict)
    operation: DifferentialExpressionOperation
    comparison_labels: list[str] = Field(default_factory=list)
    reference_label: str | None = None
    target_gene: str | None = None
    gene_column: str | None = None
    log_fold_change_column: str | None = None
    adjusted_p_value_column: str | None = None
    base_mean_column: str | None = None
    significance_threshold: float | None = None
    fold_change_threshold: float | None = None
    basemean_threshold: float | None = None
    use_lfc_shrinkage: bool = False
    correction_methods: list[str] = Field(default_factory=list)
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "DifferentialExpressionResolvedPlan":
        """Validate differential-expression-specific resolved fields."""
        if not self.result_table_files:
            raise ValueError("precomputed_results requires result_table_files.")
        if self.operation == "gene_log2_fold_change" and not self.target_gene:
            raise ValueError("gene_log2_fold_change requires target_gene.")
        if (
            self.operation
            in {"shared_overlap_pattern", "unique_significant_gene_count"}
            and len(self.comparison_labels) < 2
        ):
            raise ValueError(
                f"{self.operation} requires at least two comparison_labels."
            )
        if self.operation == "correction_ratio" and len(self.correction_methods) < 2:
            raise ValueError(
                "correction_ratio requires at least two correction_methods."
            )
        return self


def result_table_files_to_dict(entries: list[ResultTableFileEntry]) -> dict[str, str]:
    """Convert structured result table file entries into a mapping."""
    return {entry.label: entry.filename for entry in entries}


def build_differential_expression_plan_from_decision(
    decision: DifferentialExpressionResolutionDecision,
    *,
    require_value: Callable[[object | None, str], object],
) -> DifferentialExpressionResolvedPlan:
    """Convert a flat finalize decision into a differential-expression plan."""
    return DifferentialExpressionResolvedPlan(
        mode=cast(
            Literal["precomputed_results"],
            require_value(decision.mode, "mode"),
        ),
        result_table_files=result_table_files_to_dict(decision.result_table_files),
        operation=cast(
            DifferentialExpressionOperation,
            require_value(decision.operation, "operation"),
        ),
        comparison_labels=decision.comparison_labels,
        reference_label=decision.reference_label,
        target_gene=decision.target_gene,
        gene_column=decision.gene_column,
        log_fold_change_column=decision.log_fold_change_column,
        adjusted_p_value_column=decision.adjusted_p_value_column,
        base_mean_column=decision.base_mean_column,
        significance_threshold=decision.significance_threshold,
        fold_change_threshold=decision.fold_change_threshold,
        basemean_threshold=decision.basemean_threshold,
        use_lfc_shrinkage=decision.use_lfc_shrinkage,
        correction_methods=decision.correction_methods,
        decimal_places=decision.decimal_places,
        round_to=decision.round_to,
    )
