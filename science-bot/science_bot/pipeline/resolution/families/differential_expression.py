"""Differential-expression-family resolution models and plan builders."""

from collections.abc import Callable
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from science_bot.pipeline.contracts import DifferentialExpressionOperation
from science_bot.pipeline.resolution.planning import BaseResolutionDecision

SUPPORTED_RAW_COUNT_OPERATIONS = frozenset(
    {
        "significant_gene_count",
        "gene_log2_fold_change",
        "significant_marker_count",
    }
)


class ResultTableFileEntry(BaseModel):
    """One differential-expression comparison label to file mapping."""

    model_config = ConfigDict(extra="forbid")

    label: str
    filename: str


class DifferentialExpressionResolutionDecision(BaseResolutionDecision):
    """Flat resolver response for differential-expression questions."""

    mode: Literal["precomputed_results", "raw_counts"] | None = None
    result_table_files: list[ResultTableFileEntry] = Field(default_factory=list)
    count_matrix_file: str | None = None
    sample_metadata_file: str | None = None
    sample_metadata_sample_id_column: str | None = None
    design_factor_column: str | None = None
    tested_level: str | None = None
    reference_level: str | None = None
    count_matrix_orientation: Literal["genes_by_samples", "samples_by_genes"] | None = (
        None
    )
    count_matrix_gene_id_column: str | None = None
    count_matrix_sample_id_column: str | None = None
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
    """Resolved differential-expression plan."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["differential_expression"] = "differential_expression"
    mode: Literal["precomputed_results", "raw_counts"]
    result_table_files: dict[str, str] = Field(default_factory=dict)
    count_matrix_file: str | None = None
    sample_metadata_file: str | None = None
    sample_metadata_sample_id_column: str | None = None
    design_factor_column: str | None = None
    tested_level: str | None = None
    reference_level: str | None = None
    count_matrix_orientation: Literal["genes_by_samples", "samples_by_genes"] | None = (
        None
    )
    count_matrix_gene_id_column: str | None = None
    count_matrix_sample_id_column: str | None = None
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
        if self.mode == "precomputed_results":
            if not self.result_table_files:
                raise ValueError("precomputed_results requires result_table_files.")
        else:
            if self.operation not in SUPPORTED_RAW_COUNT_OPERATIONS:
                raise ValueError(
                    f"raw_counts mode does not support operation '{self.operation}'."
                )
            if self.count_matrix_file is None:
                raise ValueError("raw_counts requires count_matrix_file.")
            if self.sample_metadata_file is None:
                raise ValueError("raw_counts requires sample_metadata_file.")
            if self.sample_metadata_sample_id_column is None:
                raise ValueError(
                    "raw_counts requires sample_metadata_sample_id_column."
                )
            if self.design_factor_column is None:
                raise ValueError("raw_counts requires design_factor_column.")
            if self.tested_level is None:
                raise ValueError("raw_counts requires tested_level.")
            if self.reference_level is None:
                raise ValueError("raw_counts requires reference_level.")
            if self.count_matrix_orientation is None:
                raise ValueError("raw_counts requires count_matrix_orientation.")
            if len(self.comparison_labels) != 1:
                raise ValueError("raw_counts requires exactly one comparison_label.")
            if self.count_matrix_orientation == "genes_by_samples":
                if self.count_matrix_gene_id_column is None:
                    raise ValueError(
                        "genes_by_samples raw_counts requires "
                        "count_matrix_gene_id_column."
                    )
            elif self.count_matrix_sample_id_column is None:
                raise ValueError(
                    "samples_by_genes raw_counts requires "
                    "count_matrix_sample_id_column."
                )
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
            Literal["precomputed_results", "raw_counts"],
            require_value(decision.mode, "mode"),
        ),
        result_table_files=result_table_files_to_dict(decision.result_table_files),
        count_matrix_file=decision.count_matrix_file,
        sample_metadata_file=decision.sample_metadata_file,
        sample_metadata_sample_id_column=decision.sample_metadata_sample_id_column,
        design_factor_column=decision.design_factor_column,
        tested_level=decision.tested_level,
        reference_level=decision.reference_level,
        count_matrix_orientation=decision.count_matrix_orientation,
        count_matrix_gene_id_column=decision.count_matrix_gene_id_column,
        count_matrix_sample_id_column=decision.count_matrix_sample_id_column,
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
