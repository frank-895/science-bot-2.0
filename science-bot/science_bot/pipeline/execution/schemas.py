"""Schemas local to the execution stage."""

from typing import Annotated, Literal, TypeAlias

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from science_bot.pipeline.contracts import (
    AggregateOperation,
    AggregateReturnFormat,
    DifferentialExpressionOperation,
    HypothesisTestReturnField,
    HypothesisTestType,
    QuestionFamily,
    RegressionModelType,
    RegressionReturnField,
    VariantFilteringOperation,
    VariantFilteringReturnFormat,
)

FilterValue: TypeAlias = str | int | float | bool | list[str] | list[int] | list[float]
DifferentialExpressionExecutionMode: TypeAlias = Literal[
    "precomputed_results",
    "raw_counts",
]
CountMatrixOrientation: TypeAlias = Literal["genes_by_samples", "samples_by_genes"]


class ResolvedFilter(BaseModel):
    """Resolved filter condition ready for deterministic execution."""

    model_config = ConfigDict(extra="forbid")

    column: str
    operator: Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains"]
    value: FilterValue


class AggregateExecutionInput(BaseModel):
    """Execution payload for aggregate questions."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    family: Literal["aggregate"]
    operation: AggregateOperation
    data: pd.DataFrame
    value_column: str | None = None
    numerator_mask_column: str | None = None
    denominator_mask_column: str | None = None
    numerator_filters: list[ResolvedFilter] = Field(default_factory=list)
    denominator_filters: list[ResolvedFilter] = Field(default_factory=list)
    filters: list[ResolvedFilter] = Field(default_factory=list)
    return_format: AggregateReturnFormat
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "AggregateExecutionInput":
        """Validate aggregate execution input consistency.

        Returns:
            AggregateExecutionInput: Validated aggregate execution input.

        Raises:
            ValueError: If required fields are missing.
        """
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
                    "percentage, proportion, and ratio require numerator_mask_column "
                    "or numerator_filters."
                )
        if (
            self.operation == "ratio"
            and not self.denominator_mask_column
            and not self.denominator_filters
        ):
            raise ValueError(
                "ratio requires denominator_mask_column or denominator_filters."
            )
        return self


class HypothesisTestExecutionInput(BaseModel):
    """Execution payload for classical hypothesis tests."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    family: Literal["hypothesis_test"]
    test: HypothesisTestType
    data: pd.DataFrame
    value_column: str | None = None
    second_value_column: str | None = None
    group_column: str | None = None
    group_a_value: str | int | float | bool | None = None
    group_b_value: str | int | float | bool | None = None
    filters: list[ResolvedFilter] = Field(default_factory=list)
    return_field: HypothesisTestReturnField
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "HypothesisTestExecutionInput":
        """Validate hypothesis test execution input consistency.

        Returns:
            HypothesisTestExecutionInput: Validated hypothesis test execution input.

        Raises:
            ValueError: If required fields are missing.
        """
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


class RegressionExecutionInput(BaseModel):
    """Execution payload for regression questions."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    family: Literal["regression"]
    model_type: RegressionModelType
    data: pd.DataFrame
    outcome_column: str
    predictor_column: str
    covariate_columns: list[str] = Field(default_factory=list)
    degree: int | None = None
    prediction_inputs: dict[str, str | int | float | bool] = Field(default_factory=dict)
    filters: list[ResolvedFilter] = Field(default_factory=list)
    return_field: RegressionReturnField
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "RegressionExecutionInput":
        """Validate regression execution input consistency.

        Returns:
            RegressionExecutionInput: Validated regression execution input.

        Raises:
            ValueError: If selected options are inconsistent.
        """
        if self.model_type == "polynomial":
            if self.degree is None:
                raise ValueError("polynomial regression requires degree.")
        elif self.degree is not None:
            raise ValueError("Only polynomial regression may set degree.")

        if self.return_field == "predicted_probability" and not self.prediction_inputs:
            raise ValueError("predicted_probability requires prediction_inputs.")

        if self.return_field in {"odds_ratio", "percent_increase_in_odds"} and (
            self.model_type not in {"logistic", "ordinal_logistic"}
        ):
            raise ValueError(
                "odds_ratio and percent_increase_in_odds require logistic "
                "or ordinal_logistic."
            )

        if self.return_field == "r_squared" and self.model_type not in {
            "linear",
            "polynomial",
        }:
            raise ValueError("r_squared requires linear or polynomial regression.")
        return self


class DifferentialExpressionExecutionInput(BaseModel):
    """Execution payload for differential expression questions."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    family: Literal["differential_expression"]
    mode: DifferentialExpressionExecutionMode
    operation: DifferentialExpressionOperation
    result_tables: dict[str, pd.DataFrame] = Field(default_factory=dict)
    count_matrix: pd.DataFrame | None = None
    sample_metadata: pd.DataFrame | None = None
    sample_metadata_sample_id_column: str | None = None
    design_factor_column: str | None = None
    tested_level: str | None = None
    reference_level: str | None = None
    count_matrix_orientation: CountMatrixOrientation | None = None
    count_matrix_gene_id_column: str | None = None
    count_matrix_sample_id_column: str | None = None
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
    def validate_fields(self) -> "DifferentialExpressionExecutionInput":
        """Validate differential expression execution input consistency.

        Returns:
            DifferentialExpressionExecutionInput: Validated DE execution input.

        Raises:
            ValueError: If required fields are missing.
        """
        if self.mode == "precomputed_results":
            if not self.result_tables:
                raise ValueError("precomputed_results mode requires result_tables.")
        else:
            if self.count_matrix is None or self.sample_metadata is None:
                raise ValueError(
                    "raw_counts mode requires count_matrix and sample_metadata."
                )
            if self.sample_metadata_sample_id_column is None:
                raise ValueError(
                    "raw_counts mode requires sample_metadata_sample_id_column."
                )
            if self.design_factor_column is None:
                raise ValueError("raw_counts mode requires design_factor_column.")
            if self.tested_level is None:
                raise ValueError("raw_counts mode requires tested_level.")
            if self.reference_level is None:
                raise ValueError("raw_counts mode requires reference_level.")
            if self.count_matrix_orientation is None:
                raise ValueError("raw_counts mode requires count_matrix_orientation.")
            if len(self.comparison_labels) != 1:
                raise ValueError(
                    "raw_counts mode requires exactly one comparison_label."
                )
            if self.operation not in {
                "significant_gene_count",
                "gene_log2_fold_change",
                "significant_marker_count",
            }:
                raise ValueError(
                    f"raw_counts mode does not support operation '{self.operation}'."
                )
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
        if self.operation in {
            "shared_overlap_pattern",
            "unique_significant_gene_count",
        }:
            if len(self.comparison_labels) < 2:
                raise ValueError(
                    f"{self.operation} requires at least two comparison_labels."
                )
        if self.operation == "correction_ratio" and len(self.correction_methods) < 2:
            raise ValueError(
                "correction_ratio requires at least two correction_methods."
            )
        return self


class VariantFilteringExecutionInput(BaseModel):
    """Execution payload for variant filtering questions."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    family: Literal["variant_filtering"]
    operation: VariantFilteringOperation
    data: pd.DataFrame
    sample_column: str | None = None
    sample_value: str | int | float | bool | None = None
    gene_column: str | None = None
    effect_column: str | None = None
    vaf_column: str | None = None
    vaf_min: float | None = None
    vaf_max: float | None = None
    filters: list[ResolvedFilter] = Field(default_factory=list)
    return_format: VariantFilteringReturnFormat
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "VariantFilteringExecutionInput":
        """Validate variant filtering execution input consistency.

        Returns:
            VariantFilteringExecutionInput: Validated variant filtering execution input.

        Raises:
            ValueError: If required fields are missing.
        """
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


ExecutionPayload: TypeAlias = Annotated[
    AggregateExecutionInput
    | HypothesisTestExecutionInput
    | RegressionExecutionInput
    | DifferentialExpressionExecutionInput
    | VariantFilteringExecutionInput,
    Field(discriminator="family"),
]


class ExecutionStageInput(BaseModel):
    """Input contract for the execution stage."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    payload: ExecutionPayload


class ExecutionStageOutput(BaseModel):
    """Output contract for the execution stage."""

    model_config = ConfigDict(extra="forbid")

    family: QuestionFamily
    answer: str
    raw_result: dict[str, object] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
