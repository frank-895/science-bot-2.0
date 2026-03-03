import pytest
from science_bot.pipeline.resolution.families.differential_expression import (
    DifferentialExpressionResolvedPlan,
)


def test_de_resolved_plan_requires_result_tables():
    with pytest.raises(ValueError):
        DifferentialExpressionResolvedPlan(
            mode="precomputed_results",
            result_table_files={},
            operation="significant_gene_count",
            comparison_labels=[],
        )


def test_raw_counts_plan_requires_count_matrix_file() -> None:
    with pytest.raises(ValueError):
        DifferentialExpressionResolvedPlan(
            mode="raw_counts",
            sample_metadata_file="metadata.csv",
            sample_metadata_sample_id_column="sample",
            design_factor_column="condition",
            tested_level="B",
            reference_level="A",
            count_matrix_orientation="genes_by_samples",
            count_matrix_gene_id_column="gene",
            operation="significant_gene_count",
            comparison_labels=["B vs A"],
        )


def test_raw_counts_plan_requires_metadata_file() -> None:
    with pytest.raises(ValueError):
        DifferentialExpressionResolvedPlan(
            mode="raw_counts",
            count_matrix_file="counts.csv",
            sample_metadata_sample_id_column="sample",
            design_factor_column="condition",
            tested_level="B",
            reference_level="A",
            count_matrix_orientation="genes_by_samples",
            count_matrix_gene_id_column="gene",
            operation="significant_gene_count",
            comparison_labels=["B vs A"],
        )


def test_raw_counts_plan_requires_design_factor_and_levels() -> None:
    with pytest.raises(ValueError):
        DifferentialExpressionResolvedPlan(
            mode="raw_counts",
            count_matrix_file="counts.csv",
            sample_metadata_file="metadata.csv",
            sample_metadata_sample_id_column="sample",
            tested_level="B",
            reference_level="A",
            count_matrix_orientation="genes_by_samples",
            count_matrix_gene_id_column="gene",
            operation="significant_gene_count",
            comparison_labels=["B vs A"],
        )


def test_raw_counts_plan_requires_orientation_specific_identifier_column() -> None:
    with pytest.raises(ValueError):
        DifferentialExpressionResolvedPlan(
            mode="raw_counts",
            count_matrix_file="counts.csv",
            sample_metadata_file="metadata.csv",
            sample_metadata_sample_id_column="sample",
            design_factor_column="condition",
            tested_level="B",
            reference_level="A",
            count_matrix_orientation="samples_by_genes",
            operation="significant_gene_count",
            comparison_labels=["B vs A"],
        )


def test_raw_counts_plan_requires_single_comparison_label() -> None:
    with pytest.raises(ValueError):
        DifferentialExpressionResolvedPlan(
            mode="raw_counts",
            count_matrix_file="counts.csv",
            sample_metadata_file="metadata.csv",
            sample_metadata_sample_id_column="sample",
            design_factor_column="condition",
            tested_level="B",
            reference_level="A",
            count_matrix_orientation="genes_by_samples",
            count_matrix_gene_id_column="gene",
            operation="significant_gene_count",
            comparison_labels=["B vs A", "C vs A"],
        )
