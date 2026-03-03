import pandas as pd
import pytest
import science_bot.pipeline.execution.differential_expression as de_execution
from science_bot.pipeline.execution.schemas import DifferentialExpressionExecutionInput


def test_significant_gene_count() -> None:
    table = pd.DataFrame(
        {
            "gene": ["a", "b", "c"],
            "log2FoldChange": [2.0, 0.2, -3.0],
            "padj": [0.01, 0.01, 0.2],
            "baseMean": [11, 11, 11],
        }
    )

    result = de_execution.run_differential_expression_execution(
        DifferentialExpressionExecutionInput(
            family="differential_expression",
            mode="precomputed_results",
            operation="significant_gene_count",
            result_tables={"comp": table},
            comparison_labels=["comp"],
            adjusted_p_value_column="padj",
            log_fold_change_column="log2FoldChange",
            base_mean_column="baseMean",
            significance_threshold=0.05,
            fold_change_threshold=1.0,
            basemean_threshold=10.0,
        )
    )

    assert result.answer == "1"


def test_unique_significant_gene_count() -> None:
    primary = pd.DataFrame(
        {
            "gene": ["a", "b"],
            "log2FoldChange": [2.0, 2.0],
            "padj": [0.01, 0.01],
            "baseMean": [11, 11],
        }
    )
    secondary = pd.DataFrame(
        {
            "gene": ["b", "c"],
            "log2FoldChange": [2.0, 2.0],
            "padj": [0.01, 0.01],
            "baseMean": [11, 11],
        }
    )

    result = de_execution.run_differential_expression_execution(
        DifferentialExpressionExecutionInput(
            family="differential_expression",
            mode="precomputed_results",
            operation="unique_significant_gene_count",
            result_tables={"primary": primary, "secondary": secondary},
            comparison_labels=["primary", "secondary"],
            gene_column="gene",
            adjusted_p_value_column="padj",
            log_fold_change_column="log2FoldChange",
            base_mean_column="baseMean",
            significance_threshold=0.05,
            fold_change_threshold=1.0,
            basemean_threshold=10.0,
        )
    )

    assert result.answer == "1"


def test_shared_overlap_pattern_label() -> None:
    first = pd.DataFrame(
        {
            "gene": ["a"],
            "log2FoldChange": [2.0],
            "padj": [0.01],
            "baseMean": [11],
        }
    )
    second = pd.DataFrame(
        {
            "gene": ["b"],
            "log2FoldChange": [2.0],
            "padj": [0.01],
            "baseMean": [11],
        }
    )

    result = de_execution.run_differential_expression_execution(
        DifferentialExpressionExecutionInput(
            family="differential_expression",
            mode="precomputed_results",
            operation="shared_overlap_pattern",
            result_tables={"first": first, "second": second},
            comparison_labels=["first", "second"],
            gene_column="gene",
            adjusted_p_value_column="padj",
            log_fold_change_column="log2FoldChange",
            base_mean_column="baseMean",
            significance_threshold=0.05,
            fold_change_threshold=1.0,
            basemean_threshold=10.0,
        )
    )

    assert result.answer == "No overlap between any groups"


def test_gene_log2_fold_change_lookup() -> None:
    table = pd.DataFrame(
        {
            "gene": ["PA14_35160"],
            "log2FoldChange": [-4.1],
            "padj": [0.01],
            "baseMean": [11],
        }
    )

    result = de_execution.run_differential_expression_execution(
        DifferentialExpressionExecutionInput(
            family="differential_expression",
            mode="precomputed_results",
            operation="gene_log2_fold_change",
            result_tables={"comp": table},
            comparison_labels=["comp"],
            target_gene="PA14_35160",
            gene_column="gene",
            log_fold_change_column="log2FoldChange",
            decimal_places=2,
        )
    )

    assert result.answer == "-4.10"


def test_correction_ratio() -> None:
    bonferroni = pd.DataFrame({"adjusted_p_value": [0.2, 0.3]})
    by = pd.DataFrame({"adjusted_p_value": [0.2, 0.3]})

    result = de_execution.run_differential_expression_execution(
        DifferentialExpressionExecutionInput(
            family="differential_expression",
            mode="precomputed_results",
            operation="correction_ratio",
            result_tables={"bonferroni": bonferroni, "by": by},
            comparison_labels=["markers"],
            correction_methods=["bonferroni", "by"],
            adjusted_p_value_column="adjusted_p_value",
            significance_threshold=0.05,
        )
    )

    assert result.answer == "0:0"


def test_raw_counts_significant_gene_count_compiles_to_canonical_table(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        de_execution,
        "_run_raw_counts_differential_expression",
        lambda payload: pd.DataFrame(
            {
                "gene": ["a", "b"],
                "baseMean": [11, 11],
                "log2FoldChange": [2.0, 0.2],
                "pvalue": [0.001, 0.2],
                "padj": [0.01, 0.2],
            }
        ),
    )

    result = de_execution.run_differential_expression_execution(
        _raw_counts_payload(operation="significant_gene_count")
    )

    assert result.answer == "1"


def test_raw_counts_gene_log2_fold_change_compiles_to_canonical_table(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        de_execution,
        "_run_raw_counts_differential_expression",
        lambda payload: pd.DataFrame(
            {
                "gene": ["PA14_35160"],
                "baseMean": [11],
                "log2FoldChange": [-4.1],
                "pvalue": [0.001],
                "padj": [0.01],
            }
        ),
    )

    result = de_execution.run_differential_expression_execution(
        _raw_counts_payload(
            operation="gene_log2_fold_change",
            target_gene="PA14_35160",
            decimal_places=2,
        )
    )

    assert result.answer == "-4.10"


def test_raw_counts_significant_marker_count_compiles_to_canonical_table(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        de_execution,
        "_run_raw_counts_differential_expression",
        lambda payload: pd.DataFrame(
            {
                "gene": ["a", "b", "c"],
                "baseMean": [11, 11, 11],
                "log2FoldChange": [2.0, 0.2, -3.0],
                "pvalue": [0.001, 0.2, 0.001],
                "padj": [0.01, 0.2, 0.01],
            }
        ),
    )

    result = de_execution.run_differential_expression_execution(
        _raw_counts_payload(operation="significant_marker_count")
    )

    assert result.answer == "2"


def test_raw_counts_unique_significant_gene_count_is_deferred() -> None:
    with pytest.raises(
        ValueError,
        match="raw_counts mode does not support operation",
    ):
        _raw_counts_payload(operation="unique_significant_gene_count")


def test_raw_counts_shared_overlap_pattern_is_deferred() -> None:
    with pytest.raises(
        ValueError,
        match="raw_counts mode does not support operation",
    ):
        _raw_counts_payload(operation="shared_overlap_pattern")


def test_raw_counts_correction_ratio_is_deferred() -> None:
    with pytest.raises(
        ValueError,
        match="raw_counts mode does not support operation",
    ):
        _raw_counts_payload(operation="correction_ratio")


def test_significant_gene_count_supports_noncanonical_columns() -> None:
    table = pd.DataFrame(
        {
            "feature_id": ["a", "b", "c"],
            "lfc": [2.0, 0.2, -3.0],
            "fdr": [0.01, 0.01, 0.2],
            "mean_count": [11, 11, 11],
        }
    )

    result = de_execution.run_differential_expression_execution(
        DifferentialExpressionExecutionInput(
            family="differential_expression",
            mode="precomputed_results",
            operation="significant_gene_count",
            result_tables={"comp": table},
            comparison_labels=["comp"],
            adjusted_p_value_column="fdr",
            log_fold_change_column="lfc",
            base_mean_column="mean_count",
            significance_threshold=0.05,
            fold_change_threshold=1.0,
            basemean_threshold=10.0,
        )
    )

    assert result.answer == "1"


def test_gene_log2_fold_change_requires_resolved_columns() -> None:
    table = pd.DataFrame({"gene": ["PA14_35160"], "log2FoldChange": [-4.1]})

    with pytest.raises(
        ValueError,
        match="Differential expression execution requires gene_column.",
    ):
        de_execution.run_differential_expression_execution(
            DifferentialExpressionExecutionInput(
                family="differential_expression",
                mode="precomputed_results",
                operation="gene_log2_fold_change",
                result_tables={"comp": table},
                comparison_labels=["comp"],
                target_gene="PA14_35160",
            )
        )


def test_prepare_deseq2_inputs_raises_on_sample_mismatch() -> None:
    payload = DifferentialExpressionExecutionInput(
        family="differential_expression",
        mode="raw_counts",
        operation="significant_gene_count",
        count_matrix=pd.DataFrame({"gene": ["g1"], "s1": [1]}),
        sample_metadata=pd.DataFrame(
            {
                "sample": ["s1", "s2"],
                "condition": ["A", "B"],
            }
        ),
        sample_metadata_sample_id_column="sample",
        design_factor_column="condition",
        tested_level="A",
        reference_level="B",
        count_matrix_orientation="genes_by_samples",
        count_matrix_gene_id_column="gene",
        comparison_labels=["A vs B"],
    )

    with pytest.raises(
        ValueError,
        match="Count matrix is missing samples referenced by metadata",
    ):
        de_execution._prepare_deseq2_inputs(payload)


def test_prepare_deseq2_inputs_raises_on_missing_tested_level() -> None:
    payload = DifferentialExpressionExecutionInput(
        family="differential_expression",
        mode="raw_counts",
        operation="significant_gene_count",
        count_matrix=pd.DataFrame({"gene": ["g1"], "s1": [1], "s2": [2]}),
        sample_metadata=pd.DataFrame(
            {
                "sample": ["s1", "s2"],
                "condition": ["A", "A"],
            }
        ),
        sample_metadata_sample_id_column="sample",
        design_factor_column="condition",
        tested_level="B",
        reference_level="A",
        count_matrix_orientation="genes_by_samples",
        count_matrix_gene_id_column="gene",
        comparison_labels=["B vs A"],
    )

    with pytest.raises(
        ValueError,
        match="Sample metadata is missing contrast levels",
    ):
        de_execution._prepare_deseq2_inputs(payload)


def _raw_counts_payload(
    *,
    operation: str,
    target_gene: str | None = None,
    decimal_places: int | None = None,
) -> DifferentialExpressionExecutionInput:
    return DifferentialExpressionExecutionInput(
        family="differential_expression",
        mode="raw_counts",
        operation=operation,
        count_matrix=pd.DataFrame({"gene": ["g1"], "s1": [1], "s2": [2]}),
        sample_metadata=pd.DataFrame(
            {
                "sample": ["s1", "s2"],
                "condition": ["A", "B"],
            }
        ),
        sample_metadata_sample_id_column="sample",
        design_factor_column="condition",
        tested_level="B",
        reference_level="A",
        count_matrix_orientation="genes_by_samples",
        count_matrix_gene_id_column="gene",
        comparison_labels=["B vs A"],
        target_gene=target_gene,
        significance_threshold=0.05,
        fold_change_threshold=1.0,
        basemean_threshold=10.0,
        decimal_places=decimal_places,
    )
