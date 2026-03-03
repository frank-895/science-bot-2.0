"""Deterministic differential expression execution implementation."""

from typing import Final

import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

from science_bot.pipeline.contracts import DifferentialExpressionOperation
from science_bot.pipeline.execution.schemas import (
    DifferentialExpressionExecutionInput,
    DifferentialExpressionExecutionMode,
    ExecutionStageOutput,
)
from science_bot.pipeline.execution.utils import format_scalar_answer

IMPLEMENTED_DIFFERENTIAL_EXPRESSION_OPERATIONS: Final[
    frozenset[DifferentialExpressionOperation]
] = frozenset(
    {
        "significant_gene_count",
        "unique_significant_gene_count",
        "shared_overlap_pattern",
        "gene_log2_fold_change",
        "significant_marker_count",
        "correction_ratio",
    }
)
IMPLEMENTED_DIFFERENTIAL_EXPRESSION_EXECUTION_MODES: Final[
    frozenset[DifferentialExpressionExecutionMode]
] = frozenset({"precomputed_results", "raw_counts"})
SUPPORTED_RAW_COUNT_OPERATIONS: Final[frozenset[DifferentialExpressionOperation]] = (
    frozenset(
        {
            "significant_gene_count",
            "gene_log2_fold_change",
            "significant_marker_count",
        }
    )
)


def run_differential_expression_execution(
    payload: DifferentialExpressionExecutionInput,
) -> ExecutionStageOutput:
    """Execute a resolved differential expression question.

    Args:
        payload: Resolved differential expression execution payload.

    Returns:
        ExecutionStageOutput: Deterministic differential expression result.

    Raises:
        KeyError: If a required comparison table is missing.
        NotImplementedError: If the requested raw-count operation is deferred.
    """
    if payload.mode == "raw_counts":
        if payload.operation not in SUPPORTED_RAW_COUNT_OPERATIONS:
            raise NotImplementedError(
                "Differential expression operation "
                f"'{payload.operation}' is not supported for raw_counts mode."
            )
        compiled_payload = _compile_raw_counts_payload(payload)
        return _run_precomputed_differential_expression_execution(compiled_payload)

    return _run_precomputed_differential_expression_execution(payload)


def _run_precomputed_differential_expression_execution(
    payload: DifferentialExpressionExecutionInput,
) -> ExecutionStageOutput:
    """Execute differential expression against canonical result tables."""
    if payload.operation == "significant_gene_count":
        table = payload.result_tables[payload.comparison_labels[0]]
        filtered = _filter_significant(table, payload)
        count = int(len(filtered))
        return ExecutionStageOutput(
            family=payload.family,
            answer=str(count),
            raw_result={"significant_gene_count": count},
        )

    if payload.operation == "unique_significant_gene_count":
        primary = _significant_gene_set(
            payload.result_tables[payload.comparison_labels[0]], payload
        )
        others = set()
        for label in payload.comparison_labels[1:]:
            others |= _significant_gene_set(payload.result_tables[label], payload)
        count = int(len(primary - others))
        return ExecutionStageOutput(
            family=payload.family,
            answer=str(count),
            raw_result={"unique_significant_gene_count": count},
        )

    if payload.operation == "shared_overlap_pattern":
        sets = [
            _significant_gene_set(payload.result_tables[label], payload)
            for label in payload.comparison_labels
        ]
        shared_all = set.intersection(*sets)
        if shared_all:
            answer = "Complete overlap between all groups"
        elif _has_partial_overlap(sets):
            answer = "Partial overlap between specific groups"
        else:
            answer = "No overlap between any groups"
        return ExecutionStageOutput(
            family=payload.family,
            answer=answer,
            raw_result={"shared_overlap_pattern": answer},
        )

    if payload.operation == "gene_log2_fold_change":
        table = payload.result_tables[payload.comparison_labels[0]]
        gene_column = _require_gene_column(payload)
        log_fold_change_column = _require_log_fold_change_column(payload)
        row = table.loc[table[gene_column] == payload.target_gene].iloc[0]
        value = float(row[log_fold_change_column])
        return ExecutionStageOutput(
            family=payload.family,
            answer=format_scalar_answer(
                value, payload.decimal_places, payload.round_to
            ),
            raw_result={"gene_log2_fold_change": value},
        )

    if payload.operation == "significant_marker_count":
        total = 0
        for label in payload.comparison_labels:
            total += len(_filter_significant(payload.result_tables[label], payload))
        return ExecutionStageOutput(
            family=payload.family,
            answer=str(int(total)),
            raw_result={"significant_marker_count": int(total)},
        )

    counts: list[int] = []
    adjusted_p_value_column = _require_adjusted_p_value_column(payload)
    for method in payload.correction_methods:
        table = payload.result_tables[method]
        counts.append(
            int((table[adjusted_p_value_column] < payload.significance_threshold).sum())
        )
    answer = f"{counts[0]}:{counts[1]}"
    return ExecutionStageOutput(
        family=payload.family,
        answer=answer,
        raw_result={"correction_ratio": answer, "counts": counts},
    )


def _compile_raw_counts_payload(
    payload: DifferentialExpressionExecutionInput,
) -> DifferentialExpressionExecutionInput:
    """Compile one raw-count differential expression request to a canonical table."""
    canonical_table = _run_raw_counts_differential_expression(payload)
    return DifferentialExpressionExecutionInput(
        family=payload.family,
        mode="precomputed_results",
        operation=payload.operation,
        result_tables={payload.comparison_labels[0]: canonical_table},
        comparison_labels=payload.comparison_labels,
        reference_label=payload.reference_label,
        target_gene=payload.target_gene,
        gene_column="gene",
        log_fold_change_column="log2FoldChange",
        adjusted_p_value_column="padj",
        base_mean_column="baseMean",
        significance_threshold=payload.significance_threshold,
        fold_change_threshold=payload.fold_change_threshold,
        basemean_threshold=payload.basemean_threshold,
        use_lfc_shrinkage=payload.use_lfc_shrinkage,
        correction_methods=payload.correction_methods,
        decimal_places=payload.decimal_places,
        round_to=payload.round_to,
    )


def _run_raw_counts_differential_expression(
    payload: DifferentialExpressionExecutionInput,
) -> pd.DataFrame:
    """Run one pairwise PyDESeq2 contrast and return a canonical results table."""
    counts, metadata = _prepare_deseq2_inputs(payload)
    design_factor_column = _require_design_factor_column(payload)
    tested_level = _require_tested_level(payload)
    reference_level = _require_reference_level(payload)
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design=design_factor_column,
        ref_level=[design_factor_column, reference_level],
        quiet=True,
    )
    dds.deseq2()
    stats = DeseqStats(
        dds,
        contrast=[design_factor_column, tested_level, reference_level],
        alpha=payload.significance_threshold or 0.05,
        quiet=True,
    )
    stats.summary()

    if payload.use_lfc_shrinkage:
        stats.lfc_shrink(
            coeff=_resolve_lfc_shrinkage_coefficient(
                dds=dds,
                design_factor_column=design_factor_column,
                tested_level=tested_level,
            )
        )

    return _to_canonical_result_table(stats.results_df)


def _prepare_deseq2_inputs(
    payload: DifferentialExpressionExecutionInput,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize raw counts and metadata into the shape PyDESeq2 expects."""
    count_matrix = _require_count_matrix(payload)
    sample_metadata = _require_sample_metadata(payload)
    metadata_sample_id_column = _require_metadata_sample_id_column(payload)
    design_factor_column = _require_design_factor_column(payload)
    tested_level = _require_tested_level(payload)
    reference_level = _require_reference_level(payload)

    metadata = sample_metadata.copy()
    required_columns = [metadata_sample_id_column, design_factor_column]
    missing_columns = [
        column for column in required_columns if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Sample metadata is missing required columns: {missing_columns}."
        )
    metadata = metadata[required_columns]
    metadata[metadata_sample_id_column] = metadata[metadata_sample_id_column].astype(
        str
    )
    if metadata[metadata_sample_id_column].duplicated().any():
        raise ValueError("Sample metadata contains duplicate sample IDs.")
    metadata = metadata.set_index(metadata_sample_id_column, drop=True)

    factor_values = metadata[design_factor_column].astype(str)
    observed_levels = set(factor_values.tolist())
    missing_levels = [
        level
        for level in (tested_level, reference_level)
        if level not in observed_levels
    ]
    if missing_levels:
        raise ValueError(
            f"Sample metadata is missing contrast levels: {missing_levels}."
        )
    metadata = metadata.loc[
        factor_values.isin({tested_level, reference_level}),
        [design_factor_column],
    ].copy()
    metadata[design_factor_column] = metadata[design_factor_column].astype(str)

    counts = _prepare_count_matrix(
        count_matrix=count_matrix,
        payload=payload,
        expected_sample_ids=list(metadata.index),
    )
    missing_sample_ids = sorted(set(metadata.index) - set(counts.index))
    if missing_sample_ids:
        raise ValueError(
            "Count matrix is missing samples referenced by metadata: "
            f"{missing_sample_ids}."
        )
    counts = counts.loc[metadata.index]
    return counts, metadata


def _prepare_count_matrix(
    *,
    count_matrix: pd.DataFrame,
    payload: DifferentialExpressionExecutionInput,
    expected_sample_ids: list[str],
) -> pd.DataFrame:
    """Normalize a count matrix to sample-by-gene orientation."""
    if payload.count_matrix_orientation == "genes_by_samples":
        gene_id_column = _require_count_matrix_gene_id_column(payload)
        if gene_id_column not in count_matrix.columns:
            raise ValueError(
                f"Count matrix is missing gene identifier column '{gene_id_column}'."
            )
        gene_ids = count_matrix[gene_id_column].astype(str)
        if gene_ids.duplicated().any():
            raise ValueError("Count matrix contains duplicate gene identifiers.")
        candidate_sample_columns = [
            column for column in count_matrix.columns if column != gene_id_column
        ]
        matched_sample_columns = [
            column
            for column in candidate_sample_columns
            if str(column) in expected_sample_ids
        ]
        if not matched_sample_columns:
            raise ValueError(
                "Count matrix does not contain any expected sample columns."
            )
        counts = count_matrix[[gene_id_column, *matched_sample_columns]].copy()
        counts[gene_id_column] = counts[gene_id_column].astype(str)
        counts = counts.set_index(gene_id_column, drop=True).transpose()
        counts.index = counts.index.astype(str)
        return _coerce_counts_to_integer_dataframe(counts)

    sample_id_column = _require_count_matrix_sample_id_column(payload)
    if sample_id_column not in count_matrix.columns:
        raise ValueError(
            f"Count matrix is missing sample identifier column '{sample_id_column}'."
        )
    sample_ids = count_matrix[sample_id_column].astype(str)
    if sample_ids.duplicated().any():
        raise ValueError("Count matrix contains duplicate sample identifiers.")
    matched_rows = sample_ids.isin(expected_sample_ids)
    if not matched_rows.any():
        raise ValueError("Count matrix does not contain any expected sample rows.")
    gene_columns = [
        column for column in count_matrix.columns if column != sample_id_column
    ]
    counts = count_matrix.loc[matched_rows, [sample_id_column, *gene_columns]].copy()
    counts[sample_id_column] = counts[sample_id_column].astype(str)
    counts = counts.set_index(sample_id_column, drop=True)
    counts.index = counts.index.astype(str)
    if counts.columns.astype(str).duplicated().any():
        raise ValueError("Count matrix contains duplicate gene columns.")
    counts.columns = counts.columns.astype(str)
    return _coerce_counts_to_integer_dataframe(counts)


def _coerce_counts_to_integer_dataframe(counts: pd.DataFrame) -> pd.DataFrame:
    """Validate count values and coerce them to integer dtype."""
    numeric_counts = counts.apply(pd.to_numeric, errors="coerce")
    if numeric_counts.isna().any().any():
        raise ValueError("Count matrix contains non-numeric values.")
    if (numeric_counts < 0).any().any():
        raise ValueError("Count matrix contains negative values.")
    fractional_values = (numeric_counts % 1 != 0) & numeric_counts.notna()
    if fractional_values.any().any():
        raise ValueError("Count matrix contains non-integer count values.")
    return numeric_counts.astype(int)


def _resolve_lfc_shrinkage_coefficient(
    *,
    dds: DeseqDataSet,
    design_factor_column: str,
    tested_level: str,
) -> str:
    """Resolve the fitted coefficient name required for LFC shrinkage."""
    expected_name = f"{design_factor_column}[T.{tested_level}]"
    coefficient_names = [str(column) for column in dds.varm["LFC"].columns]
    if expected_name not in coefficient_names:
        raise ValueError(
            "Could not resolve LFC shrinkage coefficient "
            f"'{expected_name}'. Available coefficients: {coefficient_names}."
        )
    return expected_name


def _to_canonical_result_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a PyDESeq2 results dataframe into the canonical DE schema."""
    canonical = results_df.reset_index().copy()
    first_column = str(canonical.columns[0])
    canonical = canonical.rename(columns={first_column: "gene"})
    required_columns = ["gene", "baseMean", "log2FoldChange", "pvalue", "padj"]
    missing_columns = [
        column for column in required_columns if column not in canonical.columns
    ]
    if missing_columns:
        raise ValueError(
            f"PyDESeq2 results are missing canonical columns: {missing_columns}."
        )
    canonical["gene"] = canonical["gene"].astype(str)
    return canonical[required_columns]


def _filter_significant(
    table: pd.DataFrame, payload: DifferentialExpressionExecutionInput
) -> pd.DataFrame:
    """Apply significance thresholds to a DE results table.

    Args:
        table: Differential expression results table.
        payload: Differential expression execution payload.

    Returns:
        pd.DataFrame: Thresholded DE result table.
    """
    filtered = table.copy()
    if payload.significance_threshold is not None:
        adjusted_p_value_column = _require_adjusted_p_value_column(payload)
        filtered = filtered[
            filtered[adjusted_p_value_column] < payload.significance_threshold
        ]
    if payload.fold_change_threshold is not None:
        log_fold_change_column = _require_log_fold_change_column(payload)
        filtered = filtered[
            filtered[log_fold_change_column].abs() > payload.fold_change_threshold
        ]
    if payload.basemean_threshold is not None:
        base_mean_column = _require_base_mean_column(payload)
        filtered = filtered[filtered[base_mean_column] >= payload.basemean_threshold]
    return filtered


def _significant_gene_set(
    table: pd.DataFrame, payload: DifferentialExpressionExecutionInput
) -> set[str]:
    """Build the significant gene set for one comparison table.

    Args:
        table: Differential expression results table.
        payload: Differential expression execution payload.

    Returns:
        set[str]: Significant gene identifiers.
    """
    gene_column = _require_gene_column(payload)
    return set(_filter_significant(table, payload)[gene_column].astype(str))


def _require_count_matrix(
    payload: DifferentialExpressionExecutionInput,
) -> pd.DataFrame:
    """Return the raw-count matrix required for DE execution."""
    if payload.count_matrix is None:
        raise ValueError("Differential expression raw_counts requires count_matrix.")
    return payload.count_matrix


def _require_sample_metadata(
    payload: DifferentialExpressionExecutionInput,
) -> pd.DataFrame:
    """Return the sample metadata required for raw-count DE execution."""
    if payload.sample_metadata is None:
        raise ValueError("Differential expression raw_counts requires sample_metadata.")
    return payload.sample_metadata


def _require_metadata_sample_id_column(
    payload: DifferentialExpressionExecutionInput,
) -> str:
    """Return the metadata sample identifier column for raw-count execution."""
    if payload.sample_metadata_sample_id_column is None:
        raise ValueError(
            "Differential expression raw_counts requires "
            "sample_metadata_sample_id_column."
        )
    return payload.sample_metadata_sample_id_column


def _require_design_factor_column(
    payload: DifferentialExpressionExecutionInput,
) -> str:
    """Return the design factor column for raw-count execution."""
    if payload.design_factor_column is None:
        raise ValueError(
            "Differential expression raw_counts requires design_factor_column."
        )
    return payload.design_factor_column


def _require_tested_level(payload: DifferentialExpressionExecutionInput) -> str:
    """Return the tested contrast level for raw-count execution."""
    if payload.tested_level is None:
        raise ValueError("Differential expression raw_counts requires tested_level.")
    return payload.tested_level


def _require_reference_level(payload: DifferentialExpressionExecutionInput) -> str:
    """Return the reference contrast level for raw-count execution."""
    if payload.reference_level is None:
        raise ValueError("Differential expression raw_counts requires reference_level.")
    return payload.reference_level


def _require_count_matrix_gene_id_column(
    payload: DifferentialExpressionExecutionInput,
) -> str:
    """Return the gene identifier column for gene-by-sample count matrices."""
    if payload.count_matrix_gene_id_column is None:
        raise ValueError(
            "Differential expression raw_counts with genes_by_samples "
            "requires count_matrix_gene_id_column."
        )
    return payload.count_matrix_gene_id_column


def _require_count_matrix_sample_id_column(
    payload: DifferentialExpressionExecutionInput,
) -> str:
    """Return the sample identifier column for sample-by-gene count matrices."""
    if payload.count_matrix_sample_id_column is None:
        raise ValueError(
            "Differential expression raw_counts with samples_by_genes "
            "requires count_matrix_sample_id_column."
        )
    return payload.count_matrix_sample_id_column


def _require_gene_column(payload: DifferentialExpressionExecutionInput) -> str:
    """Return the resolved gene column for DE execution.

    Args:
        payload: Differential expression execution payload.

    Returns:
        str: Resolved gene column name.

    Raises:
        ValueError: If the gene column was not resolved.
    """
    if payload.gene_column is None:
        raise ValueError("Differential expression execution requires gene_column.")
    return payload.gene_column


def _require_log_fold_change_column(
    payload: DifferentialExpressionExecutionInput,
) -> str:
    """Return the resolved log fold change column for DE execution.

    Args:
        payload: Differential expression execution payload.

    Returns:
        str: Resolved log fold change column name.

    Raises:
        ValueError: If the log fold change column was not resolved.
    """
    if payload.log_fold_change_column is None:
        raise ValueError(
            "Differential expression execution requires log_fold_change_column."
        )
    return payload.log_fold_change_column


def _require_adjusted_p_value_column(
    payload: DifferentialExpressionExecutionInput,
) -> str:
    """Return the resolved adjusted p-value column for DE execution.

    Args:
        payload: Differential expression execution payload.

    Returns:
        str: Resolved adjusted p-value column name.

    Raises:
        ValueError: If the adjusted p-value column was not resolved.
    """
    if payload.adjusted_p_value_column is None:
        raise ValueError(
            "Differential expression execution requires adjusted_p_value_column."
        )
    return payload.adjusted_p_value_column


def _require_base_mean_column(payload: DifferentialExpressionExecutionInput) -> str:
    """Return the resolved base mean column for DE execution.

    Args:
        payload: Differential expression execution payload.

    Returns:
        str: Resolved base mean column name.

    Raises:
        ValueError: If the base mean column was not resolved.
    """
    if payload.base_mean_column is None:
        raise ValueError("Differential expression execution requires base_mean_column.")
    return payload.base_mean_column


def _has_partial_overlap(sets: list[set[str]]) -> bool:
    """Determine whether any pairwise overlap exists without full overlap.

    Args:
        sets: Significant gene sets per comparison.

    Returns:
        bool: True when a partial overlap exists.
    """
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            if left & right:
                return True
    return False
