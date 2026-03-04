"""Convert resolved plans into execution payloads."""

from pathlib import Path
from typing import TypeAlias

import pandas as pd

from science_bot.pipeline.execution.schemas import (
    AggregateExecutionInput,
    DifferentialExpressionExecutionInput,
    ExecutionPayload,
    HypothesisTestExecutionInput,
    RegressionExecutionInput,
    ResolvedFilter,
    VariantFilteringExecutionInput,
)
from science_bot.pipeline.resolution.families import (
    AggregateResolvedPlan,
    DifferentialExpressionResolvedPlan,
    HypothesisTestResolvedPlan,
    RegressionResolvedPlan,
    VariantFilteringResolvedPlan,
)
from science_bot.pipeline.resolution.planning import (
    MetadataJoinPlan,
    MultiFileMergePlan,
    MultiFileSourceEntry,
    ResolvedFilterPlan,
)
from science_bot.pipeline.resolution.tools import load_dataframe

FamilyResolutionPlan: TypeAlias = (
    AggregateResolvedPlan
    | HypothesisTestResolvedPlan
    | RegressionResolvedPlan
    | DifferentialExpressionResolvedPlan
    | VariantFilteringResolvedPlan
)


def summarize_resolved_plan(plan: FamilyResolutionPlan) -> dict[str, object]:
    """Build a lightweight trace summary for a finalized resolution plan."""
    if isinstance(plan, AggregateResolvedPlan):
        if plan.merge_plan is not None:
            return {
                "family": plan.family,
                "merge_plan": True,
                "data_source_count": len(plan.merge_plan.data_sources),
                "metadata_file": (
                    None
                    if plan.merge_plan.join is None
                    else plan.merge_plan.join.metadata_file
                ),
                "sample_ids": [
                    source.sample_id for source in plan.merge_plan.data_sources
                ],
                "operation": plan.operation,
                "value_column": plan.value_column,
                "filter_columns": _filter_columns(plan.filters),
            }
        return {
            "family": plan.family,
            "filename": plan.filename,
            "operation": plan.operation,
            "value_column": plan.value_column,
            "filter_columns": _filter_columns(plan.filters),
        }
    if isinstance(plan, HypothesisTestResolvedPlan):
        return {
            "family": plan.family,
            "filename": plan.filename,
            "test": plan.test,
            "value_column": plan.value_column,
            "group_column": plan.group_column,
            "return_field": plan.return_field,
        }
    if isinstance(plan, RegressionResolvedPlan):
        return {
            "family": plan.family,
            "filename": plan.filename,
            "model_type": plan.model_type,
            "outcome_column": plan.outcome_column,
            "predictor_column": plan.predictor_column,
            "covariate_columns": plan.covariate_columns,
            "return_field": plan.return_field,
        }
    if isinstance(plan, DifferentialExpressionResolvedPlan):
        if plan.mode == "raw_counts":
            return {
                "family": plan.family,
                "mode": plan.mode,
                "operation": plan.operation,
                "count_matrix_file": plan.count_matrix_file,
                "sample_metadata_file": plan.sample_metadata_file,
                "design_factor_column": plan.design_factor_column,
                "comparison_labels": plan.comparison_labels,
            }
        return {
            "family": plan.family,
            "mode": plan.mode,
            "operation": plan.operation,
            "result_table_files": plan.result_table_files,
            "comparison_labels": plan.comparison_labels,
        }
    return (
        {
            "family": plan.family,
            "filename": plan.filename,
            "operation": plan.operation,
            "sample_column": plan.sample_column,
            "gene_column": plan.gene_column,
            "effect_column": plan.effect_column,
            "vaf_column": plan.vaf_column,
        }
        if plan.merge_plan is None
        else {
            "family": plan.family,
            "merge_plan": True,
            "data_source_count": len(plan.merge_plan.data_sources),
            "metadata_file": (
                None
                if plan.merge_plan.join is None
                else plan.merge_plan.join.metadata_file
            ),
            "sample_ids": [source.sample_id for source in plan.merge_plan.data_sources],
            "operation": plan.operation,
            "sample_column": plan.sample_column,
            "gene_column": plan.gene_column,
            "effect_column": plan.effect_column,
            "vaf_column": plan.vaf_column,
        }
    )


def assemble_payload(
    *,
    capsule_path: Path,
    plan: FamilyResolutionPlan,
) -> tuple[ExecutionPayload, list[str], list[str]]:
    """Convert a validated resolved plan into an execution payload."""
    if isinstance(plan, AggregateResolvedPlan):
        required = _ordered_columns(
            plan.value_column,
            plan.numerator_mask_column,
            plan.denominator_mask_column,
            *_filter_columns(plan.filters),
            *_filter_columns(plan.numerator_filters),
            *_filter_columns(plan.denominator_filters),
        )
        if plan.merge_plan is None:
            if plan.filename is None:
                raise ValueError("Aggregate plan requires filename or merge_plan.")
            data, notes = _load_required_dataframe(
                capsule_path=capsule_path,
                filename=plan.filename,
                required_columns=required,
            )
            selected_files = [plan.filename]
        else:
            data, selected_files, notes = _assemble_merged_dataframe(
                capsule_path=capsule_path,
                merge_plan=plan.merge_plan,
                required_data_columns=required,
            )
        payload = AggregateExecutionInput(
            family=plan.family,
            operation=plan.operation,
            data=data,
            value_column=plan.value_column,
            numerator_mask_column=plan.numerator_mask_column,
            denominator_mask_column=plan.denominator_mask_column,
            numerator_filters=_to_execution_filters(plan.numerator_filters),
            denominator_filters=_to_execution_filters(plan.denominator_filters),
            filters=_to_execution_filters(plan.filters),
            return_format=plan.return_format,
            decimal_places=plan.decimal_places,
            round_to=plan.round_to,
        )
        return payload, selected_files, notes

    if isinstance(plan, HypothesisTestResolvedPlan):
        required = _ordered_columns(
            plan.value_column,
            plan.second_value_column,
            plan.group_column,
            *_filter_columns(plan.filters),
        )
        data, notes = _load_required_dataframe(
            capsule_path=capsule_path,
            filename=plan.filename,
            required_columns=required,
        )
        payload = HypothesisTestExecutionInput(
            family=plan.family,
            test=plan.test,
            data=data,
            value_column=plan.value_column,
            second_value_column=plan.second_value_column,
            group_column=plan.group_column,
            group_a_value=plan.group_a_value,
            group_b_value=plan.group_b_value,
            filters=_to_execution_filters(plan.filters),
            return_field=plan.return_field,
            decimal_places=plan.decimal_places,
            round_to=plan.round_to,
        )
        return payload, [plan.filename], notes

    if isinstance(plan, RegressionResolvedPlan):
        required = _ordered_columns(
            plan.outcome_column,
            plan.predictor_column,
            *plan.covariate_columns,
            *_filter_columns(plan.filters),
            *plan.prediction_inputs.keys(),
        )
        data, notes = _load_required_dataframe(
            capsule_path=capsule_path,
            filename=plan.filename,
            required_columns=required,
        )
        payload = RegressionExecutionInput(
            family=plan.family,
            model_type=plan.model_type,
            data=data,
            outcome_column=plan.outcome_column,
            predictor_column=plan.predictor_column,
            covariate_columns=plan.covariate_columns,
            degree=plan.degree,
            prediction_inputs=plan.prediction_inputs,
            filters=_to_execution_filters(plan.filters),
            return_field=plan.return_field,
            decimal_places=plan.decimal_places,
            round_to=plan.round_to,
        )
        return payload, [plan.filename], notes

    if isinstance(plan, DifferentialExpressionResolvedPlan):
        result_tables = {}
        notes: list[str] = []
        if plan.mode == "precomputed_results":
            required = _ordered_columns(
                plan.gene_column,
                plan.log_fold_change_column,
                plan.adjusted_p_value_column,
                plan.base_mean_column,
            )
            for label, filename in plan.result_table_files.items():
                data, file_notes = _load_required_dataframe(
                    capsule_path=capsule_path,
                    filename=filename,
                    required_columns=required,
                )
                result_tables[label] = data
                notes.extend(file_notes)
            selected_files = list(plan.result_table_files.values())
            count_matrix = None
            sample_metadata = None
        else:
            if plan.count_matrix_file is None:
                raise ValueError("raw_counts plan requires count_matrix_file.")
            if plan.sample_metadata_file is None:
                raise ValueError("raw_counts plan requires sample_metadata_file.")
            count_matrix, count_notes = _load_full_dataframe(
                capsule_path=capsule_path,
                filename=plan.count_matrix_file,
            )
            metadata_columns = _ordered_columns(
                plan.sample_metadata_sample_id_column,
                plan.design_factor_column,
            )
            sample_metadata, metadata_notes = _load_required_dataframe(
                capsule_path=capsule_path,
                filename=plan.sample_metadata_file,
                required_columns=metadata_columns,
            )
            notes.extend(count_notes)
            notes.extend(metadata_notes)
            selected_files = [plan.count_matrix_file, plan.sample_metadata_file]
        payload = DifferentialExpressionExecutionInput(
            family=plan.family,
            mode=plan.mode,
            operation=plan.operation,
            result_tables=result_tables,
            count_matrix=count_matrix,
            sample_metadata=sample_metadata,
            sample_metadata_sample_id_column=plan.sample_metadata_sample_id_column,
            design_factor_column=plan.design_factor_column,
            tested_level=plan.tested_level,
            reference_level=plan.reference_level,
            count_matrix_orientation=plan.count_matrix_orientation,
            count_matrix_gene_id_column=plan.count_matrix_gene_id_column,
            count_matrix_sample_id_column=plan.count_matrix_sample_id_column,
            comparison_labels=plan.comparison_labels,
            reference_label=plan.reference_label,
            target_gene=plan.target_gene,
            gene_column=plan.gene_column,
            log_fold_change_column=plan.log_fold_change_column,
            adjusted_p_value_column=plan.adjusted_p_value_column,
            base_mean_column=plan.base_mean_column,
            significance_threshold=plan.significance_threshold,
            fold_change_threshold=plan.fold_change_threshold,
            basemean_threshold=plan.basemean_threshold,
            use_lfc_shrinkage=plan.use_lfc_shrinkage,
            correction_methods=plan.correction_methods,
            decimal_places=plan.decimal_places,
            round_to=plan.round_to,
        )
        return payload, selected_files, notes

    if isinstance(plan, VariantFilteringResolvedPlan):
        required = _ordered_columns(
            plan.sample_column,
            plan.gene_column,
            plan.effect_column,
            plan.vaf_column,
            *_filter_columns(plan.filters),
        )
        if plan.merge_plan is None:
            if plan.filename is None:
                raise ValueError(
                    "Variant filtering plan requires filename or merge_plan."
                )
            data, notes = _load_required_dataframe(
                capsule_path=capsule_path,
                filename=plan.filename,
                required_columns=required,
            )
            selected_files = [plan.filename]
        else:
            data, selected_files, notes = _assemble_merged_dataframe(
                capsule_path=capsule_path,
                merge_plan=plan.merge_plan,
                required_data_columns=required,
            )
        payload = VariantFilteringExecutionInput(
            family=plan.family,
            operation=plan.operation,
            data=data,
            sample_column=plan.sample_column,
            sample_value=plan.sample_value,
            gene_column=plan.gene_column,
            effect_column=plan.effect_column,
            vaf_column=plan.vaf_column,
            vaf_min=plan.vaf_min,
            vaf_max=plan.vaf_max,
            filters=_to_execution_filters(plan.filters),
            return_format=plan.return_format,
            decimal_places=plan.decimal_places,
            round_to=plan.round_to,
        )
        return payload, selected_files, notes

    raise ValueError(f"Unsupported resolved plan type: {type(plan)!r}")


def _load_required_dataframe(
    *,
    capsule_path: Path,
    filename: str,
    required_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Load only the resolved columns required for execution.

    Args:
        capsule_path: Absolute path to the capsule directory.
        filename: Resolved data filename.
        required_columns: Columns that must be loaded.

    Returns:
        tuple[pd.DataFrame, list[str]]: Loaded dataframe and trace notes.
    """
    if not required_columns:
        raise ValueError(
            f"Resolution plan for {filename!r} did not identify required columns."
        )
    notes: list[str] = []
    if len(required_columns) < 200:
        notes.append(
            f"Loaded {filename} with explicit column subset of "
            f"{len(required_columns)} columns."
        )
    data = load_dataframe(capsule_path, filename, columns=required_columns)
    return data, notes


def _load_full_dataframe(
    *,
    capsule_path: Path,
    filename: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Load a full dataframe for execution paths that require all columns."""
    data = load_dataframe(capsule_path, filename, columns=None)
    notes = [
        f"Loaded {filename} with all columns for raw-count differential expression."
    ]
    return data, notes


def _assemble_merged_dataframe(
    *,
    capsule_path: Path,
    merge_plan: MultiFileMergePlan,
    required_data_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build one dataframe from many source files plus optional metadata."""
    if not merge_plan.data_sources:
        raise ValueError("Multi-file merge requires at least one data source.")

    merged_frames: list[pd.DataFrame] = []
    notes: list[str] = []
    selected_files: list[str] = []
    seen_sample_ids: set[str] = set()

    for source in merge_plan.data_sources:
        if source.sample_id in seen_sample_ids:
            raise ValueError(
                f"Duplicate data-source sample_id {source.sample_id!r} in merge plan."
            )
        seen_sample_ids.add(source.sample_id)
        data_columns = source.selected_columns or required_data_columns
        data, source_notes = _load_data_source_dataframe(
            capsule_path=capsule_path,
            source=source,
            required_columns=data_columns,
            output_sample_id_column=merge_plan.output_sample_id_column,
        )
        merged_frames.append(data)
        notes.extend(source_notes)
        selected_files.append(source.filename)

    merged = pd.concat(merged_frames, ignore_index=True)

    if merge_plan.join is None:
        return merged, selected_files, notes

    metadata, metadata_notes = _load_metadata_dataframe(
        capsule_path=capsule_path,
        join_plan=merge_plan.join,
        output_sample_id_column=merge_plan.output_sample_id_column,
    )
    merged = _join_metadata(
        data=merged,
        metadata=metadata,
        join_plan=merge_plan.join,
        output_sample_id_column=merge_plan.output_sample_id_column,
    )
    notes.extend(metadata_notes)
    selected_files.append(merge_plan.join.metadata_file)
    return merged, selected_files, notes


def _load_data_source_dataframe(
    *,
    capsule_path: Path,
    source: MultiFileSourceEntry,
    required_columns: list[str],
    output_sample_id_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Load one per-sample file and stamp the resolved sample ID onto its rows."""
    data, notes = _load_required_dataframe(
        capsule_path=capsule_path,
        filename=source.filename,
        required_columns=required_columns,
    )
    stamped = data.copy()
    stamped[output_sample_id_column] = source.sample_id
    notes.append(
        "Stamped "
        f"{source.filename} with "
        f"{output_sample_id_column}={source.sample_id!r}."
    )
    return stamped, notes


def _load_metadata_dataframe(
    *,
    capsule_path: Path,
    join_plan: MetadataJoinPlan,
    output_sample_id_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Load metadata needed for a merged dataframe join."""
    metadata_columns = _ordered_columns(
        join_plan.metadata_sample_id_column,
        *join_plan.metadata_columns,
    )
    metadata, notes = _load_required_dataframe(
        capsule_path=capsule_path,
        filename=join_plan.metadata_file,
        required_columns=metadata_columns,
    )
    normalized = metadata.copy()
    normalized[join_plan.metadata_sample_id_column] = (
        normalized[join_plan.metadata_sample_id_column].astype(str).str.strip()
    )
    if normalized[join_plan.metadata_sample_id_column].duplicated().any():
        raise ValueError(
            f"Metadata file {join_plan.metadata_file!r} contains duplicate sample IDs."
        )
    if join_plan.metadata_sample_id_column != output_sample_id_column:
        normalized = normalized.rename(
            columns={join_plan.metadata_sample_id_column: output_sample_id_column}
        )
    return normalized, notes


def _join_metadata(
    *,
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    join_plan: MetadataJoinPlan,
    output_sample_id_column: str,
) -> pd.DataFrame:
    """Join metadata columns onto concatenated data rows by sample ID."""
    merged = data.copy()
    merged[output_sample_id_column] = (
        merged[output_sample_id_column].astype(str).str.strip()
    )
    joined = merged.merge(
        metadata,
        on=output_sample_id_column,
        how="left",
        validate="many_to_one",
    )
    unresolved = joined[joined[join_plan.metadata_columns].isnull().all(axis=1)][
        output_sample_id_column
    ].unique()
    if len(unresolved) > 0:
        raise ValueError(
            "Metadata join did not resolve sample IDs: "
            f"{sorted(str(value) for value in unresolved)}"
        )
    return joined


def _ordered_columns(*columns: str | None) -> list[str]:
    """Return unique non-empty column names in first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for column in columns:
        if column and column not in seen:
            ordered.append(column)
            seen.add(column)
    return ordered


def _filter_columns(filters: list[ResolvedFilterPlan]) -> list[str]:
    """Extract referenced column names from resolved filter plans."""
    return [filter_item.column for filter_item in filters]


def _to_execution_filters(filters: list[ResolvedFilterPlan]) -> list[ResolvedFilter]:
    """Convert resolved filter plans into execution-stage filter models."""
    return [
        ResolvedFilter(
            column=filter_item.column,
            operator=filter_item.operator,
            value=filter_item.value,
        )
        for filter_item in filters
    ]
