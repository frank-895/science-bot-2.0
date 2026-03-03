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
from science_bot.pipeline.resolution.planning import ResolvedFilterPlan
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
        return {
            "family": plan.family,
            "mode": plan.mode,
            "operation": plan.operation,
            "result_table_files": plan.result_table_files,
            "comparison_labels": plan.comparison_labels,
        }
    return {
        "family": plan.family,
        "filename": plan.filename,
        "operation": plan.operation,
        "sample_column": plan.sample_column,
        "gene_column": plan.gene_column,
        "effect_column": plan.effect_column,
        "vaf_column": plan.vaf_column,
    }


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
        data, notes = _load_required_dataframe(
            capsule_path=capsule_path,
            filename=plan.filename,
            required_columns=required,
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
        return payload, [plan.filename], notes

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
        if plan.mode != "precomputed_results":
            raise ValueError(
                "Only precomputed_results differential expression mode is supported."
            )
        result_tables = {}
        notes: list[str] = []
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
        payload = DifferentialExpressionExecutionInput(
            family=plan.family,
            mode=plan.mode,
            operation=plan.operation,
            result_tables=result_tables,
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
        return payload, list(plan.result_table_files.values()), notes

    if isinstance(plan, VariantFilteringResolvedPlan):
        required = _ordered_columns(
            plan.sample_column,
            plan.gene_column,
            plan.effect_column,
            plan.vaf_column,
            *_filter_columns(plan.filters),
        )
        data, notes = _load_required_dataframe(
            capsule_path=capsule_path,
            filename=plan.filename,
            required_columns=required,
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
        return payload, [plan.filename], notes

    raise ValueError(f"Unsupported resolved plan type: {type(plan)!r}")


def _load_required_dataframe(
    *,
    capsule_path: Path,
    filename: str,
    required_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
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


def _ordered_columns(*columns: str | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for column in columns:
        if column and column not in seen:
            ordered.append(column)
            seen.add(column)
    return ordered


def _filter_columns(filters: list[ResolvedFilterPlan]) -> list[str]:
    return [filter_item.column for filter_item in filters]


def _to_execution_filters(filters: list[ResolvedFilterPlan]) -> list[ResolvedFilter]:
    return [
        ResolvedFilter(
            column=filter_item.column,
            operator=filter_item.operator,
            value=filter_item.value,
        )
        for filter_item in filters
    ]
