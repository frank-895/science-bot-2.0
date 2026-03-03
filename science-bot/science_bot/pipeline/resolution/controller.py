"""Iterative controller for the resolution stage."""

from pathlib import Path
from typing import TypeVar, cast

from pydantic import BaseModel, ValidationError

from science_bot.pipeline.contracts import QuestionFamily
from science_bot.pipeline.resolution.assembly import (
    assemble_payload,
    summarize_resolved_plan,
)
from science_bot.pipeline.resolution.families import (
    AggregateResolutionDecision,
    DifferentialExpressionResolutionDecision,
    HypothesisTestResolutionDecision,
    RegressionResolutionDecision,
    VariantFilteringResolutionDecision,
)
from science_bot.pipeline.resolution.planning import (
    BaseResolutionDecision,
    FamilyResolutionDecisionResponse,
    FamilyResolutionPlan,
    MultiFileMergePlan,
    ResolutionScratchpad,
    build_resolved_plan,
    shortlist_candidate_files,
    summarize_discovery,
    summarize_finalize,
    summarize_tool_result,
    update_scratchpad_from_tool_result,
)
from science_bot.pipeline.resolution.prompts import (
    build_resolution_prompt,
    build_system_prompt,
)
from science_bot.pipeline.resolution.schemas import (
    ResolutionStageInput,
    ResolutionStageOutput,
    ResolutionStepSummary,
)
from science_bot.pipeline.resolution.tools import (
    find_files_with_column,
    get_column_stats,
    get_column_values,
    get_file_schema,
    get_row_sample,
    list_all_capsule_files,
    list_excel_sheets,
    list_zip_contents,
    search_column_for_value,
    search_columns,
    search_filenames,
    summarize_fasta_file,
)
from science_bot.providers import parse_structured

MAX_RESOLUTION_ITERATIONS = 12
T = TypeVar("T")


class ResolutionError(Exception):
    """Base exception for resolution-stage failures."""


class ResolutionIterationLimitError(ResolutionError):
    """Raised when the resolver fails to converge within the step budget."""


class ResolutionValidationError(ResolutionError):
    """Raised when a plan or tool action is invalid."""


def _initial_scratchpad(stage_input: ResolutionStageInput) -> ResolutionScratchpad:
    manifest = list_all_capsule_files(stage_input.capsule_path)
    candidates = shortlist_candidate_files(
        manifest,
        stage_input.classification.family,
        capsule_path=stage_input.capsule_path,
    )
    return ResolutionScratchpad(
        family=stage_input.classification.family,
        question=stage_input.question,
        candidate_files=candidates,
    )


async def run_resolution_controller(
    stage_input: ResolutionStageInput,
) -> ResolutionStageOutput:
    """Resolve a question into a concrete execution payload."""
    scratchpad = _initial_scratchpad(stage_input)
    steps = [summarize_discovery(scratchpad, step_index=1)]
    notes: list[str] = []
    previous_signature: tuple[str, tuple[tuple[str, object], ...]] | None = None
    if stage_input.trace_writer is not None:
        stage_input.trace_writer.write_event(
            event="resolution_discovery",
            stage="resolution",
            question=stage_input.question,
            family=stage_input.classification.family,
            payload=steps[0].model_dump(mode="python"),
        )

    for iteration in range(MAX_RESOLUTION_ITERATIONS):
        scratchpad.iterations_used = iteration
        decision = cast(
            FamilyResolutionDecisionResponse,
            await parse_structured(
                system_prompt=build_system_prompt(stage_input.classification.family),
                user_prompt=build_resolution_prompt(
                    question=stage_input.question,
                    scratchpad=scratchpad,
                    iterations_remaining=MAX_RESOLUTION_ITERATIONS - iteration,
                ),
                response_model=_response_model_for_family(
                    stage_input.classification.family
                ),
                trace_writer=stage_input.trace_writer,
                trace_stage="resolution",
            ),
        )

        if decision.action == "fail":
            raise ResolutionError(decision.reason)

        if decision.action != "finalize":
            tool_name, arguments = _decision_to_tool_call(decision)
            if stage_input.trace_writer is not None:
                stage_input.trace_writer.write_event(
                    event="resolution_tool_call",
                    stage="resolution",
                    question=stage_input.question,
                    family=stage_input.classification.family,
                    payload={"tool_name": tool_name, "arguments": arguments},
                )
            signature = _tool_signature(tool_name, arguments)
            if signature == previous_signature:
                raise ResolutionIterationLimitError(
                    "Resolution repeated the same tool call without new progress."
                )
            previous_signature = signature
            result = _execute_tool(
                capsule_path=stage_input.capsule_path,
                tool_name=tool_name,
                arguments=arguments,
            )
            update_scratchpad_from_tool_result(
                scratchpad=scratchpad,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
            )
            step_summary = summarize_tool_result(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                scratchpad=scratchpad,
                step_index=len(steps) + 1,
            )
            steps.append(step_summary)
            if stage_input.trace_writer is not None:
                stage_input.trace_writer.write_event(
                    event="resolution_tool_result",
                    stage="resolution",
                    question=stage_input.question,
                    family=stage_input.classification.family,
                    payload=step_summary.model_dump(mode="python"),
                )
            continue

        try:
            plan = build_resolved_plan(
                stage_input.classification.family,
                decision,
                require_text=_require_text,
                require_value=_require_value,
            )
            plan = cast(
                FamilyResolutionPlan,
                _normalize_and_validate_plan(
                    scratchpad=scratchpad,
                    plan=plan,
                ),
            )
        except (ResolutionValidationError, ValueError, ValidationError) as exc:
            if _is_recoverable_finalize_issue(str(exc)):
                rejection_step = _finalize_rejection_step(
                    scratchpad=scratchpad,
                    message=str(exc),
                    step_index=len(steps) + 1,
                )
                steps.append(rejection_step)
                _record_finalize_rejection(scratchpad=scratchpad, message=str(exc))
                if stage_input.trace_writer is not None:
                    stage_input.trace_writer.write_event(
                        event="resolution_finalize",
                        stage="resolution",
                        question=stage_input.question,
                        family=stage_input.classification.family,
                        payload={
                            "rejected": True,
                            "step": rejection_step.model_dump(mode="python"),
                        },
                    )
                continue
            raise ResolutionValidationError(str(exc)) from exc
        try:
            payload, selected_files, extra_notes = assemble_payload(
                capsule_path=stage_input.capsule_path,
                plan=plan,
            )
        except ValueError as exc:
            raise ResolutionValidationError(str(exc)) from exc
        notes.extend(extra_notes)
        finalize_step = summarize_finalize(
            family=stage_input.classification.family,
            selected_files=selected_files,
            resolved_field_keys=list(plan.model_dump(exclude_none=True).keys()),
            step_index=len(steps) + 1,
        )
        steps.append(finalize_step)
        if stage_input.trace_writer is not None:
            stage_input.trace_writer.write_event(
                event="resolution_finalize",
                stage="resolution",
                question=stage_input.question,
                family=stage_input.classification.family,
                payload={
                    "step": finalize_step.model_dump(mode="python"),
                    "plan": summarize_resolved_plan(plan),
                    "notes": extra_notes,
                },
            )
        return ResolutionStageOutput(
            payload=payload,
            iterations_used=iteration + 1,
            selected_files=selected_files,
            notes=notes,
            steps=steps,
        )

    raise ResolutionIterationLimitError(
        f"Resolution exceeded {MAX_RESOLUTION_ITERATIONS} iterations."
    )


def _response_model_for_family(family: QuestionFamily) -> type[BaseModel]:
    """Return the structured response model for one supported family."""
    if family == "aggregate":
        return AggregateResolutionDecision
    if family == "hypothesis_test":
        return HypothesisTestResolutionDecision
    if family == "regression":
        return RegressionResolutionDecision
    if family == "differential_expression":
        return DifferentialExpressionResolutionDecision
    if family == "variant_filtering":
        return VariantFilteringResolutionDecision
    raise ResolutionValidationError(f"Unsupported resolution family: {family}")


def _execute_tool(
    *,
    capsule_path: Path,
    tool_name: str,
    arguments: dict[str, object],
) -> object:
    """Execute one allowed inspection tool."""
    try:
        if tool_name == "list_all_capsule_files":
            return list_all_capsule_files(capsule_path)
        if tool_name == "list_zip_contents":
            return list_zip_contents(
                capsule_path,
                zip_filename=_expect_str(arguments, "zip_filename"),
            )
        if tool_name == "search_filenames":
            return search_filenames(
                capsule_path,
                query=_expect_str(arguments, "query"),
            )
        if tool_name == "list_excel_sheets":
            return list_excel_sheets(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
            )
        if tool_name == "find_files_with_column":
            return find_files_with_column(
                capsule_path,
                query=_expect_str(arguments, "query"),
            )
        if tool_name == "get_file_schema":
            return get_file_schema(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
            )
        if tool_name == "search_columns":
            return search_columns(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
                query=_expect_str(arguments, "query"),
            )
        if tool_name == "get_column_values":
            return get_column_values(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
                column=_expect_str(arguments, "column"),
                max_values=_expect_int(arguments, "max_values", default=50),
            )
        if tool_name == "get_column_stats":
            return get_column_stats(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
                column=_expect_str(arguments, "column"),
            )
        if tool_name == "search_column_for_value":
            return search_column_for_value(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
                column=_expect_str(arguments, "column"),
                query=_expect_str(arguments, "query"),
                max_matches=_expect_int(arguments, "max_matches", default=50),
            )
        if tool_name == "get_row_sample":
            return get_row_sample(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
                columns=_expect_str_list(arguments, "columns"),
                n=_expect_int(arguments, "n", default=10),
                random_sample=_expect_bool(arguments, "random_sample", default=False),
            )
        if tool_name == "summarize_fasta_file":
            return summarize_fasta_file(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
            )
        raise ResolutionValidationError(f"Unsupported tool requested: {tool_name}")
    except TypeError as exc:
        raise ResolutionValidationError(
            f"Invalid arguments for {tool_name}: {arguments}"
        ) from exc


def _tool_signature(
    tool_name: str,
    arguments: dict[str, object],
) -> tuple[str, tuple[tuple[str, object], ...]]:
    return (tool_name, tuple(sorted(arguments.items(), key=lambda item: item[0])))


def _expect_str(arguments: dict[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ResolutionValidationError(f"{key} must be a string.")
    return value


def _expect_int(arguments: dict[str, object], key: str, *, default: int) -> int:
    value = arguments.get(key, default)
    if not isinstance(value, int):
        raise ResolutionValidationError(f"{key} must be an integer.")
    return value


def _expect_bool(arguments: dict[str, object], key: str, *, default: bool) -> bool:
    value = arguments.get(key, default)
    if not isinstance(value, bool):
        raise ResolutionValidationError(f"{key} must be a boolean.")
    return value


def _expect_str_list(arguments: dict[str, object], key: str) -> list[str]:
    value = arguments.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ResolutionValidationError(f"{key} must be a list of strings.")
    return cast(list[str], value)


def _decision_to_tool_call(
    decision: BaseResolutionDecision,
) -> tuple[str, dict[str, object]]:
    """Convert a flat use-tool decision into a tool call tuple."""
    if decision.action == "use_list_all_capsule_files":
        return ("list_all_capsule_files", {})
    if decision.action == "use_list_zip_contents":
        return (
            "list_zip_contents",
            {"zip_filename": _require_text(decision.zip_filename, "zip_filename")},
        )
    if decision.action == "use_search_filenames":
        return (
            "search_filenames",
            {"query": _require_text(decision.query, "query")},
        )
    if decision.action == "use_list_excel_sheets":
        return (
            "list_excel_sheets",
            {"filename": _require_text(decision.filename, "filename")},
        )
    if decision.action == "use_find_files_with_column":
        return (
            "find_files_with_column",
            {"query": _require_text(decision.query, "query")},
        )
    if decision.action == "use_get_file_schema":
        return (
            "get_file_schema",
            {"filename": _require_text(decision.filename, "filename")},
        )
    if decision.action == "use_search_columns":
        return (
            "search_columns",
            {
                "filename": _require_text(decision.filename, "filename"),
                "query": _require_text(decision.query, "query"),
            },
        )
    if decision.action == "use_get_column_values":
        return (
            "get_column_values",
            {
                "filename": _require_text(decision.filename, "filename"),
                "column": _require_text(decision.column, "column"),
                "max_values": decision.max_values,
            },
        )
    if decision.action == "use_get_column_stats":
        return (
            "get_column_stats",
            {
                "filename": _require_text(decision.filename, "filename"),
                "column": _require_text(decision.column, "column"),
            },
        )
    if decision.action == "use_search_column_for_value":
        return (
            "search_column_for_value",
            {
                "filename": _require_text(decision.filename, "filename"),
                "column": _require_text(decision.column, "column"),
                "query": _require_text(decision.query, "query"),
                "max_matches": decision.max_matches,
            },
        )
    if decision.action == "use_get_row_sample":
        return (
            "get_row_sample",
            {
                "filename": _require_text(decision.filename, "filename"),
                "columns": decision.columns,
                "n": decision.n,
                "random_sample": decision.random_sample,
            },
        )
    if decision.action == "use_summarize_fasta_file":
        return (
            "summarize_fasta_file",
            {"filename": _require_text(decision.filename, "filename")},
        )
    raise ResolutionValidationError(f"Unsupported decision action: {decision.action}")


def _require_text(value: str | None, field_name: str) -> str:
    if value is None:
        raise ResolutionValidationError(f"{field_name} is required for this action.")
    return value


def _require_value(value: T | None, field_name: str) -> T:
    if value is None:
        raise ResolutionValidationError(f"{field_name} is required for finalize.")
    return value


def _is_recoverable_finalize_issue(message: str) -> bool:
    """Return whether a finalize validation failure should bounce back."""
    recoverable_patterns = (
        "require value_column",
        "requires group_column",
        "requires group_a_value and group_b_value",
        "is required for finalize",
        "requires result_table_files",
        "requires at least two comparison_labels",
        "requires at least two correction_methods",
        "references unobserved column",
        "references unknown zip entr",
        "references ambiguous zip entr",
        "requires exactly one of filename or merge_plan",
    )
    return any(pattern in message for pattern in recoverable_patterns)


def _record_finalize_rejection(
    *,
    scratchpad: ResolutionScratchpad,
    message: str,
) -> None:
    """Persist a rejected finalize attempt into compact scratchpad state."""
    scratchpad.last_tool_name = "finalize"
    scratchpad.last_tool_summary = message
    scratchpad.open_questions = (scratchpad.open_questions + [message])[-8:]


def _finalize_rejection_step(
    *,
    scratchpad: ResolutionScratchpad,
    message: str,
    step_index: int,
) -> ResolutionStepSummary:
    """Build a concise step summary for a rejected finalize attempt."""
    return ResolutionStepSummary(
        step_index=step_index,
        kind="finalize",
        message=f"Finalize rejected: {message}",
        selected_files=scratchpad.selected_files,
        resolved_field_keys=sorted(scratchpad.resolved_fields.keys()),
    )


def _normalize_and_validate_plan(
    *,
    scratchpad: ResolutionScratchpad,
    plan: BaseModel,
) -> BaseModel:
    """Normalize resolved filenames and validate observed columns before assembly."""
    normalized_plan = _normalize_plan_filenames(scratchpad=scratchpad, plan=plan)
    _validate_plan_columns(scratchpad=scratchpad, plan=normalized_plan)
    return normalized_plan


def _normalize_plan_filenames(
    *,
    scratchpad: ResolutionScratchpad,
    plan: BaseModel,
) -> BaseModel:
    """Normalize archive-backed filenames against observed zip entries."""
    update: dict[str, object] = {}
    if hasattr(plan, "filename") and plan.filename is not None:
        filename = cast(str, plan.filename)
        update["filename"] = _normalize_observed_filename(
            scratchpad=scratchpad,
            filename=filename,
        )
    if hasattr(plan, "result_table_files"):
        result_table_files = cast(dict[str, str], plan.result_table_files)
        update["result_table_files"] = {
            label: _normalize_observed_filename(
                scratchpad=scratchpad,
                filename=filename,
            )
            for label, filename in result_table_files.items()
        }
    merge_plan = getattr(plan, "merge_plan", None)
    if merge_plan is not None:
        normalized_merge_plan = merge_plan.model_copy(
            update={
                "data_sources": [
                    source.model_copy(
                        update={
                            "filename": _normalize_observed_filename(
                                scratchpad=scratchpad,
                                filename=source.filename,
                            )
                        }
                    )
                    for source in merge_plan.data_sources
                ],
                "join": (
                    None
                    if merge_plan.join is None
                    else merge_plan.join.model_copy(
                        update={
                            "metadata_file": _normalize_observed_filename(
                                scratchpad=scratchpad,
                                filename=merge_plan.join.metadata_file,
                            )
                        }
                    )
                ),
            }
        )
        update["merge_plan"] = normalized_merge_plan
    if not update:
        return plan
    return plan.model_copy(update=update)


def _normalize_observed_filename(
    *,
    scratchpad: ResolutionScratchpad,
    filename: str,
) -> str:
    """Normalize a finalized filename against observed archive entries."""
    if ".zip/" in filename:
        zip_filename, inner_path = filename.split(".zip/", 1)
        zip_filename = f"{zip_filename}.zip"
        known_entries = scratchpad.known_zip_entries.get(zip_filename)
        if known_entries is None:
            return filename
        if inner_path not in known_entries:
            raise ResolutionValidationError(
                "Finalize references unknown zip entries in "
                f"{zip_filename}: {inner_path}"
            )
        return filename

    if filename.endswith(".zip"):
        return filename

    matches: list[str] = []
    for zip_filename, entries in scratchpad.known_zip_entries.items():
        for entry in entries:
            if entry == filename or entry.endswith(f"/{filename}"):
                matches.append(f"{zip_filename}/{entry}")
    if not matches:
        return filename
    if len(matches) > 1:
        raise ResolutionValidationError(
            f"Finalize references ambiguous zip entries for {filename!r}: {matches}"
        )
    return matches[0]


def _validate_plan_columns(
    *,
    scratchpad: ResolutionScratchpad,
    plan: BaseModel,
) -> None:
    """Reject finalize when referenced columns were never observed."""
    filename_to_columns = _observed_columns_by_filename(scratchpad)
    missing: dict[str, list[str]] = {}

    merge_plan = getattr(plan, "merge_plan", None)

    if merge_plan is not None:
        _validate_merge_plan_columns(
            scratchpad=scratchpad,
            plan=plan,
            merge_plan=merge_plan,
            filename_to_columns=filename_to_columns,
        )
        return

    if hasattr(plan, "filename"):
        filename = cast(str, plan.filename)
        if filename is None:
            return
        required = _required_columns_for_plan(plan)
        missing_columns = [
            column
            for column in required
            if column not in filename_to_columns.get(filename, set())
        ]
        if missing_columns:
            missing[filename] = missing_columns

    if hasattr(plan, "result_table_files"):
        result_table_files = cast(dict[str, str], plan.result_table_files)
        required = _required_columns_for_plan(plan)
        for filename in result_table_files.values():
            missing_columns = [
                column
                for column in required
                if column is not None
                and column not in filename_to_columns.get(filename, set())
            ]
            if missing_columns:
                missing[filename] = missing_columns

    if not missing:
        return

    detail = ", ".join(
        f"{filename}: {sorted(set(columns))}" for filename, columns in missing.items()
    )
    raise ResolutionValidationError(f"Finalize references unobserved columns. {detail}")


def _validate_merge_plan_columns(
    *,
    scratchpad: ResolutionScratchpad,
    plan: BaseModel,
    merge_plan: MultiFileMergePlan,
    filename_to_columns: dict[str, set[str]],
) -> None:
    """Validate observed columns for a merge-based resolved plan."""
    missing: dict[str, list[str]] = {}
    available_columns: set[str] = {merge_plan.output_sample_id_column}

    for source in merge_plan.data_sources:
        source_filename = source.filename
        selected_columns = source.selected_columns
        source_columns = filename_to_columns.get(source_filename, set())
        source_missing = [
            column for column in selected_columns if column not in source_columns
        ]
        if source_missing:
            missing[source_filename] = source_missing
        available_columns.update(source_columns)

    join = merge_plan.join
    if join is not None:
        metadata_file = join.metadata_file
        metadata_columns = _unique_strings(
            [join.metadata_sample_id_column] + join.metadata_columns
        )
        observed_metadata_columns = filename_to_columns.get(metadata_file, set())
        metadata_missing = [
            column
            for column in metadata_columns
            if column not in observed_metadata_columns
        ]
        if metadata_missing:
            missing[metadata_file] = metadata_missing
        available_columns.update(observed_metadata_columns)

    required = _required_columns_for_plan(plan)
    unresolved_required = [
        column for column in required if column not in available_columns
    ]
    if unresolved_required:
        missing["merge_plan"] = unresolved_required

    if not missing:
        return

    detail = ", ".join(
        f"{filename}: {sorted(set(columns))}" for filename, columns in missing.items()
    )
    raise ResolutionValidationError(f"Finalize references unobserved columns. {detail}")


def _observed_columns_by_filename(
    scratchpad: ResolutionScratchpad,
) -> dict[str, set[str]]:
    """Return all observed columns keyed by filename."""
    observed: dict[str, set[str]] = {
        filename: set(columns) for filename, columns in scratchpad.known_columns.items()
    }
    for candidate in scratchpad.candidate_files:
        if not candidate.first_sheet_columns:
            continue
        candidate_name = candidate.path or candidate.filename
        observed.setdefault(candidate_name, set()).update(candidate.first_sheet_columns)
        if candidate.path is not None:
            observed.setdefault(candidate.filename, set()).update(
                candidate.first_sheet_columns
            )
    return observed


def _required_columns_for_plan(plan: BaseModel) -> list[str]:
    """Return all column names referenced by a resolved plan."""
    direct_fields = (
        "value_column",
        "second_value_column",
        "group_column",
        "outcome_column",
        "predictor_column",
        "sample_column",
        "gene_column",
        "effect_column",
        "vaf_column",
        "numerator_mask_column",
        "denominator_mask_column",
        "log_fold_change_column",
        "adjusted_p_value_column",
        "base_mean_column",
    )
    columns: list[str] = []
    for field_name in direct_fields:
        value = getattr(plan, field_name, None)
        if isinstance(value, str):
            columns.append(value)
    for field_name in ("covariate_columns",):
        values = getattr(plan, field_name, [])
        columns.extend(cast(list[str], values))
    prediction_inputs = getattr(plan, "prediction_inputs", {})
    if isinstance(prediction_inputs, dict):
        columns.extend(cast(list[str], list(prediction_inputs.keys())))
    for field_name in ("filters", "numerator_filters", "denominator_filters"):
        filters = getattr(plan, field_name, [])
        for filter_plan in filters:
            column = getattr(filter_plan, "column", None)
            if isinstance(column, str):
                columns.append(column)
    return _unique_strings(columns)


def _unique_strings(values: list[str]) -> list[str]:
    """Return values in order without duplicates."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
