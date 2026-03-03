"""Iterative controller for the resolution stage."""

from pathlib import Path
from typing import TypeVar, cast

from pydantic import BaseModel

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
    candidates = shortlist_candidate_files(manifest, stage_input.classification.family)
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
        except ValueError as exc:
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
