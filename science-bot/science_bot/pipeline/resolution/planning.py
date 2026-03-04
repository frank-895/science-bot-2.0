"""Internal planning models and deterministic helpers for resolution."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from science_bot.pipeline.contracts import QuestionFamily
from science_bot.pipeline.resolution.schemas import ResolutionStepSummary
from science_bot.pipeline.resolution.tools.excel import list_excel_sheets
from science_bot.pipeline.resolution.tools.reader import parse_filename, read_header
from science_bot.pipeline.resolution.tools.schemas import (
    CapsuleManifest,
    ColumnSearchResult,
    ColumnStats,
    ColumnValues,
    ColumnValueSearchResult,
    FastaSummary,
    FilenameSearchResult,
    FileSchema,
    FullCapsuleManifest,
    RowSample,
    ZipManifest,
)

if TYPE_CHECKING:
    from science_bot.pipeline.resolution.families import (
        AggregateResolutionDecision,
        AggregateResolvedPlan,
        DifferentialExpressionResolutionDecision,
        DifferentialExpressionResolvedPlan,
        HypothesisTestResolutionDecision,
        HypothesisTestResolvedPlan,
        RegressionResolutionDecision,
        RegressionResolvedPlan,
        VariantFilteringResolutionDecision,
        VariantFilteringResolvedPlan,
    )


ResolvedFilterValue: TypeAlias = (
    str | int | float | bool | list[str] | list[int] | list[float]
)
ResolutionAction: TypeAlias = Literal[
    "use_list_all_capsule_files",
    "use_list_zip_contents",
    "use_search_filenames",
    "use_list_excel_sheets",
    "use_find_files_with_column",
    "use_get_file_schema",
    "use_search_columns",
    "use_get_column_values",
    "use_get_column_stats",
    "use_search_column_for_value",
    "use_get_row_sample",
    "use_summarize_fasta_file",
    "finalize",
    "fail",
]


class ResolvedFilterPlan(BaseModel):
    """Resolved filter ready to convert into execution-stage filters."""

    model_config = ConfigDict(extra="forbid")

    column: str
    operator: Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains"]
    value: ResolvedFilterValue


class MultiFileSourceEntry(BaseModel):
    """One per-sample data source used to build a merged dataframe."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    sample_id: str
    selected_columns: list[str] = Field(default_factory=list)


class MetadataJoinPlan(BaseModel):
    """Metadata join description for merged resolution plans."""

    model_config = ConfigDict(extra="forbid")

    metadata_file: str
    metadata_sample_id_column: str
    metadata_columns: list[str] = Field(default_factory=list)


class MultiFileMergePlan(BaseModel):
    """Internal plan for building one dataframe from multiple source files."""

    model_config = ConfigDict(extra="forbid")

    data_sources: list[MultiFileSourceEntry] = Field(default_factory=list)
    join: MetadataJoinPlan | None = None
    output_sample_id_column: str = "sample_id"


class CandidateFileSummary(BaseModel):
    """Compact file candidate kept in scratchpad state."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    filename: str
    file_type: str
    size_human: str
    row_count: int | None = None
    column_count: int | None = None
    is_wide: bool | None = None
    sheet_names: list[str] = Field(default_factory=list)
    first_sheet_name: str | None = None
    first_sheet_columns: list[str] = Field(default_factory=list)
    sheet_names_truncated: bool = False
    first_sheet_columns_truncated: bool = False
    relevance_score: int


class ColumnEvidence(BaseModel):
    """Evidence linking a file and columns to a semantic role."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    columns: list[str]
    reason: str


class ValueEvidence(BaseModel):
    """Evidence linking a file column to representative values."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    column: str
    values: list[str]
    reason: str


class SearchAttempt(BaseModel):
    """Compact record of an unsuccessful search-like tool call."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    query: str | None = None
    filename: str | None = None
    column: str | None = None
    outcome: Literal["no_matches", "invalid_target"]


class ResolutionScratchpad(BaseModel):
    """Compact structured memory carried across resolution iterations."""

    model_config = ConfigDict(extra="forbid")

    family: QuestionFamily
    question: str
    candidate_files: list[CandidateFileSummary] = Field(default_factory=list)
    selected_files: list[str] = Field(default_factory=list)
    known_sheets: dict[str, list[str]] = Field(default_factory=dict)
    known_columns: dict[str, list[str]] = Field(default_factory=dict)
    known_zip_entries: dict[str, list[str]] = Field(default_factory=dict)
    column_evidence: list[ColumnEvidence] = Field(default_factory=list)
    value_evidence: list[ValueEvidence] = Field(default_factory=list)
    resolved_fields: dict[str, object] = Field(default_factory=dict)
    open_questions: list[str] = Field(default_factory=list)
    failed_searches: list[SearchAttempt] = Field(default_factory=list)
    last_tool_name: str | None = None
    last_tool_summary: str | None = None
    iterations_used: int = 0
    notebook_summary: str | None = None


class BaseResolutionDecision(BaseModel):
    """Flat structured decision returned by the resolver LLM."""

    model_config = ConfigDict(extra="forbid")

    action: ResolutionAction
    reason: str
    zip_filename: str | None = None
    filename: str | None = None
    query: str | None = None
    column: str | None = None
    columns: list[str] = Field(default_factory=list)
    n: int = 10
    random_sample: bool = False
    max_values: int = 50
    max_matches: int = 50

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        """Validate the decision rationale."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must be non-empty.")
        return stripped

    @field_validator("zip_filename", "filename", "query", "column")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Normalize optional text fields."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


if TYPE_CHECKING:
    FamilyResolutionDecisionResponse: TypeAlias = (
        AggregateResolutionDecision
        | HypothesisTestResolutionDecision
        | RegressionResolutionDecision
        | DifferentialExpressionResolutionDecision
        | VariantFilteringResolutionDecision
    )
    FamilyResolutionPlan: TypeAlias = (
        AggregateResolvedPlan
        | HypothesisTestResolvedPlan
        | RegressionResolvedPlan
        | DifferentialExpressionResolvedPlan
        | VariantFilteringResolvedPlan
    )
else:
    FamilyResolutionDecisionResponse: TypeAlias = BaseResolutionDecision
    FamilyResolutionPlan: TypeAlias = BaseModel

MAX_CANDIDATE_FILES = 15
MAX_KNOWN_COLUMNS = 40
MAX_TOOL_SUMMARY_LENGTH = 800
MAX_STEP_MESSAGE_LENGTH = 200
MAX_ZIP_ENTRIES = 20
MAX_VALUES = 20
MAX_ROWS = 5
MAX_STARTUP_SHEET_NAMES = 10
MAX_STARTUP_FIRST_SHEET_COLUMNS = 20

_FAMILY_KEYWORDS: dict[QuestionFamily, tuple[str, ...]] = {
    "aggregate": (
        "metadata",
        "phenotype",
        "clinical",
        "expression",
        "covariate",
        "table",
    ),
    "hypothesis_test": (
        "metadata",
        "phenotype",
        "clinical",
        "expression",
        "group",
        "table",
    ),
    "regression": (
        "metadata",
        "phenotype",
        "clinical",
        "covariate",
        "expression",
        "table",
    ),
    "differential_expression": (
        "result",
        "differential",
        "de",
        "marker",
        "gene",
        "count",
        "counts",
        "metadata",
        "sample",
        "condition",
        "xlsx",
        "tsv",
        "zip",
    ),
    "variant_filtering": ("variant", "mutation", "maf", "vaf", "gene", "chip"),
}


def shortlist_candidate_files(
    manifest: CapsuleManifest | FullCapsuleManifest,
    family: QuestionFamily,
    *,
    capsule_path: Path | None = None,
) -> list[CandidateFileSummary]:
    """Reduce a manifest to a family-aware shortlist."""
    keywords = _FAMILY_KEYWORDS[family]
    candidates: list[CandidateFileSummary] = []
    for info in manifest.files:
        score = 0
        filename_lower = info.filename.lower()
        path_lower = getattr(info, "path", info.filename).lower()
        for keyword in keywords:
            if keyword in filename_lower or keyword in path_lower:
                score += 5
        file_type = getattr(info, "file_type", getattr(info, "category", "other"))
        row_count = getattr(info, "row_count", None)
        column_count = getattr(info, "column_count", None)
        is_wide = getattr(info, "is_wide", None)
        size_human = getattr(info, "size_human", "unknown")
        path = getattr(info, "path", None)
        if file_type == "zip":
            score += 2 if family == "differential_expression" else 0
        if file_type == "excel":
            score += 3 if family == "differential_expression" else 1
        if is_wide:
            score -= 1
        candidates.append(
            CandidateFileSummary(
                path=path,
                filename=info.filename,
                file_type=file_type,
                size_human=size_human,
                row_count=row_count,
                column_count=column_count,
                is_wide=is_wide,
                relevance_score=score,
            )
        )
    candidates.sort(
        key=lambda candidate: (
            candidate.relevance_score,
            candidate.column_count or -1,
            candidate.row_count or -1,
        ),
        reverse=True,
    )
    shortlisted = candidates[:MAX_CANDIDATE_FILES]
    if capsule_path is None:
        return shortlisted
    return enrich_candidate_files(capsule_path, shortlisted)


def enrich_candidate_files(
    capsule_path: Path,
    candidates: list[CandidateFileSummary],
) -> list[CandidateFileSummary]:
    """Enrich shortlisted candidates with lightweight startup metadata.

    Args:
        capsule_path: Absolute path to the capsule directory.
        candidates: Ranked candidate files.

    Returns:
        list[CandidateFileSummary]: Enriched candidate summaries.
    """
    enriched_candidates: list[CandidateFileSummary] = []
    for candidate in candidates:
        if candidate.file_type == "excel":
            enriched_candidates.append(_enrich_excel_candidate(capsule_path, candidate))
        else:
            enriched_candidates.append(candidate)
    return enriched_candidates


def _enrich_excel_candidate(
    capsule_path: Path,
    candidate: CandidateFileSummary,
) -> CandidateFileSummary:
    """Add workbook structure to a shortlisted Excel candidate.

    Args:
        capsule_path: Absolute path to the capsule directory.
        candidate: Candidate file summary.

    Returns:
        CandidateFileSummary: Updated summary, or the original candidate if
        enrichment could not be completed safely.
    """
    candidate_name = candidate.path or candidate.filename
    try:
        sheet_names = list_excel_sheets(capsule_path, candidate_name)
    except Exception:
        return candidate

    limited_sheet_names = sheet_names[:MAX_STARTUP_SHEET_NAMES]
    first_sheet_name = limited_sheet_names[0] if limited_sheet_names else None
    first_sheet_columns: list[str] = []
    first_sheet_columns_truncated = False

    if first_sheet_name is not None:
        try:
            header_ref = parse_filename(
                capsule_path,
                f"{candidate_name}::{first_sheet_name}",
            )
            headers = read_header(header_ref)
            first_sheet_columns = headers[:MAX_STARTUP_FIRST_SHEET_COLUMNS]
            first_sheet_columns_truncated = (
                len(headers) > MAX_STARTUP_FIRST_SHEET_COLUMNS
            )
        except Exception:
            first_sheet_columns = []
            first_sheet_columns_truncated = False

    return candidate.model_copy(
        update={
            "sheet_names": limited_sheet_names,
            "first_sheet_name": first_sheet_name,
            "first_sheet_columns": first_sheet_columns,
            "sheet_names_truncated": len(sheet_names) > MAX_STARTUP_SHEET_NAMES,
            "first_sheet_columns_truncated": first_sheet_columns_truncated,
        }
    )


def summarize_discovery(
    scratchpad: ResolutionScratchpad,
    *,
    step_index: int,
) -> ResolutionStepSummary:
    """Build the initial discovery step summary."""
    message = (
        f"Shortlisted {len(scratchpad.candidate_files)} candidate files for "
        f"{scratchpad.family} resolution."
    )
    return ResolutionStepSummary(
        step_index=step_index,
        kind="discover",
        message=_truncate(message, MAX_STEP_MESSAGE_LENGTH),
        selected_files=scratchpad.selected_files,
        resolved_field_keys=sorted(scratchpad.resolved_fields.keys()),
    )


def summarize_tool_result(
    *,
    tool_name: str,
    arguments: dict[str, object],
    result: object,
    scratchpad: ResolutionScratchpad,
    step_index: int,
) -> ResolutionStepSummary:
    """Build a compact step summary from one tool result."""
    message, truncated = tool_result_message(
        tool_name,
        result,
        arguments=arguments,
    )
    return ResolutionStepSummary(
        step_index=step_index,
        kind="tool",
        tool_name=tool_name,
        message=_truncate(message, MAX_STEP_MESSAGE_LENGTH),
        selected_files=scratchpad.selected_files,
        resolved_field_keys=sorted(scratchpad.resolved_fields.keys()),
        truncated=truncated,
    )


def summarize_finalize(
    *,
    family: QuestionFamily,
    selected_files: list[str],
    resolved_field_keys: list[str],
    step_index: int,
) -> ResolutionStepSummary:
    """Build the final step summary."""
    message = (
        f"Finalized {family} payload from {len(selected_files)} selected "
        f"file{'s' if len(selected_files) != 1 else ''}."
    )
    return ResolutionStepSummary(
        step_index=step_index,
        kind="finalize",
        message=_truncate(message, MAX_STEP_MESSAGE_LENGTH),
        selected_files=selected_files,
        resolved_field_keys=sorted(resolved_field_keys),
    )


def tool_result_message(
    tool_name: str,
    result: object,
    *,
    arguments: dict[str, object] | None = None,
) -> tuple[str, bool]:
    """Create a compact human-readable summary for a tool result."""
    if tool_name == "list_all_capsule_files" and isinstance(
        result, FullCapsuleManifest
    ):
        shown = [file_info.path for file_info in result.files[:MAX_VALUES]]
        truncated = len(result.files) > MAX_VALUES
        if not shown:
            return ("Capsule contains no files.", False)
        return (
            f"Discovered {len(result.files)} files across the capsule: {shown}",
            truncated,
        )

    if tool_name == "list_zip_contents" and isinstance(result, ZipManifest):
        entries = [entry.inner_path for entry in result.entries if entry.is_readable]
        truncated = len(entries) > MAX_ZIP_ENTRIES
        shown = entries[:MAX_ZIP_ENTRIES]
        if not shown:
            return (f"No readable entries found in zip {result.zip_filename}.", False)
        return (
            f"Inspected zip {result.zip_filename}; readable entries: {shown}",
            truncated,
        )

    if tool_name == "list_excel_sheets" and isinstance(result, list):
        truncated = len(result) > MAX_VALUES
        if not result:
            return ("Excel file has no discoverable sheets.", False)
        return (f"Excel sheets: {result[:MAX_VALUES]}", truncated)

    if isinstance(result, FileSchema):
        columns = [column.name for column in result.columns[:MAX_KNOWN_COLUMNS]]
        truncated = result.columns_truncated or result.column_count > MAX_KNOWN_COLUMNS
        return (
            f"Schema for {result.filename}: {result.row_count} rows, "
            f"{result.column_count} columns, sample columns={columns}",
            truncated,
        )

    if isinstance(result, ColumnSearchResult):
        if result.total_matches == 0:
            return (
                f"No column matches in {result.filename} for {result.query!r}.",
                False,
            )
        truncated = result.total_matches > MAX_VALUES
        return (
            f"Column matches in {result.filename} for {result.query!r}: "
            f"{result.matches[:MAX_VALUES]}",
            truncated,
        )

    if tool_name == "search_filenames" and isinstance(result, list):
        filename_results = cast(list[FilenameSearchResult], result)
        if not filename_results:
            query = cast(
                str | None,
                None if arguments is None else arguments.get("query"),
            )
            if query is not None:
                return (f"No matching filenames were found for {query!r}.", False)
            return ("No matching filenames were found.", False)
        shown = [entry.path for entry in filename_results[:MAX_VALUES]]
        truncated = len(filename_results) > MAX_VALUES
        return (f"Matching filenames: {shown}", truncated)

    if tool_name == "find_files_with_column" and isinstance(result, list):
        search_results = cast(list[ColumnSearchResult], result)
        if not search_results:
            return ("No files with matching columns were found.", False)
        filenames = [entry.filename for entry in search_results[:MAX_VALUES]]
        truncated = len(search_results) > MAX_VALUES
        return (f"Files with matching columns: {filenames}", truncated)

    if isinstance(result, ColumnValues):
        values = result.values[:MAX_VALUES]
        return (
            f"Observed values for {result.filename}:{result.column} -> {values}",
            result.truncated or result.unique_count > MAX_VALUES,
        )

    if isinstance(result, ColumnStats):
        if result.most_common is not None:
            return (
                f"Stats for {result.filename}:{result.column} -> "
                f"most_common={result.most_common[:MAX_VALUES]}",
                len(result.most_common) > MAX_VALUES,
            )
        return (
            f"Stats for {result.filename}:{result.column} -> "
            f"min={result.min}, max={result.max}, mean={result.mean}, std={result.std}",
            False,
        )

    if isinstance(result, ColumnValueSearchResult):
        if result.total_matches == 0:
            return (
                f"No matching values for {result.filename}:{result.column}.",
                False,
            )
        return (
            f"Matching values for {result.filename}:{result.column} -> "
            f"{result.matches[:MAX_VALUES]}",
            result.truncated or result.total_matches > MAX_VALUES,
        )

    if isinstance(result, RowSample):
        truncated = len(result.rows) > MAX_ROWS
        return (
            f"Row sample for {result.filename} using columns {result.columns}: "
            f"{result.rows[:MAX_ROWS]}",
            truncated,
        )

    if isinstance(result, FastaSummary):
        return (
            f"FASTA summary for {result.filename}: sequences={result.sequence_count}, "
            f"mean_length={result.mean_length}, gap_fraction={result.gap_fraction}",
            False,
        )

    return (f"Completed {tool_name}.", False)


def update_scratchpad_from_tool_result(
    *,
    scratchpad: ResolutionScratchpad,
    tool_name: str,
    arguments: dict[str, object],
    result: object,
) -> None:
    """Update scratchpad state from one tool result."""
    scratchpad.last_tool_name = tool_name
    tool_summary, truncated = tool_result_message(
        tool_name,
        result,
        arguments=arguments,
    )
    scratchpad.last_tool_summary = _truncate(tool_summary, MAX_TOOL_SUMMARY_LENGTH)

    if isinstance(result, FileSchema):
        scratchpad.known_columns[result.filename] = [
            column.name for column in result.columns[:MAX_KNOWN_COLUMNS]
        ]
        if result.filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.filename)
    elif isinstance(result, FullCapsuleManifest):
        scratchpad.candidate_files = shortlist_candidate_files(
            result,
            scratchpad.family,
        )
    elif isinstance(result, ColumnSearchResult):
        if result.total_matches == 0:
            _record_failed_search(
                scratchpad=scratchpad,
                tool_name=tool_name,
                filename=result.filename,
                query=result.query,
            )
        else:
            if result.filename not in scratchpad.selected_files:
                scratchpad.selected_files.append(result.filename)
            scratchpad.column_evidence.append(
                ColumnEvidence(
                    filename=result.filename,
                    columns=result.matches[:MAX_VALUES],
                    reason=f"Matches query {result.query!r}",
                )
            )
            scratchpad.column_evidence = scratchpad.column_evidence[-20:]
    elif tool_name == "find_files_with_column" and isinstance(result, list):
        search_results = cast(list[ColumnSearchResult], result)
        if not search_results:
            _record_failed_search(
                scratchpad=scratchpad,
                tool_name=tool_name,
                query=cast(str | None, arguments.get("query")),
            )
        else:
            for entry in search_results[:MAX_VALUES]:
                if entry.filename not in scratchpad.selected_files:
                    scratchpad.selected_files.append(entry.filename)
    elif tool_name == "search_filenames" and isinstance(result, list):
        filename_results = cast(list[FilenameSearchResult], result)
        if not filename_results:
            _record_failed_search(
                scratchpad=scratchpad,
                tool_name=tool_name,
                query=cast(str | None, arguments.get("query")),
            )
        else:
            for entry in filename_results[:MAX_VALUES]:
                if entry.path not in scratchpad.selected_files:
                    scratchpad.selected_files.append(entry.path)
    elif isinstance(result, ColumnValues):
        if result.filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.filename)
        scratchpad.value_evidence.append(
            ValueEvidence(
                filename=result.filename,
                column=result.column,
                values=[str(value) for value in result.values[:MAX_VALUES]],
                reason="Observed distinct values.",
            )
        )
        scratchpad.value_evidence = scratchpad.value_evidence[-20:]
    elif isinstance(result, ColumnStats):
        if result.filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.filename)
    elif isinstance(result, ColumnValueSearchResult):
        if result.total_matches == 0:
            _record_failed_search(
                scratchpad=scratchpad,
                tool_name=tool_name,
                filename=result.filename,
                column=result.column,
                query=result.query,
            )
        elif result.filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.filename)
    elif isinstance(result, RowSample):
        if result.filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.filename)
    elif isinstance(result, FastaSummary):
        if result.filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.filename)
    elif isinstance(result, ZipManifest):
        if result.zip_filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.zip_filename)
        scratchpad.known_zip_entries[result.zip_filename] = [
            entry.inner_path for entry in result.entries if entry.is_readable
        ]
        if not any(entry.is_readable for entry in result.entries):
            _record_failed_search(
                scratchpad=scratchpad,
                tool_name=tool_name,
                filename=result.zip_filename,
            )
    elif isinstance(result, list) and result and isinstance(result[0], str):
        filename = cast(str | None, arguments.get("filename"))
        if filename is not None:
            scratchpad.known_sheets[filename] = cast(list[str], result[:MAX_VALUES])
            if filename not in scratchpad.selected_files:
                scratchpad.selected_files.append(filename)

    if truncated:
        scratchpad.open_questions = (
            scratchpad.open_questions + ["More detail was available but truncated."]
        )[-8:]


def build_resolved_plan(
    family: QuestionFamily,
    decision: FamilyResolutionDecisionResponse,
    *,
    require_text: Callable[[str | None, str], str],
    require_value: Callable[[object | None, str], object],
) -> FamilyResolutionPlan:
    """Convert a family-specific finalize decision into a validated plan."""
    if decision.action != "finalize":
        raise ValueError("Only finalize decisions can build a plan.")
    if family == "aggregate":
        from science_bot.pipeline.resolution.families import (
            AggregateResolutionDecision,
            build_aggregate_plan_from_decision,
        )

        plan = build_aggregate_plan_from_decision(
            cast(AggregateResolutionDecision, decision),
            require_text=require_text,
            require_value=require_value,
        )
    elif family == "hypothesis_test":
        from science_bot.pipeline.resolution.families import (
            HypothesisTestResolutionDecision,
            build_hypothesis_test_plan_from_decision,
        )

        plan = build_hypothesis_test_plan_from_decision(
            cast(HypothesisTestResolutionDecision, decision),
            require_text=require_text,
            require_value=require_value,
        )
    elif family == "regression":
        from science_bot.pipeline.resolution.families import (
            RegressionResolutionDecision,
            build_regression_plan_from_decision,
        )

        plan = build_regression_plan_from_decision(
            cast(RegressionResolutionDecision, decision),
            require_text=require_text,
            require_value=require_value,
        )
    elif family == "differential_expression":
        from science_bot.pipeline.resolution.families import (
            DifferentialExpressionResolutionDecision,
            build_differential_expression_plan_from_decision,
        )

        plan = build_differential_expression_plan_from_decision(
            cast(DifferentialExpressionResolutionDecision, decision),
            require_value=require_value,
        )
    elif family == "variant_filtering":
        from science_bot.pipeline.resolution.families import (
            VariantFilteringResolutionDecision,
            build_variant_filtering_plan_from_decision,
        )

        plan = build_variant_filtering_plan_from_decision(
            cast(VariantFilteringResolutionDecision, decision),
            require_text=require_text,
            require_value=require_value,
        )
    else:
        raise ValueError(f"Unsupported finalize family: {family}")
    plan_family = plan.family
    if plan_family != family:
        raise ValueError(
            f"finalize plan family {plan_family!r} does not match {family!r}."
        )
    return plan


def _record_failed_search(
    *,
    scratchpad: ResolutionScratchpad,
    tool_name: str,
    query: str | None = None,
    filename: str | None = None,
    column: str | None = None,
) -> None:
    """Append one failed search attempt to bounded scratchpad history."""
    scratchpad.failed_searches = (
        scratchpad.failed_searches
        + [
            SearchAttempt(
                tool_name=tool_name,
                query=query,
                filename=filename,
                column=column,
                outcome="no_matches",
            )
        ]
    )[-20:]


def _truncate(value: str, max_length: int) -> str:
    """Shorten text with an ellipsis when it exceeds the limit."""
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."
