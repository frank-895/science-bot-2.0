"""Prompt helpers for the resolution stage."""

from science_bot.pipeline.contracts import QuestionFamily
from science_bot.pipeline.resolution.planning import ResolutionScratchpad
from science_bot.pipeline.resolution.tools import AVAILABLE_TOOLS_TEXT

MAX_PROMPT_COLUMNS = 40

RESOLUTION_SYSTEM_PROMPT = """
You are resolving a life-science analysis question into a deterministic execution
plan for one supported family.

Rules:
- You must choose exactly one next action.
- Allowed actions are:
  - use_list_all_capsule_files
  - use_list_zip_contents
  - use_search_filenames
  - use_list_excel_sheets
  - use_find_files_with_column
  - use_get_file_schema
  - use_search_columns
  - use_get_column_values
  - use_get_column_stats
  - use_search_column_for_value
  - use_get_row_sample
  - use_summarize_fasta_file
  - finalize
  - fail
- Use tools to inspect files, sheets, columns, and values.
- Never ask to load a full dataframe. Data loading happens after finalize.
- Finalize as soon as the minimum required fields for the family are visible.
- Do not gather additional evidence once the required fields are identified.
- Use one targeted tool call only when a specific required field is still
  missing.
- Do not repeat the same tool call with the same arguments.
- If a search returns no matches, do not repeat a similar search. Change
  strategy: inspect a different file schema, inspect workbook sheets, inspect
  zip contents, use filename search, or fail if no safe path remains.
- Prefer direct file or schema inspection over repeated broad search
  reformulations.
- Use startup candidate metadata before spending a tool call.
- Use only real filenames, real sheet names, and real column names already
  observed through tools.
- If one metadata file defines cohorts or statuses and many similarly named
  sample files hold the measurements, you may finalize with a merge plan instead
  of a single filename.
- Use a merge plan only when you have two or more per-sample data files and can
  pair each file with exactly one sample ID.
- Do not use a merge plan for a single matrix plus metadata file. In that case,
  keep a normal filename-based plan and use metadata only to reason about the
  needed columns and filters.
- If the question cannot be resolved safely within the available evidence,
  return fail.
- Avoid random sampling unless it is clearly required.
""".strip()

FAMILY_PROMPT_SUPPLEMENTS: dict[QuestionFamily, str] = {
    "aggregate": """
You need:
- filename
- operation
- any required filters
- value_column when the summary-statistic operation requires it
- numerator or denominator fields when ratio or proportion logic requires them
- return_format
If these are already visible from the candidate metadata, known columns, and
question wording, finalize now.
If the question depends on one metadata workbook plus many per-sample files,
identify the metadata file, the sample files, and the sample IDs, then finalize
with a merge plan instead of repeating single-file inspection.
Use merge only when there are two or more per-sample files and you can provide
aligned data_source_files and data_source_sample_ids lists.
""".strip(),
    "hypothesis_test": """
You need:
- filename
- test
- value_column
- group_column and group values when the test requires grouping
- optional second_value_column for paired or two-column questions
- return_field
If these are already visible from the candidate metadata, known columns, and
question wording, finalize now.
""".strip(),
    "regression": """
You need:
- filename
- model_type
- outcome_column
- predictor_column
- covariate_columns
- return_field
- prediction_inputs only when the return field requires prediction
If all required fields are already visible in the known columns and question,
finalize now.
""".strip(),
    "differential_expression": """
You need:
- either a precomputed DE result table or a raw-count setup
- operation
- comparison labels when required
- relevant gene, fold-change, and adjusted-p columns when required
- thresholds when the question explicitly specifies them
- prefer mode="precomputed_results" when a suitable DE result table is clearly
  available
- otherwise use mode="raw_counts" only when you can identify:
  - count_matrix_file
  - sample_metadata_file
  - metadata sample ID column
  - metadata design factor column
  - tested and reference levels from observed values
  - count matrix orientation and its identifier column
  - exactly one comparison label
If those are already visible in workbook headers, known columns, and observed
values, finalize now.
Do not keep exploring once the minimum required fields for the chosen mode are
known. Fail rather than guess.
""".strip(),
    "variant_filtering": """
You need:
- filename
- operation
- relevant sample, gene, effect, or VAF columns when the operation requires
  them
- any requested filters
- return_format
If these are already visible from the candidate metadata, known columns, and
question wording, finalize now.
If the question depends on one metadata workbook plus many per-sample files,
identify the metadata file, the sample files, and the sample IDs, then finalize
with a merge plan instead of repeating single-file inspection.
Use merge only when there are two or more per-sample files and you can provide
aligned data_source_files and data_source_sample_ids lists.
""".strip(),
}


def build_resolution_prompt(
    *,
    question: str,
    scratchpad: ResolutionScratchpad,
    iterations_remaining: int,
) -> str:
    """Build the compact user prompt for one resolver iteration."""
    candidate_lines = []
    for candidate in scratchpad.candidate_files:
        display_name = candidate.path or candidate.filename
        if candidate.file_type == "excel":
            candidate_lines.append(
                "- "
                f"{display_name} "
                f"(type={candidate.file_type}, size={candidate.size_human}, "
                f"sheets={candidate.sheet_names}, "
                f"first_sheet={candidate.first_sheet_name}, "
                f"first_columns={candidate.first_sheet_columns})"
            )
        else:
            candidate_lines.append(
                "- "
                f"{display_name} "
                f"(type={candidate.file_type}, size={candidate.size_human}, "
                f"rows={candidate.row_count}, cols={candidate.column_count}, "
                f"wide={candidate.is_wide})"
            )
    if not candidate_lines:
        candidate_lines.append("- none")

    column_lines = []
    for filename, columns in scratchpad.known_columns.items():
        shown = columns[:MAX_PROMPT_COLUMNS]
        column_lines.append(f"- {filename}: {shown}")
    if not column_lines:
        column_lines.append("- none")

    sheet_lines = []
    for filename, sheets in scratchpad.known_sheets.items():
        sheet_lines.append(f"- {filename}: {sheets}")
    if not sheet_lines:
        sheet_lines.append("- none")

    return "\n".join(
        [
            f"Question: {question}",
            f"Family: {scratchpad.family}",
            f"Iteration: {scratchpad.iterations_used + 1}",
            f"Iterations remaining after this turn: {iterations_remaining - 1}",
            "",
            "Candidate files:",
            *candidate_lines,
            "",
            "Known sheets:",
            *sheet_lines,
            "",
            "Known columns:",
            *column_lines,
            "",
            f"Selected files: {scratchpad.selected_files}",
            f"Resolved fields: {sorted(scratchpad.resolved_fields.keys())}",
            f"Open questions: {scratchpad.open_questions}",
            f"Last tool: {scratchpad.last_tool_name}",
            f"Last tool summary: {scratchpad.last_tool_summary}",
            "",
            AVAILABLE_TOOLS_TEXT,
        ]
    )


def build_system_prompt(family: QuestionFamily) -> str:
    """Build the system prompt for the selected question family."""
    return "\n\n".join([RESOLUTION_SYSTEM_PROMPT, FAMILY_PROMPT_SUPPLEMENTS[family]])
