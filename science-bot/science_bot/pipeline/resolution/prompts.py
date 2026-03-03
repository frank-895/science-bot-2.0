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
  - use_list_zip_contents
  - use_list_excel_sheets
  - use_find_files_with_column
  - use_get_file_schema
  - use_search_columns
  - use_get_column_values
  - use_get_column_stats
  - use_search_column_for_value
  - use_get_row_sample
  - finalize
  - fail
- Use tools to inspect files, sheets, columns, and values.
- Never ask to load a full dataframe. Data loading happens after finalize.
- Prefer concise progress. Do not repeat the same tool call with the same
  arguments unless necessary.
- Use only real filenames, real sheet names, and real column names already
  observed through tools.
- If the question cannot be resolved safely within the available evidence,
  return fail.
- For differential_expression, only produce mode="precomputed_results".
- Avoid random sampling unless it is clearly required.
""".strip()

FAMILY_PROMPT_SUPPLEMENTS: dict[QuestionFamily, str] = {
    "aggregate": """
Focus on resolving one primary file plus the exact columns and filters required
for count, summary-statistic, percentage, proportion, or ratio questions.
""".strip(),
    "hypothesis_test": """
Focus on resolving the value column, optional second value column, group column,
group labels, and any filters needed for the statistical test.
""".strip(),
    "regression": """
Focus on resolving the outcome column, predictor column, covariates, optional
prediction inputs, and any filters needed for the regression model.
""".strip(),
    "differential_expression": """
Focus on resolving precomputed result tables, comparison labels, gene/log fold
change/p-value columns, and thresholds. Do not choose raw_counts mode.
""".strip(),
    "variant_filtering": """
Focus on resolving the variant file, sample/gene/effect/VAF columns, and any
cohort-style filters needed for the question.
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
        candidate_lines.append(
            "- "
            f"{candidate.filename} "
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
