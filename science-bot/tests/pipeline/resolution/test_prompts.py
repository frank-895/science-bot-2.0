from science_bot.pipeline.resolution.planning import (
    CandidateFileSummary,
    ResolutionScratchpad,
)
from science_bot.pipeline.resolution.prompts import (
    build_resolution_prompt,
    build_system_prompt,
)


def test_build_system_prompt_includes_family_guidance():
    prompt = build_system_prompt("aggregate")
    assert "deterministic execution" in prompt
    assert "summary-statistic" in prompt
    assert "Finalize as soon as the minimum required fields" in prompt
    assert "If a search returns no matches" in prompt


def test_build_resolution_prompt_includes_state():
    prompt = build_resolution_prompt(
        question="How many samples?",
        scratchpad=ResolutionScratchpad(
            family="aggregate",
            question="How many samples?",
            candidate_files=[
                CandidateFileSummary(
                    filename="data.csv",
                    file_type="csv",
                    size_human="1 KB",
                    row_count=10,
                    column_count=2,
                    is_wide=False,
                    relevance_score=5,
                )
            ],
            known_columns={"data.csv": ["sample_id", "value"]},
            selected_files=["data.csv"],
            last_tool_name="get_file_schema",
            last_tool_summary="Schema found.",
        ),
        iterations_remaining=8,
    )

    assert "How many samples?" in prompt
    assert "data.csv" in prompt
    assert "sample_id" in prompt
    assert "Schema found." in prompt


def test_build_resolution_prompt_includes_excel_candidate_metadata():
    prompt = build_resolution_prompt(
        question="Which proteins are differential?",
        scratchpad=ResolutionScratchpad(
            family="differential_expression",
            question="Which proteins are differential?",
            candidate_files=[
                CandidateFileSummary(
                    filename="Proteomic_data.xlsx",
                    file_type="excel",
                    size_human="10 KB",
                    sheet_names=["Tumor vs Normal"],
                    first_sheet_name="Tumor vs Normal",
                    first_sheet_columns=["protein", "gene", "log2FC", "adj.Pval"],
                    relevance_score=10,
                )
            ],
        ),
        iterations_remaining=12,
    )

    assert "sheets=['Tumor vs Normal']" in prompt
    assert "first_sheet=Tumor vs Normal" in prompt
    assert "first_columns=['protein', 'gene', 'log2FC', 'adj.Pval']" in prompt


def test_build_system_prompt_includes_regression_finalize_checklist():
    prompt = build_system_prompt("regression")

    assert "filename" in prompt
    assert "outcome_column" in prompt
    assert "predictor_column" in prompt
    assert "covariate_columns" in prompt
    assert "return_field" in prompt
    assert "finalize now" in prompt.lower()


def test_build_system_prompt_includes_merge_guardrails():
    prompt = build_system_prompt("aggregate")

    assert "two or more per-sample data files" in prompt
    assert "Do not use a merge plan for a single matrix plus metadata file" in prompt
