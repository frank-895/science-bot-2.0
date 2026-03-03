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
