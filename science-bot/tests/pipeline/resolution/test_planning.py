from science_bot.pipeline.resolution.planning import (
    ResolutionScratchpad,
    SearchAttempt,
    shortlist_candidate_files,
    tool_result_message,
    update_scratchpad_from_tool_result,
)
from science_bot.pipeline.resolution.tools.schemas import (
    CapsuleManifest,
    ColumnSearchResult,
    FileInfo,
)


def test_shortlist_candidate_files_prefers_family_keywords():
    manifest = CapsuleManifest(
        capsule_path="/tmp/capsule",
        total_size_bytes=3,
        files=[
            FileInfo(
                filename="clinical_table.csv",
                size_bytes=1,
                size_human="1 B",
                row_count=10,
                column_count=5,
                is_wide=False,
                file_type="csv",
            ),
            FileInfo(
                filename="random.csv",
                size_bytes=1,
                size_human="1 B",
                row_count=10,
                column_count=5,
                is_wide=False,
                file_type="csv",
            ),
        ],
    )

    candidates = shortlist_candidate_files(manifest, "aggregate")

    assert candidates[0].filename == "clinical_table.csv"


def test_tool_result_message_reports_empty_search_results():
    message, truncated = tool_result_message("find_files_with_column", [])

    assert message == "No files with matching columns were found."
    assert truncated is False


def test_update_scratchpad_tracks_failed_search():
    scratchpad = ResolutionScratchpad(family="aggregate", question="question")

    update_scratchpad_from_tool_result(
        scratchpad=scratchpad,
        tool_name="find_files_with_column",
        arguments={"query": "gap"},
        result=[],
    )

    assert scratchpad.failed_searches == [
        SearchAttempt(
            tool_name="find_files_with_column",
            query="gap",
            outcome="no_matches",
        )
    ]
    assert scratchpad.last_tool_summary == "No files with matching columns were found."


def test_update_scratchpad_records_column_matches():
    scratchpad = ResolutionScratchpad(family="aggregate", question="question")
    result = ColumnSearchResult(
        filename="data.csv",
        query="gene",
        matches=["gene"],
        total_matches=1,
    )

    update_scratchpad_from_tool_result(
        scratchpad=scratchpad,
        tool_name="search_columns",
        arguments={"filename": "data.csv", "query": "gene"},
        result=result,
    )

    assert "data.csv" in scratchpad.selected_files
    assert scratchpad.column_evidence[0].columns == ["gene"]
