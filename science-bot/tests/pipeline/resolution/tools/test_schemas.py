from science_bot.pipeline.resolution.tools.schemas import (
    AllFileInfo,
    FastaSummary,
    FilenameSearchResult,
    FullCapsuleManifest,
)


def test_full_capsule_manifest_accepts_all_file_info():
    manifest = FullCapsuleManifest(
        capsule_path="/tmp/capsule",
        files=[
            AllFileInfo(
                path="notes/run.ipynb",
                filename="run.ipynb",
                extension=".ipynb",
                size_bytes=10,
                size_human="10 B",
                category="notebook",
                is_supported_for_deeper_inspection=False,
            )
        ],
        total_size_bytes=10,
    )

    assert manifest.files[0].category == "notebook"


def test_filename_search_result_accepts_filename_match():
    result = FilenameSearchResult(
        query="merip",
        path="MeRIP_RNA_result.xlsx",
        filename="MeRIP_RNA_result.xlsx",
        category="excel",
        size_bytes=100,
        matched_on="filename",
        matched_texts=["MeRIP_RNA_result.xlsx"],
    )

    assert result.matched_on == "filename"


def test_fasta_summary_accepts_gap_metrics():
    summary = FastaSummary(
        filename="alignment.faa",
        sequence_count=2,
        min_length=4,
        max_length=6,
        mean_length=5.0,
        total_characters=10,
        gap_count=2,
        gap_fraction=0.2,
        alphabet_hint="protein",
    )

    assert summary.gap_fraction == 0.2
