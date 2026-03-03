import zipfile
from pathlib import Path

from science_bot.pipeline.resolution.tools.sequence import summarize_fasta_file


def test_summarize_fasta_file_reports_lengths_and_gap_fraction(tmp_path: Path):
    fasta_path = tmp_path / "alignment.faa"
    fasta_path.write_text(">seq1\nAC-T\n>seq2\nA--T\n", encoding="utf-8")

    summary = summarize_fasta_file(tmp_path, "alignment.faa")

    assert summary.sequence_count == 2
    assert summary.min_length == 4
    assert summary.max_length == 4
    assert summary.mean_length == 4.0
    assert summary.gap_count == 3
    assert summary.gap_fraction == 3 / 8


def test_summarize_fasta_file_supports_zip_paths(tmp_path: Path):
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("inner/alignment.fasta", ">seq1\nACGT\n")

    summary = summarize_fasta_file(tmp_path, "bundle.zip/inner/alignment.fasta")

    assert summary.sequence_count == 1
    assert summary.alphabet_hint == "dna"


def test_summarize_fasta_file_raises_for_invalid_content(tmp_path: Path):
    fasta_path = tmp_path / "invalid.faa"
    fasta_path.write_text("not fasta", encoding="utf-8")

    try:
        summarize_fasta_file(tmp_path, "invalid.faa")
    except ValueError as exc:
        assert "does not contain FASTA records" in str(exc)
    else:
        raise AssertionError("Expected invalid FASTA content to raise ValueError.")
