from pathlib import Path

from science_bot.pipeline.resolution.tools.discovery import (
    list_all_capsule_files,
    list_capsule_files,
    search_filenames,
)


def test_list_all_capsule_files_includes_non_tabular_files(tmp_path: Path):
    (tmp_path / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "analysis.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "alignment.faa").write_text(">a\nAC-T\n", encoding="utf-8")

    manifest = list_all_capsule_files(tmp_path)

    paths = [file_info.path for file_info in manifest.files]
    categories = {file_info.path: file_info.category for file_info in manifest.files}
    assert "notes/analysis.ipynb" in paths
    assert "alignment.faa" in paths
    assert categories["notes/analysis.ipynb"] == "notebook"
    assert categories["alignment.faa"] == "sequence"


def test_search_filenames_prefers_filename_matches(tmp_path: Path):
    (tmp_path / "Proteomic_data.xlsx").write_text("placeholder", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "analysis.txt").write_text("placeholder", encoding="utf-8")

    results = search_filenames(tmp_path, "proteomic")

    assert len(results) == 1
    assert results[0].filename == "Proteomic_data.xlsx"
    assert results[0].matched_on == "filename"


def test_search_filenames_invalid_regex_raises_value_error(tmp_path: Path):
    (tmp_path / "a.txt").write_text("placeholder", encoding="utf-8")

    try:
        search_filenames(tmp_path, "[")
    except ValueError as exc:
        assert "Invalid filename regex" in str(exc)
    else:
        raise AssertionError("Expected invalid regex to raise ValueError.")


def test_list_capsule_files_remains_top_level_and_tabular(tmp_path: Path):
    (tmp_path / "data.csv").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("note", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "other.csv").write_text("a\n1\n", encoding="utf-8")

    manifest = list_capsule_files(tmp_path)

    assert [file_info.filename for file_info in manifest.files] == ["data.csv"]
