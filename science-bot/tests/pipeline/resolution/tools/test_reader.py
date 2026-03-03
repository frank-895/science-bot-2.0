import zipfile
from pathlib import Path

from science_bot.pipeline.resolution.tools.reader import parse_filename


def test_parse_filename_supports_sheet_suffix(tmp_path: Path):
    file_path = tmp_path / "results.xlsx"
    file_path.write_text("placeholder", encoding="utf-8")

    ref = parse_filename(tmp_path, "results.xlsx::Sheet1")

    assert ref.file_path == file_path.resolve()
    assert ref.sheet_name == "Sheet1"


def test_parse_filename_supports_zip_inner_paths(tmp_path: Path):
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("inner/data.tsv", "a\tb\n1\t2\n")

    ref = parse_filename(tmp_path, "bundle.zip/inner/data.tsv")

    assert ref.zip_path == archive_path.resolve()
    assert ref.inner_path == "inner/data.tsv"
