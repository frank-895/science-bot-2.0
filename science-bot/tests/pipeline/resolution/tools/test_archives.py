import zipfile
from pathlib import Path

from science_bot.pipeline.resolution.tools.archives import list_zip_contents


def test_list_zip_contents_reports_readable_entries(tmp_path: Path):
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("inner/data.tsv", "a\tb\n1\t2\n")
        archive.writestr("inner/readme.txt", "note")

    manifest = list_zip_contents(tmp_path, "bundle.zip")

    assert manifest.zip_filename == "bundle.zip"
    assert {entry.inner_path for entry in manifest.entries} == {
        "inner/data.tsv",
        "inner/readme.txt",
    }
    readable = {entry.inner_path: entry.is_readable for entry in manifest.entries}
    assert readable["inner/data.tsv"] is True
    assert readable["inner/readme.txt"] is False
