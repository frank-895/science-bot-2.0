import gzip
import zipfile
from pathlib import Path
from typing import cast

import pandas as pd
from science_bot.pipeline.resolution.tools.reader import parse_filename
from science_bot.pipeline.resolution.tools.tabular import get_file_schema


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


def test_get_file_schema_reads_gzip_compressed_csv(tmp_path: Path):
    file_path = tmp_path / "data.csv"
    with gzip.open(file_path, "wt", encoding="utf-8") as handle:
        handle.write("sample_id,value\ns1,1\ns2,2\n")

    schema = get_file_schema(tmp_path, "data.csv")

    assert [column.name for column in schema.columns] == ["sample_id", "value"]
    assert schema.row_count == 2


def test_get_file_schema_normalizes_duplicate_headers(tmp_path: Path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("gene,gene,value\nA,B,1\n", encoding="utf-8")

    schema = get_file_schema(tmp_path, "data.csv")

    assert [column.name for column in schema.columns] == [
        "gene",
        "gene__2",
        "value",
    ]


def test_get_file_schema_treats_text_xls_as_tabular(tmp_path: Path):
    file_path = tmp_path / "clinical.xls"
    file_path.write_text("sample_id\tGroup\r\ns1\tA\r\n", encoding="utf-8")

    schema = get_file_schema(tmp_path, "clinical.xls")

    assert [column.name for column in schema.columns] == ["sample_id", "Group"]


def test_get_file_schema_strips_outer_quotes_and_whitespace(tmp_path: Path):
    file_path = tmp_path / "quoted.csv"
    file_path.write_text(' "AEDECOD" ,value\r\nfoo,1\r\n', encoding="utf-8")

    schema = get_file_schema(tmp_path, "quoted.csv")

    assert [column.name for column in schema.columns] == ["AEDECOD", "value"]


def test_reader_retries_csv_with_python_engine(monkeypatch):
    calls = []

    class FakeParserError(pd.errors.ParserError):
        pass

    def fake_read_csv(source, **kwargs):
        calls.append(kwargs.get("engine", "c"))
        if len(calls) == 1:
            raise FakeParserError("bad parse")
        return pd.DataFrame({"a": [1], "b": [2]})

    monkeypatch.setattr(
        "science_bot.pipeline.resolution.tools.reader.pd.read_csv",
        fake_read_csv,
    )

    from science_bot.pipeline.resolution.tools import reader

    result = cast(
        pd.DataFrame,
        reader._read_csv_with_fallback(
            source=Path(__file__).open("rb"),
            sep=",",
            names=["a", "b"],
            skiprows=0,
            usecols=None,
            nrows=None,
        ),
    )

    assert list(result.columns) == ["a", "b"]
    assert calls == [None, "python"]
