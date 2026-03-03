from pathlib import Path

from science_bot.pipeline.resolution.tools.tabular import (
    get_column_values,
    get_file_schema,
)


def test_get_file_schema_reads_basic_csv(tmp_path: Path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("sample_id,value\ns1,1\ns2,2\n", encoding="utf-8")

    schema = get_file_schema(tmp_path, "data.csv")

    assert schema.filename == "data.csv"
    assert schema.row_count == 2
    assert schema.column_count == 2
    assert [column.name for column in schema.columns] == ["sample_id", "value"]


def test_get_column_values_returns_distinct_values(tmp_path: Path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("group\nA\nB\nA\n", encoding="utf-8")

    values = get_column_values(tmp_path, "data.csv", "group")

    assert values.unique_count == 2
    assert values.values == ["A", "B"]
