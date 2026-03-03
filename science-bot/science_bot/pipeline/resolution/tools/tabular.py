"""Tabular inspection and loading tools for resolution."""

import random
import re
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from science_bot.pipeline.resolution.tools.reader import (
    TABULAR_EXTENSIONS,
    count_rows,
    parse_filename,
    read_header,
    read_tabular,
)
from science_bot.pipeline.resolution.tools.schemas import (
    ColumnInfo,
    ColumnSearchResult,
    ColumnStats,
    ColumnValues,
    ColumnValueSearchResult,
    FileSchema,
    RowSample,
)

WIDE_FILE_THRESHOLD: int = 200
MAX_SCHEMA_COLUMNS: int = 200
MAX_VALUE_COUNT: int = 50


def find_files_with_column(capsule_path: Path, query: str) -> list[ColumnSearchResult]:
    """Search column headers across all top-level readable files.

    Args:
        capsule_path: Absolute path to the capsule directory.
        query: Substring or regex pattern to match against column names.

    Returns:
        list[ColumnSearchResult]: One entry per file with at least one match.
    """
    results: list[ColumnSearchResult] = []
    for path in sorted(capsule_path.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TABULAR_EXTENSIONS:
            continue
        try:
            ref = parse_filename(capsule_path, path.name)
            matches = _filter_columns(read_header(ref), query)
        except Exception:
            continue
        if matches:
            results.append(
                ColumnSearchResult(
                    filename=path.name,
                    query=query,
                    matches=matches,
                    total_matches=len(matches),
                )
            )
    return results


def get_file_schema(capsule_path: Path, filename: str) -> FileSchema:
    """Return a compact schema derived from at most five data rows.

    Args:
        capsule_path: Absolute path to the capsule directory.
        filename: Extended filename supporting zip and sheet syntax.

    Returns:
        FileSchema: Column names, dtypes, and small sample values.
    """
    ref = parse_filename(capsule_path, filename)
    dataframe = cast(pd.DataFrame, read_tabular(ref, nrows=5))
    all_columns = list(dataframe.columns)
    column_count = len(all_columns)

    columns_truncated = column_count > MAX_SCHEMA_COLUMNS
    described_columns = all_columns[:MAX_SCHEMA_COLUMNS]
    dataframe = dataframe[described_columns]
    row_count = count_rows(ref)

    column_infos: list[ColumnInfo] = []
    for column in described_columns:
        series = dataframe[column]
        column_infos.append(
            ColumnInfo(
                name=column,
                dtype=_infer_dtype_label(series),
                sample_values=[_sanitize_value(value) for value in series.tolist()],
                null_count_in_sample=int(series.isna().sum()),
            )
        )

    return FileSchema(
        filename=filename,
        row_count=row_count,
        column_count=column_count,
        columns=column_infos,
        columns_truncated=columns_truncated,
        max_schema_columns=MAX_SCHEMA_COLUMNS,
    )


def search_columns(capsule_path: Path, filename: str, query: str) -> ColumnSearchResult:
    """Search column headers in one file without loading data rows.

    Args:
        capsule_path: Absolute path to the capsule directory.
        filename: Extended filename.
        query: Substring or regex pattern.

    Returns:
        ColumnSearchResult: Matching column names.
    """
    ref = parse_filename(capsule_path, filename)
    headers = read_header(ref)
    matches = _filter_columns(headers, query)
    return ColumnSearchResult(
        filename=filename,
        query=query,
        matches=matches,
        total_matches=len(matches),
    )


def get_column_values(
    capsule_path: Path,
    filename: str,
    column: str,
    max_values: int = MAX_VALUE_COUNT,
) -> ColumnValues:
    """Return unique values observed in a single column.

    Args:
        capsule_path: Absolute path to the capsule directory.
        filename: Extended filename.
        column: Column name to inspect.
        max_values: Maximum distinct values to return.

    Returns:
        ColumnValues: Unique values with truncation metadata.
    """
    ref = parse_filename(capsule_path, filename)
    dataframe = cast(pd.DataFrame, read_tabular(ref, usecols=[column]))
    series = dataframe[column]
    unique_values = series.dropna().unique().tolist()
    unique_count = len(unique_values)
    return ColumnValues(
        filename=filename,
        column=column,
        dtype=_infer_dtype_label(series),
        unique_count=unique_count,
        values=[_sanitize_value(value) for value in unique_values[:max_values]],
        truncated=unique_count > max_values,
    )


def get_column_stats(capsule_path: Path, filename: str, column: str) -> ColumnStats:
    """Return descriptive statistics for a single column.

    Args:
        capsule_path: Absolute path to the capsule directory.
        filename: Extended filename.
        column: Column name to inspect.

    Returns:
        ColumnStats: Numeric or categorical summary statistics.
    """
    ref = parse_filename(capsule_path, filename)
    dataframe = cast(pd.DataFrame, read_tabular(ref, usecols=[column]))
    series = dataframe[column]
    dtype_label = _infer_dtype_label(series)
    numeric = pd.to_numeric(series, errors="coerce")

    min_value = max_value = mean_value = std_value = None
    most_common = None
    if numeric.notna().sum() > 0 and dtype_label in {"integer", "float"}:
        min_value = float(numeric.min())
        max_value = float(numeric.max())
        mean_value = float(numeric.mean())
        std_value = float(numeric.std(ddof=1)) if len(numeric.dropna()) > 1 else None
    else:
        counts = series.dropna().astype(str).value_counts().head(10)
        most_common = [(str(key), int(value)) for key, value in counts.items()]

    return ColumnStats(
        filename=filename,
        column=column,
        dtype=dtype_label,
        row_count=len(series),
        null_count=int(series.isna().sum()),
        unique_count=int(series.nunique(dropna=True)),
        min=min_value,
        max=max_value,
        mean=mean_value,
        std=std_value,
        most_common=most_common,
    )


def search_column_for_value(
    capsule_path: Path,
    filename: str,
    column: str,
    query: str,
    max_matches: int = 50,
) -> ColumnValueSearchResult:
    """Find distinct column values that contain the query string.

    Args:
        capsule_path: Absolute path to the capsule directory.
        filename: Extended filename.
        column: Column name to search.
        query: Case-insensitive substring to search for.
        max_matches: Maximum distinct matching values to return.

    Returns:
        ColumnValueSearchResult: Matching values with truncation metadata.
    """
    ref = parse_filename(capsule_path, filename)
    dataframe = cast(pd.DataFrame, read_tabular(ref, usecols=[column]))
    series = dataframe[column]
    query_lower = query.lower()

    matches: list[str | int | float | bool] = []
    for value in series.dropna().unique():
        if query_lower in str(value).lower():
            matches.append(_sanitize_value(value))  # type: ignore[arg-type]

    return ColumnValueSearchResult(
        filename=filename,
        column=column,
        query=query,
        matches=matches[:max_matches],
        total_matches=len(matches),
        truncated=len(matches) > max_matches,
    )


def get_row_sample(
    capsule_path: Path,
    filename: str,
    columns: list[str],
    n: int = 10,
    random_sample: bool = False,
) -> RowSample:
    """Return a head or random sample of named columns.

    Args:
        capsule_path: Absolute path to the capsule directory.
        filename: Extended filename.
        columns: Column names to include.
        n: Number of rows to return.
        random_sample: If True, use streamed reservoir sampling.

    Returns:
        RowSample: Sampled rows with metadata.

    Raises:
        ValueError: If columns is empty.
    """
    if not columns:
        raise ValueError("columns must be non-empty.")

    ref = parse_filename(capsule_path, filename)
    total_rows = count_rows(ref)

    if not random_sample:
        dataframe = cast(pd.DataFrame, read_tabular(ref, usecols=columns, nrows=n))
        rows = _df_to_row_dicts(dataframe)
        return RowSample(
            filename=filename,
            columns=columns,
            rows=rows,
            total_rows_in_file=total_rows,
            sample_size=len(rows),
            sampled_randomly=False,
        )

    reservoir: list[dict[str, str | int | float | bool | None]] = []
    row_index = 0
    chunked = read_tabular(ref, usecols=columns, chunksize=1000)
    chunks = [chunked] if isinstance(chunked, pd.DataFrame) else chunked

    for chunk in chunks:
        for _, row in chunk.iterrows():
            row_dict = {
                column_name: _sanitize_value(row[column_name])
                for column_name in columns
            }
            if row_index < n:
                reservoir.append(row_dict)
            else:
                replacement_index = random.randint(0, row_index)
                if replacement_index < n:
                    reservoir[replacement_index] = row_dict
            row_index += 1

    return RowSample(
        filename=filename,
        columns=columns,
        rows=reservoir,
        total_rows_in_file=total_rows,
        sample_size=len(reservoir),
        sampled_randomly=True,
    )


def load_dataframe(
    capsule_path: Path,
    filename: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a tabular file into a dataframe for deterministic execution.

    Args:
        capsule_path: Absolute path to the capsule directory.
        filename: Extended filename.
        columns: Optional explicit column subset.

    Returns:
        pd.DataFrame: Loaded dataframe.

    Raises:
        ValueError: If a wide file is loaded without explicit columns or if
            requested columns are missing.
    """
    ref = parse_filename(capsule_path, filename)

    if columns is None:
        headers = read_header(ref)
        if len(headers) > WIDE_FILE_THRESHOLD:
            raise ValueError(
                f"File '{filename}' has {len(headers)} columns "
                f"(>{WIDE_FILE_THRESHOLD}). Specify 'columns' explicitly."
            )
        return cast(pd.DataFrame, read_tabular(ref))

    headers = read_header(ref)
    missing_columns = [column for column in columns if column not in headers]
    if missing_columns:
        raise ValueError(f"Columns not found in '{filename}': {missing_columns}")

    return cast(pd.DataFrame, read_tabular(ref, usecols=columns))


def _filter_columns(columns: list[str], query: str) -> list[str]:
    query_lower = query.lower()
    substring_matches = [column for column in columns if query_lower in column.lower()]
    if substring_matches:
        return substring_matches
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        return []
    return [column for column in columns if pattern.search(column)]


def _infer_dtype_label(
    series: pd.Series,
) -> Literal["string", "integer", "float", "boolean", "mixed"]:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    sample = series.dropna().head(20)
    if sample.empty:
        return "string"
    types = {type(value) for value in sample}
    if types <= {int}:
        return "integer"
    if types <= {float}:
        return "float"
    if types <= {bool}:
        return "boolean"
    if types <= {int, float}:
        return "float"
    if len(types) > 1:
        return "mixed"
    return "string"


def _sanitize_value(value: object) -> str | int | float | bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return str(value)


def _df_to_row_dicts(
    dataframe: pd.DataFrame,
) -> list[dict[str, str | int | float | bool | None]]:
    return [
        {column: _sanitize_value(row[column]) for column in dataframe.columns}
        for _, row in dataframe.iterrows()
    ]
