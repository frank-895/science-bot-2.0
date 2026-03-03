"""Pydantic output models for capsule inspection tools."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

FileCategory = Literal[
    "tabular",
    "excel",
    "zip",
    "sequence",
    "notebook",
    "json",
    "text",
    "other",
]


class FileInfo(BaseModel):
    """Summary information about a single capsule file."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    size_bytes: int
    size_human: str
    row_count: int | None
    column_count: int | None
    is_wide: bool | None
    file_type: Literal["csv", "tsv", "excel", "zip"]


class AllFileInfo(BaseModel):
    """Summary information about any file discovered in a capsule."""

    model_config = ConfigDict(extra="forbid")

    path: str
    filename: str
    extension: str
    size_bytes: int
    size_human: str
    category: FileCategory
    is_supported_for_deeper_inspection: bool


class CapsuleManifest(BaseModel):
    """Directory listing for a capsule."""

    model_config = ConfigDict(extra="forbid")

    capsule_path: str
    files: list[FileInfo]
    total_size_bytes: int


class FullCapsuleManifest(BaseModel):
    """Recursive file listing for a capsule."""

    model_config = ConfigDict(extra="forbid")

    capsule_path: str
    files: list[AllFileInfo]
    total_size_bytes: int


class ZipEntry(BaseModel):
    """A single entry within a zip archive."""

    model_config = ConfigDict(extra="forbid")

    inner_path: str
    size_bytes: int
    file_type: FileCategory | Literal["csv", "tsv"]
    is_readable: bool


class ZipManifest(BaseModel):
    """Contents listing for a zip archive."""

    model_config = ConfigDict(extra="forbid")

    zip_filename: str
    entries: list[ZipEntry]


class ColumnInfo(BaseModel):
    """Per-column metadata derived from a small sample of rows."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: Literal["string", "integer", "float", "boolean", "mixed"]
    sample_values: list[str | int | float | bool | None]
    null_count_in_sample: int


class FileSchema(BaseModel):
    """Column-level schema for a tabular file."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    row_count: int
    column_count: int
    columns: list[ColumnInfo]
    columns_truncated: bool
    max_schema_columns: int


class ColumnSearchResult(BaseModel):
    """Header search results within a single file."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    query: str
    matches: list[str]
    total_matches: int


class ColumnValues(BaseModel):
    """Unique values observed in a single column."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    column: str
    dtype: str
    unique_count: int
    values: list[str | int | float | bool | None]
    truncated: bool


class ColumnStats(BaseModel):
    """Descriptive statistics for a single column."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    column: str
    dtype: str
    row_count: int
    null_count: int
    unique_count: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    most_common: list[tuple[str, int]] | None = None


class ColumnValueSearchResult(BaseModel):
    """Results of a substring search within a column's values."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    column: str
    query: str
    matches: list[str | int | float | bool]
    total_matches: int
    truncated: bool


class RowSample(BaseModel):
    """A sample of rows from a tabular file."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    columns: list[str]
    rows: list[dict[str, str | int | float | bool | None]]
    total_rows_in_file: int
    sample_size: int
    sampled_randomly: bool


class FilenameSearchResult(BaseModel):
    """Filename or path search result for one capsule file."""

    model_config = ConfigDict(extra="forbid")

    query: str
    path: str
    filename: str
    category: FileCategory
    size_bytes: int
    matched_on: Literal["filename", "path"]
    matched_texts: list[str]


class FastaSummary(BaseModel):
    """Compact summary statistics for a FASTA-like file."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    sequence_count: int
    min_length: int | None
    max_length: int | None
    mean_length: float | None
    total_characters: int
    gap_count: int
    gap_fraction: float | None
    alphabet_hint: Literal["dna", "protein", "unknown"]
    truncated: bool = False


class NotebookCellOutput(BaseModel):
    """A single executed code-cell output from a Jupyter notebook."""

    model_config = ConfigDict(extra="forbid")

    cell_index: int
    source_preview: str
    output_text: str


class NotebookOutputs(BaseModel):
    """Executed outputs collected from a capsule's Jupyter notebook."""

    model_config = ConfigDict(extra="forbid")

    notebook_found: bool
    total_cells_with_output: int
    outputs: list[NotebookCellOutput]
    truncated: bool
