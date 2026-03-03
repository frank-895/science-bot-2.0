"""Public interface for capsule inspection and data-loading tools."""

from science_bot.pipeline.resolution.tools.archives import list_zip_contents
from science_bot.pipeline.resolution.tools.discovery import (
    list_all_capsule_files,
    list_capsule_files,
    search_filenames,
)
from science_bot.pipeline.resolution.tools.excel import list_excel_sheets
from science_bot.pipeline.resolution.tools.notebook import (
    format_notebook_summary,
    get_notebook_outputs,
)
from science_bot.pipeline.resolution.tools.schemas import (
    AllFileInfo,
    CapsuleManifest,
    ColumnInfo,
    ColumnSearchResult,
    ColumnStats,
    ColumnValues,
    ColumnValueSearchResult,
    FastaSummary,
    FileCategory,
    FileInfo,
    FilenameSearchResult,
    FileSchema,
    FullCapsuleManifest,
    NotebookCellOutput,
    NotebookOutputs,
    RowSample,
    ZipEntry,
    ZipManifest,
)
from science_bot.pipeline.resolution.tools.sequence import summarize_fasta_file
from science_bot.pipeline.resolution.tools.tabular import (
    find_files_with_column,
    get_column_stats,
    get_column_values,
    get_file_schema,
    get_row_sample,
    load_dataframe,
    search_column_for_value,
    search_columns,
)

AVAILABLE_TOOLS_TEXT = """
Available tools:
- list_all_capsule_files()
- search_filenames(query)
- list_zip_contents(zip_filename)
- list_excel_sheets(filename)
- find_files_with_column(query)
- get_file_schema(filename)
- search_columns(filename, query)
- get_column_values(filename, column)
- get_column_stats(filename, column)
- search_column_for_value(filename, column, query)
- get_row_sample(filename, columns, n=10, random_sample=False)
- summarize_fasta_file(filename)
""".strip()

__all__ = [
    "AVAILABLE_TOOLS_TEXT",
    # Discovery
    "list_capsule_files",
    "list_all_capsule_files",
    "search_filenames",
    "list_zip_contents",
    "list_excel_sheets",
    "find_files_with_column",
    # Schema inspection
    "get_file_schema",
    "search_columns",
    # Value inspection
    "get_column_values",
    "get_column_stats",
    "search_column_for_value",
    # Row preview
    "get_row_sample",
    # Sequence inspection
    "summarize_fasta_file",
    # Notebook inspection
    "get_notebook_outputs",
    "format_notebook_summary",
    # Data assembly
    "load_dataframe",
    # Schemas
    "AllFileInfo",
    "CapsuleManifest",
    "ColumnInfo",
    "ColumnSearchResult",
    "ColumnStats",
    "ColumnValueSearchResult",
    "ColumnValues",
    "FastaSummary",
    "FileCategory",
    "FileInfo",
    "FilenameSearchResult",
    "FileSchema",
    "FullCapsuleManifest",
    "NotebookCellOutput",
    "NotebookOutputs",
    "RowSample",
    "ZipEntry",
    "ZipManifest",
]
