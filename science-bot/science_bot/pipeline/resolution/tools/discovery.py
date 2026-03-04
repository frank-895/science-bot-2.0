"""Discovery-oriented capsule tools for resolution."""

import re
from pathlib import Path
from typing import Literal

from science_bot.pipeline.resolution.tools.reader import (
    EXCEL_EXTENSIONS,
    TABULAR_EXTENSIONS,
    count_rows,
    parse_filename,
    read_header,
)
from science_bot.pipeline.resolution.tools.schemas import (
    AllFileInfo,
    CapsuleManifest,
    FileCategory,
    FileInfo,
    FilenameSearchResult,
    FullCapsuleManifest,
)

WIDE_FILE_THRESHOLD: int = 200
SEQUENCE_EXTENSIONS: frozenset[str] = frozenset({".faa", ".fasta", ".fa", ".aln"})
TEXT_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".log"})


def list_capsule_files(capsule_path: Path) -> CapsuleManifest:
    """Enumerate top-level readable files and zip containers.

    Args:
        capsule_path: Absolute path to the capsule directory.

    Returns:
        CapsuleManifest: Manifest sorted by file size descending.
    """
    infos: list[FileInfo] = []
    for path in sorted(capsule_path.iterdir()):
        if not path.is_file():
            continue
        extension = path.suffix.lower()
        if extension not in TABULAR_EXTENSIONS and extension != ".zip":
            continue

        size_bytes = path.stat().st_size
        if extension == ".zip":
            infos.append(
                FileInfo(
                    filename=path.name,
                    size_bytes=size_bytes,
                    size_human=human_size(size_bytes),
                    row_count=None,
                    column_count=None,
                    is_wide=None,
                    file_type="zip",
                )
            )
            continue

        file_type = _tabular_file_type(extension)
        try:
            ref = parse_filename(capsule_path, path.name)
            columns = read_header(ref)
            column_count = len(columns)
            row_count = count_rows(ref)
        except Exception:
            column_count = None
            row_count = None

        infos.append(
            FileInfo(
                filename=path.name,
                size_bytes=size_bytes,
                size_human=human_size(size_bytes),
                row_count=row_count,
                column_count=column_count,
                is_wide=column_count is not None and column_count > WIDE_FILE_THRESHOLD,
                file_type=file_type,
            )
        )

    infos.sort(key=lambda file_info: file_info.size_bytes, reverse=True)
    return CapsuleManifest(
        capsule_path=str(capsule_path),
        files=infos,
        total_size_bytes=sum(file_info.size_bytes for file_info in infos),
    )


def list_all_capsule_files(capsule_path: Path) -> FullCapsuleManifest:
    """Enumerate every file under a capsule with a coarse category.

    Args:
        capsule_path: Absolute path to the capsule directory.

    Returns:
        FullCapsuleManifest: Recursive manifest sorted by relative path.
    """
    capsule_resolved = capsule_path.resolve()
    files: list[AllFileInfo] = []

    for path in sorted(capsule_resolved.rglob("*")):
        if not path.is_file():
            continue
        relative_path = str(path.relative_to(capsule_resolved))
        extension = path.suffix.lower()
        category = file_category_for_extension(extension)
        size_bytes = path.stat().st_size
        files.append(
            AllFileInfo(
                path=relative_path,
                filename=path.name,
                extension=extension,
                size_bytes=size_bytes,
                size_human=human_size(size_bytes),
                category=category,
                is_supported_for_deeper_inspection=_supports_deeper_inspection(
                    category
                ),
            )
        )

    return FullCapsuleManifest(
        capsule_path=str(capsule_resolved),
        files=files,
        total_size_bytes=sum(file_info.size_bytes for file_info in files),
    )


def search_filenames(capsule_path: Path, query: str) -> list[FilenameSearchResult]:
    """Search filenames and relative paths across the full capsule tree.

    Args:
        capsule_path: Absolute path to the capsule directory.
        query: Case-insensitive substring or regex pattern.

    Returns:
        list[FilenameSearchResult]: Matching file paths.

    Raises:
        ValueError: If the regex query is invalid.
    """
    manifest = list_all_capsule_files(capsule_path)
    query_lower = query.lower()
    substring_matches: list[FilenameSearchResult] = []

    for file_info in manifest.files:
        matched_texts: list[str] = []
        matched_on: str | None = None
        if query_lower in file_info.filename.lower():
            matched_on = "filename"
            matched_texts.append(file_info.filename)
        if query_lower in file_info.path.lower():
            if matched_on is None:
                matched_on = "path"
            matched_texts.append(file_info.path)
        if matched_on is not None:
            substring_matches.append(
                FilenameSearchResult(
                    query=query,
                    path=file_info.path,
                    filename=file_info.filename,
                    category=file_info.category,
                    size_bytes=file_info.size_bytes,
                    matched_on=matched_on,
                    matched_texts=sorted(set(matched_texts)),
                )
            )

    if substring_matches:
        return _sort_filename_matches(substring_matches)

    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"Invalid filename regex: {query}") from exc

    regex_matches: list[FilenameSearchResult] = []
    for file_info in manifest.files:
        matched_texts = []
        matched_on: str | None = None
        if pattern.search(file_info.filename):
            matched_on = "filename"
            matched_texts.append(file_info.filename)
        if pattern.search(file_info.path):
            if matched_on is None:
                matched_on = "path"
            matched_texts.append(file_info.path)
        if matched_on is not None:
            regex_matches.append(
                FilenameSearchResult(
                    query=query,
                    path=file_info.path,
                    filename=file_info.filename,
                    category=file_info.category,
                    size_bytes=file_info.size_bytes,
                    matched_on=matched_on,
                    matched_texts=sorted(set(matched_texts)),
                )
            )
    return _sort_filename_matches(regex_matches)


def human_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        size_bytes: Raw byte count.

    Returns:
        str: Human-readable size.
    """
    value = size_bytes
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value} {unit}"
        value //= 1024
    return f"{value} TB"


def file_category_for_extension(extension: str) -> FileCategory:
    """Map a file extension to a coarse discovery category.

    Args:
        extension: Lowercase file extension including the leading dot.

    Returns:
        FileCategory: Coarse category for resolver discovery.
    """
    if extension in {".csv", ".tsv", ".tab"}:
        return "tabular"
    if extension in EXCEL_EXTENSIONS:
        return "excel"
    if extension == ".zip":
        return "zip"
    if extension in SEQUENCE_EXTENSIONS:
        return "sequence"
    if extension == ".ipynb":
        return "notebook"
    if extension == ".json":
        return "json"
    if extension in TEXT_EXTENSIONS:
        return "text"
    return "other"


def _supports_deeper_inspection(category: FileCategory) -> bool:
    """Return whether discovery should advertise richer follow-up tools."""
    return category in {"tabular", "excel", "zip", "sequence"}


def _sort_filename_matches(
    matches: list[FilenameSearchResult],
) -> list[FilenameSearchResult]:
    """Sort filename matches by relevance and deterministic path order."""
    return sorted(
        matches,
        key=lambda result: (
            0 if result.matched_on == "filename" else 1,
            len(result.path),
            result.path.lower(),
        ),
    )


def _tabular_file_type(extension: str) -> Literal["csv", "tsv", "excel"]:
    """Map a tabular extension to the manifest file-type label."""
    if extension in EXCEL_EXTENSIONS:
        return "excel"
    if extension in {".tsv", ".tab"}:
        return "tsv"
    return "csv"
