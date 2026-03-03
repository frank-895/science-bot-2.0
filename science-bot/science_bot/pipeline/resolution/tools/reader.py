"""Shared filename parsing and tabular reader helpers for resolution tools."""

import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import IO, Literal

import pandas as pd
from pandas.io.parsers.readers import TextFileReader

TABULAR_EXTENSIONS = {".csv", ".tsv", ".tab", ".xlsx", ".xls"}
EXCEL_EXTENSIONS = {".xlsx", ".xls"}


def excel_engine(extension: str) -> Literal["xlrd", "openpyxl"]:
    """Return the pandas Excel engine for a file extension.

    Args:
        extension: Lowercase file extension including the leading dot.

    Returns:
        Literal["xlrd", "openpyxl"]: Pandas engine name.
    """
    return "xlrd" if extension == ".xls" else "openpyxl"


class ManagedChunkReader:
    """Iterator wrapper that closes backing resources when exhausted."""

    def __init__(
        self,
        reader: TextFileReader,
        *,
        raw_file: IO[bytes] | None = None,
        zip_file: zipfile.ZipFile | None = None,
    ) -> None:
        """Initialize a managed chunk reader.

        Args:
            reader: Pandas chunk reader.
            raw_file: Optional open file handle for zip members.
            zip_file: Optional owning zip file.
        """
        self._reader = reader
        self._raw_file = raw_file
        self._zip_file = zip_file

    def __iter__(self) -> "ManagedChunkReader":
        """Return the chunk iterator.

        Returns:
            ManagedChunkReader: Iterator instance.
        """
        return self

    def __next__(self) -> pd.DataFrame:
        """Return the next dataframe chunk.

        Returns:
            pd.DataFrame: Next chunk from the underlying reader.

        Raises:
            StopIteration: When no more chunks remain.
        """
        try:
            return next(self._reader)
        except StopIteration:
            self.close()
            raise

    def close(self) -> None:
        """Close the underlying reader and any zip resources."""
        self._reader.close()
        if self._raw_file is not None:
            self._raw_file.close()
            self._raw_file = None
        if self._zip_file is not None:
            self._zip_file.close()
            self._zip_file = None


@dataclass
class FileRef:
    """Resolved reference to a file, optionally inside a zip archive."""

    capsule_path: Path
    file_path: Path
    zip_path: Path | None = None
    inner_path: str | None = None
    sheet_name: str | int = field(default=0)

    @property
    def extension(self) -> str:
        """Return the lowercase extension of the logical file.

        Returns:
            str: Lowercase file extension.
        """
        if self.inner_path is not None:
            return Path(self.inner_path.split("::")[0]).suffix.lower()
        raw = str(self.file_path)
        sheet_sep = raw.rfind("::")
        base = raw[:sheet_sep] if sheet_sep != -1 else raw
        return Path(base).suffix.lower()

    @property
    def is_excel(self) -> bool:
        """Return whether the logical file is Excel-like.

        Returns:
            bool: True when the file extension is an Excel extension.
        """
        return self.extension in EXCEL_EXTENSIONS

    @property
    def separator(self) -> str:
        """Return the CSV/TSV separator for a logical file.

        Returns:
            str: Tab for TSV/TAB files, comma otherwise.
        """
        return "\t" if self.extension in {".tsv", ".tab"} else ","


def parse_filename(capsule_path: Path, filename: str) -> FileRef:
    """Parse an extended filename into a resolved file reference.

    Args:
        capsule_path: Absolute path to the capsule directory.
        filename: Extended filename string.

    Returns:
        FileRef: Resolved file reference.

    Raises:
        FileNotFoundError: If the referenced file does not exist.
        ValueError: If the path escapes the capsule directory.
    """
    capsule_resolved = capsule_path.resolve()
    zip_marker = ".zip/"
    zip_pos = filename.lower().find(zip_marker)

    if zip_pos != -1:
        zip_filename = filename[: zip_pos + 4]
        inner_raw = filename[zip_pos + len(zip_marker) :]

        zip_file_path = (capsule_path / zip_filename).resolve()
        _assert_within_capsule(capsule_resolved, zip_file_path)
        if not zip_file_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_file_path}")

        sheet_name: str | int = 0
        inner_path_base = inner_raw
        if "::" in inner_raw:
            inner_path_base, sheet_name = inner_raw.rsplit("::", 1)

        return FileRef(
            capsule_path=capsule_resolved,
            file_path=zip_file_path,
            zip_path=zip_file_path,
            inner_path=inner_path_base,
            sheet_name=sheet_name,
        )

    sheet_name = 0
    filename_base = filename
    if "::" in filename:
        filename_base, sheet_name = filename.rsplit("::", 1)

    file_path = (capsule_path / filename_base).resolve()
    _assert_within_capsule(capsule_resolved, file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return FileRef(
        capsule_path=capsule_resolved,
        file_path=file_path,
        sheet_name=sheet_name,
    )


def scan_csv_preamble(ref: FileRef) -> tuple[int, str]:
    """Detect leading comment rows and return the effective header line.

    Args:
        ref: Resolved CSV/TSV file reference.

    Returns:
        tuple[int, str]: Number of comment-only rows to skip and the header text.
    """
    sep = ref.separator

    def iter_lines() -> Iterator[str]:
        if ref.zip_path is not None and ref.inner_path is not None:
            with zipfile.ZipFile(ref.zip_path) as zf:
                with zf.open(ref.inner_path) as fobj:
                    for raw in fobj:
                        yield raw.decode("utf-8", errors="replace")
        else:
            with open(ref.file_path, encoding="utf-8", errors="replace") as fobj:
                yield from fobj

    skip = 0
    for line in iter_lines():
        stripped = line.rstrip("\n")
        if stripped.startswith("#") and len(stripped.split(sep)) <= 1:
            skip += 1
            continue
        if stripped.startswith("#"):
            for prefix in ("# ", "#"):
                if stripped.startswith(prefix):
                    return skip, stripped[len(prefix) :]
        return skip, stripped
    return skip, ""


def read_header(ref: FileRef) -> list[str]:
    """Return column names without loading any data rows.

    Args:
        ref: Resolved file reference.

    Returns:
        list[str]: Column names.
    """
    if ref.is_excel:
        engine = excel_engine(ref.extension)
        if ref.zip_path is not None and ref.inner_path is not None:
            with zipfile.ZipFile(ref.zip_path) as zf:
                data = zf.read(ref.inner_path)
            source = BytesIO(data)
        else:
            source = ref.file_path
        df = pd.read_excel(
            source,
            sheet_name=ref.sheet_name,
            nrows=0,
            engine=engine,
        )
        return list(df.columns)

    _, header_line = scan_csv_preamble(ref)
    return header_line.split(ref.separator)


def count_rows(ref: FileRef) -> int:
    """Count data rows (excluding header) without loading the full dataset.

    Args:
        ref: Resolved file reference.

    Returns:
        int: Number of data rows.
    """
    if ref.is_excel:
        engine = excel_engine(ref.extension)
        if ref.zip_path is not None and ref.inner_path is not None:
            with zipfile.ZipFile(ref.zip_path) as zf:
                data = zf.read(ref.inner_path)
            source = BytesIO(data)
        else:
            source = ref.file_path
        df = pd.read_excel(
            source,
            sheet_name=ref.sheet_name,
            usecols=[0],
            engine=engine,
        )
        return len(df)

    skip, _ = scan_csv_preamble(ref)
    row_count = 0
    if ref.zip_path is not None and ref.inner_path is not None:
        with zipfile.ZipFile(ref.zip_path) as zf:
            with zf.open(ref.inner_path) as fobj:
                for _ in fobj:
                    row_count += 1
    else:
        with open(ref.file_path, "rb") as fobj:
            for _ in fobj:
                row_count += 1
    return max(0, row_count - 1 - skip)


def read_tabular(
    ref: FileRef,
    *,
    usecols: list[str] | None = None,
    nrows: int | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Read a tabular file into a dataframe or chunk iterator.

    Args:
        ref: Resolved file reference.
        usecols: Columns to load. None loads all columns.
        nrows: Maximum number of rows to read.
        chunksize: If set, return a chunk iterator for CSV-like files.

    Returns:
        pd.DataFrame | Iterator[pd.DataFrame]: Loaded dataframe or chunk iterator.
    """
    if ref.is_excel:
        engine = excel_engine(ref.extension)
        if ref.zip_path is not None and ref.inner_path is not None:
            with zipfile.ZipFile(ref.zip_path) as zf:
                data = zf.read(ref.inner_path)
            source = BytesIO(data)
        else:
            source = ref.file_path
        return pd.read_excel(
            source,
            sheet_name=ref.sheet_name,
            usecols=usecols,
            nrows=nrows,
            engine=engine,
        )

    skip, header_line = scan_csv_preamble(ref)
    header_cols = header_line.split(ref.separator)
    if ref.zip_path is not None and ref.inner_path is not None:
        zf = zipfile.ZipFile(ref.zip_path)
        raw_file = zf.open(ref.inner_path)
        if chunksize is not None:
            reader = pd.read_csv(
                raw_file,
                sep=ref.separator,
                names=header_cols,
                skiprows=skip + 1,
                usecols=usecols,
                nrows=nrows,
                low_memory=False,
                chunksize=chunksize,
            )
            return ManagedChunkReader(reader, raw_file=raw_file, zip_file=zf)
        reader = pd.read_csv(
            raw_file,
            sep=ref.separator,
            names=header_cols,
            skiprows=skip + 1,
            usecols=usecols,
            nrows=nrows,
            low_memory=False,
        )
        raw_file.close()
        zf.close()
        return reader

    if chunksize is not None:
        return pd.read_csv(
            ref.file_path,
            sep=ref.separator,
            names=header_cols,
            skiprows=skip + 1,
            usecols=usecols,
            nrows=nrows,
            low_memory=False,
            chunksize=chunksize,
        )
    return pd.read_csv(
        ref.file_path,
        sep=ref.separator,
        names=header_cols,
        skiprows=skip + 1,
        usecols=usecols,
        nrows=nrows,
        low_memory=False,
    )


def _assert_within_capsule(capsule_resolved: Path, target: Path) -> None:
    """Raise if a resolved path escapes the capsule root.

    Args:
        capsule_resolved: Resolved capsule root.
        target: Candidate path.

    Raises:
        ValueError: If the target path escapes the capsule root.
    """
    try:
        target.relative_to(capsule_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Path {target} escapes the capsule directory {capsule_resolved}"
        ) from exc
