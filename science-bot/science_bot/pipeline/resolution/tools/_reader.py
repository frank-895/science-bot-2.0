"""Internal file-reference and reader abstractions for capsule tools."""

import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import IO, Literal

import pandas as pd
from pandas.io.parsers.readers import TextFileReader

_TABULAR_EXTENSIONS = {".csv", ".tsv", ".tab", ".xlsx", ".xls"}
_EXCEL_EXTENSIONS = {".xlsx", ".xls"}


def _excel_engine(extension: str) -> Literal["xlrd", "openpyxl"]:
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
    file_path: Path  # absolute path on disk (the zip itself, or the direct file)
    zip_path: Path | None = None  # set when file is inside a zip
    inner_path: str | None = None  # path within the zip
    sheet_name: str | int = field(default=0)  # for Excel; ignored otherwise

    @property
    def extension(self) -> str:
        """Return the lowercase extension of the logical file."""
        if self.inner_path is not None:
            return Path(self.inner_path.split("::")[0]).suffix.lower()
        raw = str(self.file_path)
        sheet_sep = raw.rfind("::")
        base = raw[:sheet_sep] if sheet_sep != -1 else raw
        return Path(base).suffix.lower()

    @property
    def is_excel(self) -> bool:
        return self.extension in _EXCEL_EXTENSIONS

    @property
    def separator(self) -> str:
        return "\t" if self.extension in {".tsv", ".tab"} else ","


def parse_filename(capsule_path: Path, filename: str) -> "FileRef":
    """Parse an extended filename into a FileRef.

    Supported syntax:
        "data.csv"                         – top-level CSV
        "data.tsv"                         – top-level TSV
        "results.xlsx"                     – Excel, first sheet
        "results.xlsx::Sheet2"             – Excel, named sheet
        "archive.zip/full_table.tsv"       – TSV inside zip
        "archive.zip/results.xlsx::Sheet1" – Excel inside zip, named sheet

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

    # detect zip containment
    zip_marker = ".zip/"
    zip_pos = filename.lower().find(zip_marker)

    if zip_pos != -1:
        zip_filename = filename[: zip_pos + 4]  # includes ".zip"
        inner_raw = filename[zip_pos + len(zip_marker) :]

        zip_file_path = (capsule_path / zip_filename).resolve()
        _assert_within_capsule(capsule_resolved, zip_file_path)
        if not zip_file_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_file_path}")

        # parse optional sheet from inner path
        sheet_name: str | int = 0
        inner_path_base = inner_raw
        if "::" in inner_raw:
            parts = inner_raw.rsplit("::", 1)
            inner_path_base, sheet_name = parts[0], parts[1]

        return FileRef(
            capsule_path=capsule_resolved,
            file_path=zip_file_path,
            zip_path=zip_file_path,
            inner_path=inner_path_base,
            sheet_name=sheet_name,
        )

    # top-level file (possibly with sheet suffix)
    sheet_name = 0
    filename_base = filename
    if "::" in filename:
        parts = filename.rsplit("::", 1)
        filename_base, sheet_name = parts[0], parts[1]

    file_path = (capsule_path / filename_base).resolve()
    _assert_within_capsule(capsule_resolved, file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return FileRef(
        capsule_path=capsule_resolved,
        file_path=file_path,
        sheet_name=sheet_name,
    )


def _assert_within_capsule(capsule_resolved: Path, target: Path) -> None:
    """Raise ValueError if target is outside the capsule directory."""
    try:
        target.relative_to(capsule_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Path {target} escapes the capsule directory {capsule_resolved}"
        ) from exc


def _open_file_object(ref: FileRef):
    """Return a file-like object for the logical file referenced by ref.

    For zip inner files, opens via zipfile without extraction.
    For direct files, returns the Path (pandas accepts Path directly).
    """
    if ref.zip_path is not None and ref.inner_path is not None:
        zf = zipfile.ZipFile(ref.zip_path)
        return zf.open(ref.inner_path)
    return ref.file_path


def _scan_csv_preamble(ref: FileRef) -> tuple[int, str]:
    """Scan a CSV/TSV file to detect leading comment-only rows.

    A "comment-only" row is one that starts with ``#`` and produces only
    a single field when split by the file's separator (i.e. no tab-separated
    data). The first row with more than one field is the actual header line
    (even if it starts with ``#``).

    Args:
        ref: Resolved CSV/TSV file reference.

    Returns:
        tuple[int, str]: Number of comment-only rows to skip and the raw
            header line text (with any leading ``# `` stripped).
    """
    sep = ref.separator

    def _iter_lines():
        if ref.zip_path is not None and ref.inner_path is not None:
            with zipfile.ZipFile(ref.zip_path) as zf:
                with zf.open(ref.inner_path) as fobj:
                    for raw in fobj:
                        yield raw.decode("utf-8", errors="replace")
        else:
            with open(ref.file_path, encoding="utf-8", errors="replace") as fobj:
                yield from fobj

    skip = 0
    for line in _iter_lines():
        stripped = line.rstrip("\n")
        if stripped.startswith("#") and len(stripped.split(sep)) <= 1:
            skip += 1
        else:
            # This is the header line; strip a leading "# " if present
            header = (
                stripped.lstrip("# ").lstrip("#").strip()
                if stripped.startswith("#")
                else stripped
            )
            # Re-parse using the original stripped line to preserve field values
            if stripped.startswith("#"):
                # Remove the very first "# " or "#" prefix only
                for prefix in ("# ", "#"):
                    if stripped.startswith(prefix):
                        header = stripped[len(prefix) :]
                        break
            else:
                header = stripped
            return skip, header

    return skip, ""


def read_header(ref: FileRef) -> list[str]:
    """Return column names without loading any data rows.

    Args:
        ref: Resolved file reference.

    Returns:
        list[str]: Column names.
    """
    if ref.is_excel:
        engine = _excel_engine(ref.extension)
        if ref.zip_path is not None and ref.inner_path is not None:
            with zipfile.ZipFile(ref.zip_path) as zf:
                data = zf.read(ref.inner_path)
            buf = BytesIO(data)
            df = pd.read_excel(buf, sheet_name=ref.sheet_name, nrows=0, engine=engine)
        else:
            df = pd.read_excel(
                ref.file_path,
                sheet_name=ref.sheet_name,
                nrows=0,
                engine=engine,
            )
        return list(df.columns)

    _, header_line = _scan_csv_preamble(ref)
    return header_line.split(ref.separator)


def count_rows(ref: FileRef) -> int:
    """Count data rows (excluding header) without loading the full dataset.

    Args:
        ref: Resolved file reference.

    Returns:
        int: Number of data rows.
    """
    if ref.is_excel:
        engine = _excel_engine(ref.extension)
        if ref.zip_path is not None and ref.inner_path is not None:
            with zipfile.ZipFile(ref.zip_path) as zf:
                data = zf.read(ref.inner_path)
            buf = BytesIO(data)
            df = pd.read_excel(
                buf,
                sheet_name=ref.sheet_name,
                usecols=[0],
                engine=engine,
            )
        else:
            df = pd.read_excel(
                ref.file_path,
                sheet_name=ref.sheet_name,
                usecols=[0],
                engine=engine,
            )
        return len(df)

    # CSV/TSV — stream-count newlines
    skip, _ = _scan_csv_preamble(ref)
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

    # subtract header line and comment-only lines
    return max(0, row_count - 1 - skip)


def read_tabular(
    ref: FileRef,
    *,
    usecols: list[str] | None = None,
    nrows: int | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Read a tabular file into a DataFrame.

    Routes to read_csv or read_excel based on file extension.
    For zip inner files, uses zipfile to obtain a file-like object.

    Args:
        ref: Resolved file reference.
        usecols: Columns to load. None loads all columns.
        nrows: Maximum number of rows to read.
        chunksize: If set, returns a TextFileReader (CSV only).

    Returns:
        pd.DataFrame | Iterator[pd.DataFrame]: Loaded data or chunk iterator.
    """
    if ref.is_excel:
        engine = _excel_engine(ref.extension)
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

    # CSV/TSV — detect leading comment rows
    skip, header_line = _scan_csv_preamble(ref)

    # Build column names from the cleaned header line so pandas gets correct names
    header_cols = header_line.split(ref.separator)

    kwargs: dict = {
        "sep": ref.separator,
        "names": header_cols,
        "skiprows": skip + 1,  # skip comment rows + the original header row
        "usecols": usecols,
        "nrows": nrows,
        "low_memory": False,
    }
    if chunksize is not None:
        kwargs["chunksize"] = chunksize

    if ref.zip_path is not None and ref.inner_path is not None:
        zf = zipfile.ZipFile(ref.zip_path)
        raw_file = zf.open(ref.inner_path)
        reader = pd.read_csv(raw_file, **kwargs)
        if chunksize is not None:
            return ManagedChunkReader(reader, raw_file=raw_file, zip_file=zf)
        raw_file.close()
        zf.close()
        return reader

    return pd.read_csv(ref.file_path, **kwargs)
