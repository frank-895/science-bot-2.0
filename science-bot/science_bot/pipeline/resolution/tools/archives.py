"""Archive inspection tools for resolution."""

import zipfile
from pathlib import Path

from science_bot.pipeline.resolution.tools.discovery import file_category_for_extension
from science_bot.pipeline.resolution.tools.reader import TABULAR_EXTENSIONS
from science_bot.pipeline.resolution.tools.schemas import ZipEntry, ZipManifest


def list_zip_contents(capsule_path: Path, zip_filename: str) -> ZipManifest:
    """List all entries inside a zip archive without extracting it.

    Args:
        capsule_path: Absolute path to the capsule directory.
        zip_filename: Name or relative path of the zip file within the capsule.

    Returns:
        ZipManifest: Archive contents with coarse type metadata.
    """
    zip_path = (capsule_path / zip_filename).resolve()
    entries: list[ZipEntry] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            extension = Path(info.filename).suffix.lower()
            category = file_category_for_extension(extension)
            if extension in {".tsv", ".tab"}:
                file_type = "tsv"
            elif extension == ".csv":
                file_type = "csv"
            else:
                file_type = category
            entries.append(
                ZipEntry(
                    inner_path=info.filename,
                    size_bytes=info.file_size,
                    file_type=file_type,
                    is_readable=extension in TABULAR_EXTENSIONS,
                )
            )
    return ZipManifest(zip_filename=zip_filename, entries=entries)
