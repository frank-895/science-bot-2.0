"""Benchmark runtime helpers for CSV loading, execution, scoring, and tracing."""

import asyncio
import csv
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from science_bot.agent.orchestrator import OrchestratorRequest, run_orchestrator
from science_bot.tracing import (
    BenchmarkRowTraceSummary,
    BenchmarkTraceManifest,
    BenchmarkTraceSummary,
    TraceWriter,
)

REQUIRED_BENCHMARK_COLUMNS = frozenset(
    {
        "question",
        "data_folder",
        "capsule_uuid",
        "question_id",
        "ideal",
        "eval_mode",
    }
)
OPTIONAL_BENCHMARK_COLUMNS = frozenset({"id"})
BENCHMARK_COLUMNS = REQUIRED_BENCHMARK_COLUMNS | OPTIONAL_BENCHMARK_COLUMNS
BENCHMARK_CONCURRENCY = 20
DEFAULT_TRACE_ROOT = Path(".science-bot/traces")
NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+")
RANGE_PATTERN = re.compile(
    r"^\s*[\(\[]\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*[\)\]]\s*$"
)


class BenchmarkRow(BaseModel):
    """Raw benchmark input row."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    row_id: str | None = Field(default=None, alias="id")
    question: str
    data_folder: str
    capsule_uuid: str
    question_id: str
    ideal: str
    eval_mode: Literal["str_verifier", "range_verifier", "llm_verifier"]

    @field_validator(
        "question",
        "data_folder",
        "capsule_uuid",
        "question_id",
        "ideal",
    )
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        """Validate that required text fields are non-empty.

        Args:
            value: Candidate field value.

        Returns:
            str: Stripped field value.

        Raises:
            ValueError: If the value is empty after stripping.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("Benchmark fields must be non-empty.")
        return stripped

    @field_validator("row_id")
    @classmethod
    def validate_row_id(cls, value: str | None) -> str | None:
        """Normalize an optional benchmark row identifier.

        Args:
            value: Candidate row identifier.

        Returns:
            str | None: Stripped identifier or `None` when absent.
        """
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def populate_row_id(self) -> "BenchmarkRow":
        """Fill the row identifier from the capsule UUID when omitted.

        Returns:
            BenchmarkRow: Row with a resolved identifier.
        """
        if self.row_id is None:
            self.row_id = self.capsule_uuid
        return self


class BenchmarkRowResult(BaseModel):
    """Result for one benchmark row execution."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    capsule_path: Path | None
    eval_mode: Literal["str_verifier", "range_verifier", "llm_verifier"]
    ideal: str
    response: str | None
    classification_family: str | None = None
    selected_files: list[str] = Field(default_factory=list)
    resolution_iterations_used: int | None = None
    is_correct: bool
    status: Literal["completed", "failed"]
    error: str | None = None


class BenchmarkSummary(BaseModel):
    """Aggregate summary for a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    total_rows: int
    completed_rows: int
    failed_rows: int
    correct_rows: int
    incorrect_rows: int
    accuracy: float
    elapsed_seconds: float
    rows: list[BenchmarkRowResult] = Field(default_factory=list)


def initialize_fixed_trace_writer(trace_root: Path | None = None) -> TraceWriter:
    """Clean and initialize the fixed benchmark trace directory.

    Args:
        trace_root: Optional trace root override.

    Returns:
        TraceWriter: Trace writer rooted at the cleaned trace directory.
    """
    resolved_trace_root = (
        DEFAULT_TRACE_ROOT.expanduser().resolve()
        if trace_root is None
        else trace_root.expanduser().resolve()
    )
    if resolved_trace_root.exists():
        shutil.rmtree(resolved_trace_root)
    resolved_trace_root.mkdir(parents=True, exist_ok=True)
    return TraceWriter(resolved_trace_root)


def is_ignored_zip_member(member_name: str) -> bool:
    """Determine whether a zip entry should be skipped during extraction.

    Args:
        member_name: Raw archive member name.

    Returns:
        bool: Whether the member should be ignored.
    """
    parts = [part for part in Path(member_name).parts if part not in {".", ""}]
    return any(part == "__MACOSX" or part.startswith("._") for part in parts)


def load_benchmark_rows(csv_path: Path) -> list[BenchmarkRow]:
    """Load benchmark rows from disk.

    Args:
        csv_path: Path to the benchmark CSV file.

    Returns:
        list[BenchmarkRow]: Parsed benchmark rows.

    Raises:
        FileNotFoundError: If the benchmark CSV does not exist.
        ValueError: If the CSV schema or content is invalid.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = REQUIRED_BENCHMARK_COLUMNS - fieldnames
        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(f"Benchmark CSV missing required columns: {missing_list}")

        rows: list[BenchmarkRow] = []
        for index, raw_row in enumerate(reader, start=2):
            try:
                filtered_row = {
                    key: value
                    for key, value in raw_row.items()
                    if key in BENCHMARK_COLUMNS
                }
                rows.append(BenchmarkRow.model_validate(filtered_row))
            except ValidationError as exc:
                raise ValueError(f"Invalid benchmark row {index}: {exc}") from exc

    return rows


def resolve_benchmark_capsule_path(
    row: BenchmarkRow,
    extracted_capsules_root: Path,
) -> Path:
    """Resolve the data folder for a benchmark row.

    Args:
        row: Benchmark row metadata.
        extracted_capsules_root: Root directory for extracted capsules.

    Returns:
        Path: Resolved capsule data path.

    Raises:
        FileNotFoundError: If the capsule directory cannot be resolved.
        ValueError: If the row's folder format is invalid or ambiguous.
    """
    folder_name = Path(row.data_folder).name
    if not folder_name.startswith("CapsuleFolder-") or not folder_name.endswith(".zip"):
        raise ValueError(f"Unsupported data_folder value: {row.data_folder}")

    row_id = row.row_id if row.row_id is not None else row.capsule_uuid
    benchmark_root = extracted_capsules_root / row_id
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"Capsule directory not found: {benchmark_root}")

    expected_path = benchmark_root / f"CapsuleData-{row.capsule_uuid}"
    if expected_path.is_dir():
        return expected_path

    matches = sorted(
        path for path in benchmark_root.glob("CapsuleData-*") if path.is_dir()
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No CapsuleData directory found under {benchmark_root}"
        )
    raise ValueError(f"Multiple CapsuleData directories found under {benchmark_root}")


def is_extracted_capsule_tree(directory_path: Path) -> bool:
    """Detect whether a directory already contains extracted benchmark capsules.

    Args:
        directory_path: Candidate benchmark directory.

    Returns:
        bool: Whether the directory matches the extracted capsule tree shape.
    """
    if not directory_path.is_dir():
        return False

    for child in directory_path.iterdir():
        if not child.is_dir():
            continue
        if any(
            grandchild.is_dir() and grandchild.name.startswith("CapsuleData-")
            for grandchild in child.iterdir()
        ):
            return True
    return False


def extract_outer_archive(archive_path: Path) -> Path:
    """Extract the outer capsule archive beside the provided zip file.

    Args:
        archive_path: Zip archive containing capsule zip files.

    Returns:
        Path: Directory containing the extracted outer archive content.

    Raises:
        FileNotFoundError: If the archive does not exist.
        ValueError: If the path is not a zip archive.
        zipfile.BadZipFile: If the archive is invalid.
    """
    if not archive_path.is_file():
        raise FileNotFoundError(f"Benchmark directory not found: {archive_path}")
    if archive_path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a .zip file: {archive_path}")

    extracted_directory = archive_path.with_suffix("")
    if extracted_directory.is_dir():
        return extracted_directory

    extracted_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if is_ignored_zip_member(member.filename):
                continue
            archive.extract(member, archive_path.parent)
    return extracted_directory


def extract_inner_capsules(source_directory: Path) -> Path:
    """Extract inner capsule archives into the benchmark resolver layout.

    Args:
        source_directory: Directory containing `CapsuleFolder-*.zip` files.

    Returns:
        Path: Extracted capsule root compatible with benchmark resolution.

    Raises:
        FileNotFoundError: If the source directory does not exist.
    """
    if not source_directory.is_dir():
        raise FileNotFoundError(f"Benchmark directory not found: {source_directory}")

    extracted_root = source_directory.parent / "extracted_capsules"
    extracted_root.mkdir(parents=True, exist_ok=True)

    capsule_archives = sorted(source_directory.rglob("CapsuleFolder-*.zip"))
    if not capsule_archives:
        raise ValueError(
            f"No CapsuleFolder-*.zip archives were found under {source_directory}."
        )

    for capsule_zip in capsule_archives:
        capsule_uuid = capsule_zip.stem.removeprefix("CapsuleFolder-")
        target_directory = extracted_root / capsule_uuid
        existing_data_directories = [
            child for child in target_directory.glob("CapsuleData-*") if child.is_dir()
        ]
        if existing_data_directories:
            continue

        target_directory.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(capsule_zip) as archive:
            for member in archive.infolist():
                if is_ignored_zip_member(member.filename):
                    continue
                archive.extract(member, target_directory)

    return extracted_root


def prepare_benchmark_directory(directory_path: Path) -> Path:
    """Prepare a benchmark directory for row execution.

    Args:
        directory_path: User-provided benchmark data path.

    Returns:
        Path: Extracted capsule root in the benchmark resolver layout.

    Raises:
        FileNotFoundError: If the input path does not exist.
        ValueError: If the path does not match a supported benchmark shape.
    """
    if not directory_path.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {directory_path}")

    if directory_path.is_file():
        if directory_path.suffix.lower() != ".zip":
            raise ValueError(f"Unsupported benchmark directory file: {directory_path}")
        extracted_directory = extract_outer_archive(directory_path)
        if any(extracted_directory.rglob("CapsuleFolder-*.zip")):
            return extract_inner_capsules(extracted_directory)
        if is_extracted_capsule_tree(extracted_directory):
            return extracted_directory
        return extract_inner_capsules(extracted_directory)

    if any(directory_path.rglob("CapsuleFolder-*.zip")):
        return extract_inner_capsules(directory_path)

    if is_extracted_capsule_tree(directory_path):
        return directory_path

    raise ValueError(
        "Unsupported benchmark directory. Provide a zip archive, a directory of "
        "CapsuleFolder-*.zip files, or an extracted capsule tree."
    )


def normalize_text(value: str) -> str:
    """Normalize text for deterministic comparisons.

    Args:
        value: Raw text.

    Returns:
        str: Normalized text.
    """
    return " ".join(value.strip().split()).lower()


def score_benchmark_response(eval_mode: str, ideal: str, response: str) -> bool:
    """Score a benchmark response with a deterministic temporary rule.

    Args:
        eval_mode: Benchmark evaluation mode.
        ideal: Expected answer string.
        response: Orchestrator response text.

    Returns:
        bool: Whether the response is considered correct.

    Raises:
        ValueError: If the evaluation mode or expected range is invalid.
    """
    normalized_ideal = normalize_text(ideal)
    normalized_response = normalize_text(response)

    if eval_mode == "str_verifier":
        return normalized_response == normalized_ideal

    if eval_mode == "llm_verifier":
        return normalized_ideal in normalized_response

    if eval_mode == "range_verifier":
        match = RANGE_PATTERN.fullmatch(ideal)
        if match is None:
            raise ValueError(f"Invalid range_verifier ideal value: {ideal}")
        lower_bound = float(match.group(1))
        upper_bound = float(match.group(2))
        response_match = NUMERIC_PATTERN.search(response)
        if response_match is None:
            return False
        numeric_value = float(response_match.group(0))
        return lower_bound <= numeric_value <= upper_bound

    raise ValueError(f"Unsupported benchmark eval_mode: {eval_mode}")


async def run_benchmark(
    csv_path: Path,
    benchmark_directory: Path,
    trace_writer: TraceWriter,
) -> BenchmarkSummary:
    """Run the fixed benchmark suite.

    Args:
        csv_path: Benchmark CSV location.
        benchmark_directory: User-provided benchmark data location.
        trace_writer: Trace writer used for run and row artifacts.

    Returns:
        BenchmarkSummary: Aggregate benchmark outcomes.

    Raises:
        FileNotFoundError: If the benchmark inputs are missing.
        ValueError: If the benchmark inputs are malformed.
    """
    extracted_capsules_root = prepare_benchmark_directory(benchmark_directory)
    trace_writer.write_manifest(
        BenchmarkTraceManifest(
            command="benchmark",
            csv_path=str(csv_path),
            benchmark_directory=str(benchmark_directory),
            prepared_capsule_root=str(extracted_capsules_root),
            trace_root=str(trace_writer.root_dir),
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            concurrency=BENCHMARK_CONCURRENCY,
        )
    )
    trace_writer.write_event(
        event="benchmark_started",
        stage="cli",
        payload={
            "csv_path": csv_path,
            "benchmark_directory": benchmark_directory,
            "prepared_capsule_root": extracted_capsules_root,
        },
    )
    rows = load_benchmark_rows(csv_path)
    semaphore = asyncio.Semaphore(BENCHMARK_CONCURRENCY)
    start_time = time.perf_counter()

    async def run_row(row: BenchmarkRow) -> BenchmarkRowResult:
        """Execute one benchmark row under the shared concurrency limit.

        Args:
            row: Benchmark row to execute.

        Returns:
            BenchmarkRowResult: Completed row result.
        """
        async with semaphore:
            row_trace_writer = trace_writer.create_row_writer(row.question_id)
            row_trace_writer.write_event(
                event="benchmark_row_started",
                stage="cli",
                question_id=row.question_id,
                question=row.question,
                payload={
                    "row_id": row.row_id,
                    "capsule_uuid": row.capsule_uuid,
                    "data_folder": row.data_folder,
                },
            )
            try:
                capsule_path = resolve_benchmark_capsule_path(
                    row,
                    extracted_capsules_root,
                )
                orchestrator_result = await run_orchestrator(
                    OrchestratorRequest(
                        question=row.question,
                        capsule_path=capsule_path,
                    )
                )
                is_correct = score_benchmark_response(
                    row.eval_mode,
                    row.ideal,
                    orchestrator_result.answer,
                )
                result = BenchmarkRowResult(
                    question_id=row.question_id,
                    question=row.question,
                    capsule_path=capsule_path,
                    eval_mode=row.eval_mode,
                    ideal=row.ideal,
                    response=orchestrator_result.answer,
                    classification_family=_extract_metadata_string(
                        orchestrator_result.metadata,
                        "classification_family",
                    ),
                    selected_files=_extract_metadata_str_list(
                        orchestrator_result.metadata,
                        "resolution_selected_files",
                    ),
                    resolution_iterations_used=_extract_metadata_int(
                        orchestrator_result.metadata,
                        "resolution_iterations_used",
                    ),
                    is_correct=is_correct,
                    status="completed",
                )
                row_trace_writer.write_event(
                    event="benchmark_row_finished",
                    stage="cli",
                    question_id=row.question_id,
                    question=row.question,
                    family=result.classification_family,
                    payload={
                        "status": result.status,
                        "is_correct": result.is_correct,
                        "answer": result.response,
                    },
                )
                row_trace_writer.write_summary(
                    BenchmarkRowTraceSummary(
                        question_id=row.question_id,
                        status=result.status,
                        classification_family=result.classification_family,
                        resolution_iterations_used=result.resolution_iterations_used,
                        selected_files=result.selected_files,
                        answer=result.response,
                        is_correct=result.is_correct,
                        error=result.error,
                    )
                )
                return result
            except Exception as exc:
                result = BenchmarkRowResult(
                    question_id=row.question_id,
                    question=row.question,
                    capsule_path=None,
                    eval_mode=row.eval_mode,
                    ideal=row.ideal,
                    response=None,
                    is_correct=False,
                    status="failed",
                    error=str(exc),
                )
                row_trace_writer.write_event(
                    event="benchmark_row_finished",
                    stage="cli",
                    question_id=row.question_id,
                    question=row.question,
                    payload={
                        "status": result.status,
                        "error": result.error,
                    },
                )
                row_trace_writer.write_summary(
                    BenchmarkRowTraceSummary(
                        question_id=row.question_id,
                        status=result.status,
                        error=result.error,
                        is_correct=result.is_correct,
                    )
                )
                row_trace_writer.write_error(exc)
                return result

    row_results = await asyncio.gather(*(run_row(row) for row in rows))
    elapsed_seconds = time.perf_counter() - start_time
    completed_rows = sum(result.status == "completed" for result in row_results)
    failed_rows = len(row_results) - completed_rows
    correct_rows = sum(result.is_correct for result in row_results)
    incorrect_rows = len(row_results) - correct_rows
    accuracy = correct_rows / len(row_results) if row_results else 0.0
    summary = BenchmarkSummary(
        total_rows=len(row_results),
        completed_rows=completed_rows,
        failed_rows=failed_rows,
        correct_rows=correct_rows,
        incorrect_rows=incorrect_rows,
        accuracy=accuracy,
        elapsed_seconds=elapsed_seconds,
        rows=row_results,
    )
    trace_writer.write_event(
        event="run_finished",
        stage="cli",
        payload={
            "status": "completed",
            "total_rows": summary.total_rows,
            "completed_rows": summary.completed_rows,
            "failed_rows": summary.failed_rows,
            "accuracy": summary.accuracy,
        },
    )
    trace_writer.write_summary(
        BenchmarkTraceSummary(
            total_rows=summary.total_rows,
            completed_rows=summary.completed_rows,
            failed_rows=summary.failed_rows,
            correct_rows=summary.correct_rows,
            incorrect_rows=summary.incorrect_rows,
            accuracy=summary.accuracy,
            elapsed_seconds=summary.elapsed_seconds,
            rows=[
                {
                    "question_id": row.question_id,
                    "status": row.status,
                    "row_trace_dir": str(trace_writer.root_dir / row.question_id),
                }
                for row in summary.rows
            ],
        )
    )
    return summary


def format_benchmark_output(summary: BenchmarkSummary) -> str:
    """Format benchmark output for terminal display.

    Args:
        summary: Benchmark summary to render.

    Returns:
        str: Human-readable output.
    """
    lines = [
        "Benchmark Summary",
        f"Total rows: {summary.total_rows}",
        f"Completed rows: {summary.completed_rows}",
        f"Failed rows: {summary.failed_rows}",
        f"Correct rows: {summary.correct_rows}",
        f"Incorrect rows: {summary.incorrect_rows}",
        f"Accuracy: {summary.accuracy:.2%}",
        f"Elapsed seconds: {summary.elapsed_seconds:.3f}",
        "Rows:",
    ]
    for row in summary.rows:
        outcome = "correct" if row.is_correct else "incorrect"
        detail = row.error if row.error else (row.response or "")
        detail_preview = detail.replace("\n", " ")[:80]
        metadata_bits: list[str] = []
        if row.classification_family:
            metadata_bits.append(f"family={row.classification_family}")
        if row.resolution_iterations_used is not None:
            metadata_bits.append(f"iters={row.resolution_iterations_used}")
        if row.selected_files:
            metadata_bits.append(f"files={','.join(row.selected_files[:2])}")
        metadata_suffix = f" [{'; '.join(metadata_bits)}]" if metadata_bits else ""
        lines.append(
            f"- {row.question_id}: {row.status}, {outcome}{metadata_suffix}, "
            f"{detail_preview}"
        )
    return "\n".join(lines)


def _extract_metadata_string(
    metadata: dict[str, object],
    key: str,
) -> str | None:
    """Extract a string value from orchestrator metadata.

    Args:
        metadata: Orchestrator metadata mapping.
        key: Metadata key to extract.

    Returns:
        str | None: String value when present and valid.
    """
    value = metadata.get(key)
    return value if isinstance(value, str) else None


def _extract_metadata_int(
    metadata: dict[str, object],
    key: str,
) -> int | None:
    """Extract an integer value from orchestrator metadata.

    Args:
        metadata: Orchestrator metadata mapping.
        key: Metadata key to extract.

    Returns:
        int | None: Integer value when present and valid.
    """
    value = metadata.get(key)
    return value if isinstance(value, int) else None


def _extract_metadata_str_list(
    metadata: dict[str, object],
    key: str,
) -> list[str]:
    """Extract a list of strings from orchestrator metadata.

    Args:
        metadata: Orchestrator metadata mapping.
        key: Metadata key to extract.

    Returns:
        list[str]: Validated string list or an empty list.
    """
    value = metadata.get(key)
    if not isinstance(value, list):
        return []
    if not all(isinstance(item, str) for item in value):
        return []
    return cast(list[str], value)
