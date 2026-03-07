import asyncio
import zipfile
from pathlib import Path

import pytest
from science_bot import benchmark
from science_bot.agent.orchestrator import OrchestratorResult
from science_bot.benchmark import (
    BenchmarkRow,
    BenchmarkSummary,
    extract_inner_capsules,
    extract_outer_archive,
    initialize_fixed_trace_writer,
    is_extracted_capsule_tree,
    load_benchmark_rows,
    prepare_benchmark_directory,
    resolve_benchmark_capsule_path,
    run_benchmark,
    score_benchmark_response,
)


def write_benchmark_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(
            [
                "question,data_folder,capsule_uuid,question_id,ideal,eval_mode",
                *[
                    ",".join(
                        [
                            row["question"],
                            row["data_folder"],
                            row["capsule_uuid"],
                            row["question_id"],
                            row["ideal"],
                            row["eval_mode"],
                        ]
                    )
                    for row in rows
                ],
            ]
        ),
        encoding="utf-8",
    )


def test_initialize_fixed_trace_writer_cleans_existing_contents(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir(parents=True)
    stale = trace_root / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    trace_writer = initialize_fixed_trace_writer(trace_root)

    assert trace_writer.root_dir == trace_root
    assert trace_root.is_dir()
    assert not stale.exists()


def test_load_benchmark_rows_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "benchmark.csv"
    csv_path.write_text("question,data_folder\nq1,folder\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_benchmark_rows(csv_path)


def test_load_benchmark_rows_defaults_id_to_capsule_uuid(tmp_path: Path) -> None:
    csv_path = tmp_path / "benchmark.csv"
    write_benchmark_csv(
        csv_path,
        [
            {
                "question": "What?",
                "data_folder": "CapsuleFolder-inner-uuid.zip",
                "capsule_uuid": "cap-uuid",
                "question_id": "q1",
                "ideal": "answer",
                "eval_mode": "str_verifier",
            }
        ],
    )

    rows = load_benchmark_rows(csv_path)

    assert len(rows) == 1
    assert rows[0].row_id == "cap-uuid"


def test_resolve_benchmark_capsule_path_prefers_expected_match(tmp_path: Path) -> None:
    root = tmp_path / "extracted"
    expected = root / "cap-uuid" / "CapsuleData-inner-uuid"
    expected.mkdir(parents=True)

    row = BenchmarkRow(
        question="What?",
        data_folder="CapsuleFolder-inner-uuid.zip",
        capsule_uuid="cap-uuid",
        question_id="q1",
        ideal="answer",
        eval_mode="str_verifier",
    )

    resolved = resolve_benchmark_capsule_path(row, root)

    assert resolved == expected


def test_resolve_benchmark_capsule_path_uses_single_fallback(tmp_path: Path) -> None:
    root = tmp_path / "extracted"
    fallback = root / "cap-uuid" / "CapsuleData-other"
    fallback.mkdir(parents=True)

    row = BenchmarkRow(
        question="What?",
        data_folder="CapsuleFolder-inner-uuid.zip",
        capsule_uuid="cap-uuid",
        question_id="q1",
        ideal="answer",
        eval_mode="str_verifier",
    )

    resolved = resolve_benchmark_capsule_path(row, root)

    assert resolved == fallback


def test_resolve_benchmark_capsule_path_rejects_ambiguous_fallback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "extracted"
    first = root / "cap-uuid" / "CapsuleData-a"
    second = root / "cap-uuid" / "CapsuleData-b"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    row = BenchmarkRow(
        question="What?",
        data_folder="CapsuleFolder-inner-uuid.zip",
        capsule_uuid="cap-uuid",
        question_id="q1",
        ideal="answer",
        eval_mode="str_verifier",
    )

    with pytest.raises(ValueError, match="Multiple CapsuleData"):
        resolve_benchmark_capsule_path(row, root)


def test_is_extracted_capsule_tree_detects_expected_layout(tmp_path: Path) -> None:
    extracted_root = tmp_path / "extracted"
    (extracted_root / "cap-uuid" / "CapsuleData-inner").mkdir(parents=True)

    assert is_extracted_capsule_tree(extracted_root)


def test_extract_outer_archive_uses_adjacent_directory(tmp_path: Path) -> None:
    archive_path = tmp_path / "capsule_folders.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("__MACOSX/._ignored", "ignored")
        archive.writestr("capsule_folders/CapsuleFolder-cap-1.zip", "placeholder")

    extracted_directory = extract_outer_archive(archive_path)

    assert extracted_directory == tmp_path / "capsule_folders"
    assert extracted_directory.is_dir()
    assert (extracted_directory / "CapsuleFolder-cap-1.zip").is_file()
    assert not (tmp_path / "__MACOSX").exists()


def test_extract_inner_capsules_creates_extracted_capsule_tree(tmp_path: Path) -> None:
    source_directory = tmp_path / "capsule_folders"
    source_directory.mkdir()
    capsule_zip = source_directory / "CapsuleFolder-cap-1.zip"
    with zipfile.ZipFile(capsule_zip, "w") as archive:
        archive.writestr("__MACOSX/._ignored", "ignored")
        archive.writestr("CapsuleData-inner/data.txt", "payload")

    extracted_root = extract_inner_capsules(source_directory)

    assert extracted_root == tmp_path / "extracted_capsules"
    assert (extracted_root / "cap-1" / "CapsuleData-inner" / "data.txt").is_file()
    assert not (extracted_root / "cap-1" / "__MACOSX").exists()


def test_extract_inner_capsules_finds_archives_recursively(tmp_path: Path) -> None:
    source_directory = tmp_path / "capsule_folders"
    nested_directory = source_directory / "nested"
    nested_directory.mkdir(parents=True)
    capsule_zip = nested_directory / "CapsuleFolder-cap-1.zip"
    with zipfile.ZipFile(capsule_zip, "w") as archive:
        archive.writestr("CapsuleData-inner/data.txt", "payload")

    extracted_root = extract_inner_capsules(source_directory)

    assert extracted_root == tmp_path / "extracted_capsules"
    assert (extracted_root / "cap-1" / "CapsuleData-inner" / "data.txt").is_file()


def test_extract_inner_capsules_reuses_existing_data(tmp_path: Path) -> None:
    source_directory = tmp_path / "capsule_folders"
    source_directory.mkdir()
    capsule_zip = source_directory / "CapsuleFolder-cap-1.zip"
    with zipfile.ZipFile(capsule_zip, "w") as archive:
        archive.writestr("CapsuleData-inner/data.txt", "new-payload")

    existing_file = (
        tmp_path / "extracted_capsules" / "cap-1" / "CapsuleData-existing" / "data.txt"
    )
    existing_file.parent.mkdir(parents=True)
    existing_file.write_text("existing", encoding="utf-8")

    extracted_root = extract_inner_capsules(source_directory)

    assert extracted_root == tmp_path / "extracted_capsules"
    assert existing_file.read_text(encoding="utf-8") == "existing"


def test_prepare_benchmark_directory_accepts_extracted_tree(tmp_path: Path) -> None:
    extracted_root = tmp_path / "extracted_capsules"
    (extracted_root / "cap-1" / "CapsuleData-inner").mkdir(parents=True)

    prepared_root = prepare_benchmark_directory(extracted_root)

    assert prepared_root == extracted_root


def test_prepare_benchmark_directory_extracts_zip_input(tmp_path: Path) -> None:
    outer_zip = tmp_path / "capsule_folders.zip"
    inner_zip_bytes = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner_zip_bytes, "w") as inner_archive:
        inner_archive.writestr("CapsuleData-inner/data.txt", "payload")

    with zipfile.ZipFile(outer_zip, "w") as outer_archive:
        outer_archive.writestr(
            "capsule_folders/CapsuleFolder-cap-1.zip",
            inner_zip_bytes.read_bytes(),
        )

    prepared_root = prepare_benchmark_directory(outer_zip)

    assert prepared_root == tmp_path / "extracted_capsules"
    assert (prepared_root / "cap-1" / "CapsuleData-inner" / "data.txt").is_file()


def test_prepare_benchmark_directory_extracts_already_unzipped_outer_directory(
    tmp_path: Path,
) -> None:
    outer_directory = tmp_path / "capsule_folders"
    nested_directory = outer_directory / "capsules"
    nested_directory.mkdir(parents=True)
    capsule_zip = nested_directory / "CapsuleFolder-cap-1.zip"
    with zipfile.ZipFile(capsule_zip, "w") as archive:
        archive.writestr("CapsuleData-inner/data.txt", "payload")

    prepared_root = prepare_benchmark_directory(outer_directory)

    assert prepared_root == tmp_path / "extracted_capsules"
    assert (prepared_root / "cap-1" / "CapsuleData-inner" / "data.txt").is_file()


def test_score_benchmark_response_for_supported_modes() -> None:
    assert score_benchmark_response("str_verifier", "  Hello  ", "hello")
    assert score_benchmark_response("range_verifier", "(1.5,1.7)", "value 1.6")
    assert not score_benchmark_response("range_verifier", "(1.5,1.7)", "value 2.1")
    assert score_benchmark_response("llm_verifier", "35%", "The result is 35%")


def test_run_benchmark_continues_after_row_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "benchmark.csv"
    extracted_root = tmp_path / "extracted"
    success_path = extracted_root / "cap-1" / "CapsuleData-inner-1"
    success_path.mkdir(parents=True)
    (extracted_root / "cap-2").mkdir(parents=True)
    write_benchmark_csv(
        csv_path,
        [
            {
                "question": "What is one?",
                "data_folder": "CapsuleFolder-inner-1.zip",
                "capsule_uuid": "cap-1",
                "question_id": "q1",
                "ideal": "ORCHESTRATOR_STUB_RESPONSE",
                "eval_mode": "str_verifier",
            },
            {
                "question": "What is two?",
                "data_folder": "CapsuleFolder-inner-2.zip",
                "capsule_uuid": "cap-2",
                "question_id": "q2",
                "ideal": "ORCHESTRATOR_STUB_RESPONSE",
                "eval_mode": "str_verifier",
            },
        ],
    )

    async def fake_run_orchestrator(request: object) -> OrchestratorResult:
        del request
        return OrchestratorResult(
            question="What is one?",
            capsule_path=success_path,
            status="completed",
            answer="ORCHESTRATOR_STUB_RESPONSE",
            metadata={
                "classification_family": "aggregate",
                "resolution_iterations_used": 2,
                "resolution_selected_files": ["clinical.csv"],
            },
            error=None,
        )

    monkeypatch.setattr(benchmark, "run_orchestrator", fake_run_orchestrator)
    trace_writer = initialize_fixed_trace_writer(tmp_path / "traces")

    summary = asyncio.run(
        run_benchmark(
            csv_path=csv_path,
            benchmark_directory=extracted_root,
            trace_writer=trace_writer,
        )
    )

    assert isinstance(summary, BenchmarkSummary)
    assert summary.total_rows == 2
    assert summary.completed_rows == 1
    assert summary.failed_rows == 1
    assert summary.correct_rows == 1
    assert summary.incorrect_rows == 1
    assert any(
        row.question_id == "q2" and row.status == "failed" for row in summary.rows
    )


def test_run_benchmark_writes_trace_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "benchmark.csv"
    extracted_root = tmp_path / "extracted"
    capsule_path = extracted_root / "row-1" / "CapsuleData-cap-1"
    capsule_path.mkdir(parents=True)
    write_benchmark_csv(
        csv_path,
        [
            {
                "question": "What is one?",
                "data_folder": "CapsuleFolder-cap-1.zip",
                "capsule_uuid": "cap-1",
                "question_id": "q1",
                "ideal": "answer",
                "eval_mode": "str_verifier",
            }
        ],
    )

    monkeypatch.setattr(
        benchmark,
        "prepare_benchmark_directory",
        lambda _: extracted_root,
    )

    async def fake_run_orchestrator(request: object) -> OrchestratorResult:
        del request
        return OrchestratorResult(
            question="What is one?",
            capsule_path=capsule_path,
            status="completed",
            answer="answer",
            metadata={
                "classification_family": "aggregate",
                "resolution_iterations_used": 1,
                "resolution_selected_files": ["clinical.csv"],
            },
            error=None,
        )

    monkeypatch.setattr(benchmark, "run_orchestrator", fake_run_orchestrator)
    trace_writer = initialize_fixed_trace_writer(tmp_path / "traces")

    summary = asyncio.run(
        run_benchmark(
            csv_path=csv_path,
            benchmark_directory=tmp_path / "capsules",
            trace_writer=trace_writer,
        )
    )

    assert summary.total_rows == 1
    assert (trace_writer.root_dir / "manifest.json").is_file()
    assert (trace_writer.root_dir / "summary.json").is_file()
    assert (trace_writer.root_dir / "q1" / "events.jsonl").is_file()
    assert (trace_writer.root_dir / "q1" / "summary.json").is_file()
