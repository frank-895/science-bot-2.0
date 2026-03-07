from pathlib import Path

import pytest
from science_bot import cli
from science_bot.benchmark import BenchmarkSummary


def test_build_parser_accepts_benchmark_without_trace_dir() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        ["benchmark", "--directory", "/tmp/data", "--csv", "/tmp/bench.csv"]
    )

    assert args.command == "benchmark"
    assert args.directory == "/tmp/data"
    assert args.csv == "/tmp/bench.csv"
    assert not hasattr(args, "trace_dir")


def test_main_benchmark_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = BenchmarkSummary(
        total_rows=2,
        completed_rows=2,
        failed_rows=0,
        correct_rows=1,
        incorrect_rows=1,
        accuracy=0.5,
        elapsed_seconds=0.123,
        rows=[],
    )

    async def fake_run_benchmark(
        *,
        csv_path: Path,
        benchmark_directory: Path,
        trace_writer: object,
    ) -> BenchmarkSummary:
        del trace_writer
        assert csv_path == Path("/tmp/benchmark.csv").resolve()
        assert benchmark_directory == Path("/tmp/data").resolve()
        return summary

    monkeypatch.setattr(cli, "ensure_python_executor_ready", lambda: None)
    monkeypatch.setattr(cli, "initialize_fixed_trace_writer", lambda: object())
    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    exit_code = cli.main(
        ["benchmark", "--directory", "/tmp/data", "--csv", "/tmp/benchmark.csv"]
    )

    output = capsys.readouterr().out
    resolved_directory = Path("/tmp/data").expanduser().resolve()
    assert exit_code == 0
    assert f"Preparing benchmark data from: {resolved_directory}" in output
    assert "Benchmark Summary" in output
    assert "Total rows: 2" in output
    assert "Accuracy: 50.00%" in output


def test_main_returns_error_for_preflight_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "ensure_python_executor_ready",
        lambda: (_ for _ in ()).throw(RuntimeError("executor not ready")),
    )
    monkeypatch.setattr(cli, "initialize_fixed_trace_writer", lambda: object())

    exit_code = cli.main(
        ["benchmark", "--directory", "/tmp/data", "--csv", "/tmp/benchmark.csv"]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Error: executor not ready" in output


def test_main_writes_error_trace_when_writer_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeTraceWriter:
        def write_event(self, *, event: str, stage: str, payload: object) -> None:
            del stage
            del payload
            events.append(event)

        def write_error(self, exc: Exception) -> None:
            del exc
            events.append("error_written")

    async def fake_run_benchmark(
        *,
        csv_path: Path,
        benchmark_directory: Path,
        trace_writer: object,
    ) -> BenchmarkSummary:
        del csv_path
        del benchmark_directory
        del trace_writer
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "ensure_python_executor_ready", lambda: None)
    monkeypatch.setattr(cli, "initialize_fixed_trace_writer", lambda: FakeTraceWriter())
    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    exit_code = cli.main(
        ["benchmark", "--directory", "/tmp/data", "--csv", "/tmp/benchmark.csv"]
    )

    assert exit_code == 1
    assert events == ["run_failed", "error_written"]


def test_main_recovers_when_trace_writer_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "ensure_python_executor_ready", lambda: None)

    calls = {"count": 0}

    def fail_trace_init() -> object:
        calls["count"] += 1
        raise RuntimeError("trace init failed")

    async def fake_run_benchmark(
        *,
        csv_path: Path,
        benchmark_directory: Path,
        trace_writer: object,
    ) -> BenchmarkSummary:
        del csv_path
        del benchmark_directory
        del trace_writer
        return BenchmarkSummary(
            total_rows=0,
            completed_rows=0,
            failed_rows=0,
            correct_rows=0,
            incorrect_rows=0,
            accuracy=0.0,
            elapsed_seconds=0.0,
            rows=[],
        )

    monkeypatch.setattr(cli, "initialize_fixed_trace_writer", fail_trace_init)
    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    exit_code = cli.main(
        ["benchmark", "--directory", "/tmp/data", "--csv", "/tmp/benchmark.csv"]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Error: trace init failed" in output
    assert calls["count"] == 2
