"""Command-line interface for science-bot."""

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from science_bot.benchmark import (
    format_benchmark_output,
    initialize_fixed_trace_writer,
    run_benchmark,
)
from science_bot.providers.executor import ensure_python_executor_ready
from science_bot.tracing import TraceWriter


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser.

    Returns:
        argparse.ArgumentParser: Configured parser for supported arguments.
    """
    parser = argparse.ArgumentParser(prog="science-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--directory", required=True)
    benchmark_parser.add_argument("--csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the top-level science-bot CLI.

    Args:
        argv: Optional command-line arguments.

    Returns:
        int: Process exit code for the CLI.
    """
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)
    trace_writer: TraceWriter | None = None

    try:
        if args.command != "benchmark":
            parser.error(f"Unsupported command: {args.command}")
            return 1

        ensure_python_executor_ready()
        benchmark_directory = Path(args.directory).expanduser().resolve()
        csv_path = Path(args.csv).expanduser().resolve()
        trace_writer = initialize_fixed_trace_writer()

        print(f"Preparing benchmark data from: {benchmark_directory}")
        summary = asyncio.run(
            run_benchmark(
                csv_path=csv_path,
                benchmark_directory=benchmark_directory,
                trace_writer=trace_writer,
            )
        )
        print(format_benchmark_output(summary))
        return 0
    except Exception as exc:
        if trace_writer is None:
            try:
                trace_writer = initialize_fixed_trace_writer()
            except Exception:
                trace_writer = None
        if (
            trace_writer is not None
            and hasattr(trace_writer, "write_event")
            and hasattr(trace_writer, "write_error")
        ):
            trace_writer.write_event(
                event="run_failed",
                stage="cli",
                payload={"error": str(exc)},
            )
            trace_writer.write_error(exc)
        print(f"Error: {exc}")
        return 1
