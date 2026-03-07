import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from science_bot.agent.contracts import AgentIterationResponse, AgentStepRecord
from science_bot.agent.runtime import run_agent
from science_bot.agent.summary import summarize_steps
from science_bot.providers.executor import PythonExecutionResult
from science_bot.providers.llm import LLMResponseFormatError
from science_bot.tracing import TraceWriter


def test_iteration_response_requires_exactly_one_choice() -> None:
    with pytest.raises(ValidationError):
        AgentIterationResponse.model_validate(
            {"python": "print(1)", "final_answer": "42"}
        )

    with pytest.raises(ValidationError):
        AgentIterationResponse.model_validate({"python": None, "final_answer": None})


def test_iteration_response_ignores_extra_fields() -> None:
    response = AgentIterationResponse.model_validate(
        {"python": "print(1)", "final_answer": None, "extra": "ignored"}
    )

    assert response.python_code == "print(1)"


def test_run_agent_executes_then_returns_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        AgentIterationResponse.model_validate({"python": "print('step1')"}),
        AgentIterationResponse.model_validate({"final_answer": "42"}),
    ]

    run_calls = {"count": 0}

    async def fake_parse_structured(**_: object) -> AgentIterationResponse:
        return responses.pop(0)

    async def fake_run_python(
        script: str,
        *,
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> PythonExecutionResult:
        del timeout_seconds
        run_calls["count"] += 1
        assert script.startswith("print")
        assert run_id == "q1-iter-1"
        return PythonExecutionResult(
            status="completed",
            answer="runtime-value",
            error_type=None,
            error_message=None,
            stdout_tail="ok stdout",
            stderr_tail="warn stderr",
            duration_ms=5,
            worker="runner-1",
        )

    monkeypatch.setattr(
        "science_bot.agent.runtime.list_available_python_packages",
        lambda: ["numpy"],
    )
    monkeypatch.setattr(
        "science_bot.agent.runtime.parse_structured",
        fake_parse_structured,
    )
    monkeypatch.setattr("science_bot.agent.runtime.run_python", fake_run_python)

    result = asyncio.run(
        run_agent(
            question="How many rows?",
            capsule_path=Path("/capsules/row1"),
            capsule_manifest="/capsules/row1/file.csv",
            execution_id="q1",
            max_iterations=6,
        )
    )

    assert result.status == "completed"
    assert result.answer == "42"
    assert result.iterations_used == 2
    assert len(result.steps) == 2
    assert run_calls["count"] == 1
    assert result.steps[0].execution_status == "completed"
    assert result.steps[0].execution_answer == "runtime-value"
    assert result.steps[1].execution_status is None
    assert result.steps[1].proposed_final_answer == "42"


def test_run_agent_fails_after_budget_without_final_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_parse_structured(**_: object) -> AgentIterationResponse:
        return AgentIterationResponse.model_validate({"python": "print(1)"})

    async def fake_run_python(
        script: str,
        *,
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> PythonExecutionResult:
        del script
        del timeout_seconds
        del run_id
        return PythonExecutionResult(
            status="failed",
            answer=None,
            error_type="runtime_error",
            error_message="boom",
            stdout_tail="",
            stderr_tail="trace",
            duration_ms=10,
            worker="runner-1",
        )

    monkeypatch.setattr(
        "science_bot.agent.runtime.list_available_python_packages",
        lambda: ["numpy"],
    )
    monkeypatch.setattr(
        "science_bot.agent.runtime.parse_structured",
        fake_parse_structured,
    )
    monkeypatch.setattr("science_bot.agent.runtime.run_python", fake_run_python)

    result = asyncio.run(
        run_agent(
            question="Question?",
            capsule_path=tmp_path,
            capsule_manifest="/capsules/row1/file.csv",
            max_iterations=2,
        )
    )

    assert result.status == "failed"
    assert result.failure_reason == "max_iterations_no_answer"
    assert result.failure_detail is not None
    assert "last_had_python=True" in result.failure_detail
    assert "last_execution_status=failed" in result.failure_detail


def test_run_agent_repairs_invalid_output_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = {"count": 0}

    async def fake_parse_structured(**_: object) -> AgentIterationResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise LLMResponseFormatError("invalid json")
        return AgentIterationResponse.model_validate({"final_answer": "42"})

    async def fake_run_python(
        script: str,
        *,
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> PythonExecutionResult:
        raise AssertionError(
            "run_python should not be called when final_answer is provided "
            f"({script}, {timeout_seconds}, {run_id})"
        )

    monkeypatch.setattr(
        "science_bot.agent.runtime.list_available_python_packages",
        lambda: ["numpy"],
    )
    monkeypatch.setattr(
        "science_bot.agent.runtime.parse_structured",
        fake_parse_structured,
    )
    monkeypatch.setattr("science_bot.agent.runtime.run_python", fake_run_python)

    result = asyncio.run(
        run_agent(
            question="Question?",
            capsule_path=tmp_path,
            capsule_manifest="/capsules/row1/file.csv",
            max_iterations=1,
        )
    )

    assert calls["count"] == 2
    assert result.status == "completed"
    assert result.answer == "42"


def test_run_agent_writes_iteration_trace_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    responses = [AgentIterationResponse.model_validate({"final_answer": "done"})]

    async def fake_parse_structured(**_: object) -> AgentIterationResponse:
        return responses.pop(0)

    async def fake_run_python(
        script: str,
        *,
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> PythonExecutionResult:
        raise AssertionError(
            "run_python should not be called when final_answer is provided "
            f"({script}, {timeout_seconds}, {run_id})"
        )

    monkeypatch.setattr(
        "science_bot.agent.runtime.list_available_python_packages",
        lambda: ["numpy"],
    )
    monkeypatch.setattr(
        "science_bot.agent.runtime.parse_structured",
        fake_parse_structured,
    )
    monkeypatch.setattr("science_bot.agent.runtime.run_python", fake_run_python)
    trace_writer = TraceWriter(tmp_path / "agent-traces")

    result = asyncio.run(
        run_agent(
            question="Question?",
            capsule_path=tmp_path,
            capsule_manifest="/capsules/row1/file.csv",
            execution_id="trace-q",
            trace_writer=trace_writer,
            max_iterations=2,
        )
    )

    assert result.status == "completed"
    events = (trace_writer.root_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "agent_iteration_started" in events
    assert "agent_decision" in events
    assert "agent_execution_result" not in events
    assert "agent_terminated" in events


def test_summarize_steps_includes_execution_output_fields() -> None:
    summary = summarize_steps(
        [
            AgentStepRecord(
                iteration=1,
                script="print(1)",
                execution_status="completed",
                execution_answer="answer-value",
                execution_stdout_tail="stdout-value",
                execution_stderr_tail="stderr-value",
                execution_duration_ms=15,
                execution_worker="runner-2",
                proposed_final_answer="candidate",
            )
        ]
    )

    assert "iter=1" in summary
    assert "exec_status=completed" in summary
    assert "exec_answer=answer-value" in summary
    assert "exec_stdout=stdout-value" in summary
    assert "exec_stderr=stderr-value" in summary
    assert "exec_ms=15" in summary
    assert "exec_worker=runner-2" in summary
    assert "proposed_final=candidate" in summary
