"""Real-mode runtime loop for iterative question answering."""

import re
from pathlib import Path

from pydantic import ValidationError

from science_bot.agent.contracts import (
    AgentIterationResponse,
    AgentRunRequest,
    AgentRunResult,
    AgentStepRecord,
)
from science_bot.agent.prompts import (
    build_repair_prompt,
    build_system_prompt,
    build_user_prompt,
)
from science_bot.agent.summary import summarize_steps
from science_bot.providers.executor import (
    list_available_python_packages,
    run_python,
)
from science_bot.providers.llm import LLMResponseFormatError, parse_structured
from science_bot.tracing import TraceWriter

DEFAULT_MAX_ITERATIONS = 6
DEFAULT_PYTHON_TIMEOUT_SECONDS = 30


async def run_agent(
    *,
    question: str,
    capsule_path: Path,
    capsule_manifest: str | None = None,
    execution_id: str | None = None,
    trace_writer: TraceWriter | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AgentRunResult:
    """Run the real-mode agent loop.

    Args:
        question: Natural language question to answer.
        capsule_path: Capsule path used by generated scripts.
        capsule_manifest: Optional precomputed recursive file listing.
        execution_id: Optional question-scoped execution identifier.
        trace_writer: Optional trace writer for iteration-level diagnostics.
        max_iterations: Maximum number of decision iterations.

    Returns:
        AgentRunResult: Terminal runtime result with all recorded steps.

    Raises:
        PythonExecutorUnavailableError: If package discovery or execution fails.
    """
    request = AgentRunRequest(
        question=question,
        capsule_path=capsule_path,
        capsule_manifest=capsule_manifest,
        max_iterations=max_iterations,
    )
    available_packages = _safe_list_packages()
    prompt_capsule_path = request.capsule_path
    resolved_manifest = request.capsule_manifest or "(manifest unavailable)"
    steps: list[AgentStepRecord] = []
    system_prompt = build_system_prompt(request.max_iterations)

    for iteration in range(1, request.max_iterations + 1):
        remaining = request.max_iterations - iteration + 1
        _write_trace_event(
            trace_writer=trace_writer,
            event="agent_iteration_started",
            payload={
                "iteration": iteration,
                "remaining_budget": remaining,
                "execution_id": execution_id,
            },
        )
        user_prompt = build_user_prompt(
            question=request.question,
            capsule_path=prompt_capsule_path,
            capsule_manifest=resolved_manifest,
            available_packages=available_packages,
            step_summary=summarize_steps(steps),
            iteration=iteration,
            max_iterations=request.max_iterations,
        )
        try:
            decision = await parse_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=AgentIterationResponse,
                trace_writer=trace_writer,
                trace_stage="agent",
            )
        except (LLMResponseFormatError, ValidationError) as first_error:
            _write_trace_event(
                trace_writer=trace_writer,
                event="agent_repair_attempted",
                payload={
                    "iteration": iteration,
                    "error": str(first_error),
                },
            )
            repair_prompt = build_repair_prompt(previous_error=str(first_error))
            try:
                decision = await parse_structured(
                    system_prompt=system_prompt,
                    user_prompt=repair_prompt,
                    response_model=AgentIterationResponse,
                    trace_writer=trace_writer,
                    trace_stage="agent",
                )
            except (LLMResponseFormatError, ValidationError) as second_error:
                _write_trace_event(
                    trace_writer=trace_writer,
                    event="agent_repair_failed",
                    payload={
                        "iteration": iteration,
                        "first_error": str(first_error),
                        "second_error": str(second_error),
                    },
                )
                failure_detail = (
                    "Decision parsing failed after one repair attempt. "
                    f"first_error={first_error}; second_error={second_error}"
                )
                _write_trace_event(
                    trace_writer=trace_writer,
                    event="agent_terminated",
                    payload={
                        "iteration": iteration,
                        "reason": "invalid_decision_output",
                        "detail": failure_detail,
                    },
                )
                return AgentRunResult(
                    status="failed",
                    iterations_used=iteration,
                    steps=steps,
                    failure_reason="invalid_decision_output",
                    failure_detail=failure_detail,
                )

        _write_trace_event(
            trace_writer=trace_writer,
            event="agent_decision",
            payload={
                "iteration": iteration,
                "has_python": bool(
                    decision.python_code and decision.python_code.strip()
                ),
                "has_final_answer": decision.final_answer is not None,
            },
        )
        if decision.final_answer is not None:
            final_answer = _stringify_final_answer(decision.final_answer)
            step = AgentStepRecord(
                iteration=iteration,
                script=decision.python_code or "",
                proposed_final_answer=final_answer,
            )
            steps.append(step)
            _write_trace_event(
                trace_writer=trace_writer,
                event="agent_terminated",
                payload={
                    "iteration": iteration,
                    "reason": "completed",
                    "answer_preview": final_answer[:200] if final_answer else None,
                },
            )
            return AgentRunResult(
                status="completed",
                answer=final_answer,
                iterations_used=iteration,
                steps=steps,
                failure_reason=None,
                failure_detail=None,
            )

        run_id = _build_execution_run_id(execution_id, iteration)
        execution_result = await run_python(
            decision.python_code or "",
            timeout_seconds=DEFAULT_PYTHON_TIMEOUT_SECONDS,
            run_id=run_id,
        )
        step = AgentStepRecord(
            iteration=iteration,
            script=decision.python_code or "",
            proposed_final_answer=_stringify_final_answer(decision.final_answer),
            execution_status=execution_result.status,
            execution_error=execution_result.error_message,
            execution_answer=execution_result.answer,
            execution_stdout_tail=execution_result.stdout_tail,
            execution_stderr_tail=execution_result.stderr_tail,
            execution_duration_ms=execution_result.duration_ms,
            execution_worker=execution_result.worker,
        )
        steps.append(step)
        _write_trace_event(
            trace_writer=trace_writer,
            event="agent_execution_result",
            payload={
                "iteration": iteration,
                "execution_id": execution_id,
                "run_id": run_id,
                "status": execution_result.status,
                "error": execution_result.error_message,
                "duration_ms": execution_result.duration_ms,
                "worker": execution_result.worker,
                "stdout_tail": execution_result.stdout_tail[:240],
                "stderr_tail": execution_result.stderr_tail[:240],
                "answer": execution_result.answer[:200]
                if execution_result.answer
                else None,
            },
        )

    last_step = steps[-1] if steps else None
    no_answer_detail = (
        f"last_had_python={bool(last_step.script.strip())}; "
        f"last_execution_status={last_step.execution_status}; "
        f"had_candidate_answer={last_step.proposed_final_answer is not None}"
        if last_step is not None
        else (
            "last_had_python=None; "
            "last_execution_status=None; "
            "had_candidate_answer=False"
        )
    )
    _write_trace_event(
        trace_writer=trace_writer,
        event="agent_terminated",
        payload={
            "iteration": request.max_iterations,
            "reason": "max_iterations_no_answer",
            "detail": no_answer_detail,
        },
    )
    return AgentRunResult(
        status="failed",
        iterations_used=request.max_iterations,
        steps=steps,
        failure_reason="max_iterations_no_answer",
        failure_detail=no_answer_detail,
    )


def _safe_list_packages() -> list[str]:
    """Load available execution packages for prompt conditioning.

    Returns:
        list[str]: Sorted package list.

    Raises:
        PythonExecutorUnavailableError: If the executor package cannot be queried.
    """
    packages = list_available_python_packages()
    return sorted(packages)


def _build_execution_run_id(execution_id: str | None, iteration: int) -> str:
    """Build a row-attributed execution run identifier.

    Args:
        execution_id: Optional identifier tied to one benchmark row.
        iteration: Current 1-based iteration.

    Returns:
        str: Stable run identifier used for script artifacts.
    """
    if execution_id is None:
        return f"iter-{iteration}"
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", execution_id).strip("_")
    if not cleaned:
        cleaned = "row"
    return f"{cleaned}-iter-{iteration}"


def _write_trace_event(
    *,
    trace_writer: TraceWriter | None,
    event: str,
    payload: dict[str, object],
) -> None:
    """Write one optional runtime trace event.

    Args:
        trace_writer: Optional trace writer.
        event: Stable event name.
        payload: Event payload mapping.
    """
    if trace_writer is None:
        return
    trace_writer.write_event(event=event, stage="agent", payload=payload)


def _stringify_final_answer(value: object | None) -> str | None:
    """Convert any model-emitted final answer value into string form."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
