"""Real-mode runtime loop for iterative question answering."""

from pathlib import Path

from science_bot.agent.contracts import (
    AgentDecision,
    AgentRunRequest,
    AgentRunResult,
    AgentStepRecord,
)
from science_bot.agent.prompts import build_system_prompt, build_user_prompt
from science_bot.agent.summary import summarize_steps
from science_bot.providers.executor import (
    list_available_python_packages,
    run_python,
)
from science_bot.providers.llm import parse_structured

DEFAULT_MAX_ITERATIONS = 6
DEFAULT_PYTHON_TIMEOUT_SECONDS = 30


async def run_agent(
    *,
    question: str,
    capsule_path: Path,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AgentRunResult:
    """Run the real-mode agent loop without oracle correctness feedback.

    Args:
        question: Natural language question to answer.
        capsule_path: Capsule data directory.
        max_iterations: Maximum number of decision iterations.

    Returns:
        AgentRunResult: Terminal runtime result with all recorded steps.

    Raises:
        PythonExecutorUnavailableError: If package discovery or execution fails.
    """
    request = AgentRunRequest(
        question=question,
        capsule_path=capsule_path,
        max_iterations=max_iterations,
    )
    available_packages = _safe_list_packages()
    steps: list[AgentStepRecord] = []
    latest_candidate_answer: str | None = None
    system_prompt = build_system_prompt(request.max_iterations)

    for iteration in range(1, request.max_iterations + 1):
        user_prompt = build_user_prompt(
            question=request.question,
            capsule_path=request.capsule_path,
            available_packages=available_packages,
            step_summary=summarize_steps(steps),
            iteration=iteration,
            max_iterations=request.max_iterations,
        )
        decision = await parse_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=AgentDecision,
        )

        if decision.decision == "run_python":
            execution_result = await run_python(
                decision.script or "",
                timeout_seconds=DEFAULT_PYTHON_TIMEOUT_SECONDS,
                run_id=f"iter-{iteration}",
            )
            steps.append(
                AgentStepRecord(
                    iteration=iteration,
                    decision=decision.decision,
                    script=decision.script,
                    execution_status=execution_result.status,
                    execution_error=execution_result.error_message,
                )
            )
            continue

        if decision.decision == "respond":
            latest_candidate_answer = decision.answer
            steps.append(
                AgentStepRecord(
                    iteration=iteration,
                    decision=decision.decision,
                    answer=decision.answer,
                )
            )
            continue

        steps.append(
            AgentStepRecord(
                iteration=iteration,
                decision=decision.decision,
                reason=decision.reason,
            )
        )
        return AgentRunResult(
            status="failed",
            iterations_used=iteration,
            steps=steps,
            failure_reason="need_info",
        )

    if latest_candidate_answer is not None:
        return AgentRunResult(
            status="completed",
            answer=latest_candidate_answer,
            iterations_used=request.max_iterations,
            steps=steps,
            failure_reason=None,
        )

    return AgentRunResult(
        status="failed",
        iterations_used=request.max_iterations,
        steps=steps,
        failure_reason="max_iterations_no_answer",
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
