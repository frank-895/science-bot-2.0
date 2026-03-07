"""Utilities for compact step-history summaries."""

from science_bot.agent.contracts import AgentStepRecord


def summarize_steps(steps: list[AgentStepRecord], max_chars: int = 2400) -> str:
    """Summarize prior agent steps into a bounded text payload.

    Args:
        steps: Historical step records.
        max_chars: Maximum summary length.

    Returns:
        str: Compact summary text for prompting.
    """
    if not steps:
        return "No prior steps."

    lines: list[str] = []
    for step in steps[-8:]:
        fields = [f"iter={step.iteration}"]
        if step.execution_status:
            fields.append(f"exec_status={step.execution_status}")
        if step.execution_duration_ms is not None:
            fields.append(f"exec_ms={step.execution_duration_ms}")
        if step.execution_worker:
            fields.append(f"exec_worker={step.execution_worker[:40]}")
        if step.execution_error:
            fields.append(f"exec_error={step.execution_error[:140]}")
        if step.execution_answer:
            fields.append(f"exec_answer={step.execution_answer[:200]}")
        if step.execution_stdout_tail:
            fields.append(f"exec_stdout={step.execution_stdout_tail[:240]}")
        if step.execution_stderr_tail:
            fields.append(f"exec_stderr={step.execution_stderr_tail[:240]}")
        if step.proposed_final_answer:
            fields.append(f"proposed_final={step.proposed_final_answer[:140]}")
        lines.append("; ".join(fields))

    summary = "\n".join(lines)
    if len(summary) <= max_chars:
        return summary
    return summary[-max_chars:]
