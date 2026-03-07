"""Prompt builders for the iterative agent runtime."""

from pathlib import Path

DECISION_SCHEMA_TEXT = """Return one JSON object with this schema:
{"python": "<python code>|null", "final_answer": "<string>|null"}
Rules:
- Provide exactly one of "python" or "final_answer".
- Use "python" when you need another execution step.
- Use "final_answer" only when you are ready to answer.
- Extra keys are ignored.
"""


REPAIR_PROMPT_TEXT = """Your previous output did not match the required schema.
Return one JSON object only:
{"python": "<python code>|null", "final_answer": "<string>|null"}
Do not include markdown, prose, or code fences.
"""


def build_system_prompt(max_iterations: int) -> str:
    """Build the system prompt for one decision turn.

    Args:
        max_iterations: Maximum allowed iterations for this run.

    Returns:
        str: System prompt text.
    """
    return (
        "You are a life-science data analysis agent.\n"
        "You must answer the question by using short, targeted Python scripts.\n"
        f"You have at most {max_iterations} iterations total.\n"
        "Start executing quickly, avoid broad exploration, and provide a candidate "
        "answer as soon as evidence is sufficient.\n"
        "If you decide to stop or give up, return your best available final_answer.\n"
        "Prefer deterministic scripts and concise outputs.\n"
        f"{DECISION_SCHEMA_TEXT}"
    )


def build_user_prompt(
    *,
    question: str,
    capsule_path: Path,
    capsule_manifest: str,
    available_packages: list[str],
    step_summary: str,
    iteration: int,
    max_iterations: int,
) -> str:
    """Build the user prompt for one decision turn.

    Args:
        question: Natural language question.
        capsule_path: Capsule filesystem path.
        capsule_manifest: Recursive file listing for the capsule.
        available_packages: Packages available for execution.
        step_summary: Compact summary of prior steps.
        iteration: Current 1-based iteration.
        max_iterations: Maximum iteration budget.

    Returns:
        str: User prompt text.
    """
    remaining = max_iterations - iteration + 1
    package_list = ", ".join(available_packages) if available_packages else "(none)"
    return (
        f"Question:\n{question}\n\n"
        f"Capsule path:\n{capsule_path}\n\n"
        "Capsule files (recursive):\n"
        f"{capsule_manifest}\n\n"
        f"Available packages:\n{package_list}\n\n"
        f"Iteration:\n{iteration}/{max_iterations} (remaining={remaining})\n\n"
        "Step summary:\n"
        f"{step_summary}\n\n"
        "All required data is already available at the capsule path. "
        "Do not ask the user to upload data or provide additional files.\n\n"
        "Respond with one valid JSON object only."
    )


def build_repair_prompt(*, previous_error: str) -> str:
    """Build a strict correction prompt after invalid decision output.

    Args:
        previous_error: Validation or parse failure message.

    Returns:
        str: Repair instruction prompt.
    """
    return f"{REPAIR_PROMPT_TEXT}\nPrevious error:\n{previous_error}\n"
