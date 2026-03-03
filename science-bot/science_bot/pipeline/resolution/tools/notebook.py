"""Notebook inspection tool for the resolution stage.

Reads executed Jupyter notebook outputs from a capsule directory.
These outputs often contain pre-computed statistical results that
directly answer the benchmark question, or reveal correct file names
and column names used by the original authors.
"""

import json
import re
from pathlib import Path

from science_bot.pipeline.resolution.tools.schemas import (
    NotebookCellOutput,
    NotebookOutputs,
)

_MAX_NOTEBOOK_FILE_BYTES = 20 * 1024 * 1024  # 20 MB
_KEEP_FIRST_CELLS = 5
_KEEP_LAST_CELLS = 30
_MAX_CELL_OUTPUT_CHARS = 600
_MAX_SUMMARY_CELL_CHARS = 250

# Maximum total characters for the formatted notebook summary string.
MAX_NOTEBOOK_SUMMARY_CHARS = 8000

# Strip ANSI escape codes produced by R / Python terminals.
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Prefixes that indicate noisy pip / conda install output.
_NOISY_PREFIXES = (
    "Collecting ",
    "Downloading ",
    "Installing ",
    "Requirement already",
    "Successfully installed",
    "Looking in indexes",
    "Note: you may need",
    "Using cached",
    "Obtaining ",
    "Building wheels",
    "Building wheel",
    "  Building wheel",
    "  Downloading ",
    "  Installing ",
    "  Obtaining ",
)


def _clean_text(text: str) -> str:
    """Remove ANSI escape sequences and strip surrounding whitespace."""
    return _ANSI_RE.sub("", text).strip()


def _is_noisy(text: str) -> bool:
    """Return True if *text* is installation / warning noise with no data value."""
    # Pip / conda installation lines
    for prefix in _NOISY_PREFIXES:
        if text.startswith(prefix):
            return True
    # Matplotlib / seaborn figure object reprs with no numeric content
    if re.match(r"^<(Figure|seaborn|matplotlib)", text):
        return True
    # Pure progress bars
    if re.match(r"^\s*\|[█ ]+\|", text):
        return True
    return False


def get_notebook_outputs(capsule_path: Path) -> NotebookOutputs:
    """Collect executed cell outputs from the capsule's Jupyter notebook.

    The notebook was created and executed by the original study authors.
    Its outputs may contain:
    - Pre-computed statistical results (odds ratios, p-values, correlations,
      Cohen's d, DESeq2 counts, etc.) that directly answer the question.
    - The exact column names and file names used in the analysis.
    - The statistical method and significance thresholds applied.

    Call this during resolution to orient yourself before inspecting data
    files. If an answer is already computed in the notebook outputs, use
    it to finalize the plan immediately.

    Args:
        capsule_path: Absolute path to the capsule directory.

    Returns:
        NotebookOutputs: Structured list of executed code-cell outputs.
    """
    notebooks = sorted(capsule_path.glob("**/*.ipynb"))
    if not notebooks:
        return NotebookOutputs(
            notebook_found=False,
            total_cells_with_output=0,
            outputs=[],
            truncated=False,
        )

    nb_path = notebooks[0]

    # Guard against extremely large notebooks bloating the scratchpad.
    try:
        nb_size = nb_path.stat().st_size
    except OSError:
        nb_size = 0
    if nb_size > _MAX_NOTEBOOK_FILE_BYTES:
        return NotebookOutputs(
            notebook_found=False,
            total_cells_with_output=0,
            outputs=[],
            truncated=False,
        )

    try:
        with open(nb_path, encoding="utf-8") as fh:
            nb = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return NotebookOutputs(
            notebook_found=False,
            total_cells_with_output=0,
            outputs=[],
            truncated=False,
        )

    collected: list[NotebookCellOutput] = []

    for cell_index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        source_preview = source[:120].replace("\n", " ").strip()

        for out in cell.get("outputs", []):
            output_type = out.get("output_type", "")

            if output_type == "stream":
                raw = "".join(out.get("text", []))
            elif output_type in ("execute_result", "display_data"):
                data = out.get("data", {})
                raw = "".join(data.get("text/plain", []))
                # Skip outputs that are only images with no text repr.
                if not raw and ("image/png" in data or "image/svg+xml" in data):
                    continue
            else:
                continue

            text = _clean_text(raw)
            if not text or len(text) < 4:
                continue
            if _is_noisy(text):
                continue

            collected.append(
                NotebookCellOutput(
                    cell_index=cell_index,
                    source_preview=source_preview,
                    output_text=text[:_MAX_CELL_OUTPUT_CHARS],
                )
            )

    total = len(collected)
    max_cells = _KEEP_FIRST_CELLS + _KEEP_LAST_CELLS

    if total <= max_cells:
        shown = collected
        truncated = False
    else:
        # Keep the first few cells (data loading context) and the last many
        # cells (where final computed results are most likely to appear).
        first = collected[:_KEEP_FIRST_CELLS]
        last = collected[total - _KEEP_LAST_CELLS :]
        shown = first + last
        truncated = True

    return NotebookOutputs(
        notebook_found=True,
        total_cells_with_output=total,
        outputs=shown,
        truncated=truncated,
    )


def format_notebook_summary(result: NotebookOutputs) -> str:
    """Format NotebookOutputs into a compact string for the scratchpad.

    Each cell output is shown truncated to _MAX_SUMMARY_CELL_CHARS so that
    more cells fit within the total budget while preserving the key values
    (scalars, short tables) that appear at the start of each output.

    Args:
        result: Collected notebook outputs.

    Returns:
        str: A human-readable summary suitable for embedding in the prompt.
    """
    if not result.notebook_found:
        return "No executed notebook found for this capsule."

    if not result.outputs:
        return "Notebook found but no code-cell outputs were recorded."

    header = (
        f"Notebook outputs ({result.total_cells_with_output} output cells"
        + (", showing first+last cells" if result.truncated else "")
        + "):"
    )
    lines: list[str] = [header]
    for out in result.outputs:
        display_text = out.output_text
        if len(display_text) > _MAX_SUMMARY_CELL_CHARS:
            display_text = display_text[:_MAX_SUMMARY_CELL_CHARS] + "..."
        lines.append(
            f"  [cell {out.cell_index}] {out.source_preview[:70]}\n"
            f"    => {display_text}"
        )

    summary = "\n".join(lines)
    if len(summary) > MAX_NOTEBOOK_SUMMARY_CHARS:
        summary = summary[:MAX_NOTEBOOK_SUMMARY_CHARS] + "\n  [... truncated]"
    return summary
