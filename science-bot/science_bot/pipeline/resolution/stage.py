"""Public resolution stage entrypoint."""

from science_bot.pipeline.resolution.controller import run_resolution_controller
from science_bot.pipeline.resolution.schemas import (
    ResolutionStageInput,
    ResolutionStageOutput,
)


async def run_resolution_stage(
    stage_input: ResolutionStageInput,
) -> ResolutionStageOutput:
    """Resolve a classified question into an execution payload.

    Args:
        stage_input: Resolution stage input.

    Returns:
        ResolutionStageOutput: Final payload and compact debug summaries.
    """
    return await run_resolution_controller(stage_input)
