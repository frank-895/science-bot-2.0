from typing import get_args

import science_bot.pipeline.execution.stage as execution_stage
from science_bot.pipeline.contracts import QuestionFamily


def test_all_question_families_have_execution_implementations() -> None:
    families = set(get_args(QuestionFamily))
    implemented_families = {
        family
        for family in families
        if hasattr(execution_stage, f"run_{family}_execution")
    }

    assert implemented_families == families
