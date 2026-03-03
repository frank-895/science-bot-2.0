import pytest
from science_bot.pipeline.resolution.families.differential_expression import (
    DifferentialExpressionResolvedPlan,
)


def test_de_resolved_plan_requires_result_tables():
    with pytest.raises(ValueError):
        DifferentialExpressionResolvedPlan(
            mode="precomputed_results",
            result_table_files={},
            operation="significant_gene_count",
            comparison_labels=[],
        )
