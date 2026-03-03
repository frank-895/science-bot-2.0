import pytest
from science_bot.pipeline.resolution.families.variant_filtering import (
    VariantFilteringResolvedPlan,
)


def test_variant_filtering_resolved_plan_requires_sample_value_for_sample_count():
    with pytest.raises(ValueError):
        VariantFilteringResolvedPlan(
            filename="variants.csv",
            operation="sample_variant_count",
            sample_column="sample",
            sample_value=None,
            filters=[],
            return_format="number",
        )
