import pytest
from science_bot.pipeline.resolution.families.variant_filtering import (
    VariantFilteringResolutionDecision,
    VariantFilteringResolvedPlan,
    build_variant_filtering_plan_from_decision,
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


def test_build_variant_filtering_merge_plan_from_decision():
    plan = build_variant_filtering_plan_from_decision(
        VariantFilteringResolutionDecision(
            action="finalize",
            reason="done",
            use_merge=True,
            data_source_files=["a.csv", "b.csv"],
            data_source_sample_ids=["A", "B"],
            data_source_selected_columns=["gene", "effect", "vaf"],
            metadata_file="metadata.csv",
            metadata_sample_id_column="Sample ID",
            metadata_columns=["Status"],
            operation="variant_proportion",
            gene_column="gene",
            effect_column="effect",
            vaf_column="vaf",
            filters=[],
            return_format="percentage",
        ),
        require_text=lambda value, _name: value,
        require_value=lambda value, _name: value,
    )

    assert plan.filename is None
    assert plan.merge_plan is not None
    assert [source.sample_id for source in plan.merge_plan.data_sources] == ["A", "B"]
    assert plan.merge_plan.join is not None
    assert plan.merge_plan.join.metadata_columns == ["Status"]


def test_variant_filtering_merge_plan_requires_data_sources():
    with pytest.raises(ValueError):
        build_variant_filtering_plan_from_decision(
            VariantFilteringResolutionDecision(
                action="finalize",
                reason="done",
                use_merge=True,
                data_source_files=[],
                data_source_sample_ids=[],
                operation="filtered_variant_count",
                filters=[],
                return_format="number",
            ),
            require_text=lambda value, _name: value,
            require_value=lambda value, _name: value,
        )
