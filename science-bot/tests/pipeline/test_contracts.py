from typing import get_args

import pytest
from pydantic import ValidationError
from science_bot.pipeline.contracts import (
    AggregateOperation,
    AggregateQuestionSpec,
    DifferentialExpressionOperation,
    DifferentialExpressionQuestionSpec,
    HypothesisTestQuestionSpec,
    HypothesisTestType,
    RegressionModelType,
    RegressionQuestionSpec,
    RegressionReturnField,
    UnsupportedQuestionClassification,
    VariantFilteringOperation,
    VariantFilteringQuestionSpec,
    parse_question_classification,
    parse_question_execution_spec,
)
from science_bot.pipeline.execution.aggregate import IMPLEMENTED_AGGREGATE_OPERATIONS
from science_bot.pipeline.execution.differential_expression import (
    IMPLEMENTED_DIFFERENTIAL_EXPRESSION_EXECUTION_MODES,
    IMPLEMENTED_DIFFERENTIAL_EXPRESSION_OPERATIONS,
    SUPPORTED_RAW_COUNT_OPERATIONS,
)
from science_bot.pipeline.execution.hypothesis_test import (
    IMPLEMENTED_HYPOTHESIS_TESTS,
)
from science_bot.pipeline.execution.regression import (
    IMPLEMENTED_REGRESSION_MODEL_TYPES,
    IMPLEMENTED_REGRESSION_RETURN_FIELDS,
)
from science_bot.pipeline.execution.schemas import DifferentialExpressionExecutionMode
from science_bot.pipeline.execution.variant_filtering import (
    IMPLEMENTED_VARIANT_FILTERING_OPERATIONS,
)


def test_supported_question_classification_parses() -> None:
    result = parse_question_classification({"family": "aggregate"})

    assert result.family == "aggregate"


def test_unsupported_question_classification_parses() -> None:
    result = parse_question_classification(
        {
            "family": "unsupported",
            "reason": "requires clustering workflow not yet implemented",
        }
    )

    assert result.family == "unsupported"
    assert isinstance(result, UnsupportedQuestionClassification)
    assert result.reason == "requires clustering workflow not yet implemented"


def test_unsupported_question_classification_requires_reason() -> None:
    with pytest.raises(ValidationError):
        parse_question_classification({"family": "unsupported"})


def test_unsupported_question_classification_rejects_blank_reason() -> None:
    with pytest.raises(ValidationError):
        parse_question_classification({"family": "unsupported", "reason": "   "})


def test_supported_question_classification_ignores_reason() -> None:
    result = parse_question_classification(
        {
            "family": "aggregate",
            "reason": "The model included extra commentary that should be ignored.",
        }
    )

    assert result.family == "aggregate"


def test_unknown_family_fails_validation() -> None:
    with pytest.raises(ValidationError):
        parse_question_classification({"family": "machine_learning"})


def test_valid_aggregate_spec_parses() -> None:
    parsed = parse_question_execution_spec(
        {
            "family": "aggregate",
            "operation": "median",
            "value_field_hint": "treeness",
            "filters": [{"field_hint": "kingdom", "operator": "==", "value": "fungi"}],
            "return_format": "number",
        }
    )

    assert isinstance(parsed, AggregateQuestionSpec)
    assert parsed.operation == "median"


def test_aggregate_spec_parses_round_to() -> None:
    parsed = parse_question_execution_spec(
        {
            "family": "aggregate",
            "operation": "mean",
            "value_field_hint": "swarming area",
            "round_to": 1000,
            "return_format": "number",
        }
    )

    assert isinstance(parsed, AggregateQuestionSpec)
    assert parsed.round_to == 1000


def test_invalid_aggregate_operation_fails() -> None:
    with pytest.raises(ValidationError):
        parse_question_execution_spec(
            {
                "family": "aggregate",
                "operation": "sum",
                "value_field_hint": "treeness",
                "return_format": "number",
            }
        )


def test_invalid_aggregate_return_format_fails() -> None:
    with pytest.raises(ValidationError):
        parse_question_execution_spec(
            {
                "family": "aggregate",
                "operation": "count",
                "return_format": "json",
            }
        )


def test_aggregate_percentage_allows_missing_value_field_hint() -> None:
    parsed = AggregateQuestionSpec(
        family="aggregate",
        operation="percentage",
        return_format="percentage",
    )

    assert parsed.operation == "percentage"


def test_aggregate_median_requires_value_field_hint() -> None:
    with pytest.raises(ValidationError):
        AggregateQuestionSpec(
            family="aggregate",
            operation="median",
            return_format="number",
        )


def test_aggregate_skewness_requires_value_field_hint() -> None:
    with pytest.raises(ValidationError):
        AggregateQuestionSpec(
            family="aggregate",
            operation="skewness",
            return_format="number",
        )


def test_valid_mann_whitney_spec_parses() -> None:
    parsed = parse_question_execution_spec(
        {
            "family": "hypothesis_test",
            "test": "mann_whitney_u",
            "value_field_hint": "parsimony_informative_percentage",
            "group_field_hint": "group",
            "group_a_hint": "animals",
            "group_b_hint": "fungi",
            "return_field": "u_statistic",
        }
    )

    assert isinstance(parsed, HypothesisTestQuestionSpec)
    assert parsed.test == "mann_whitney_u"


def test_invalid_hypothesis_test_name_fails() -> None:
    with pytest.raises(ValidationError):
        parse_question_execution_spec(
            {
                "family": "hypothesis_test",
                "test": "anova",
                "return_field": "statistic",
            }
        )


def test_effect_size_return_field_parses_for_cohens_d() -> None:
    parsed = HypothesisTestQuestionSpec(
        family="hypothesis_test",
        test="cohens_d",
        value_field_hint="neun_count",
        group_field_hint="condition",
        group_a_hint="KD",
        group_b_hint="control",
        return_field="effect_size",
    )

    assert parsed.return_field == "effect_size"


def test_valid_cohens_d_spec_parses() -> None:
    parsed = parse_question_execution_spec(
        {
            "family": "hypothesis_test",
            "test": "cohens_d",
            "value_field_hint": "neun_count",
            "group_field_hint": "condition",
            "group_a_hint": "KD",
            "group_b_hint": "control",
            "return_field": "effect_size",
        }
    )

    assert isinstance(parsed, HypothesisTestQuestionSpec)
    assert parsed.test == "cohens_d"
    assert parsed.return_field == "effect_size"


def test_shapiro_wilk_rejects_group_field_hint() -> None:
    with pytest.raises(ValidationError):
        HypothesisTestQuestionSpec(
            family="hypothesis_test",
            test="shapiro_wilk",
            value_field_hint="NeuN",
            group_field_hint="condition",
            return_field="w_statistic",
        )


def test_valid_ordinal_logistic_regression_spec_parses() -> None:
    parsed = parse_question_execution_spec(
        {
            "family": "regression",
            "model_type": "ordinal_logistic",
            "outcome_field_hint": "AESEV",
            "predictor_field_hint": "BCG_vaccination",
            "covariate_field_hints": ["interaction_frequency"],
            "return_field": "odds_ratio",
        }
    )

    assert isinstance(parsed, RegressionQuestionSpec)
    assert parsed.model_type == "ordinal_logistic"


def test_polynomial_regression_requires_degree() -> None:
    with pytest.raises(ValidationError):
        RegressionQuestionSpec(
            family="regression",
            model_type="polynomial",
            outcome_field_hint="swarming_area",
            predictor_field_hint="rhlI_frequency",
            return_field="r_squared",
        )


def test_invalid_regression_return_field_fails() -> None:
    with pytest.raises(ValidationError):
        parse_question_execution_spec(
            {
                "family": "regression",
                "model_type": "linear",
                "outcome_field_hint": "y",
                "predictor_field_hint": "x",
                "return_field": "p_value",
            }
        )


def test_predicted_probability_regression_spec_parses_prediction_inputs() -> None:
    parsed = parse_question_execution_spec(
        {
            "family": "regression",
            "model_type": "logistic",
            "outcome_field_hint": "y",
            "predictor_field_hint": "x",
            "prediction_inputs": [{"field_hint": "x", "value": 1}],
            "return_field": "predicted_probability",
        }
    )

    assert isinstance(parsed, RegressionQuestionSpec)
    assert parsed.prediction_inputs[0].field_hint == "x"


def test_valid_differential_expression_count_spec_parses() -> None:
    parsed = parse_question_execution_spec(
        {
            "family": "differential_expression",
            "method": "deseq2_like",
            "operation": "significant_gene_count",
            "comparison_label_hints": ["CBD/cisplatin vs DMSO"],
            "significance_threshold": 0.05,
            "fold_change_threshold": 1.0,
            "basemean_threshold": 10.0,
            "return_format": "number",
        }
    )

    assert isinstance(parsed, DifferentialExpressionQuestionSpec)
    assert parsed.operation == "significant_gene_count"


def test_valid_differential_expression_log2fc_spec_parses() -> None:
    parsed = parse_question_execution_spec(
        {
            "family": "differential_expression",
            "method": "precomputed_de_table",
            "operation": "gene_log2_fold_change",
            "target_gene_hint": "PA14_35160",
            "comparison_label_hints": ["delta_rhlI"],
            "return_format": "signed_number",
        }
    )

    assert isinstance(parsed, DifferentialExpressionQuestionSpec)
    assert parsed.target_gene_hint == "PA14_35160"


def test_invalid_differential_expression_operation_fails() -> None:
    with pytest.raises(ValidationError):
        parse_question_execution_spec(
            {
                "family": "differential_expression",
                "method": "deseq2_like",
                "operation": "pathway_enrichment",
                "return_format": "label",
            }
        )


def test_valid_variant_filtering_count_spec_parses() -> None:
    parsed = parse_question_execution_spec(
        {
            "family": "variant_filtering",
            "operation": "filtered_variant_count",
            "cohort_hint": "CHIP genes",
            "vaf_max": 0.3,
            "return_format": "number",
        }
    )

    assert isinstance(parsed, VariantFilteringQuestionSpec)
    assert parsed.operation == "filtered_variant_count"


def test_valid_variant_filtering_gene_with_max_spec_parses() -> None:
    parsed = parse_question_execution_spec(
        {
            "family": "variant_filtering",
            "operation": "gene_with_max_variants",
            "sample_hint": "oldest male carrier",
            "return_format": "label",
        }
    )

    assert isinstance(parsed, VariantFilteringQuestionSpec)
    assert parsed.operation == "gene_with_max_variants"


def test_invalid_variant_filtering_return_format_fails() -> None:
    with pytest.raises(ValidationError):
        parse_question_execution_spec(
            {
                "family": "variant_filtering",
                "operation": "sample_variant_count",
                "return_format": "signed_number",
            }
        )


def test_aggregate_operation_parity() -> None:
    assert set(get_args(AggregateOperation)) == set(IMPLEMENTED_AGGREGATE_OPERATIONS)


def test_hypothesis_test_parity() -> None:
    assert set(get_args(HypothesisTestType)) == set(IMPLEMENTED_HYPOTHESIS_TESTS)


def test_regression_model_type_parity() -> None:
    assert set(get_args(RegressionModelType)) == set(IMPLEMENTED_REGRESSION_MODEL_TYPES)


def test_regression_return_field_parity() -> None:
    assert set(get_args(RegressionReturnField)) == set(
        IMPLEMENTED_REGRESSION_RETURN_FIELDS
    )


def test_differential_expression_operation_parity() -> None:
    assert set(get_args(DifferentialExpressionOperation)) == set(
        IMPLEMENTED_DIFFERENTIAL_EXPRESSION_OPERATIONS
    )


def test_differential_expression_execution_modes_are_partially_implemented() -> None:
    assert set(get_args(DifferentialExpressionExecutionMode)) == {
        "precomputed_results",
        "raw_counts",
    }
    assert set(IMPLEMENTED_DIFFERENTIAL_EXPRESSION_EXECUTION_MODES) == {
        "precomputed_results",
        "raw_counts",
    }
    assert set(SUPPORTED_RAW_COUNT_OPERATIONS) == {
        "significant_gene_count",
        "gene_log2_fold_change",
        "significant_marker_count",
    }


def test_variant_filtering_operation_parity() -> None:
    assert set(get_args(VariantFilteringOperation)) == set(
        IMPLEMENTED_VARIANT_FILTERING_OPERATIONS
    )
