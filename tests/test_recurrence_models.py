import numpy as np
import pytest

from lungml.recurrence_models import CensoringDistribution


@pytest.mark.parametrize("kind", [[1, 0, 2], [2, 0, 1]])
def test_reverse_km_removes_tied_failures_before_censoring(kind):
    # Three start at risk. One failure leaves before the tied censor, so
    # reverse KM has a censoring hazard of 1/2, rather than 1/3.
    censoring = CensoringDistribution().fit([1, 1, 2], kind)
    assert censoring.survival_probability(1, left_limit=True) == 1
    assert censoring.survival_probability(1) == pytest.approx(0.5)
    assert censoring.survival_probability(2) == pytest.approx(0.5)


def test_reverse_km_handles_both_causes_tied_with_censoring_and_zero_tail():
    censoring = CensoringDistribution().fit([1, 1, 1, 2], [1, 2, 0, 0])
    np.testing.assert_allclose(censoring.after_, [0.5, 0])
    assert censoring.survival_probability(2, left_limit=True) == pytest.approx(0.5)


def test_reverse_km_without_ties_preserves_left_limits():
    censoring = CensoringDistribution().fit([1, 2, 3, 4], [1, 0, 2, 0])
    assert censoring.survival_probability(2, left_limit=True) == 1
    assert censoring.survival_probability(2) == pytest.approx(2 / 3)
    assert censoring.survival_probability(3) == pytest.approx(2 / 3)
    assert censoring.survival_probability(4) == 0


@pytest.mark.parametrize("policy", [
    {"min_survival": 0}, {"min_survival": 1.1}, {"min_survival": np.nan},
    {"min_at_risk": 0}, {"min_at_risk": 1.5},
])
def test_invalid_censoring_support_policy_is_rejected(policy):
    with pytest.raises(ValueError, match="Minimum"):
        CensoringDistribution(**policy)
