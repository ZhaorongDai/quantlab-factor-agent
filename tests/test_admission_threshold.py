"""The t threshold an evaluated trial needs, against the trial count."""

import pytest

from factor_agent import _admission
from factor_agent.config import AdmissionConfig


@pytest.mark.parametrize(("trials", "threshold"), [(1, 3.21), (10, 3.82), (100, 4.35), (600, 4.73)])
def test_the_plain_bonferroni_threshold_divides_by_twice_the_trial_count(trials, threshold):
    assert _admission.t_threshold(AdmissionConfig(), trials) == pytest.approx(threshold, abs=0.005)


def test_with_exponent_one_half_a_hundred_trials_count_as_ten():
    assert _admission.t_threshold(AdmissionConfig(trial_exponent=0.5), 100) == pytest.approx(
        _admission.t_threshold(AdmissionConfig(), 10))


def test_the_first_trial_needs_the_same_t_whatever_the_exponent():
    assert _admission.t_threshold(AdmissionConfig(trial_exponent=0.5), 1) == pytest.approx(3.21, abs=0.005)
