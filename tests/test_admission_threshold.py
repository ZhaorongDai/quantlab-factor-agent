"""The t threshold an evaluated trial needs, against the trial count, and where a refusal for coverage says values are lost."""

import pytest

from factor_agent import _admission
from factor_agent.config import AdmissionConfig
from factor_agent.evaluator import NodeCoverage


@pytest.mark.parametrize(("trials", "threshold"), [(1, 3.21), (10, 3.82), (100, 4.35), (600, 4.73)])
def test_the_plain_bonferroni_threshold_divides_by_twice_the_trial_count(trials, threshold):
    assert _admission.t_threshold(AdmissionConfig(), trials) == pytest.approx(threshold, abs=0.005)


def test_with_exponent_one_half_a_hundred_trials_count_as_ten():
    assert _admission.t_threshold(AdmissionConfig(trial_exponent=0.5), 100) == pytest.approx(
        _admission.t_threshold(AdmissionConfig(), 10))


def test_the_first_trial_needs_the_same_t_whatever_the_exponent():
    assert _admission.t_threshold(AdmissionConfig(trial_exponent=0.5), 1) == pytest.approx(3.21, abs=0.005)


#: How the coverage tests write a node: by a name of its own.
NAMES = {0: "amount", 1: "high - low", 2: "amount / (high - low)", 3: "CrossSectionalZScore(amount / (high - low))"}


def test_the_cause_names_the_step_that_empties_a_day_and_its_infinite_input():
    trace = [
        NodeCoverage(0, 0.99, 0.0, 0.0, (), column=True), NodeCoverage(1, 0.99, 0.0, 0.0, ()),
        NodeCoverage(2, 0.97, 0.004, 0.63, (0, 1)), NodeCoverage(3, 0.0, 0.0, 0.0, (2,)),
    ]
    assert _admission.coverage_cause(trace, NAMES.get, 0.5) == (
        "; it falls at CrossSectionalZScore(amount / (high - low)), to 0.00 from 0.97 for amount / (high - low); "
        "amount / (high - low) is infinite on 0.4% of the member days and on 63% of the days at least once "
        "(a division by zero is infinite)"
    )


def test_a_step_infinite_itself_is_named_with_its_own_share():
    trace = [NodeCoverage(0, 0.99, 0.0, 0.0, (), column=True), NodeCoverage(2, 0.3, 0.69, 1.0, (0,))]
    assert _admission.coverage_cause(trace, NAMES.get, 0.5) == (
        "; it falls at amount / (high - low), to 0.30 from 0.99 for amount; it is infinite on 69% of the member "
        "days and on 100% of the days at least once (a division by zero is infinite)"
    )


def test_a_sparse_data_column_is_named_as_the_data():
    trace = [NodeCoverage(0, 0.42, 0.0, 0.0, (), column=True), NodeCoverage(3, 0.41, 0.0, 0.0, (0,))]
    assert _admission.coverage_cause(trace, NAMES.get, 0.5) == "; the data column amount has a value on 0.42 of the member days"


def test_without_a_traced_step_below_the_threshold_nothing_is_added():
    assert _admission.coverage_cause([NodeCoverage(3, 0.9, 0.0, 0.0, ())], NAMES.get, 0.5) == ""
