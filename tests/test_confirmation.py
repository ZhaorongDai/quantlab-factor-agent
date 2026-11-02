"""A confirmation segment: the later part of the training data, which admission reads and no prompt shows.

The search adapts to what its feedback shows, so a factor tuned to the statistics of the training data
can fit their noise. With ``segments.confirmation`` a candidate is evaluated there as well, and is
accepted only if its partial IC holds there in the factor's direction (``admission.confirm_t``); the
refusal says so without a number, and no prompt holds a statistic of the confirmation segment.
"""

import re

import pytest

from factor_agent.config import AdmissionConfig
from factor_agent.segments import Segment, Segments
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM
from tests.test_campaign_loop import market, noise  # noqa: F401  (the module's synthetic store)

DISCOVERY = (synthetic_market.TRAINING[0], "2010-12-31")
CONFIRMATION = ("2011-01-03", synthetic_market.TRAINING[1])
SIGNAL = candidate("volume shock", "alpha = volume / WindowedAvg(volume, 20)")


def _propose(number, directions):
    return [SIGNAL if number == 1 and slot == 0 else noise(number, slot) for slot in range(len(directions))]


def _campaign(tmp_path, market, confirm_t):  # noqa: F811
    path = write_config(
        tmp_path, market["components"],
        segments={"training": {"start": DISCOVERY[0], "end": DISCOVERY[1]},
                  "confirmation": {"start": CONFIRMATION[0], "end": CONFIRMATION[1]}},
        admission={"confirm_t": confirm_t},
        round={"candidates": 10, "min_per_type": 1, "retries": 0, "max_rounds": 2},
    )
    fake = FakeLLM(_propose)
    return run(path, fake, market["cache"]), fake


@pytest.fixture(scope="module")
def held(tmp_path_factory, market):  # noqa: F811
    return _campaign(tmp_path_factory.mktemp("held"), market, confirm_t=2.0)


@pytest.fixture(scope="module")
def not_held(tmp_path_factory, market):  # noqa: F811
    return _campaign(tmp_path_factory.mktemp("not_held"), market, confirm_t=1000.0)


def test_a_signal_that_holds_on_the_confirmation_segment_is_accepted_and_its_statistics_kept(held):
    result, _ = held
    first = result.records[0]
    assert first.status == "accepted"
    assert first.confirmation["partial_ic_t"] > 2.0 and first.confirmation["start"] >= CONFIRMATION[0]
    assert first.metrics["end"] <= DISCOVERY[1]


def test_a_candidate_that_does_not_hold_is_refused_without_a_number(not_held):
    result, _ = not_held
    first = result.records[0]
    assert first.status == "rejected" and first.metrics is not None
    [reason] = first.reasons
    assert reason.startswith("the factor did not hold on the confirmation period")
    assert not re.search(r"\d", reason)


#: The statistics a prompt shows, with the digits it shows them in.
SHOWN = (("ic_mean", 4), ("ic_t", 2), ("partial_ic_mean", 4), ("partial_ic_t", 2))


def _shown(metrics: dict) -> set[str]:
    """The statistics as a prompt would write them, but those rounding to zero, which any correlation may read."""
    return {text for key, digits in SHOWN if (text := f"{metrics[key]:+.{digits}f}").strip("+-0.")}


def test_no_prompt_holds_a_statistic_of_the_confirmation_segment(held, not_held):
    for result, fake in (held, not_held):
        training = set().union(*(_shown(r.metrics) for r in result.records if r.metrics))
        # a short number may also be some trial's training statistic, which the prompts do show
        hidden = set().union(*(_shown(r.confirmation) for r in result.records if r.confirmation)) - training
        assert hidden
        for request in fake.requests:
            assert not any(re.search(re.escape(text) + r"(?!\d)", request.prompt) for text in hidden), request.role


def test_the_confirmation_segment_lies_between_the_training_and_the_validation_segments():
    training = Segment("training", "2010-01-01", "2010-12-31")
    with pytest.raises(ValueError, match="segment confirmation starts on"):
        Segments(training, confirmation=Segment("confirmation", "2010-06-01", "2011-06-30"))
    with pytest.raises(ValueError, match="segment validation starts on"):
        Segments(training, confirmation=Segment("confirmation", "2011-01-03", "2011-12-30"),
                 validation=Segment("validation", "2011-06-01", "2012-06-29"))


def test_a_confirmation_threshold_needs_a_number():
    with pytest.raises(TypeError, match="confirm_t"):
        AdmissionConfig(confirm_t="high")
