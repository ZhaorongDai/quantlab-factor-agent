"""The two arms of the daily test of whether the agent learns from its trials."""

import dataclasses

from factor_agent.config import CampaignConfig
from tests.vocabulary_files import CONFIGS

A, B, A2 = (CampaignConfig.from_file(CONFIGS / f"learn_daily_{arm}.yaml") for arm in ("a", "b", "a2"))


def test_the_arms_differ_only_in_their_run_directory_and_whether_they_read_their_trials():
    assert (A.context.feedback, B.context.feedback) == (True, False)
    assert A.run_dir != B.run_dir
    assert dataclasses.replace(B, run_dir=A.run_dir, context=A.context, source=A.source) == A


def test_the_arms_search_without_a_factor_library_for_eight_rounds_on_claude():
    assert A.reference == () and A.known == ()
    assert (A.round.max_rounds, A.round.patience, A.round.memory_every) == (8, 8, 1)
    assert {(r.backend, r.model) for r in A.roles.values()} == {("anthropic", "claude-opus-5-5")}


def test_arm_a2_repeats_arm_a_in_a_run_directory_of_its_own():
    assert A2.run_dir == "/data/quantlab/runs/factor_agent_learn_daily_a2"
    assert dataclasses.replace(A2, run_dir=A.run_dir, source=A.source) == A


A3, B3 = (CampaignConfig.from_file(CONFIGS / f"learn_daily_{arm}.yaml") for arm in ("a3", "b3"))


def test_arms_a3_and_b3_are_a_and_b_with_the_training_data_split_into_what_the_search_sees_and_a_confirmation():
    for split, whole in ((A3, A), (B3, B)):
        assert (split.segments.training.end, split.segments.confirmation.start) == ("2009-12-31", "2010-01-04")
        assert split.segments.confirmation.end == whole.segments.training.end
        assert split.admission.confirm_t == 2.0
        assert dataclasses.replace(
            split, run_dir=whole.run_dir, source=whole.source, segments=whole.segments, admission=whole.admission,
        ) == whole


A5 = CampaignConfig.from_file(CONFIGS / "learn_daily_a5.yaml")


def test_arm_a5_is_arm_a3_shown_the_same_years_without_a_confirmation():
    assert A5.run_dir == "/data/quantlab/runs/factor_agent_learn_daily_a5"
    assert A5.segments.training == A3.segments.training and A5.segments.confirmation is None
    assert dataclasses.replace(
        A5, run_dir=A3.run_dir, source=A3.source, segments=A3.segments, admission=A3.admission,
    ) == A3
    assert dataclasses.replace(A5.admission, confirm_t=A3.admission.confirm_t) == A3.admission
