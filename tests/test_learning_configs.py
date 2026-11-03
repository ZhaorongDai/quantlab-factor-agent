"""The two arms of the daily test of whether the agent learns from its trials."""

import dataclasses

import pytest

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


LOCAL = CampaignConfig.from_file(CONFIGS / "learn_daily_local.yaml")


def test_the_local_arm_is_arm_a2_on_a_model_served_on_the_same_computer():
    assert LOCAL.run_dir == "/data/quantlab/runs/factor_agent_learn_daily_local"
    assert {(r.backend, r.model) for r in LOCAL.roles.values()} == {("openai_compatible", "qwen3.6-35b-a3b-ud-q4_k_xl")}
    assert dataclasses.replace(LOCAL, run_dir=A2.run_dir, source=A2.source, roles=A2.roles) == A2


LOCAL_THEMES = CampaignConfig.from_file(CONFIGS / "learn_daily_local_themes.yaml")


def test_the_themes_arm_is_the_local_arm_grouping_its_trials_by_what_they_measure():
    assert LOCAL_THEMES.run_dir == "/data/quantlab/runs/factor_agent_learn_daily_local_themes"
    assert (LOCAL_THEMES.context.theme_correlation, LOCAL_THEMES.context.send_back_repeats) == (0.5, True)
    assert dataclasses.replace(
        LOCAL_THEMES, run_dir=LOCAL.run_dir, source=LOCAL.source,
        context=dataclasses.replace(LOCAL_THEMES.context, theme_correlation=None, send_back_repeats=False),
    ) == LOCAL

LOCAL_THEMES2, LOCAL_MAP = (CampaignConfig.from_file(CONFIGS / f"learn_daily_local_{arm}.yaml") for arm in ("themes2", "map"))


def test_the_second_themes_arm_repeats_the_first_in_a_run_directory_of_its_own():
    assert LOCAL_THEMES2.run_dir == "/data/quantlab/runs/factor_agent_learn_daily_local_themes2"
    assert dataclasses.replace(LOCAL_THEMES2, run_dir=LOCAL_THEMES.run_dir, source=LOCAL_THEMES.source) == LOCAL_THEMES


def test_the_map_arm_is_the_themes_arm_sending_nothing_back():
    assert LOCAL_MAP.run_dir == "/data/quantlab/runs/factor_agent_learn_daily_local_map"
    assert (LOCAL_MAP.context.theme_correlation, LOCAL_MAP.context.send_back_repeats) == (0.5, False)
    assert dataclasses.replace(
        LOCAL_MAP, run_dir=LOCAL_THEMES.run_dir, source=LOCAL_THEMES.source,
        context=dataclasses.replace(LOCAL_MAP.context, send_back_repeats=True),
    ) == LOCAL_THEMES


LOCAL_R2, LOCAL_THEMES2_R2 = (CampaignConfig.from_file(CONFIGS / f"learn_daily_{arm}_r2.yaml") for arm in ("local", "local_themes2"))


@pytest.mark.parametrize(("second", "first"), [(LOCAL_R2, LOCAL), (LOCAL_THEMES2_R2, LOCAL_THEMES2)], ids=["local", "themes2"])
def test_the_second_run_of_an_arm_differs_from_the_first_only_in_its_run_directory_and_seed(second, first):
    assert second.run_dir == first.run_dir + "_r2"
    assert (first.round.seed, second.round.seed) == (0, 1)
    assert dataclasses.replace(
        second, run_dir=first.run_dir, source=first.source, round=dataclasses.replace(second.round, seed=0),
    ) == first
