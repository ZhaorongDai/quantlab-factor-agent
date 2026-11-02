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
