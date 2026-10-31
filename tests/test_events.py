"""Reading the events file."""

import json

from factor_agent import events

_START = {"schema": events.SCHEMA, "type": "start", "time": "2026-10-29T00:00:00+00:00", "pid": 1, "host": "h",
          "versions": {}, "rounds": 0, "candidates": 20, "max_rounds": 30, "max_trials": 600, "patience": 5,
          "t_threshold": 3.0, "max_correlation": 0.7, "min_coverage": 0.5, "min_monotonicity": 0.6}


def test_a_start_written_before_the_trial_exponent_reads_as_bonferroni(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(_START) + "\n")
    [start] = events.read(path)
    assert start.trial_exponent == 1.0


def test_a_start_reads_back_the_trial_exponent_it_was_written_with(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(_START | {"trial_exponent": 0.5}) + "\n")
    [start] = events.read(path)
    assert start.trial_exponent == 0.5
