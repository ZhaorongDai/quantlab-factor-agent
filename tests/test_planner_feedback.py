"""What the planner reads of the refused trials: their reasons in the recent history, and the near misses before it.

A near miss is an evaluated trial before the recent history that admission refused and that fails exactly
one admission check on its training statistics, all of them checked again; the confirmation does not
count. The trials are written by hand, each a negative factor (lower value, higher return).
"""

from factor_agent._feedback import history, named, near_misses
from factor_agent.config import AdmissionConfig
from factor_agent.evaluator import Metrics
from factor_agent.ledger import TrialRecord

#: Thresholds with the t check off: a training IC of 0.005, a partial IC of 0.002, 3 of 4 full years.
ADMISSION = AdmissionConfig(t_threshold=None, min_ic=0.005, min_partial_ic=0.002, min_year_share=0.75)

#: Statistics every check of ``ADMISSION`` passes.
GOOD = dict(
    ic_mean=-0.008, ic_t=-6.0, partial_ic_mean=-0.003, partial_ic_t=-5.0, nearest="x", nearest_correlation=0.1,
    nearest_known="y", nearest_known_correlation=-0.2, decile_returns=tuple(range(10, 0, -1)), coverage=0.9,
    n_days=1000, start="s", end="e", years_in_direction=4, full_years=4,
)


def trial(number, description, status="rejected", reasons=(), parent="", **change) -> TrialRecord:
    """An evaluated trial of round ``number`` with ``GOOD``'s statistics but ``change``."""
    return TrialRecord(
        round=number, candidate={"description": description, "hypothesis": f"why {description}"}, status=status,
        stage="admission", reasons=list(reasons), metrics=Metrics(**{**GOOD, **change}).to_dict(),
        direction={"mechanism_type": "earnings_quality", "action": "refinement" if parent else "new",
                   "parent": parent, "direction": ""},
    )


PARTIAL = "the training partial IC 0.0014 (residualised ...) in the factor's direction is below min_partial_ic 0.002"


def test_with_reasons_each_line_of_the_recent_history_ends_with_why_its_trial_was_refused():
    trials = named([trial(1, "thin", reasons=[PARTIAL], partial_ic_mean=-0.0014), trial(1, "kept", status="accepted")])
    lines = history(trials, reasons=True).splitlines()
    assert lines[0].endswith(f"; refused: {PARTIAL}")
    assert "refused:" not in lines[1]
    assert "refused:" not in history(trials)


def test_the_near_misses_are_the_refused_trials_before_the_recent_history_one_check_short_nearest_first():
    trials = named([
        trial(1, "a little thin", partial_ic_mean=-0.0018),                      # T1-01: a tenth short
        trial(1, "thin", partial_ic_mean=-0.0012),                               # T1-02: two fifths short
        trial(1, "thin and unsteady", partial_ic_mean=-0.0012, years_in_direction=2),   # T1-03: two checks
        trial(1, "accepted", status="accepted"),                                 # T1-04
        trial(1, "refused at the confirmation only"),                           # T1-05: no shortfall
        trial(1, "no direction", ic_mean=0.0),                                   # T1-06
        trial(2, "too small", ic_mean=-0.004),                                  # T2-01: a fifth short
        trial(2, "built on", partial_ic_mean=-0.0019),                          # T2-02: has a child
        trial(3, "recent", partial_ic_mean=-0.0019),                            # T3-01: in the recent history
        trial(3, "child", parent="T2-02"),                                      # T3-02
    ])
    shown = near_misses(trials, before_round=3, admission=ADMISSION, trial_count=10, n=10)
    assert [line.split(" [")[0] for line in shown.splitlines() if line[:1].isdigit()] == [
        "1. T1-01", "2. T2-01", "3. T1-02",
    ]
    assert "hypothesis: why a little thin" in shown and "residual IC -0.0018" in shown
    assert [line.split(" [")[0] for line in near_misses(trials, 3, ADMISSION, 10, n=2).splitlines()
            if line[:1].isdigit()] == ["1. T1-01", "2. T2-01"]
    assert near_misses(trials, before_round=1, admission=ADMISSION, trial_count=10, n=10) == ""
