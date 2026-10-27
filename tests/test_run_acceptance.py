"""Whether a run's accepted factors improve the user's model, as its Acceptance reports and hold-out lock say.

The campaign of ``tests/test_acceptance.py`` (one round accepting the
planted intraday and volume signals against a stored momentum) is judged in
a linear model with seeds 0 and 1 and both contribution modes. The run is
read before Acceptance, after validating every accepted factor and a
selection of one, after opening the hold-out on both and after opening it
again with force.
"""

import dataclasses
import json
import shutil

import pytest
from quantlab.factor.kunquant_compile import CACHE_ENV

from factor_agent.acceptance import COMPARED, Acceptance
from factor_agent.config import CampaignConfig
from factor_agent.web.run_view import RunView
from tests.test_acceptance import LINEAR, _campaign

PLANTED, VOLUME = "planted (accepted in round 1)", "volume (accepted in round 1)"


@pytest.fixture(scope="module")
def judged(tmp_path_factory):
    root = tmp_path_factory.mktemp("judged")
    config_path, _ = _campaign(root, scoring=True, model={"name": LINEAR})
    config = CampaignConfig.from_file(config_path)
    settings = dataclasses.replace(config.acceptance, seeds=(0, 1), contribution="both")
    views = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(root / "kunquant_cache"))
        acceptance = Acceptance(dataclasses.replace(config, acceptance=settings))
        views["unjudged"] = RunView.read(root / "run")
        reports = {"validation": acceptance.validate(), "selection": acceptance.validate(["planted"])}
        views["validated"] = RunView.read(root / "run")
        reports["holdout"] = acceptance.open_holdout(["planted", "volume"])
        views["opened"] = RunView.read(root / "run")
        reports["forced"] = acceptance.open_holdout(["planted", "volume"], force=True)
        views["forced"] = RunView.read(root / "run")
    return {"root": root, "views": views, "reports": reports}


def test_a_run_before_acceptance_has_no_comparison_and_no_hold_out(judged):
    acceptance = judged["views"]["unjudged"].acceptance
    assert acceptance.validation is None and acceptance.holdout is None
    assert acceptance.reports == () and acceptance.holdout_opens == ()


def test_the_validation_comparison_holds_the_seed_averaged_statistics_with_and_without_the_accepted_factors(judged):
    validation = judged["views"]["validated"].acceptance.validation
    report = judged["reports"]["validation"].to_dict()
    assert validation.segment == "validation" and validation.selection is False
    assert set(validation.factors) == {PLANTED, VOLUME}
    assert validation.seeds == (0, 1)
    for name in COMPARED:
        assert validation.baseline[name] == pytest.approx(report["mean"]["baseline"][name])
        assert validation.with_accepted[name] == pytest.approx(report["mean"]["with_accepted"][name])
        assert validation.difference[name] == pytest.approx(report["difference"][name])
    assert validation.forced is False


@pytest.mark.parametrize(("rank_ic", "top_k_excess", "verdict"), [
    (0.01, 0.002, "improves"),
    (0.01, -0.002, "mixed"),
    (-0.01, 0.002, "mixed"),
    (-0.01, 0.0, "does_not_improve"),
    (float("nan"), 0.002, None),
])
def test_the_verdict_says_whether_rank_ic_and_top_k_excess_both_rise_with_the_accepted_factors(
        judged, tmp_path, rank_ic, top_k_excess, verdict):
    shutil.copytree(judged["root"] / "run", tmp_path / "run")
    path = tmp_path / "run" / "acceptance" / "validation.json"
    report = json.loads(path.read_text())
    report["difference"].update(rank_ic=rank_ic, top_k_excess=top_k_excess)
    path.write_text(json.dumps(report))
    assert RunView.read(tmp_path / "run").acceptance.validation.verdict == verdict


def test_every_validation_report_is_listed_and_the_verdict_is_on_every_accepted_factor_not_a_selection(judged):
    acceptance = judged["views"]["validated"].acceptance
    listed = {r.name: (r.segment, r.selection, set(r.factors)) for r in acceptance.reports}
    assert listed.pop("validation") == ("validation", False, {PLANTED, VOLUME})
    assert list(listed.values()) == [("validation", True, {PLANTED})]
    assert acceptance.validation.name == "validation"


def test_each_accepted_factor_has_its_drop_one_and_add_one_contribution_and_the_seeds_agreeing(judged):
    view = judged["views"]["validated"]
    table = judged["reports"]["validation"].contribution_table()
    rows = {(row.factor, row.mode): row for row in view.acceptance.validation.contribution}
    assert sorted(rows) == sorted((f.label, mode) for f in view.factors for mode in ("drop_one", "add_one"))
    for expected in table:
        row = rows[expected["factor"], expected["mode"]]
        assert row.difference["rank_ic"] == pytest.approx(expected["difference"]["rank_ic"])
        assert row.agreeing["rank_ic"] == expected["agreeing"]["rank_ic"]
        assert 0 <= row.agreeing["rank_ic"] <= 2


def test_the_hold_out_appears_only_once_its_report_exists_with_every_open_of_the_lock(judged):
    assert judged["views"]["validated"].acceptance.holdout is None
    opened = judged["views"]["opened"].acceptance
    assert opened.holdout.segment == "holdout" and set(opened.holdout.factors) == {PLANTED, VOLUME}
    assert opened.holdout.forced is False and opened.holdout.contribution == ()
    assert opened.holdout.difference["rank_ic"] == pytest.approx(judged["reports"]["holdout"].difference["rank_ic"])
    assert [o.forced for o in opened.holdout_opens] == [False]
    lock = json.loads((judged["root"] / "run" / "holdout.lock").read_text())
    assert [o.opened_at for o in judged["views"]["forced"].acceptance.holdout_opens] == [o["opened_at"] for o in lock["opens"]]


def test_a_hold_out_opened_again_with_force_says_so(judged):
    forced = judged["views"]["forced"].acceptance
    assert forced.holdout.forced is True
    assert [o.forced for o in forced.holdout_opens] == [False, True]
    # The validation comparison is not a hold-out, whatever the lock holds.
    assert forced.validation.segment == "validation" and forced.validation.forced is False


def test_a_lock_without_a_hold_out_report_lists_its_opens_but_shows_no_hold_out(judged, tmp_path):
    shutil.copytree(judged["root"] / "run", tmp_path / "run")
    for path in (tmp_path / "run" / "acceptance").glob("holdout*.json"):
        path.unlink()
    acceptance = RunView.read(tmp_path / "run").acceptance
    assert acceptance.holdout is None and len(acceptance.holdout_opens) == 2


def test_a_report_of_an_older_version_without_contribution_or_selection_shows_what_it_holds(judged, tmp_path):
    shutil.copytree(judged["root"] / "run", tmp_path / "run")
    path = tmp_path / "run" / "acceptance" / "validation.json"
    report = json.loads(path.read_text())
    for key in ("contribution", "contributions", "selection"):
        report.pop(key)
    path.write_text(json.dumps(report))
    validation = RunView.read(tmp_path / "run").acceptance.validation
    assert validation.contribution is None and validation.selection is None
    assert set(validation.factors) == {PLANTED, VOLUME}
    assert validation.difference["rank_ic"] == pytest.approx(report["difference"]["rank_ic"])
