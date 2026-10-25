"""Acceptance: the validation comparison with a tiny xgb, the purge, the hold-out lock and the export.

The acceptance model is a template: a tiny xgb on the reference set for the
campaigns, and on a copy of the first campaign's run directory a linear model
and an ensemble of two.

A campaign on the synthetic store of ``tests/synthetic_market.py`` accepts
the planted intraday and volume signals against a stored Polars momentum,
the reference set and the baseline model's only feature; its store starts
inside the training segment, as the shared intraday stores start after
theirs. The segments put
training on ``TRAINING``, validation on 2012 and the hold-out on 2013.

A second campaign scores without neutralisation and without membership, on
the label of every symbol.
"""

import dataclasses
import json
import shutil

import numpy as np
import pytest
import xarray as xr
from KunQuant.Op import Output, Rank
from KunQuant.ops.ElewiseOp import AddConst, MulConst
from KunQuant.Stage import Function
from quantlab.core.component import rebuild
from quantlab.core.store_folder import load_component
from quantlab.factor.config import FactorConfig, NeutralizedConfig, PolarsFactorConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.kunquant_compile import CACHE_ENV
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from quantlab.runs.trained_run import TrainedRun

from factor_agent.acceptance import COMPARED, Acceptance, AcceptanceReport, Contribution, ModelResult
from factor_agent.config import CampaignConfig
from factor_agent.gate.graph import Graph
from factor_agent.segments import HoldoutLocked
from tests import synthetic_market
from tests.campaign_runner import XGB, candidate, run, write_config
from tests.vocabulary_files import price_volume
from tests.fake_llm import FakeLLM

SEGMENTS = {
    "training": {"start": synthetic_market.TRAINING[0], "end": synthetic_market.TRAINING[1]},
    "validation": {"start": "2012-01-02", "end": "2012-12-31"},
    "holdout": {"start": "2013-01-02", "end": "2013-12-31"},
}
#: A tiny xgb on the reference set, the acceptance model.
TINY_XGB = {"name": XGB, "hyperparameters": {
    "training_target": "cs_rank", "num_boost_round": 20, "max_depth": 3, "eta": 0.3, "nthread": 4, "device": "cpu",
}}
CANDIDATES = [
    candidate("planted", "alpha = close / open"),
    # A volume shock written backwards: a higher value goes with a lower return, so its direction is -1.
    candidate("volume", "alpha = WindowedAvg(volume, 20) / volume"),
]
PACKAGE = "exported_for_test"
#: The first bar of the momentum store, inside the training segment.
MOMENTUM_START = "2011-01-03"


def _campaign(root, scoring: bool, model: dict = TINY_XGB):
    """A campaign of ``CANDIDATES``, scored as the default campaign or ranked only on every symbol, accepted in ``model``."""
    components = synthetic_market.build(root / "data")
    momentum = synthetic_market.AdjustedMomentum(PolarsFactorConfig(
        warmup_bars=20, dataset=rebuild(components["prices"]),
        file_path=str(root / "data" / "factors" / "momentum" / "momentum.zarr"),
    ))
    momentum.build(MOMENTUM_START, synthetic_market.END)
    (root / "packages" / PACKAGE).mkdir(parents=True)
    (root / "packages" / PACKAGE / "__init__.py").write_text("")
    unscored = {} if scoring else {
        "data": {"prices": components["prices"], "label": components["market_label"]},
        "scoring": {"neutralize": None},
    }
    config_path = write_config(
        root, components, [momentum.get_config()], segments=SEGMENTS,
        round={"candidates": len(CANDIDATES), "min_per_type": 0, "max_rounds": 1},
        acceptance={"model": model, "top_k": 20, "factor_root": str(root / "factors"), "package": PACKAGE},
        **unscored,
    )
    result = run(config_path, FakeLLM(lambda number, directions: CANDIDATES), root / "kunquant_cache")
    return config_path, result


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("acceptance")
    config_path, result = _campaign(root, scoring=True)
    assert [r.candidate["description"] for r in result.accepted] == ["planted", "volume"]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(root / "kunquant_cache"))
        patch.syspath_prepend(str(root / "packages"))
        yield {"root": root, "acceptance": Acceptance(CampaignConfig.from_file(config_path)), "result": result}


@pytest.fixture(scope="module")
def validation(campaign):
    return campaign["acceptance"].validate()


def test_validate_compares_the_baseline_with_the_accepted_factors_for_three_seeds(campaign, validation):
    assert validation.segment == "validation"
    assert validation.seeds == (0, 1, 2)
    assert [r.seed for r in validation.baseline] == [r.seed for r in validation.with_accepted] == [0, 1, 2]
    assert set(validation.accepted) == {"planted (accepted in round 1)", "volume (accepted in round 1)"}
    for result in (*validation.baseline, *validation.with_accepted):
        assert result.n_days > 200
        assert all(np.isfinite(getattr(result, name)) for name in COMPARED)
        assert len(result.decile_returns) == 10
    assert set(validation.difference) == set(COMPARED)
    written = json.loads((campaign["root"] / "run" / "acceptance" / "validation.json").read_text())
    assert written["difference"] == pytest.approx(validation.difference)
    assert "| rank_ic |" in (campaign["root"] / "run" / "acceptance" / "validation.md").read_text()
    # contribution: none (the default) trains the two feature sets only and reports no contribution.
    assert validation.contributions == () and written["contribution"] == []
    assert _model_dirs(campaign["acceptance"], "validation") == [
        "baseline_s0", "baseline_s1", "baseline_s2", "with_accepted_s0", "with_accepted_s1", "with_accepted_s2",
    ]


def test_validate_on_a_selection_trains_only_it_and_keeps_its_report_apart(campaign, validation):
    out = campaign["root"] / "run" / "acceptance"
    whole = (out / "validation.json").read_text()
    report = campaign["acceptance"].validate(["planted"])
    assert report.accepted == {"planted (accepted in round 1)": "accepted_0"}
    for kind, added in (("baseline", []), ("with_accepted", ["accepted_0"])):
        for result in getattr(report, kind):
            assert TrainedRun.open(result.run).trained_on["factor_names"] == ["momentum_20", *added]
    [written] = [p for p in out.glob("validation_*.json")]
    assert json.loads(written.read_text())["accepted"] == report.accepted
    assert "planted (accepted in round 1)" in written.with_suffix(".md").read_text()
    assert (out / "validation.json").read_text() == whole
    # The same selection named by its label is the same selection, reported in the same file.
    assert campaign["acceptance"].validate(["planted (accepted in round 1)"]).accepted == report.accepted
    assert [p for p in out.glob("validation_*.json")] == [written]


def test_a_selection_naming_no_accepted_factor_or_one_twice_is_refused(campaign):
    with pytest.raises(KeyError, match="names 0 accepted"):
        campaign["acceptance"].validate(["never accepted"])
    with pytest.raises(ValueError, match="twice"):
        campaign["acceptance"].validate(["planted", "planted (accepted in round 1)"])


def test_a_signal_that_holds_only_on_the_training_segment_fails_validation(validation):
    # The planted intraday signal reverses after the training segment, so the model that learnt it ranks backwards.
    assert validation.difference["rank_ic"] < -0.1
    assert all(r.rank_ic < b.rank_ic for r, b in zip(validation.with_accepted, validation.baseline))


def test_the_purge_leaves_no_label_crossing_a_segment_boundary(campaign, validation):
    acceptance = campaign["acceptance"]
    segments = acceptance.config.segments
    calendar = acceptance.config.data.load("prices").calendar(synthetic_market.START, synthetic_market.END)
    lookahead = acceptance.config.data.load("label").lookahead_bars()
    for segment in (segments.training, segments.validation, segments.holdout):
        first, last = segments.scored(segment, calendar, lookahead)
        position = calendar.get_loc(last)
        assert calendar[position + lookahead] <= np.datetime64(segment.end)
        if position + lookahead + 1 < len(calendar):  # one more date would read past the segment
            assert calendar[position + lookahead + 1] > np.datetime64(segment.end)
        assert first >= segment.start
    # The model trained on labels inside the training segment and was scored on labels inside validation.
    for (start, end), segment in ((validation.train_window, segments.training), (validation.test_window, segments.validation)):
        assert calendar[calendar.get_loc(end) + lookahead] <= np.datetime64(segment.end)


def test_the_hold_out_opens_once_and_a_forced_reopen_is_stated(campaign, validation):
    acceptance = campaign["acceptance"]
    first = acceptance.open_holdout(["planted", "volume"])
    lock = json.loads(acceptance.lock_path.read_text())
    assert [o["forced"] for o in lock["opens"]] == [False]
    assert lock["opens"][0]["code_version"]["factor_agent"]
    assert not first.forced
    assert first.segment == "holdout" and first.train_window[1] < "2013-01-01" <= first.test_window[0]

    with pytest.raises(HoldoutLocked, match="opened on"):
        acceptance.open_holdout(["planted", "volume"])
    assert json.loads(acceptance.lock_path.read_text()) == lock

    forced = acceptance.open_holdout(["planted", "volume"], force=True)
    assert forced.forced
    assert [o["forced"] for o in forced.holdout_opens] == [False, True]
    assert "opened again with force" in (campaign["root"] / "run" / "acceptance" / "holdout.md").read_text()
    assert json.loads((campaign["root"] / "run" / "acceptance" / "holdout.json").read_text())["forced"] is True


LINEAR = "quantlab.model.predefined.linear.LinearRegressor"


def _in_model(campaign, tmp_path, model: dict, contribution: str = "none") -> Acceptance:
    """The campaign's Acceptance with ``model`` as the acceptance model and seeds 0 and 1, on a copy of its run directory."""
    run_dir = tmp_path / "run"
    shutil.copytree(campaign["root"] / "run", run_dir, ignore=shutil.ignore_patterns("acceptance", "holdout.lock"))
    config = campaign["acceptance"].config
    acceptance = dataclasses.replace(config.acceptance, model=model, seeds=(0, 1), contribution=contribution)
    return Acceptance(dataclasses.replace(config, acceptance=acceptance), run_dir)


def _model_dirs(acceptance: Acceptance, segment: str) -> list[str]:
    """The model directories Acceptance trained into on ``segment`` (its model cache aside)."""
    return sorted(p.name for p in (acceptance.run_dir / "acceptance" / segment).iterdir() if p.name != "cache")


def _factor_names(results) -> list[list[str]]:
    return [TrainedRun.open(r.run).trained_on["factor_names"] for r in results]


def test_the_acceptance_model_is_replaceable(campaign, tmp_path):
    report = _in_model(campaign, tmp_path, {"name": LINEAR}).validate()
    for kind, results in (("baseline", report.baseline), ("with_accepted", report.with_accepted)):
        assert [r.seed for r in results] == [0, 1]
        for result in results:
            run = TrainedRun.open(result.run)
            assert run.config["name"] == LINEAR and run.path.parent.name == f"{kind}_s{result.seed}"
            assert result.n_days > 200 and np.isfinite(result.rank_ic)


def test_every_member_of_an_ensemble_gets_its_baseline_the_accepted_factors_and_the_seed(campaign, tmp_path):
    template = {"name": "quantlab.model.predefined.model_ensemble.ModelEnsemble", "members": [
        {"name": LINEAR, "random_seed": 7},  # no factors: the reference set
        {"name": LINEAR, "factors": [campaign["acceptance"].config.reference[0]],
         "hyperparameters": {"estimator": "ridge", "alpha": 1.0}},
    ]}
    report = _in_model(campaign, tmp_path, template).validate()
    for kind, added in (("baseline", []), ("with_accepted", ["accepted_0", "accepted_1"])):
        for result in getattr(report, kind):
            run = TrainedRun.open(result.run)
            assert run.kind == "ensemble" and len(run.members) == 2
            for member in run.members:
                assert member.trained_on["factor_names"] == ["momentum_20", *added]
                assert member.config["random_seed"] == result.seed
            assert run.members[1].config["hyperparameters"] == {"estimator": "ridge", "alpha": 1.0}
            assert np.isfinite(result.rank_ic)


@pytest.fixture(scope="module")
def contributed(campaign, tmp_path_factory):
    """Both contribution modes in a linear model, validated."""
    acceptance = _in_model(campaign, tmp_path_factory.mktemp("contributed"), {"name": LINEAR}, contribution="both")
    return {"acceptance": acceptance, "report": acceptance.validate()}


def test_contribution_trains_one_feature_set_per_factor_and_mode_and_reports_a_row_each(contributed):
    acceptance, report = contributed["acceptance"], contributed["report"]
    planted, volume = "planted (accepted in round 1)", "volume (accepted in round 1)"
    table = report.contribution_table()
    assert sorted((row["factor"], row["mode"]) for row in table) == [
        (planted, "add_one"), (planted, "drop_one"), (volume, "add_one"), (volume, "drop_one"),
    ]
    differences = [row["difference"]["rank_ic"] for row in table]
    assert differences == sorted(differences, reverse=True)
    assert all(0 <= row["agreeing"][name] <= 2 for row in table for name in COMPARED)

    output = report.accepted
    for c in report.contributions:
        assert [r.seed for r in c.without] == [r.seed for r in c.with_factor] == [0, 1]
        other = [o for o in output.values() if o != output[c.factor]]
        if c.mode == "drop_one":
            assert _factor_names(c.with_factor) == [["momentum_20", *output.values()]] * 2
            assert _factor_names(c.without) == [["momentum_20", *other]] * 2
        else:
            assert _factor_names(c.without) == [["momentum_20"]] * 2
            assert _factor_names(c.with_factor) == [["momentum_20", output[c.factor]]] * 2
    # Two factors: dropping one is adding the other, so baseline, the selection and two more sets, every seed.
    assert len(_model_dirs(acceptance, "validation")) == 4 * 2
    assert "## Contribution" in (acceptance.run_dir / "acceptance" / "validation.md").read_text()
    assert len(json.loads((acceptance.run_dir / "acceptance" / "validation.json").read_text())["contribution"]) == 4


def test_the_hold_out_runs_no_contribution_whatever_the_config_says(contributed):
    acceptance = contributed["acceptance"]
    report = acceptance.open_holdout(["planted", "volume"])
    assert report.contributions == ()
    assert _model_dirs(acceptance, "holdout") == ["baseline_s0", "baseline_s1", "with_accepted_s0", "with_accepted_s1"]
    assert "Contribution" not in (acceptance.run_dir / "acceptance" / "holdout.md").read_text()


def test_contribution_runs_over_the_selection_and_a_one_factor_selection_reuses_its_two_sets(campaign, tmp_path):
    acceptance = _in_model(campaign, tmp_path, {"name": LINEAR}, contribution="both")
    report = acceptance.validate(["planted"])
    assert [(c.factor, c.mode) for c in report.contributions] == [
        ("planted (accepted in round 1)", "drop_one"), ("planted (accepted in round 1)", "add_one"),
    ]
    [drop, add] = report.contributions
    assert drop.without == add.without == report.baseline
    assert drop.with_factor == add.with_factor == report.with_accepted
    assert _factor_names(report.with_accepted) == [["momentum_20", "accepted_0"]] * 2
    [folder] = (acceptance.run_dir / "acceptance").glob("validation_*/")
    assert _model_dirs(acceptance, folder.name) == ["baseline_s0", "baseline_s1", "with_accepted_s0", "with_accepted_s1"]


def test_the_hold_out_opens_only_on_a_validated_selection_and_compares_it_alone(campaign, tmp_path):
    acceptance = _in_model(campaign, tmp_path, {"name": LINEAR})
    acceptance.validate(["planted"])
    for unvalidated in (["volume"], ["planted", "volume"]):
        with pytest.raises(ValueError, match="run validate"):
            acceptance.open_holdout(unvalidated)
    with pytest.raises(KeyError, match="names 0 accepted"):
        acceptance.open_holdout(["never accepted"])
    assert not acceptance.lock_path.exists()

    report = acceptance.open_holdout(["planted"])
    assert report.segment == "holdout" and report.selection
    assert report.accepted == {"planted (accepted in round 1)": "accepted_0"}
    for kind, added in (("baseline", []), ("with_accepted", ["accepted_0"])):
        for result in getattr(report, kind):
            assert TrainedRun.open(result.run).trained_on["factor_names"] == ["momentum_20", *added]
    out = acceptance.run_dir / "acceptance"
    [written] = out.glob("holdout_*.json")
    assert written.stem.removeprefix("holdout") == next(out.glob("validation_*.json")).stem.removeprefix("validation")
    assert not (out / "holdout.json").exists()

    # One open of any selection locks the hold-out; a forced one opens another selection and says so.
    acceptance.validate()
    with pytest.raises(HoldoutLocked):
        acceptance.open_holdout(["planted", "volume"])
    forced = acceptance.open_holdout(["volume (accepted in round 1)", "planted"], force=True)
    assert forced.forced and not forced.selection and set(forced.accepted) == {
        "planted (accepted in round 1)", "volume (accepted in round 1)",
    }
    assert [o["forced"] for o in json.loads(acceptance.lock_path.read_text())["opens"]] == [False, True]
    assert (out / "holdout.json").exists()


class _Evaluated(FactorKunQuant):
    """A recorded graph ranked across the market, as the Evaluator compiles it, turned to its direction.

    The volume shock's direction is -1, so the exported factor is one minus its rank.
    """

    graph: list = []

    def _get_factor_names(self) -> tuple[str, ...]:
        return ("evaluated",)

    def _get_factor_func(self) -> Function:
        output, ops = Graph.from_json(self.graph, price_volume().classes).build({})
        ranked = Rank(output)
        flipped = MulConst(ranked, -1.0)
        turned = AddConst(flipped, 1.0)
        return Function([*ops, ranked, flipped, turned, Output(turned, "evaluated")])


@pytest.fixture(scope="module")
def exported(campaign):
    folders = campaign["acceptance"].export({"volume": "volume_shock"})
    record = next(r for r in campaign["result"].accepted if r.candidate["description"] == "volume")
    return {"folder": folders[0], "record": record}


def test_an_exported_factor_equals_the_evaluated_graph_bit_for_bit(campaign, exported):
    store = load_component(exported["folder"])
    factor = store.config.factor
    assert isinstance(factor, FactorKunQuant)
    assert type(factor).__module__ == f"{PACKAGE}.volume_shock"
    _Evaluated.graph = exported["record"].graph
    evaluated = _Evaluated(FactorConfig(
        warmup_bars=60, dataset=factor.config.dataset, mode="batch", data_columns=("adjVolume",), njobs=4,
    ))
    window = synthetic_market.TRAINING
    ours, theirs = factor.compute(*window)["volume_shock"].values, evaluated.compute(*window)["evaluated"].values
    assert np.isfinite(ours).mean() > 0.9
    assert np.array_equal(ours, theirs, equal_nan=True)

    neutral = NeutralizedFactor(NeutralizedConfig(factor=evaluated, dataset=store.config.dataset, njobs=4))
    stored = store.read(*window)["volume_shock"].values
    assert np.array_equal(stored, neutral.compute(*window)["evaluated"].values, equal_nan=True)


def test_the_exported_store_has_its_readme_and_component(exported):
    folder = exported["folder"]
    assert folder.name == "volume_shock"
    readme = (folder / "README.md").read_text()
    assert "Hypothesis behind volume." in readme and "alpha = WindowedAvg(volume, 20) / volume" in readme
    assert "direction -1 (a higher value went with a lower return); the factor is turned to it" in readme
    assert "accepted in round 1 of the factor-agent campaign" in readme
    component = (folder / "component.json").read_text()
    assert '"directions"' in component and '"volume_shock": -1' in component and '"training_ic"' in component
    assert (folder / "volume_shock.zarr").exists()


def test_an_export_refuses_an_existing_name_and_an_unknown_factor(campaign, exported):
    with pytest.raises(ValueError, match="already exists"):
        campaign["acceptance"].export({"volume": "volume_shock"})
    with pytest.raises(ValueError, match="repeat"):
        campaign["acceptance"].export({"volume": "twice", "planted": "twice"})
    assert not list((campaign["root"] / "factors").glob("twice"))
    with pytest.raises(KeyError, match="names 0 accepted"):
        campaign["acceptance"].export({"never accepted": "nothing"})


def test_an_export_without_a_factor_root_fails_and_writes_nothing(campaign):
    config = campaign["acceptance"].config
    config = dataclasses.replace(config, acceptance=dataclasses.replace(config.acceptance, factor_root=None))
    package_dir = campaign["root"] / "packages" / PACKAGE
    before = sorted(package_dir.iterdir())
    with pytest.raises(ValueError, match="acceptance.factor_root"):
        Acceptance(config).export({"volume": "no_root"})
    assert sorted(package_dir.iterdir()) == before


@pytest.fixture(scope="module")
def unscored(tmp_path_factory):
    """The campaign ranked only, on every symbol; it exports into its own package directory."""
    root = tmp_path_factory.mktemp("unscored")
    config_path, result = _campaign(root, scoring=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(root / "kunquant_cache"))
        patch.syspath_prepend(str(root / "packages"))
        acceptance = Acceptance(CampaignConfig.from_file(config_path))
        yield {"root": root, "acceptance": acceptance, "result": result}


def test_without_membership_the_signal_that_holds_on_members_only_is_not_accepted(unscored):
    # The planted intraday signal reverses off the members, so on every symbol it says nothing.
    assert [r.candidate["description"] for r in unscored["result"].accepted] == ["volume"]


def test_without_membership_acceptance_predicts_every_symbol(unscored):
    report = unscored["acceptance"].validate()
    late = f"S{synthetic_market.MEMBERS:03d}"  # a member only from LATE_JOIN, after validation
    for result in (*report.baseline, *report.with_accepted):
        assert result.n_days > 200 and np.isfinite(result.rank_ic)
        predictions = xr.open_zarr(TrainedRun.open(result.run).test_predictions)
        assert np.isfinite(predictions.sel(symbol=late).to_array().values).mean() > 0.9


def test_without_neutralisation_the_exported_store_holds_the_ranked_graph(unscored):
    [folder] = unscored["acceptance"].export({"volume": "volume_ranked"})
    store = load_component(folder)
    record = unscored["result"].accepted[0]
    _Evaluated.graph = record.graph
    evaluated = _Evaluated(FactorConfig(
        warmup_bars=60, dataset=store.config.dataset, mode="batch", data_columns=("adjVolume",), njobs=4,
    ))
    window = synthetic_market.TRAINING
    stored = store.read(*window)["volume_ranked"].values
    assert np.isfinite(stored).mean() > 0.9
    assert np.array_equal(stored, evaluated.compute(*window)["evaluated"].values, equal_nan=True)
    assert "not neutralised" in (folder / "README.md").read_text()


def test_acceptance_and_export_use_the_run_snapshot_of_the_vocabulary(unscored, tmp_path):
    # The config's field and operator files are gone; the run's snapshot still holds the ones it scored with.
    gone = str(tmp_path / "gone.yaml")
    config = dataclasses.replace(unscored["acceptance"].config, fields=(gone,), operators=gone)
    acceptance = Acceptance(config)
    assert acceptance.vocabulary.classes == unscored["acceptance"].vocabulary.classes
    [folder] = acceptance.export({"volume": "volume_from_snapshot"})
    stored = load_component(folder).read(*synthetic_market.TRAINING)["volume_from_snapshot"].values
    assert np.isfinite(stored).mean() > 0.9


def test_acceptance_of_a_run_directory_without_a_snapshot_fails(campaign, tmp_path):
    with pytest.raises(FileNotFoundError, match="no snapshot"):
        Acceptance(campaign["acceptance"].config, run_dir=tmp_path / "never_run")


def _seeded(rank_ics: list[float], top_k_excesses: list[float]) -> tuple[ModelResult, ...]:
    """Results of seeds 0, 1, 2 with these rank ICs and TopK excesses; every other statistic is the rank IC."""
    return tuple(
        ModelResult(seed=seed, rank_ic=r, ic=r, monotonicity=r, long_short=r, top_k_return=r, top_k_excess=x,
                    top_k_annualised=r, decile_returns=(0.0,) * 10, n_days=100, run="")
        for seed, (r, x) in enumerate(zip(rank_ics, top_k_excesses))
    )


def _report(contributions: tuple) -> AcceptanceReport:
    flat = _seeded([0.05] * 3, [0.001] * 3)
    return AcceptanceReport(
        segment="validation", train_window=("2010-01-04", "2011-12-20"), test_window=("2012-01-03", "2012-12-20"),
        seeds=(0, 1, 2), top_k=20, accepted={"a (accepted in round 1)": "accepted_0", "b (accepted in round 2)": "accepted_1"},
        baseline=flat, with_accepted=flat, contributions=contributions,
    )


def test_the_contribution_table_holds_the_seed_mean_differences_and_seeds_agreeing_sorted_by_rank_ic():
    report = _report((
        Contribution("a (accepted in round 1)", "drop_one",
                     without=_seeded([0.05] * 3, [0.001] * 3), with_factor=_seeded([0.08, 0.06, 0.04], [0.003, 0.002, 0.004])),
        Contribution("b (accepted in round 2)", "drop_one",
                     without=_seeded([0.05] * 3, [0.001] * 3), with_factor=_seeded([0.04, 0.05, 0.03], [0.0] * 3)),
        Contribution("a (accepted in round 1)", "add_one",
                     without=_seeded([0.02] * 3, [0.0] * 3), with_factor=_seeded([0.05, 0.04, 0.06], [-0.001, -0.002, 0.0])),
    ))
    table = report.contribution_table()
    assert [(row["factor"], row["mode"]) for row in table] == [
        ("a (accepted in round 1)", "add_one"), ("a (accepted in round 1)", "drop_one"), ("b (accepted in round 2)", "drop_one"),
    ]
    assert [row["difference"]["rank_ic"] for row in table] == pytest.approx([0.03, 0.01, -0.01])
    assert [row["difference"]["top_k_excess"] for row in table] == pytest.approx([-0.001, 0.002, -0.001])
    assert [row["agreeing"]["rank_ic"] for row in table] == [3, 2, 2]
    assert [row["agreeing"]["top_k_excess"] for row in table] == [2, 3, 3]
    assert set(table[0]["difference"]) == set(table[0]["agreeing"]) == set(COMPARED)

    written = json.loads(json.dumps(report.to_dict()))
    assert [(row["factor"], row["mode"]) for row in written["contribution"]] == [
        (row["factor"], row["mode"]) for row in table
    ]
    assert written["contribution"][1]["difference"]["monotonicity"] == pytest.approx(0.01)
    assert written["contribution"][1]["agreeing"]["monotonicity"] == 2

    markdown = report.markdown()
    assert markdown.index("| rank_ic |") < markdown.index("## Contribution")
    assert (
        "| factor | mode | rank IC difference | seeds agreeing | TopK excess difference |\n|---|---|---|---|---|\n"
        "| a (accepted in round 1) | add_one | +0.0300 | 3/3 | -0.0010 |\n"
        "| a (accepted in round 1) | drop_one | +0.0100 | 2/3 | +0.0020 |\n"
        "| b (accepted in round 2) | drop_one | -0.0100 | 2/3 | -0.0010 |\n"
    ) in markdown


def test_a_report_without_contribution_has_no_contribution_section():
    report = _report(())
    assert report.contribution_table() == [] and report.to_dict()["contribution"] == []
    assert "Contribution" not in report.markdown()


def _trained(acceptance: Acceptance, segment: str = "validation") -> set:
    """The models trained so far on ``segment``: the run directories under its acceptance directory."""
    return set((acceptance.run_dir / "acceptance" / segment).glob("*/*_trial_*"))


def test_a_second_validate_with_nothing_changed_trains_no_model_and_gives_the_same_report(campaign, tmp_path, capsys):
    acceptance = _in_model(campaign, tmp_path, {"name": LINEAR})
    first = acceptance.validate()
    trained = _trained(acceptance)
    assert len(trained) == 4
    out = capsys.readouterr().out
    assert "acceptance (validation): 4 models to train, 0 cached\n" in out  # no finished model, no estimate

    again = acceptance.validate()
    assert _trained(acceptance) == trained
    assert again == dataclasses.replace(first, code_version=again.code_version)
    assert "acceptance (validation): 0 models to train, 4 cached\n" in capsys.readouterr().out


def test_a_validate_interrupted_partway_trains_only_the_unfinished_models_when_run_again(
    campaign, tmp_path, capsys, monkeypatch,
):
    from quantlab.model.predefined.linear import LinearRegressor

    acceptance, train = _in_model(campaign, tmp_path, {"name": LINEAR}), LinearRegressor.train
    calls = []

    def interrupted(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return train(self, *args, **kwargs)

    monkeypatch.setattr(LinearRegressor, "train", interrupted)
    with pytest.raises(KeyboardInterrupt):
        acceptance.validate()
    finished = _trained(acceptance)
    assert len(finished) == 2
    capsys.readouterr()

    monkeypatch.setattr(LinearRegressor, "train", train)
    report = acceptance.validate()
    assert len(_trained(acceptance)) == 4 and finished < _trained(acceptance)
    assert "2 models to train, 2 cached, about " in capsys.readouterr().out
    assert {r.run for r in (*report.baseline, *report.with_accepted)} >= {str(p) for p in finished}


def _changed(acceptance: Acceptance, **settings) -> Acceptance:
    """``acceptance`` on the same run directory, with ``settings`` replacing the config's (``segments`` or acceptance fields)."""
    config = acceptance.config
    segments = settings.pop("segments", config.segments)
    changed = dataclasses.replace(config, segments=segments, acceptance=dataclasses.replace(config.acceptance, **settings))
    return Acceptance(changed, acceptance.run_dir)


def test_a_changed_template_seed_or_segment_misses_the_cache_for_the_models_it_changes(campaign, tmp_path):
    acceptance = _in_model(campaign, tmp_path, {"name": LINEAR})
    acceptance.validate()
    trained = _trained(acceptance)

    _changed(acceptance, seeds=(0, 1, 2)).validate()  # one more seed: its baseline and its model with the factors
    assert len(_trained(acceptance) - trained) == 2
    trained = _trained(acceptance)

    _changed(acceptance, model={"name": LINEAR, "hyperparameters": {"estimator": "ridge", "alpha": 1.0}}).validate()
    assert len(_trained(acceptance) - trained) == 4
    trained = _trained(acceptance)

    segments = acceptance.config.segments
    shorter = dataclasses.replace(segments, validation=dataclasses.replace(segments.validation, end="2012-11-30"))
    _changed(acceptance, segments=shorter).validate()
    assert len(_trained(acceptance) - trained) == 4
    trained = _trained(acceptance)

    acceptance.validate()  # back to the first settings: every model is cached
    assert _trained(acceptance) == trained


def _trained_anywhere(acceptance: Acceptance) -> set:
    """The models trained so far on any segment and selection."""
    return set((acceptance.run_dir / "acceptance").glob("*/*/*_trial_*"))


def test_a_narrowed_selection_reads_the_whole_sets_models_of_the_same_features(campaign, tmp_path, capsys):
    acceptance = _in_model(campaign, tmp_path, {"name": LINEAR}, contribution="add_one")
    acceptance.validate()
    trained = _trained_anywhere(acceptance)
    assert len(trained) == 4 * 2
    capsys.readouterr()

    # The selection's model with the factor and its add-one are the whole set's add-one: nothing new to train.
    report = acceptance.validate(["planted"])
    assert _trained_anywhere(acceptance) == trained
    assert "acceptance (validation): 0 models to train, 4 cached\n" in capsys.readouterr().out
    assert report.accepted == {"planted (accepted in round 1)": "accepted_0"}
    assert _factor_names(report.with_accepted) == [["momentum_20", "accepted_0"]] * 2

    # The selection of the other factor keeps its output name from the whole list.
    other = acceptance.validate(["volume"])
    assert other.accepted == {"volume (accepted in round 1)": "accepted_1"}
    assert _trained_anywhere(acceptance) == trained


def test_a_moved_run_directory_keeps_its_cache_and_a_code_change_misses_it(campaign, tmp_path, capsys, monkeypatch):
    import factor_agent.acceptance

    acceptance = _in_model(campaign, tmp_path / "first", {"name": LINEAR})
    acceptance.validate()
    moved = tmp_path / "moved"
    shutil.copytree(acceptance.run_dir, moved)
    capsys.readouterr()

    Acceptance(acceptance.config, moved).validate()
    assert "acceptance (validation): 0 models to train, 4 cached\n" in capsys.readouterr().out

    commits = factor_agent.acceptance.code_version()
    monkeypatch.setattr(factor_agent.acceptance, "code_version", lambda: {**commits, "factor_agent": "another"})
    Acceptance(acceptance.config, moved).validate()
    assert "acceptance (validation): 4 models to train, 0 cached" in capsys.readouterr().out


def test_the_duration_estimate_reads_only_finished_models_of_the_same_template(campaign, tmp_path, capsys):
    acceptance = _in_model(campaign, tmp_path, {"name": LINEAR})
    acceptance.validate()
    capsys.readouterr()

    ridge = {"name": LINEAR, "hyperparameters": {"estimator": "ridge", "alpha": 1.0}}
    _changed(acceptance, model=ridge).validate()  # the linear models say nothing of how long a ridge takes
    assert "acceptance (validation): 4 models to train, 0 cached\n" in capsys.readouterr().out
    _changed(acceptance, model=ridge, seeds=(0, 1, 2)).validate()
    out = capsys.readouterr().out
    assert "acceptance (validation): 2 models to train, 4 cached, about " in out and " a model over 4 finished)" in out
