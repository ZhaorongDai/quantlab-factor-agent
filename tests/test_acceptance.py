"""Acceptance: the validation comparison with a tiny xgb, the purge, the hold-out lock and the export.

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

import numpy as np
import pytest
import xarray as xr
from KunQuant.Op import Output, Rank
from KunQuant.Stage import Function
from quantlab.core.component import rebuild
from quantlab.core.store_folder import load_component
from quantlab.factor.config import FactorConfig, NeutralizedConfig, PolarsFactorConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.kunquant_compile import CACHE_ENV
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from quantlab.runs.trained_run import TrainedRun

from factor_agent.acceptance import COMPARED, Acceptance
from factor_agent.config import CampaignConfig
from factor_agent.gate.graph import Graph
from factor_agent.segments import HoldoutLocked
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.vocabulary_files import price_volume
from tests.fake_llm import FakeLLM

SEGMENTS = {
    "training": {"start": synthetic_market.TRAINING[0], "end": synthetic_market.TRAINING[1]},
    "validation": {"start": "2012-01-02", "end": "2012-12-31"},
    "holdout": {"start": "2013-01-02", "end": "2013-12-31"},
}
TINY_XGB = {"training_target": "cs_rank", "num_boost_round": 20, "max_depth": 3, "eta": 0.3, "nthread": 4, "device": "cpu"}
CANDIDATES = [
    candidate("planted", "alpha = close / open"),
    candidate("volume", "alpha = volume / WindowedAvg(volume, 20)"),
]
PACKAGE = "exported_for_test"
#: The first bar of the momentum store, inside the training segment.
MOMENTUM_START = "2011-01-03"


def _campaign(root, scoring: bool):
    """A campaign of ``CANDIDATES``, scored as the default campaign or ranked only on every symbol."""
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
        acceptance={"hyperparameters": TINY_XGB, "top_k": 20, "factor_root": str(root / "factors"), "package": PACKAGE},
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
    first = acceptance.open_holdout()
    lock = json.loads(acceptance.lock_path.read_text())
    assert [o["forced"] for o in lock["opens"]] == [False]
    assert lock["opens"][0]["code_version"]["factor_agent"]
    assert not first.forced
    assert first.segment == "holdout" and first.train_window[1] < "2013-01-01" <= first.test_window[0]

    with pytest.raises(HoldoutLocked, match="opened on"):
        acceptance.open_holdout()
    assert json.loads(acceptance.lock_path.read_text()) == lock

    forced = acceptance.open_holdout(force=True)
    assert forced.forced
    assert [o["forced"] for o in forced.holdout_opens] == [False, True]
    assert "opened again with force" in (campaign["root"] / "run" / "acceptance" / "holdout.md").read_text()
    assert json.loads((campaign["root"] / "run" / "acceptance" / "holdout.json").read_text())["forced"] is True


class _Evaluated(FactorKunQuant):
    """A recorded graph ranked across the market, as the Evaluator compiles it."""

    graph: list = []

    def _get_factor_names(self) -> tuple[str, ...]:
        return ("evaluated",)

    def _get_factor_func(self) -> Function:
        output, ops = Graph.from_json(self.graph, price_volume().classes).build({})
        ranked = Rank(output)
        return Function([*ops, ranked, Output(ranked, "evaluated")])


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
    assert "Hypothesis behind volume." in readme and "alpha = volume / WindowedAvg(volume, 20)" in readme
    assert "accepted in round 1 of the factor-agent campaign" in readme
    assert json.loads((folder / "component.json").read_text())
    assert (folder / "volume_shock.zarr").exists()


def test_an_export_refuses_an_existing_name_and_an_unknown_factor(campaign, exported):
    with pytest.raises(ValueError, match="already exists"):
        campaign["acceptance"].export({"volume": "volume_shock"})
    with pytest.raises(ValueError, match="repeat"):
        campaign["acceptance"].export({"volume": "twice", "planted": "twice"})
    assert not list((campaign["root"] / "factors").glob("twice"))
    with pytest.raises(KeyError, match="names 0 accepted"):
        campaign["acceptance"].export({"never accepted": "nothing"})


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
