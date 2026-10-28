"""Acceptance of an intraday factor: validated and opened on the hold-out like any factor, exported and recomputed.

The campaign of ``test_intraday_family`` on ``synthetic_intraday_market``,
with its training segment shortened to leave a validation segment and a
hold-out inside the minute bars, accepts the late volume share against a
20-day momentum, the reference set and the baseline model's only feature.
The exported factor is recomputed from the minute bars, in this process and
in a fresh one.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from quantlab.core.component import rebuild
from quantlab.core.store_folder import load_component
from quantlab.factor.config import FactorConfig, PolarsFactorConfig
from quantlab.factor.kunquant_compile import CACHE_ENV

from factor_agent.acceptance import Acceptance
from factor_agent.config import CampaignConfig
from factor_agent.evaluator import Evaluator
from factor_agent.exported import GraphFactor
from factor_agent.gate.graph import Graph
from factor_agent.scoring import Scoring
from tests import synthetic_intraday_market as market
from tests.campaign_runner import XGB, run, write_config
from tests.fake_llm import FakeLLM
from tests.synthetic_market import AdjustedMomentum
from tests.test_intraday_family import DAILY_PRICE, LATE_SHARE
from tests.vocabulary_files import MINUTE_BARS, PRICE_VOLUME

SEGMENTS = {
    "training": {"start": market.TRAINING[0], "end": "2020-06-30"},
    "validation": {"start": "2020-07-06", "end": "2020-08-14"},
    "holdout": {"start": "2020-08-24", "end": "2020-09-30"},
}
TINY_XGB = {"name": XGB, "hyperparameters": {
    "training_target": "cs_rank", "num_boost_round": 20, "max_depth": 3, "eta": 0.3, "nthread": 4, "device": "cpu",
}}
PACKAGE = "exported_intraday_for_test"
CANDIDATES = [LATE_SHARE, DAILY_PRICE]


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("intraday_acceptance")
    components = market.build(root / "data")
    momentum = AdjustedMomentum(PolarsFactorConfig(
        warmup_bars=20, dataset=rebuild(components["prices"]),
        file_path=str(root / "data" / "factors" / "momentum" / "momentum.zarr"),
    ))
    momentum.build(market.START, market.END)
    (root / "packages" / PACKAGE).mkdir(parents=True)
    (root / "packages" / PACKAGE / "__init__.py").write_text("")
    config_path = write_config(
        root, {**components, "exposures": None}, [momentum.get_config()], segments=SEGMENTS,
        data={name: components[name] for name in ("prices", "label", "membership", "intraday")},
        scoring={"neutralize": None}, fields=[str(PRICE_VOLUME), str(MINUTE_BARS)],
        round={"candidates": len(CANDIDATES), "min_per_type": 0, "max_rounds": 1, "retries": 0},
        acceptance={"model": TINY_XGB, "top_k": 10, "factor_root": str(root / "factors"), "package": PACKAGE},
    )
    result = run(config_path, FakeLLM(lambda number, directions: CANDIDATES[:len(directions)]), root / "kunquant_cache")
    assert [r.candidate["description"] for r in result.accepted] == ["late volume share"]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(root / "kunquant_cache"))
        patch.syspath_prepend(str(root / "packages"))
        acceptance = Acceptance(CampaignConfig.from_file(config_path))
        [folder] = acceptance.export({"late volume share": "late_share"})
        yield {"root": root, "acceptance": acceptance, "record": result.accepted[0], "folder": folder}


def test_the_exported_intraday_factor_gives_the_values_the_evaluator_scored(campaign):
    acceptance, record = campaign["acceptance"], campaign["record"]
    config = acceptance.config
    evaluator = Evaluator.for_segment(
        config.training, prices=config.data.load("prices"), label=config.data.load("label"),
        scoring=Scoring.from_config(config.scoring, config.data, njobs=4),
        warmup_bars=config.evaluator.warmup_bars, njobs=4, vocabulary=acceptance.vocabulary,
        intraday=config.data.load("intraday"),
    )
    frame = evaluator._scored_frame()
    [evaluated] = evaluator._values([Graph.from_json(record.graph, acceptance.vocabulary.classes)], frame)
    factor = load_component(campaign["folder"])  # unneutralised: the store is the exported factor itself
    recomputed = factor.compute(config.training.start, config.training.end)["late_share"]
    on_members = np.where(frame.in_rows, frame.on_axes(recomputed), np.nan)
    assert np.isfinite(on_members).mean() > 0.9
    np.testing.assert_array_equal(on_members, np.where(frame.in_rows, evaluated.values, np.nan))


def test_two_candidates_whose_aggregations_share_a_canonical_hash_each_export_the_values_they_were_scored_on(campaign):
    # Adding 1e8 and taking it off again hashes as the volume itself but loses its digits in float32.
    config, vocabulary = campaign["acceptance"].config, campaign["acceptance"].vocabulary
    codes = ["alpha = WindowedAvg(SessionMean(bar_volume), 5)", "alpha = WindowedAvg(SessionMean(bar_volume + 1e8 - 1e8), 5)"]
    graphs = [Graph.record(vocabulary.execute(code)["alpha"], vocabulary) for code in codes]
    assert graphs[0].fingerprint() == graphs[1].fingerprint()
    scoring, intraday = Scoring.from_config(config.scoring, config.data, njobs=4), config.data.load("intraday")
    evaluator = Evaluator.for_segment(
        config.training, prices=config.data.load("prices"), label=config.data.load("label"), scoring=scoring,
        warmup_bars=config.evaluator.warmup_bars, njobs=4, vocabulary=vocabulary, intraday=intraday,
    )
    frame = evaluator._scored_frame()
    evaluated = evaluator._values(graphs, frame)
    exported = []
    for graph, values in zip(graphs, evaluated):
        daily, sessions = GraphFactor.staged({"f": graph.to_json()}, vocabulary)
        factor = GraphFactor(FactorConfig(
            warmup_bars=config.evaluator.warmup_bars, dataset=config.data.load("prices"), mode="batch", njobs=4,
            data_columns=GraphFactor.data_columns(daily, sessions=sessions), kwargs={
                "graphs": daily, "classes": vocabulary.class_paths([*daily.values(), *sessions.values()]),
                "sessions": sessions, "intraday": intraday.get_config(),
            },
        ))
        recomputed = scoring.apply(factor).compute(config.training.start, config.training.end)["f"]
        on_members = np.where(frame.in_rows, frame.on_axes(recomputed), np.nan)
        assert np.isfinite(on_members).mean() > 0.9
        np.testing.assert_array_equal(on_members, np.where(frame.in_rows, values.values, np.nan))
        exported.append(on_members)
    assert not np.array_equal(exported[0], exported[1], equal_nan=True)


def test_acceptance_validates_the_intraday_factor_and_opens_the_hold_out_on_it(campaign):
    acceptance = campaign["acceptance"]
    for report in (acceptance.validate(), acceptance.open_holdout(["late volume share"])):
        assert report.accepted == {"late volume share (accepted in round 1)": "accepted_0"}
        for result in (*report.baseline, *report.with_accepted):
            assert result.n_days > 15 and np.isfinite(result.rank_ic)
        # The late share is the planted signal, which the momentum baseline cannot see.
        assert report.difference["rank_ic"] > 0.05, report.segment


def test_the_exported_class_holds_the_staged_graph_and_the_intraday_dataset(campaign):
    exported = type(load_component(campaign["folder"]))
    record, vocabulary = campaign["record"], campaign["acceptance"].vocabulary
    assert exported.__module__ == f"{PACKAGE}.late_share"
    [graph] = exported.GRAPHS.values()
    sessions = Graph.from_json(graph, vocabulary.classes).input_names
    # The daily stage reads two session columns (the late volume and the session's volume), no bar column.
    assert len(sessions) == 2 and all(name.startswith("session_") for name in sessions)
    assert set(exported.SESSIONS) == set(sessions)
    assert {"SessionMean", "Select", "WindowedAvg"} <= set(exported.CLASSES)
    intraday = campaign["acceptance"].config.data.load("intraday").get_config()
    assert exported.INTRADAY == json.loads(json.dumps(intraday))
    assert record.graph not in exported.GRAPHS.values()
    assert "An intraday factor: its 2 session column(s)" in (campaign["folder"] / "README.md").read_text()


def test_the_exported_module_imports_and_computes_in_a_fresh_process(campaign, tmp_path):
    out = tmp_path / "values.npy"
    start, end = SEGMENTS["validation"]["start"], SEGMENTS["validation"]["end"]
    script = (
        "import sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(campaign['root'] / 'packages')!r})\n"
        "from quantlab.core.component import rebuild\n"
        "from quantlab.factor.config import FactorConfig\n"
        f"from {PACKAGE}.late_share import LateShare\n"
        f"prices = rebuild({campaign['acceptance'].config.data.prices!r})\n"
        "factor = LateShare(FactorConfig(\n"
        "    warmup_bars=60, dataset=prices, mode='batch', njobs=1,\n"
        "    data_columns=LateShare.data_columns(LateShare.GRAPHS, sessions=LateShare.SESSIONS),\n"
        "))\n"
        f"np.save({str(out)!r}, factor.compute({start!r}, {end!r})['late_share'].values)\n"
    )
    env = {**os.environ, CACHE_ENV: str(tmp_path / "kunquant_cache")}
    done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    values = np.load(out)
    assert np.isfinite(values).mean() > 0.9
    np.testing.assert_array_equal(values, load_component(campaign["folder"]).read(start, end)["late_share"].values)
