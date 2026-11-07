"""CampaignConfig.from_file: components given as a store folder with chosen outputs, and the real config."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from quantlab.core.store_folder import save_component
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.alpha158 import Alpha158Stock

from factor_agent.config import DEFAULT_MECHANISMS, ROLES, CampaignConfig, NeutralizeConfig, TrialGraphConfig, load
from factor_agent.gate.vocabulary import VocabularyError
from factor_agent.prompts import PromptError
from tests.campaign_runner import XGB
from tests.vocabulary_files import CONFIGS, EXAMPLE_OPERATORS, PRICE_VOLUME

FUNDAMENTALS = CONFIGS / "fields" / "sharadar_fundamentals.yaml"


@pytest.fixture
def store(tmp_path):
    """A folder whose ``component.json`` records an Alpha158Stock of every output."""
    xr.Dataset(
        {"Close": (["timestamp", "symbol"], np.ones((3, 2)))},
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=3), "symbol": ["A", "B"]},
    ).to_zarr(tmp_path / "prices.zarr")
    dataset = StockDataset(DatasetConfig(
        zarr_file_path=str(tmp_path / "prices.zarr"), raw_data_dir_path=str(tmp_path), market="us_equity", frequency="1d",
    ))
    factor = Alpha158Stock(FactorConfig(
        warmup_bars=60, dataset=dataset, mode="batch", data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"),
        file_path=str(tmp_path / "alpha158" / "alpha158.zarr"),
    ))
    save_component(factor, tmp_path / "alpha158")
    return tmp_path / "alpha158"


def test_a_store_entry_is_the_stores_factor_pinned_to_the_named_outputs(store):
    factor = load({"store": str(store), "factor_names": ["KMID", "STD20"]})
    assert isinstance(factor, Alpha158Stock)
    assert factor.get_factor_names() == ("KMID", "STD20")
    assert factor.config.file_path == str(store / "alpha158.zarr")


def test_a_store_entry_without_names_is_the_whole_store(store):
    assert len(load({"store": str(store)}).get_factor_names()) == 169


def test_a_store_entry_naming_an_output_the_store_lacks_is_refused(store):
    with pytest.raises(ValueError, match="NOPE"):
        load({"store": str(store), "factor_names": ["KMID", "NOPE"]})


def test_the_reference_of_a_config_file_may_be_store_entries(store, tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({
        "run_dir": str(tmp_path / "run"),
        "segments": {"training": {"start": "2024-01-01", "end": "2024-01-03"}},
        "data": {"prices": "p", "label": "l", "membership": "m"},
        "scoring": {"neutralize": {"exposures": "e", "size_column": "marketcap", "industry_column": "industry"}},
        "reference": [{"store": str(store), "factor_names": ["KMID"]}],
        "fields": [str(PRICE_VOLUME)], "operators": str(EXAMPLE_OPERATORS), "market": "test bars",
        "llm": {role: {"backend": "scripted", "model": "m", "replies": "r.json"} for role in ROLES},
        "acceptance": {"model": {"name": XGB}},
    }))
    [factor] = [load(c) for c in CampaignConfig.from_file(path).reference]
    assert factor.get_factor_names() == ("KMID",)


def _store_entry(store) -> dict:
    return {"store": str(store), "factor_names": ["KMID", "STD20"]}


def _component(store) -> dict:
    """The store's factor pinned to KMID and STD20, as a ``get_config()`` dict read back from JSON."""
    return json.loads(json.dumps(load(_store_entry(store)).get_config()))


@pytest.mark.parametrize(("entry", "names"), [
    (lambda store: str(store), 169),
    (_store_entry, ("KMID", "STD20")),
    (_component, ("KMID", "STD20")),
])
def test_the_known_factors_may_be_given_in_each_entry_form(store, tmp_path, entry, names):
    config = CampaignConfig.from_file(_minimal(tmp_path, known=[entry(store)]))
    reference, [known] = config.load_factors()
    assert reference == [] and isinstance(known, Alpha158Stock)
    assert (len(known.get_factor_names()) if isinstance(names, int) else known.get_factor_names()) == names


@pytest.mark.parametrize("entry", [
    42,
    "",
    {"store": "s", "names": ["KMID"]},
    {"store": "s", "factor_names": "KMID"},
    {"store": 3},
    {"factor_names": ["KMID"]},
])
def test_a_malformed_known_entry_is_refused_when_the_config_is_read(tmp_path, entry):
    with pytest.raises(TypeError, match=r"known\[1\]"):
        CampaignConfig.from_file(_minimal(tmp_path, known=["folder", entry]))


def test_a_config_without_known_factors_has_none(tmp_path):
    config = CampaignConfig.from_file(_minimal(tmp_path))
    assert config.known == () and config.load_factors() == ([], [])


def _minimal(tmp_path, **sections) -> Path:
    """A config file of placeholder components, with ``sections`` replacing or, as None, removing its sections."""
    raw = {
        "run_dir": str(tmp_path / "run"),
        "segments": {"training": {"start": "2024-01-01", "end": "2024-01-03"}},
        "data": {"prices": "p", "label": "l"},
        "scoring": {"neutralize": None},
        "fields": [str(PRICE_VOLUME)], "operators": str(EXAMPLE_OPERATORS), "market": "test bars",
        "llm": {role: {"backend": "scripted", "model": "m", "replies": "r.json"} for role in ROLES},
        "acceptance": {"model": {"name": XGB}},
    }
    raw.update(sections)
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in raw.items() if v is not None}))
    return path


def test_a_campaign_may_score_without_neutralisation_and_without_membership(tmp_path):
    config = CampaignConfig.from_file(_minimal(tmp_path))
    assert config.scoring.neutralize is None
    assert config.data.membership is None


def test_neutralisation_names_its_exposures_and_their_columns(tmp_path):
    neutralize = {"exposures": ["d", "i"], "size_column": "cap", "industry_column": "sector"}
    config = CampaignConfig.from_file(_minimal(tmp_path, scoring={"neutralize": neutralize}))
    assert config.scoring.neutralize == NeutralizeConfig(exposures=["d", "i"], size_column="cap", industry_column="sector")


@pytest.mark.parametrize("scoring, error", [
    (None, KeyError),  # no scoring section: nothing is scored one way by default
    ({}, KeyError),  # neutralize must be given, as null to turn it off
    ({"neutralize": None, "membership": None}, TypeError),
    ({"neutralize": {"exposures": "e", "size_column": "marketcap"}}, TypeError),
])
def test_a_scoring_section_that_does_not_say_how_to_score_is_refused(tmp_path, scoring, error):
    with pytest.raises(error, match="scoring|industry_column"):
        CampaignConfig.from_file(_minimal(tmp_path, scoring=scoring))


def test_exposures_are_no_longer_data(tmp_path):
    with pytest.raises(TypeError, match="exposures"):
        CampaignConfig.from_file(_minimal(tmp_path, data={"prices": "p", "label": "l", "exposures": "e"}))


def test_the_field_and_operator_files_of_a_config_make_its_vocabulary(tmp_path):
    config = CampaignConfig.from_file(_minimal(tmp_path))
    assert config.vocabulary().prompt()["inputs"] == "open, high, low, close, volume, vwap, amount"
    with pytest.raises(VocabularyError, match="adjVolume"):
        config.vocabulary(columns=["adjOpen", "adjHigh", "adjLow", "adjClose"])


def test_vocabulary_paths_are_read_relative_to_the_config_file(tmp_path):
    (tmp_path / "vocabulary").mkdir()
    for source in (PRICE_VOLUME, EXAMPLE_OPERATORS):
        (tmp_path / "vocabulary" / source.name).write_text(source.read_text())
    config = CampaignConfig.from_file(_minimal(
        tmp_path, fields=[f"vocabulary/{PRICE_VOLUME.name}"], operators=f"vocabulary/{EXAMPLE_OPERATORS.name}",
    ))
    assert config.fields == (str(tmp_path / "vocabulary" / PRICE_VOLUME.name),)
    assert "Rank" in config.vocabulary().prompt()["operators"]


@pytest.mark.parametrize("template, text, problem", [
    ("review", "Judge {candidates} on {market} in round {round}, given {data}.", None),
    ("review", "Judge {candidates} on {market} in round {round}.", "{data}"),
    ("review", "Judge them on {market}.", "lacks the placeholder {candidates}"),
    ("data", "{market} {scoring} {horizon} {inputs} {round}", "unknown placeholder(s) ['{round}']"),
    ("judge", "Judge {candidates}.", "judge: not a template"),
])
def test_replaced_templates_are_checked_when_the_config_is_read(tmp_path, template, text, problem):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / f"{template}.txt").write_text(text)
    path = _minimal(tmp_path, prompts={template: f"prompts/{template}.txt"})
    if problem is None:
        assert CampaignConfig.from_file(path).prompts == {template: str(tmp_path / "prompts" / f"{template}.txt")}
    else:
        with pytest.raises(PromptError, match=re.escape(problem)):
            CampaignConfig.from_file(path)


@pytest.mark.parametrize("missing", ["fields", "operators"])
def test_a_config_without_field_or_operator_files_is_refused(tmp_path, missing):
    with pytest.raises(KeyError, match=missing):
        CampaignConfig.from_file(_minimal(tmp_path, **{missing: None}))


#: The example price-volume campaign's config; its data is not in the repository, so only its text is checked here.
REAL = Path(__file__).parent.parent / "configs" / "us3000_ret5.yaml"


def test_the_real_config_reads_with_191_reference_factors_and_a_direction_per_mechanism_type():
    config = CampaignConfig.from_file(REAL)
    names = [name for entry in config.reference for name in entry["factor_names"]]
    assert len(names) == len(set(names)) == 191
    assert {direction.split(":")[0] for direction in config.directions} == {m.name for m in DEFAULT_MECHANISMS}
    assert not re.search(r"\b(19|20)\d\d\b", " ".join(config.directions))  # no date reaches a prompt
    assert {role.model for role in config.roles.values()} == {"deepseek-v4-pro"}
    assert config.data.membership == "/data/quantlab/universe/us3000/membership_estu"
    assert (config.scoring.neutralize.size_column, config.scoring.neutralize.industry_column) == ("marketcap", "industry")
    assert config.fields == (str(PRICE_VOLUME),) and config.operators == str(EXAMPLE_OPERATORS)


#: The example fundamentals campaign's config, checked the same way.
REAL_FUNDAMENTALS = Path(__file__).parent.parent / "configs" / "us3000_fund_ret5.yaml"


def test_the_fundamentals_config_differs_from_the_first_only_in_its_run_dir_fields_data_and_directions():
    config, first = CampaignConfig.from_file(REAL_FUNDAMENTALS), CampaignConfig.from_file(REAL)
    names = [name for entry in config.reference for name in entry["factor_names"]]
    assert len(names) == len(set(names)) == 191 and config.reference == first.reference
    assert {direction.split(":")[0] for direction in config.directions} == {m.name for m in DEFAULT_MECHANISMS}
    assert len(config.directions) == 10 and not set(config.directions) & set(first.directions)
    assert not re.search(r"\b(19|20)\d\d\b", " ".join(config.directions))  # no date reaches a prompt
    assert config.run_dir == "/data/quantlab/runs/factor_agent_us3000_fund_ret5" != first.run_dir
    assert config.fields == (str(PRICE_VOLUME), str(FUNDAMENTALS)) and config.operators == first.operators
    assert config.admission.min_coverage == 0.5
    assert {role.model for role in config.roles.values()} == {"deepseek-v4-pro"}
    assert (config.data.label, config.data.membership, config.scoring) == (first.data.label, first.data.membership, first.scoring)
    assert config.acceptance == first.acceptance and config.acceptance.model == XGB_DEEP
    assert config.acceptance.factor_root == "/data/quantlab/factors/market"

#: The example acceptance model: a deep xgboost on cross-sectional ranks, on the reference set.
XGB_DEEP = {"name": XGB, "factors": [], "hyperparameters": {
    "training_target": "cs_rank", "early_stopping": True, "early_stopping_patience": 100,
    "objective": "reg:squarederror", "num_boost_round": 2000, "eta": 0.03, "max_depth": 6,
    "min_child_weight": 1000, "subsample": 0.8, "colsample_bytree": 0.5,
}}


def test_the_real_config_names_a_deep_xgb_as_the_acceptance_model_and_the_export_root_itself():
    acceptance = CampaignConfig.from_file(REAL).acceptance
    assert acceptance.model == XGB_DEEP
    assert acceptance.factor_root == "/data/quantlab/factors/market"


@pytest.mark.parametrize("acceptance", [None, {"seeds": [0]}])
def test_a_config_without_an_acceptance_model_is_refused(tmp_path, acceptance):
    with pytest.raises(KeyError, match="acceptance.model"):
        CampaignConfig.from_file(_minimal(tmp_path, acceptance=acceptance))


@pytest.mark.parametrize("removed", ["hyperparameters", "features"])
def test_the_removed_acceptance_fields_are_refused_with_where_they_went(tmp_path, removed):
    acceptance = {"model": {"name": XGB}, removed: {"max_depth": 3} if removed == "hyperparameters" else []}
    with pytest.raises(TypeError, match=f"acceptance.{removed} .*acceptance.model"):
        CampaignConfig.from_file(_minimal(tmp_path, acceptance=acceptance))


def test_a_seed_ensemble_acceptance_model_is_refused_for_acceptance_seeds(tmp_path):
    template = {"name": "quantlab.model.predefined.seed_ensemble.SeedEnsemble", "model": {"name": XGB}, "seeds": [0, 1]}
    with pytest.raises(ValueError, match="acceptance.seeds"):
        CampaignConfig.from_file(_minimal(tmp_path, acceptance={"model": template}))


def test_an_acceptance_model_that_is_no_model_is_refused(tmp_path):
    with pytest.raises(TypeError, match="BaseModel or a ModelEnsemble"):
        CampaignConfig.from_file(_minimal(tmp_path, acceptance={"model": {"name": "quantlab.factor.kunquant.FactorKunQuant"}}))


def test_contribution_is_off_by_default_and_takes_its_modes(tmp_path):
    assert CampaignConfig.from_file(_minimal(tmp_path)).acceptance.contribution == "none"
    for mode in ("drop_one", "add_one", "both"):
        config = CampaignConfig.from_file(_minimal(tmp_path, acceptance={"model": {"name": XGB}, "contribution": mode}))
        assert config.acceptance.contribution == mode


def test_an_unknown_contribution_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="acceptance.contribution .*drop_one"):
        CampaignConfig.from_file(_minimal(tmp_path, acceptance={"model": {"name": XGB}, "contribution": "drop-one"}))


def test_the_planner_reads_the_last_five_rounds_trial_by_trial_by_default(tmp_path):
    assert CampaignConfig.from_file(_minimal(tmp_path)).context.recent_rounds == 5


def test_the_context_section_sets_how_many_rounds_the_recent_history_lists(tmp_path):
    assert CampaignConfig.from_file(_minimal(tmp_path, context={"recent_rounds": 2})).context.recent_rounds == 2


def test_a_round_budget_is_refused_as_gone(tmp_path):
    with pytest.raises(TypeError, match="round.budget_usd is gone"):
        CampaignConfig.from_file(_minimal(tmp_path, round={"budget_usd": 10.0}))


@pytest.mark.parametrize("price", ["input_usd_per_mtok", "output_usd_per_mtok"])
def test_per_token_prices_in_a_role_are_refused_as_gone(tmp_path, price):
    llm = {role: {"backend": "scripted", "model": "m", "replies": "r.json"} for role in ROLES}
    llm["reviewer"][price] = 1.0
    with pytest.raises(TypeError, match=f"llm.reviewer.{price} is gone"):
        CampaignConfig.from_file(_minimal(tmp_path, llm=llm))


def test_the_threshold_grows_with_the_whole_trial_count_by_default_and_with_a_power_of_it_when_set(tmp_path):
    assert CampaignConfig.from_file(_minimal(tmp_path)).admission.trial_exponent == 1.0
    config = CampaignConfig.from_file(_minimal(tmp_path, admission={"trial_exponent": 0.5}))
    assert config.admission.trial_exponent == 0.5


@pytest.mark.parametrize("exponent", [0.0, -0.5, 1.5])
def test_a_trial_exponent_outside_zero_to_one_is_refused(tmp_path, exponent):
    with pytest.raises(ValueError, match=f"admission: trial_exponent is {exponent}"):
        CampaignConfig.from_file(_minimal(tmp_path, admission={"trial_exponent": exponent}))


@pytest.mark.parametrize("role, misspelt, options", [
    ("planner", {"backend": "openai_compatible", "base_url": "http://127.0.0.1:1/v1", "timeout": 60},
     "api_key_env, base_url, max_completion_tokens, max_tokens, reasoning_effort, timeout_seconds"),
    ("proposer", {"backend": "anthropic", "max_token": 100}, "base_url, effort, max_tokens, timeout_seconds"),
    ("reviewer", {"backend": "claude_code", "reasoning_effort": "low"}, "effort, executable, timeout_seconds"),
    ("distiller", {"backend": "replay", "logs": "llm_log"}, "log"),
    ("planner", {"backend": "scripted", "reply": "r.json"}, "replies"),
])
def test_an_option_a_roles_backend_does_not_take_is_refused_naming_the_options_it_does(tmp_path, role, misspelt, options):
    llm = {name: {"backend": "scripted", "model": "m", "replies": "r.json"} for name in ROLES}
    llm[role] = {"model": "m", **misspelt}
    unknown = next(key for key in misspelt if key not in ("backend", "base_url"))
    message = rf"llm\.{role}: .*{misspelt['backend']}.* does not take \['{unknown}'\]; it takes {options}"
    with pytest.raises(TypeError, match=message):
        CampaignConfig.from_file(_minimal(tmp_path, llm=llm))


def test_a_role_naming_an_unknown_backend_is_refused_when_the_config_is_read(tmp_path):
    llm = {name: {"backend": "scripted", "model": "m", "replies": "r.json"} for name in ROLES}
    llm["reviewer"] = {"backend": "openai", "model": "m"}
    with pytest.raises(ValueError, match=r"llm\.reviewer: unknown LLM backend 'openai'"):
        CampaignConfig.from_file(_minimal(tmp_path, llm=llm))


@pytest.mark.parametrize("segment", ["training", "confirmation", "validation", "holdout"])
def test_a_segment_entry_with_another_key_than_start_and_end_is_refused(tmp_path, segment):
    segments = {
        "training": {"start": "2020-01-01", "end": "2020-12-31"},
        "confirmation": {"start": "2021-01-01", "end": "2021-06-30"},
        "validation": {"start": "2021-07-01", "end": "2021-12-31"},
        "holdout": {"start": "2022-01-01", "end": "2022-12-31"},
    }
    segments[segment]["purge_bars"] = 5
    with pytest.raises(TypeError, match=rf"segments\.{segment}: unknown key\(s\) \['purge_bars'\]; it takes start and end"):
        CampaignConfig.from_file(_minimal(tmp_path, segments=segments))


@pytest.mark.parametrize("option", ["api_key", "token", "ANTHROPIC_API_KEY"])
def test_a_key_in_a_role_is_refused_when_the_config_is_read_saying_to_name_an_environment_variable(tmp_path, option):
    llm = {name: {"backend": "scripted", "model": "m", "replies": "r.json"} for name in ROLES}
    llm["planner"] = {"backend": "openai_compatible", "model": "m", "base_url": "http://127.0.0.1:1/v1",
                      option: "sk-secret-sentinel"}
    with pytest.raises(ValueError, match=r"llm\.planner: .*environment variable \(name it with api_key_env\)") as raised:
        CampaignConfig.from_file(_minimal(tmp_path, llm=llm))
    assert "sk-secret-sentinel" not in str(raised.value)


def test_every_factor_entry_that_cannot_be_rebuilt_is_named_with_its_cause(store, tmp_path):
    unimportable = tmp_path / "unimportable"
    unimportable.mkdir()
    recorded = json.loads((store / "component.json").read_text())
    recorded["name"] = "no_such_research_package.factor.Alpha"
    (unimportable / "component.json").write_text(json.dumps(recorded))
    config = CampaignConfig.from_file(_minimal(
        tmp_path, reference=[str(store), str(unimportable)], known=[{"store": str(unimportable)}],
    ))
    with pytest.raises(ValueError) as raised:
        config.load_factors()
    assert raised.value.problems == (
        f"reference[1] {unimportable}: No module named 'no_such_research_package'",
        f"known[0] {unimportable}: No module named 'no_such_research_package'",
    )


def test_a_config_without_a_trial_graph_section_takes_the_papers_values(tmp_path):
    config = CampaignConfig.from_file(_minimal(tmp_path))
    # CosmosAlpha, Tables 6 and 7.
    assert config.trial_graph == TrialGraphConfig(
        gamma=0.75, iterations=100, tolerance=1e-8,
        weights={"generation": 1 / 3, "structure": 1 / 3, "correlation": 1 / 3},
        min_jaccard=0.20, structure_neighbours=30, min_correlation=0.10, correlation_neighbours=30,
        nearest_accepted=10, max_expansions=5,
    )
    assert config.context.fertile_children == 3


def test_the_trial_graph_section_sets_its_values(tmp_path):
    section = {"gamma": 0.5, "weights": {"generation": 0.5, "structure": 0.25, "correlation": 0.25}, "max_expansions": 2}
    config = CampaignConfig.from_file(_minimal(tmp_path, trial_graph=section, context={"fertile_children": 1}))
    assert config.trial_graph.gamma == 0.5 and config.trial_graph.max_expansions == 2
    assert config.trial_graph.weights == {"generation": 0.5, "structure": 0.25, "correlation": 0.25}
    assert config.trial_graph.min_jaccard == 0.20 and config.context.fertile_children == 1


@pytest.mark.parametrize("section, message", [
    ({"gamma": 1.0}, "trial_graph: gamma is 1.0"),
    ({"gamma": -0.1}, "trial_graph: gamma is -0.1"),
    ({"iterations": 0}, "trial_graph: iterations is 0"),
    ({"tolerance": 0.0}, "trial_graph: tolerance is 0.0"),
    ({"weights": {"generation": 0.5, "structure": 0.5, "correlation": 0.5}}, "trial_graph: weights sum to 1.5"),
    ({"weights": {"generation": 1.5, "structure": -0.5, "correlation": 0.0}}, "trial_graph: weights.structure is -0.5"),
    ({"weights": {"lineage": 0.5, "structure": 0.5}}, "trial_graph: weights has"),
    ({"min_jaccard": 1.2}, "trial_graph: min_jaccard is 1.2"),
    ({"min_correlation": -0.1}, "trial_graph: min_correlation is -0.1"),
    ({"structure_neighbours": 0}, "trial_graph: structure_neighbours is 0"),
    ({"correlation_neighbours": 2.5}, "trial_graph: correlation_neighbours is 2.5"),
    ({"nearest_accepted": 0}, "trial_graph: nearest_accepted is 0"),
    ({"max_expansions": 0}, "trial_graph: max_expansions is 0"),
])
def test_an_invalid_trial_graph_value_is_refused_naming_its_key(tmp_path, section, message):
    with pytest.raises((ValueError, TypeError), match=re.escape(message)):
        CampaignConfig.from_file(_minimal(tmp_path, trial_graph=section))


def test_an_unknown_trial_graph_key_is_refused(tmp_path):
    with pytest.raises(TypeError, match="gama"):
        CampaignConfig.from_file(_minimal(tmp_path, trial_graph={"gama": 0.5}))


@pytest.mark.parametrize("count", [0, 1.5, True])
def test_a_fertile_children_count_below_one_is_refused(tmp_path, count):
    with pytest.raises(ValueError, match=re.escape(f"context: fertile_children is {count!r}")):
        CampaignConfig.from_file(_minimal(tmp_path, context={"fertile_children": count}))
