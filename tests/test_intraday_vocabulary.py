"""An intraday family and operator stages in the vocabulary files: what ``Vocabulary.load`` reads and refuses.

Each test writes a small daily field file, an intraday field file (marked
``intraday`` with its window set) and an operator file whose entries name
their stages, as ``test_vocabulary`` does for the daily files.
"""

import copy

import pytest
import yaml

from factor_agent.gate.vocabulary import Vocabulary, VocabularyError
from tests.vocabulary_files import intraday, price_volume

DAILY = {
    "family": "daily_bars",
    "description": "daily bars",
    "fields": [
        {"name": "close", "column": "adjClose", "unit": "dollars/shares", "description": "closing price"},
    ],
}

MINUTES = {
    "family": "minute_bars",
    "description": "one-minute trade bars",
    "intraday": {"windows": [5, 30]},
    "fields": [
        {"name": "minute_close", "column": "close", "unit": "dollars/shares", "description": "last trade of the bar"},
        {"name": "minute_volume", "column": "volume", "unit": "shares", "description": "shares traded in the bar"},
        {"name": "bars_left", "column": "session_bars_left", "unit": 1, "description": "bars to the close"},
    ],
}

OPERATORS = {
    "parameters": {"window": [5, 20], "lag": [1, 5]},
    "operators": [
        {"name": "BackRef", "class": "KunQuant.ops.BackRef", "params": {"window": "lag"}, "dimension": "same",
         "stages": ["intraday", "daily"]},
        {"name": "Mul", "class": "KunQuant.ops.Mul", "inputs": ["series", "series"], "dimension": "product",
         "stages": ["intraday", "daily"]},
        {"name": "ExpMovingAvg", "class": "KunQuant.ops.ExpMovingAvg", "params": {"window": "window"},
         "dimension": "same", "stages": ["daily"]},
        {"name": "SessionMean", "class": "factor_agent.aggregations.SessionMean", "dimension": "same",
         "stages": ["aggregation"]},
    ],
}


def _load(tmp_path, *families, operators=OPERATORS, columns=None, intraday_columns=None) -> Vocabulary:
    paths = []
    for i, family in enumerate(families):
        paths.append(tmp_path / f"fields_{i}.yaml")
        paths[-1].write_text(yaml.safe_dump(family))
    (tmp_path / "operators.yaml").write_text(yaml.safe_dump(operators))
    return Vocabulary.load(paths, tmp_path / "operators.yaml", columns=columns, intraday_columns=intraday_columns)


def _changed(base: dict, change) -> dict:
    changed = copy.deepcopy(base)
    change(changed)
    return changed


def test_an_intraday_field_file_makes_its_family_the_intraday_family_with_its_window_set(tmp_path):
    vocabulary = _load(tmp_path, DAILY, MINUTES)
    assert vocabulary.intraday_family == "minute_bars"
    assert vocabulary.intraday_windows == (5, 30)
    # Without lags of its own, a lag in the intraday stage takes the operator file's lag set.
    assert vocabulary.intraday_lags == (1, 5)
    with_lags = _load(tmp_path, DAILY, _changed(MINUTES, lambda f: f["intraday"].update(lags=[1, 2])))
    assert with_lags.intraday_lags == (1, 2)
    assert vocabulary.intraday_columns == frozenset({"close", "volume", "session_bars_left"})


def test_without_an_intraday_field_file_there_is_no_intraday_family(tmp_path):
    vocabulary = _load(tmp_path, DAILY)
    assert vocabulary.intraday_family is None
    assert vocabulary.intraday_columns == frozenset()


def test_each_operator_names_the_stages_it_may_be_used_in(tmp_path):
    vocabulary = _load(tmp_path, DAILY, MINUTES)
    assert vocabulary.spec("Mul").stages == ("intraday", "daily")
    assert vocabulary.spec("ExpMovingAvg").stages == ("daily",)
    assert vocabulary.spec("SessionMean").stages == ("aggregation",)


def test_an_operator_without_stages_is_daily_and_an_aggregation_without_them_an_aggregation(tmp_path):
    operators = _changed(OPERATORS, lambda o: [entry.pop("stages") for entry in o["operators"]])
    vocabulary = _load(tmp_path, DAILY, operators=operators)
    assert vocabulary.spec("Mul").stages == ("daily",)
    assert vocabulary.spec("SessionMean").stages == ("aggregation",)


def test_daily_fields_are_checked_against_the_price_dataset_and_intraday_fields_against_the_intraday_one(tmp_path):
    with pytest.raises(VocabularyError) as refused:
        _load(tmp_path, DAILY, MINUTES, columns={"adjClose", "close"}, intraday_columns={"close", "session_bars_left"})
    assert refused.value.problems == (
        "field minute_volume (minute_bars): column 'volume' is not in the intraday dataset",
    )


@pytest.mark.parametrize("case", [
    "two intraday families", "no windows", "bad window", "intraday bar mask",
    "stage unknown", "aggregation in a stage", "operator as an aggregation", "bad lag",
    "unbounded memory in the intraday stage",
])
def test_each_intraday_startup_error_is_reported(tmp_path, case):
    other = _changed(MINUTES, lambda f: f.update(family="more_minutes", fields=[
        {"name": "trades", "column": "n_trades", "unit": 1, "description": "trades in the bar"},
    ]))
    families, operators, expected = [DAILY, MINUTES], OPERATORS, None
    if case == "two intraday families":
        families.append(other)
        expected = "more_minutes is a second intraday family; a campaign opens one, minute_bars"
    elif case == "no windows":
        families[1] = _changed(MINUTES, lambda f: f.update(intraday={}))
        expected = "intraday windows must be a list of positive whole numbers of bars, got None"
    elif case == "bad window":
        families[1] = _changed(MINUTES, lambda f: f.update(intraday={"windows": [5, 0.5]}))
        expected = "intraday windows must be a list of positive whole numbers of bars, got [5, 0.5]"
    elif case == "intraday bar mask":
        families[1] = _changed(MINUTES, lambda f: f.update(bar=True))
        expected = "an intraday family cannot be bar: true; a bar without a trade has no price but is a bar"
    elif case == "stage unknown":
        operators = _changed(OPERATORS, lambda o: o["operators"][1].update(stages=["minute"]))
        expected = "stages ['minute'] must be a non-empty list of intraday and daily"
    elif case == "aggregation in a stage":
        operators = _changed(OPERATORS, lambda o: o["operators"][3].update(stages=["intraday"]))
        expected = "a session aggregation's stages are [aggregation], not ['intraday']"
    elif case == "operator as an aggregation":
        operators = _changed(OPERATORS, lambda o: o["operators"][1].update(stages=["aggregation"]))
        expected = "stages ['aggregation'] must be a non-empty list of intraday and daily"
    elif case == "bad lag":
        families[1] = _changed(MINUTES, lambda f: f["intraday"].update(lags=[0]))
        expected = "intraday lags must be a list of positive whole numbers of bars, got [0]"
    elif case == "unbounded memory in the intraday stage":
        operators = _changed(OPERATORS, lambda o: o["operators"][2].update(stages=["intraday", "daily"]))
        expected = (
            "ExpMovingAvg has unbounded memory, which would carry one session into the next: "
            "its stages are [daily]"
        )
    with pytest.raises(VocabularyError) as refused:
        _load(tmp_path, *families, operators=operators)
    assert any(expected in problem for problem in refused.value.problems), refused.value.problems


def test_every_operator_of_the_example_file_names_its_stages():
    text = yaml.safe_load(open(price_volume().sources["operators"]["path"]).read())
    unnamed = [entry["name"] for entry in text["operators"] if "stages" not in entry]
    assert not unnamed, f"operators of the example operator file that name no stages: {unnamed}"


def test_the_example_file_keeps_unbounded_memory_and_month_based_operators_daily():
    vocabulary = price_volume()
    # KunQuant's linear regressions keep running sums from bar to bar, which would carry one session into the next.
    regressions = ("WindowedLinearRegressionSlope", "WindowedLinearRegressionRSqaure", "WindowedLinearRegressionResi")
    for name in ("ExpMovingAvg", "CMRA", *regressions):
        assert vocabulary.spec(name).stages == ("daily",), name
    for name in ("Add", "WindowedAvg", "Rank", "Select"):
        assert vocabulary.spec(name).stages == ("intraday", "daily"), name
    assert vocabulary.spec("SessionMean").stages == ("aggregation",)


def test_a_vocabulary_without_an_intraday_family_neither_names_nor_shows_the_aggregations():
    daily, minutes = price_volume(), intraday()
    assert "SessionMean" not in daily.names
    assert "SessionMean" not in daily.prompt()["operators"]
    assert "SessionMean" in minutes.names
    assert "- SessionMean(x): " in minutes.prompt()["operators"]
