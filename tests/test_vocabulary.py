"""Vocabulary.load: field and operator files make what candidates may write; every problem is reported at once.

Each test writes small field and operator files of its own; the dimension
cases build their graphs through a real Gate, whose sandbox rebuilds the
same vocabulary.
"""

import copy
import re

import pytest
import yaml

from factor_agent.candidate import Candidate
from factor_agent.config import GateConfig
from factor_agent.gate import Gate, Passed, Rejected
from factor_agent.gate.vocabulary import Vocabulary, VocabularyError
from factor_agent.zoo import Zoo

FIELDS = {
    "family": "test_family",
    "description": "test bars",
    "bar": True,
    "phrases": {"volume": r"\bvolumes?\b"},
    "fields": [
        {"name": "close", "column": "adjClose", "unit": "dollars/shares", "description": "closing price"},
        {"name": "volume", "column": "adjVolume", "unit": "shares", "description": "shares traded",
         "synonyms": ["volume"]},
        {"name": "amount", "formula": "close * volume", "description": "dollars traded", "synonyms": ["volume"]},
    ],
}

FUNDAMENTALS = {
    "family": "test_fundamentals",
    "description": "statement items",
    "fields": [
        {"name": "netinc", "column": "netinc", "unit": "dollars", "description": "net income"},
        {"name": "marketcap", "column": "marketcap", "unit": "dollars", "description": "market capitalisation"},
        {"name": "sharesbas", "column": "sharesbas", "unit": "shares", "description": "shares outstanding"},
    ],
}

OPERATORS = {
    "parameters": {"window": [5, 20], "constant": [0, 1, 2], "lag": [5, 20]},
    "phrases": {"volatility": r"\bvolatility\b"},
    "operators": [
        {"name": "Add", "class": "KunQuant.ops.Add", "signature": "Add(x, y)", "inputs": ["series", "series"],
         "dimension": "match", "description": "sum"},
        {"name": "Mul", "class": "KunQuant.ops.Mul", "signature": "Mul(x, y)", "inputs": ["series", "series"],
         "dimension": "product", "description": "product"},
        {"name": "Div", "class": "KunQuant.ops.Div", "signature": "Div(x, y)", "inputs": ["series", "series"],
         "dimension": "quotient", "description": "quotient"},
        {"name": "MulConst", "class": "KunQuant.ops.MulConst", "signature": "MulConst(x, c)",
         "params": {"value": "constant", "swap": "flag"}, "dimension": "scale", "description": "x times c"},
        {"name": "WindowedStddev", "class": "KunQuant.ops.WindowedStddev", "signature": "WindowedStddev(x, window)",
         "params": {"window": "window"}, "dimension": "same", "description": "standard deviation",
         "binding": ["volatility"]},
        {"name": "Max", "class": "KunQuant.ops.Max", "call": "quantlab.factor.kunquant_nan.Max",
         "signature": "Max(x, y)", "inputs": ["series", "series"], "dimension": "match", "description": "maximum"},
        {"name": "Rank", "class": "KunQuant.Op.Rank", "signature": "Rank(x)", "dimension": "pure",
         "description": "rank across the market"},
        {"name": "Sub", "class": "KunQuant.ops.Sub", "signature": "Sub(x, y)", "inputs": ["series", "series"],
         "dimension": "match", "description": "difference"},
        {"name": "BackRef", "class": "KunQuant.ops.BackRef", "signature": "BackRef(x, lag)",
         "params": {"window": "lag"}, "dimension": "same", "description": "x lag bars ago"},
    ],
}


def _write(tmp_path, fields=FIELDS, operators=OPERATORS, *more_fields) -> tuple[list, object]:
    paths = []
    for i, family in enumerate([fields, *more_fields]):
        path = tmp_path / f"fields_{i}.yaml"
        path.write_text(yaml.safe_dump(family))
        paths.append(path)
    operator_path = tmp_path / "operators.yaml"
    operator_path.write_text(yaml.safe_dump(operators))
    return paths, operator_path


def _changed(base: dict, change) -> dict:
    changed = copy.deepcopy(base)
    change(changed)
    return changed


def _field(index, **values):
    return lambda family: family["fields"][index].update(values)


def _operator(index, **values):
    return lambda operators: operators["operators"][index].update(values)


def _add_field(**field):
    return lambda family: family["fields"].append(field)


def _candidate(code: str, hypothesis: str = "A test mechanism.") -> Candidate:
    return Candidate(
        hypothesis=hypothesis, mechanism_type="behavioural_bias", expected_sign=1,
        horizon_rationale="Within a week.", description="Test factor.", code=code,
    )


def test_a_field_file_and_an_operator_file_make_the_names_candidates_may_use(tmp_path):
    vocabulary = Vocabulary.load(*_write(tmp_path))
    prompt = vocabulary.prompt()
    assert prompt["inputs"] == "close, volume, amount"
    assert "- WindowedStddev(x, window): standard deviation" in prompt["operators"].splitlines()
    assert "- window: 5, 20" in prompt["parameters"].splitlines()


# Every start-up error: (field file change, operator file change, load arguments, what the error says).
STARTUP_ERRORS = {
    "missing column": (None, None, {"columns": ["adjClose"]}, "adjVolume"),
    "malformed unit": (_field(0, unit="dollars per share"), None, {}, "unit 'dollars per share'"),
    "unit of an unknown base": (_field(0, unit="euros/shares"), None, {}, "unit 'euros/shares'"),
    "name of an operator": (_field(0, name="Rank"), None, {}, "Rank"),
    "name of a builtin": (_field(0, name="abs"), None, {}, "abs"),
    "name of another field": (_field(1, name="close"), None, {}, "close"),
    "column of another field": (_field(1, column="adjClose"), None, {}, "adjClose"),
    "a field with neither column nor formula": (_add_field(name="spread", description="d"), None, {}, "spread"),
    "an unknown synonym": (_field(0, synonyms=["closing price"]), None, {}, "closing price"),
    "a rule on an unknown field": (lambda f: f.update(rules=[{"phrase": "volume", "needs": ["open"]}]), None, {}, "open"),
    "an unknown key": (_field(0, colour="red"), None, {}, "colour"),
    "a class that is not a KunQuant operator": (None, _operator(6, **{"class": "KunQuant.Op.Builder"}), {},
                                                "not a KunQuant operator"),
    "a class that does not exist": (None, _operator(6, **{"class": "KunQuant.Op.NoSuchOp"}), {}, "NoSuchOp"),
    "a class from an unlisted module": (None, _operator(6, **{"class": "collections.OrderedDict"}), {},
                                        "collections"),
    "a callable from an unlisted module": (None, _operator(5, call="os.system"), {}, "os.system"),
    "a callable reached through an unlisted module": (None, _operator(5, call="KunQuant.jit.cfake.subprocess.run"), {},
                                                      "through module subprocess"),
    "a class reached through an unlisted module": (None, _operator(6, **{"class": "KunQuant.Op.threading.Thread"}), {},
                                                   "through module threading"),
    "a callable an allowed module imported from an unlisted one": (None, _operator(5, call="KunQuant.Op.OrderedDict"),
                                                                   {}, "defined in collections"),
    "a class an allowed module imported from an unlisted one": (None, _operator(6, **{"class": "KunQuant.ops.Union"}),
                                                                {}, "defined in typing"),
    "an unknown dimension rule": (None, _operator(0, dimension="money"), {}, "money"),
    "an unknown parameter set": (None, _operator(4, params={"window": "windows"}), {}, "windows"),
    "an unknown input kind": (None, _operator(0, inputs=["series", "number"]), {}, "number"),
    "an unknown binding phrase": (None, _operator(4, binding=["dispersion"]), {}, "dispersion"),
    "an operator named like a builtin": (None, _operator(6, name="len"), {}, "len"),
    "two operators of one name": (None, _operator(1, name="Add"), {}, "Add"),
    "a formula with attribute access": (_field(2, formula="close.__class__"), None, {}, "attribute access"),
    "a formula with an import": (_field(2, formula="__import__('os')"), None, {}, "__import__"),
    "a formula that is a statement": (_field(2, formula="x = close"), None, {}, "syntax error"),
    "a formula over a later field": (_field(2, formula="close * spread"), None, {}, "spread"),
    "a formula calling a builtin": (_field(2, formula="abs(close)"), None, {}, "abs"),
    "a formula that fails to build": (_field(2, formula="Div(close)"), None, {}, "amount"),
}


@pytest.mark.parametrize("case", STARTUP_ERRORS)
def test_each_startup_error_is_reported_with_what_is_wrong(tmp_path, case):
    fields_change, operators_change, arguments, says = STARTUP_ERRORS[case]
    fields = _changed(FIELDS, fields_change) if fields_change else FIELDS
    operators = _changed(OPERATORS, operators_change) if operators_change else OPERATORS
    with pytest.raises(VocabularyError, match=re.escape(says)):
        Vocabulary.load(*_write(tmp_path, fields, operators), **arguments)


def test_every_startup_error_is_reported_at_once(tmp_path):
    fields = _changed(FIELDS, lambda f: (_field(0, unit="euros")(f), _field(1, name="Rank")(f)))
    operators = _changed(OPERATORS, _operator(5, call="os.system"))
    with pytest.raises(VocabularyError) as error:
        Vocabulary.load(*_write(tmp_path, fields, operators), columns=["adjVolume"])
    message = str(error.value)
    for says in ("euros", "Rank", "os.system", "adjClose"):
        assert says in message, message


def test_a_callable_from_a_module_the_config_lists_is_accepted(tmp_path):
    operators = _changed(OPERATORS, _operator(5, call="tests.vocabulary_files.price_volume"))
    with pytest.raises(VocabularyError, match="tests.vocabulary_files"):
        Vocabulary.load(*_write(tmp_path, FIELDS, operators))
    Vocabulary.load(*_write(tmp_path, FIELDS, operators), operator_modules=["tests.vocabulary_files"])


def test_a_user_operator_file_needs_no_exclusions_and_refuses_what_it_does_not_register(tmp_path):
    gate = Gate(GateConfig(timeout_seconds=10.0), Vocabulary.load(*_write(tmp_path)))
    passed = gate.check(_candidate("alpha = Rank(close / WindowedStddev(close, 20))"), Zoo())
    assert isinstance(passed, Passed), getattr(passed, "reasons", None)
    refused = gate.check(_candidate("alpha = WindowedAvg(close, 20) / close"), Zoo())
    assert isinstance(refused, Rejected)
    assert any("name WindowedAvg is not in the vocabulary" in r for r in refused.reasons), refused.reasons


def test_the_sandbox_reads_the_fields_and_stand_ins_of_the_vocabulary(tmp_path):
    gate = Gate(GateConfig(timeout_seconds=10.0), Vocabulary.load(*_write(tmp_path)))
    result = gate.check(_candidate("alpha = amount / WindowedStddev(amount, 20) + Max(close / close, close / close)"), Zoo())
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    assert result.stats["base_fields"] == ["amount", "close"]
    # The stand-in adds the bar mask over the family's columns.
    assert result.graph.input_names == ("adjClose", "adjVolume")


@pytest.mark.parametrize(
    ("code", "dimension"),
    [
        ("alpha = netinc / marketcap", "pure number"),
        ("alpha = volume / sharesbas", "pure number"),
        ("alpha = amount / marketcap", "pure number"),
        ("alpha = close * sharesbas / marketcap", "pure number"),
        ("alpha = close / netinc", "1/shares"),
        ("alpha = close", "price"),
        ("alpha = amount", "money"),
    ],
)
def test_dimensions_follow_from_dollar_and_share_units_over_mixed_families(tmp_path, code, dimension):
    gate = Gate(GateConfig(timeout_seconds=10.0), Vocabulary.load(*_write(tmp_path, FIELDS, OPERATORS, FUNDAMENTALS)))
    result = gate.check(_candidate(code), Zoo())
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    assert result.stats["dimension"] == dimension


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("alpha = close + netinc", "Add combines a price with a money"),
        ("alpha = volume + marketcap", "Add combines a volume with a money"),
    ],
)
def test_a_sum_of_different_dimensions_is_refused_across_families(tmp_path, code, reason):
    gate = Gate(GateConfig(timeout_seconds=10.0), Vocabulary.load(*_write(tmp_path, FIELDS, OPERATORS, FUNDAMENTALS)))
    result = gate.check(_candidate(code), Zoo())
    assert isinstance(result, Rejected)
    assert any(reason in r for r in result.reasons), result.reasons


def _renamed_sets(operators: dict) -> dict:
    """``operators`` with its constant and lag sets under other names."""
    renamed = {"constant": "numbers", "lag": "shifts"}
    operators = copy.deepcopy(operators)
    operators["parameters"] = {renamed.get(k, k): v for k, v in operators["parameters"].items()}
    for entry in operators["operators"]:
        entry["params"] = {a: renamed.get(k, k) for a, k in (entry.get("params") or {}).items()}
    return operators


@pytest.mark.parametrize("operators", [OPERATORS, _renamed_sets(OPERATORS)], ids=["named", "renamed"])
@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("alpha = close * 2", None),
        ("alpha = close * -2", None),
        ("alpha = close * 2 * 2", "the constants combine to MulConst scale 4"),
        ("alpha = BackRef(close, 5) / close", None),
        ("alpha = BackRef(BackRef(close, 5), 5) / close", "the constants combine to BackRef lag 10"),
    ],
)
def test_constant_and_lag_values_are_checked_whatever_the_operator_file_names_their_sets(
    tmp_path, operators, code, reason,
):
    config = GateConfig(timeout_seconds=10.0, max_free_parameter_ratio=10.0)
    gate = Gate(config, Vocabulary.load(*_write(tmp_path, FIELDS, operators)))
    result = gate.check(_candidate(code), Zoo())
    if reason is None:
        assert isinstance(result, Passed), getattr(result, "reasons", None)
    else:
        assert isinstance(result, Rejected)
        assert any(reason in r for r in result.reasons), result.reasons


@pytest.mark.parametrize("operators", [OPERATORS, _renamed_sets(OPERATORS)], ids=["named", "renamed"])
def test_hand_set_constants_count_whatever_the_operator_file_names_their_set(tmp_path, operators):
    gate = Gate(GateConfig(timeout_seconds=10.0), Vocabulary.load(*_write(tmp_path, FIELDS, operators)))
    result = gate.check(_candidate("alpha = close * 2"), Zoo())
    assert isinstance(result, Rejected)
    assert any("1 hand-set constants over 1 operators" in r for r in result.reasons), result.reasons


def test_field_synonyms_and_operator_phrases_bind_a_hypothesis(tmp_path):
    gate = Gate(GateConfig(timeout_seconds=10.0), Vocabulary.load(*_write(tmp_path)))
    unbound = gate.check(_candidate("alpha = Rank(close)", "Heavy volume and high volatility attract attention."), Zoo())
    assert isinstance(unbound, Rejected)
    assert any("names volume" in r and "never reads amount or volume" in r for r in unbound.reasons), unbound.reasons
    assert any("names volatility" in r and "WindowedStddev" in r for r in unbound.reasons), unbound.reasons
    bound = gate.check(
        _candidate("alpha = Rank(WindowedStddev(amount, 20))", "Heavy volume and high volatility attract attention."),
        Zoo(),
    )
    assert isinstance(bound, Passed), getattr(bound, "reasons", None)


# Period lags.

PERIOD_LAG = {
    "column": "{column}__q{k}",
    "k": [1, 4],
    "period": {"description": "one quarterly report", "key": "reportperiod", "days": 91,
               "tolerance_days": 45, "bars": 63},
}

LAGGED = _changed(FUNDAMENTALS, lambda f: f.update(period_lag=PERIOD_LAG))


def _lag_gate(tmp_path, **limits) -> Gate:
    vocabulary = Vocabulary.load(*_write(tmp_path, FIELDS, OPERATORS, LAGGED))
    return Gate(GateConfig(timeout_seconds=10.0, **limits), vocabulary)


def test_period_lag_reads_the_lag_column_of_a_field_whose_family_declares_one(tmp_path):
    result = _lag_gate(tmp_path).check(_candidate("alpha = (netinc - PeriodLag(netinc, 4)) / marketcap"), Zoo())
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    assert result.graph.input_names == ("marketcap", "netinc", "netinc__q4")
    assert result.stats["dimension"] == "pure number"


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("alpha = PeriodLag(close, 4) / close", "PeriodLag(close, ...): test_family declares no period lag"),
        ("alpha = PeriodLag(netinc, 2) / marketcap", "PeriodLag(netinc, 2): k must be one of 1, 4"),
        ("alpha = PeriodLag(netinc, 4.0) / marketcap", "PeriodLag(netinc, 4.0): k must be one of 1, 4"),
        ("alpha = PeriodLag(netinc / marketcap, 4)", "PeriodLag takes a field read from a column"),
        ("alpha = PeriodLag(amount, 1) / amount", "PeriodLag takes a field read from a column"),
    ],
)
def test_period_lag_is_refused_without_a_declared_lag_or_with_a_k_not_allowed(tmp_path, code, reason):
    result = _lag_gate(tmp_path).check(_candidate(code), Zoo())
    assert isinstance(result, Rejected)
    assert any(reason in r for r in result.reasons), result.reasons


def test_period_lag_is_not_a_name_when_no_family_declares_one(tmp_path):
    gate = Gate(GateConfig(timeout_seconds=10.0), Vocabulary.load(*_write(tmp_path, FIELDS, OPERATORS, FUNDAMENTALS)))
    result = gate.check(_candidate("alpha = PeriodLag(netinc, 4) / marketcap"), Zoo())
    assert isinstance(result, Rejected)
    assert any("PeriodLag" in r for r in result.reasons), result.reasons


def test_a_field_and_its_lags_count_once_toward_the_field_limit(tmp_path):
    code = "alpha = (netinc - PeriodLag(netinc, 1) + PeriodLag(netinc, 4)) / marketcap"
    result = _lag_gate(tmp_path, max_base_fields=2).check(_candidate(code), Zoo())
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    assert result.stats["base_fields"] == ["marketcap", "netinc"]
    refused = _lag_gate(tmp_path, max_base_fields=1).check(_candidate(code), Zoo())
    assert isinstance(refused, Rejected)
    assert any("reads 2 input fields (marketcap, netinc)" in r for r in refused.reasons), refused.reasons


def test_the_prompt_shows_period_lag_only_when_a_family_declares_one(tmp_path):
    without = Vocabulary.load(*_write(tmp_path, FIELDS, OPERATORS, FUNDAMENTALS)).prompt()["operators"]
    assert "PeriodLag" not in without
    lines = Vocabulary.load(*_write(tmp_path, FIELDS, OPERATORS, LAGGED)).prompt()["operators"].splitlines()
    assert any(line.startswith("- PeriodLag(x, k):") for line in lines), lines
    family = "  - netinc, marketcap, sharesbas (test_fundamentals): k one of 1, 4; a period is one quarterly report"
    assert family in lines, lines


def _lag(**values):
    return lambda family: family["period_lag"].update(values)


def _period(**values):
    return lambda family: family["period_lag"]["period"].update(values)


# Every start-up error of a period lag: (field file change, load arguments, what the error says).
PERIOD_LAG_ERRORS = {
    "a pattern without k": (_lag(column="{column}_lag"), {}, "exactly {column} and {k}"),
    "a pattern with another placeholder": (_lag(column="{column}__{n}{k}"), {}, "exactly {column} and {k}"),
    "a k that is not whole": (_lag(k=[1, 2.5]), {}, "k [1, 2.5]"),
    "a k of zero": (_lag(k=[0, 4]), {}, "k [0, 4]"),
    "no k": (_lag(k=[]), {}, "k []"),
    "a period without a key": (lambda f: f["period_lag"]["period"].pop("key"), {}, "period key"),
    "a period of no days": (_period(days=0), {}, "period days"),
    "a period of fractional bars": (_period(bars=62.5), {}, "period bars"),
    "an unknown period key": (_period(months=3), {}, "months"),
    "an unknown period lag key": (_lag(window=4), {}, "window"),
    "a key not in the price dataset": (
        None, {"columns": ["adjClose", "adjVolume", "netinc", "marketcap", "sharesbas"]}, "'reportperiod'"),
    "a lag column that is a field's column": (_lag(column="{column}{k}", k=[1]), {}, None),
}


@pytest.mark.parametrize("case", PERIOD_LAG_ERRORS)
def test_each_period_lag_startup_error_is_reported(tmp_path, case):
    change, arguments, says = PERIOD_LAG_ERRORS[case]
    lagged = _changed(LAGGED, change) if change else LAGGED
    if says is None:  # netinc1 is the lag of netinc and also a field's column
        lagged["fields"].append({"name": "other", "column": "netinc1", "unit": "dollars", "description": "d"})
        says = "PeriodLag(netinc, 1) reads 'netinc1'"
    with pytest.raises(VocabularyError, match=re.escape(says)):
        Vocabulary.load(*_write(tmp_path, FIELDS, OPERATORS, lagged), **arguments)


def test_a_rewrite_names_a_lag_as_candidate_code_writes_it(tmp_path):
    gate = _lag_gate(tmp_path)
    zoo = Zoo()
    accepted = gate.check(_candidate("alpha = (netinc - PeriodLag(netinc, 4)) / marketcap"), zoo)
    assert isinstance(accepted, Passed), getattr(accepted, "reasons", None)
    zoo.add("growth", "test", accepted.graph)
    result = gate.check(_candidate("alpha = Rank((netinc - PeriodLag(netinc, 4)) / marketcap)"), zoo)
    assert isinstance(result, Rejected)
    assert any("PeriodLag(netinc, 4)" in r and "growth (test)" in r for r in result.reasons), result.reasons
