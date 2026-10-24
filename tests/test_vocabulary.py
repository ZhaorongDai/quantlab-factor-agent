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
    "parameters": {"window": [5, 20], "constant": [0, 1, 2]},
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
        horizon_rationale="Within a week.", reference="Test reference", description="Test factor.", code=code,
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
    assert any("WindowedAvg is not in the operator whitelist" in r for r in refused.reasons), refused.reasons


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
