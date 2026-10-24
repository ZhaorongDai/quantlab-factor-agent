"""The Vocabulary: the fields, operators and parameter values a campaign's candidates may write with.

It is built from files only (``Vocabulary.load``): one field file per input
family and one operator file, named by the campaign config. The framework
holds no field, column or operator of its own; it keeps the rules a file
picks from (the dimension rules of ``_dimension``, the input kinds, the
syntax of ``_checks``), the ``Input`` a graph reads a column with, and the
``Builder`` of window loops.

A field file lists its family's fields. A field reads a column of the price
dataset, in a unit that is a product of powers of dollars and shares, or is
derived: a formula over the family's earlier fields, checked under the
syntax rules of candidate code and built with the operators. Its synonyms
name the family's phrases that bind a hypothesis to it, and the family's
rules name phrases that need several fields. A family with ``bar: true``
makes a missing value in any of its columns a missing bar: candidate code
runs inside quantlab's ``missing_bars_only`` over those columns, so the
NaN-preserving stand-ins of ``quantlab.factor.kunquant_nan`` keep such a
bar NaN.

A family may declare a period lag (``period_lag``): the column pattern of
its lags (``{column}`` and ``{k}`` fill it, ``"{column}__q{k}"`` as
quantlab's ``FundamentalAlpha`` names its report lags), the allowed ``k``
and what a period is: the dataset column holding the period end of the
report a bar shows (``key``), the spacing of consecutive periods in days and
the tolerance on it, and the most bars a period spans. Candidate code then
calls ``PeriodLag(x, k)`` on a field of the family that reads a column; it
reads the lag column of the field, which ``factor_agent.period_lag``
computes from the field's column and the key when the factor is computed.
The lags of a field count as the field itself, toward the field limit and
in dimensions.

The operator file lists each operator's class (a KunQuant ``OpBase``
subclass, what a graph records), the callable candidate code calls under
its name (the class, or a stand-in such as ``kunquant_nan.Max``), its
signature, argument and parameter kinds, dimension rule, prompt description
and binding phrases, plus the parameter value sets and, in the
repository's example file, every other operator class with why it is
excluded. A class or a callable must come from KunQuant,
``quantlab.factor`` or a module the config lists in ``operator_modules``;
the module is checked before it is imported.

``load`` reports every problem of the files at once (``VocabularyError``).
The sandbox rebuilds the same vocabulary from ``sources``, the files' text.
"""

import builtins
import contextlib
import importlib
import keyword
import math
import re
import string
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path

import yaml
from KunQuant.Op import Builder, Input, OpBase
from quantlab.factor import kunquant_nan

from factor_agent.gate import _checks, _dimension
from factor_agent.gate._canonical import canonical_hashes, effective_parameters
from factor_agent.gate.graph import Graph

#: Modules a class or a callable may always come from; the config's ``operator_modules`` add to them.
FRAMEWORK_MODULES = ("KunQuant", "quantlab.factor")

#: The builtins candidate code may call.
BUILTINS = {
    name: getattr(builtins, name)
    for name in ("abs", "bool", "dict", "enumerate", "float", "int", "len", "list", "range",
                 "round", "tuple", "zip")
}

#: Kinds of a series argument: a number series, a comparison, or a ``ForeachBackWindow`` loop.
INPUT_KINDS = ("series", "condition", "loop")

#: The parameter set of an attribute that is bookkeeping, never checked.
FLAG = "flag"

#: Names the framework binds in candidate code itself.
_RESERVED = frozenset({"Builder", "Input", "PeriodLag", "alpha"})

#: The bases of a unit, in the order of a dimension's exponents.
_UNIT_BASES = ("dollars", "shares")
_UNIT_FACTOR = re.compile(r"(dollars|shares|1)(?:\^(-?\d+))?")

_FIELD_FILE_KEYS = {"family", "description", "bar", "phrases", "rules", "fields", "period_lag"}
_PERIOD_LAG_KEYS = {"column", "k", "period"}
_PERIOD_KEYS = {"description", "key", "days", "tolerance_days", "bars"}
_FIELD_KEYS = {"name", "column", "formula", "unit", "description", "synonyms"}
_OPERATOR_FILE_KEYS = {"parameters", "phrases", "operators", "excluded"}
_OPERATOR_KEYS = {"name", "class", "call", "signature", "inputs", "params", "dimension", "description", "binding"}


class PeriodLagError(ValueError):
    """Candidate code called ``PeriodLag`` on what has no such lag; the message says why."""


class VocabularyError(ValueError):
    """The field or operator files are wrong; the message lists every problem."""

    def __init__(self, problems: list[str]):
        self.problems = tuple(problems)
        super().__init__("the vocabulary files have problems:\n" + "\n".join(f"- {p}" for p in problems))


@dataclass(frozen=True)
class Field:
    """One field: its name, family, the column it reads or the formula it is derived by, and its description.

    ``dimension`` is the unit's ``(dollars, shares)`` exponents; None for a
    derived field, whose dimension follows from its formula.
    """

    name: str
    family: str
    column: str | None
    formula: str | None
    dimension: tuple | None
    description: str


@dataclass(frozen=True)
class OpSpec:
    """One registered operator; see the module docstring.

    ``path`` is the class's dotted path; ``params`` maps each attribute to
    its parameter set, or ``FLAG``.
    """

    name: str
    cls: type
    path: str
    call: object
    signature: str
    inputs: tuple[str, ...]
    params: dict
    dimension: str
    description: str
    binding: tuple[str, ...]


@dataclass(frozen=True)
class PeriodLag:
    """A family's declared period lag; see the module docstring."""

    family: str
    column: str
    k: tuple[int, ...]
    description: str
    key: str
    days: float
    tolerance_days: float
    bars: int

    def lag_column(self, column: str, k: int) -> str:
        """The column a lag of ``k`` periods of ``column`` is read under."""
        return self.column.format(column=column, k=k)


#: ``Input``, which every graph records and no candidate calls.
INPUT = OpSpec("Input", Input, "KunQuant.Op.Input", None, "", (), {"name": FLAG}, "input", "", ())


class Vocabulary:
    """What candidates may write with; build it with ``load``, or ``from_sources`` in the sandbox.

    Raises
    ------
    VocabularyError
        If the files have any problem; every problem is listed.
    """

    def __init__(self, sources: dict, columns: Iterable[str] | None = None):
        #: The files' paths and text and the operator modules: what the sandbox rebuilds from.
        self.sources = sources
        #: The parameter sets of the operator file, by kind.
        self.parameters: dict[str, tuple] = {}
        #: The fields of every family, in file order.
        self.fields: dict[str, Field] = {}
        #: The description of every family, in file order.
        self.families: dict[str, str] = {}
        #: The operator file's excluded classes, with their reasons.
        self.excluded: dict[type, str] = {}
        self._problems: list[str] = []
        self._modules = FRAMEWORK_MODULES + tuple(sources.get("operator_modules") or ())
        self._operators: dict[str, OpSpec] = {}
        self._operator_phrases: dict[str, re.Pattern] = {}
        self._field_phrases: dict[str, re.Pattern] = {}
        self._synonyms: dict[str, list[str]] = {}
        self._rules: list[tuple[str, tuple[str, ...]]] = []
        self._bar_columns: list[str] = []
        #: The declared period lags, by family.
        self.period_lags: dict[str, PeriodLag] = {}
        operators = sources["operators"]
        self._read_operators(operators["path"], self._yaml(operators))
        for document in sources["fields"]:
            self._read_fields(document["path"], self._yaml(document))
        if not self._problems:
            self._build_derived()
        self._check_lag_columns()
        if columns is not None:
            available = set(columns)
            self._problems += [
                f"field {f.name} ({f.family}): column {f.column!r} is not in the price dataset"
                for f in self.fields.values() if f.column is not None and f.column not in available
            ]
            self._problems += [
                f"period lag of {lag.family}: key column {lag.key!r} is not in the price dataset"
                for lag in self.period_lags.values() if lag.key not in available
            ]
        if self._problems:
            raise VocabularyError(self._problems)

    @classmethod
    def load(
        cls, fields: Iterable[str | Path], operators: str | Path, operator_modules: Iterable[str] = (),
        columns: Iterable[str] | None = None,
    ) -> "Vocabulary":
        """Read the field files ``fields`` and the operator file ``operators``.

        ``operator_modules`` are the modules besides KunQuant and
        ``quantlab.factor`` a class or a callable may come from; ``columns``,
        when given, are the price dataset's columns, among which every
        field's column must be.

        Raises
        ------
        VocabularyError
            With every problem of the files.
        """
        def document(path) -> dict:
            return {"path": str(path), "text": Path(path).read_text()}

        sources = {
            "fields": [document(path) for path in fields], "operators": document(operators),
            "operator_modules": list(operator_modules),
        }
        return cls(sources, columns)

    @classmethod
    def from_sources(cls, sources: dict) -> "Vocabulary":
        """Rebuild a vocabulary from another's ``sources``."""
        return cls(sources)

    # What candidate code may use.

    @property
    def names(self) -> frozenset[str]:
        """Every name candidate code may read without binding it first."""
        lag = {"PeriodLag"} if self.period_lags else set()
        return frozenset(self.fields) | frozenset(self._operators) | frozenset(BUILTINS) | {"Builder"} | lag

    @property
    def classes(self) -> dict[str, type]:
        """Every class a graph may hold, by the name it is recorded under (``Input`` included)."""
        return {"Input": Input, **{name: spec.cls for name, spec in self._operators.items()}}

    @property
    def operator_classes(self) -> dict[type, str]:
        """The registered classes, each with the name a graph records it under."""
        return {spec.cls: name for name, spec in self._operators.items()}

    def spec(self, name: str) -> OpSpec:
        """The operator a graph records as ``name``."""
        return INPUT if name == "Input" else self._operators[name]

    def class_paths(self, graphs: Iterable[list]) -> dict[str, str]:
        """The class path of every operator recorded graphs (``Graph.to_json()``) hold, by name."""
        used = {node[0] for graph in graphs for node in graph}
        return {name: self.spec(name).path for name in sorted(used)}

    @contextlib.contextmanager
    def namespace(self) -> Iterator[dict]:
        """Fresh globals for one execution of candidate code, which runs inside the block.

        The fields are new operators every time, so two executions share no
        node; inside the block the NaN-preserving stand-ins add the bar mask.
        """
        fields = self._inputs()
        callables = {name: spec.call for name, spec in self._operators.items()}
        lag = {"PeriodLag": self._period_lag_call()} if self.period_lags else {}
        with self._bar_mask(fields):
            yield {"__builtins__": dict(BUILTINS), **callables, "Builder": Builder, **lag, **fields}

    def recorded_name(self, op_class: type) -> str | None:
        """Return the name a graph records ``op_class`` under, or None when it is not registered.

        A class made at run time (``CrossSectionalWinsorize_0p05_0p95``,
        which quantlab's public class makes per bound pair) is recorded
        under the registered class it derives from; a declared class that is
        not registered is not.
        """
        registered = self.operator_classes
        for klass in op_class.__mro__:
            if klass is Input:
                return "Input"
            if klass in registered:
                return registered[klass]
            if klass.__name__.startswith("_") or _declared(klass):
                return None
        return None

    # What the Gate checks.

    def check(self, graph: Graph) -> tuple[list[str], str]:
        """Return why ``graph``'s parameter values or dimensions are refused (empty if none) and its dimension."""
        reasons = self._parameter_reasons(graph) or self._effective_reasons(graph)
        dimension, dimension_reasons = _dimension.infer(graph, self)
        return reasons + dimension_reasons, _dimension.name(dimension)

    def bind(self, hypothesis: str, fields: Iterable[str], operators: Iterable[str]) -> list[str]:
        """Return the mechanism-binding failures of ``hypothesis`` against the graph's ``fields`` and ``operators``.

        A deterministic check of the text against the graph (after
        FactorMiner's ``mechanism_binding``, MIT): it does not judge whether
        the story is right, only that the code uses what the story says it
        uses. A field phrase the hypothesis names needs one of the fields
        listing it as a synonym, and each field of a rule on it; an operator
        phrase needs one of the operators bound to it. Negations are not
        understood: the check errs toward passing.
        """
        text, read, used = hypothesis.lower(), set(fields), set(operators)
        found = []
        for name, pattern in self._field_phrases.items():
            if not pattern.search(text):
                continue
            groups = [set(self._synonyms[name])] if name in self._synonyms else []
            groups += [{field} for phrase, needs in self._rules if phrase == name for field in needs]
            missing = [group for group in groups if not group & read]
            if missing:
                wanted = " and ".join(" or ".join(sorted(group)) for group in missing)
                found.append(f"mechanism binding: the hypothesis names {name}, but the graph never reads {wanted}")
        for name, pattern in self._operator_phrases.items():
            ops = {op for op, spec in self._operators.items() if name in spec.binding}
            if ops and pattern.search(text) and not ops & used:
                found.append(
                    f"mechanism binding: the hypothesis names {name}, but the graph has none of {', '.join(sorted(ops))}"
                )
        return found

    def allowed(self, kind: str, value) -> bool:
        """Whether ``value`` is in the parameter set ``kind``; a constant may also be negative or ``"nan"``."""
        if kind == "constant" and value == "nan":
            return True
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        magnitude = abs(value) if kind == "constant" else value
        return any(
            math.isclose(magnitude, v, rel_tol=1e-9, abs_tol=0.0) or magnitude == v
            for v in self.parameters.get(kind, ())
        )

    def field_name(self, column: str) -> str:
        """The field reading ``column`` or a period lag of it, or the column itself when no field reads it."""
        if column in self._lag_columns:
            return self._lag_columns[column][0]
        return self._column_fields.get(column, column)

    def written(self, column: str) -> str:
        """``column`` as candidate code reads it: a field's name, or ``PeriodLag(field, k)`` for a lag column."""
        if column in self._lag_columns:
            name, k = self._lag_columns[column]
            return f"PeriodLag({name}, {k})"
        return self.field_name(column)

    def column_dimension(self, column: str) -> tuple | None:
        """The ``(dollars, shares)`` exponents of the field reading ``column`` or a period lag of it, or None."""
        name = self.field_name(column)
        return self.fields[name].dimension if name in self.fields else None

    @property
    def lag_columns(self) -> dict[str, dict]:
        """How each lag column candidate code may read is computed, by its name (``factor_agent.period_lag``)."""
        found = {}
        for lag_column, (name, k) in self._lag_columns.items():
            field = self.fields[name]
            lag = self.period_lags[field.family]
            found[lag_column] = {
                "column": field.column, "k": k, "key": lag.key, "days": lag.days,
                "tolerance_days": lag.tolerance_days, "bars": lag.bars,
            }
        return found

    @cached_property
    def reference_hashes(self) -> tuple[dict[str, str], str | None]:
        """The canonical hashes of the derived fields (hash -> name) and of the bar mask (None without one)."""
        fields = self._inputs()
        derived = {
            canonical_hashes(Graph.record(fields[f.name], self, strict=False))[-1]: f.name
            for f in self.fields.values() if f.formula is not None
        }
        if not self._bar_columns:
            return derived, None
        with self._bar_mask(fields):
            probe = kunquant_nan.SetInfOrNanToValue(fields[self._column_fields[self._bar_columns[0]]])
        return derived, canonical_hashes(Graph.record(probe.inputs[1], self, strict=False))[-1]

    # What the prompts show.

    def prompt(self) -> dict[str, str]:
        """The prompt's description of the vocabulary: inputs, operators, parameter values, builtins."""
        def operator(spec: OpSpec) -> str:
            return f"- {spec.signature}: {spec.description}" if spec.description else f"- {spec.signature}"

        lines = [operator(spec) for spec in self._operators.values()]
        if self.period_lags:
            lines.append(
                "- PeriodLag(x, k): field x as of the k-th earlier period, counted in periods, not in bars; "
                "missing when that period is missing or is not k periods before the current one. "
                "x is a field of a family below, k one of its values:"
            )
            for lag in self.period_lags.values():
                names = [f.name for f in self.fields.values() if f.family == lag.family and f.column is not None]
                lines.append(
                    f"  - {', '.join(names)} ({lag.family}): k one of {', '.join(map(str, lag.k))}; "
                    f"a period is {lag.description}"
                )
        return {
            "inputs": ", ".join(self.fields),
            "operators": "\n".join(lines),
            "parameters": "\n".join(
                f"- {kind}: {', '.join(f'{v:g}' for v in values)}" for kind, values in self.parameters.items()
            ),
            "builtins": ", ".join(sorted(BUILTINS)),
        }

    # Internals.

    @cached_property
    def _column_fields(self) -> dict[str, str]:
        return {f.column: f.name for f in self.fields.values() if f.column is not None}

    @cached_property
    def _lag_columns(self) -> dict[str, tuple[str, int]]:
        """``(field, k)`` of every lag column, by its name."""
        return {
            lag.lag_column(f.column, k): (f.name, k)
            for f in self.fields.values() if f.column is not None and f.family in self.period_lags
            for lag in [self.period_lags[f.family]] for k in lag.k
        }

    def _period_lag_call(self):
        """``PeriodLag`` of one execution of candidate code: one ``Input`` per lag column."""
        made: dict[str, OpBase] = {}
        families = ", ".join(self.period_lags)

        def period_lag(x, k):
            name = self._column_fields.get(x.attrs.get("name")) if isinstance(x, Input) else None
            if name is None:
                raise PeriodLagError(
                    "PeriodLag takes a field read from a column, not an expression or a derived field"
                )
            field = self.fields[name]
            lag = self.period_lags.get(field.family)
            if lag is None:
                raise PeriodLagError(
                    f"PeriodLag({name}, ...): {field.family} declares no period lag; "
                    f"only fields of {families} have one"
                )
            if isinstance(k, bool) or not isinstance(k, int) or k not in lag.k:
                raise PeriodLagError(
                    f"PeriodLag({name}, {k!r}): k must be one of {', '.join(map(str, lag.k))} for {field.family}"
                )
            column = lag.lag_column(field.column, k)
            return made.setdefault(column, Input(column))

        return period_lag

    def _inputs(self) -> dict[str, OpBase]:
        """Fresh input operators by field name: an ``Input`` per column, each derived field built by its formula."""
        fields: dict[str, OpBase] = {}
        for f in self.fields.values():
            fields[f.name] = Input(f.column) if f.column is not None else self._derive(f, fields)
        return fields

    def _derive(self, f: Field, fields: dict[str, OpBase]) -> OpBase:
        """Build a derived field over the ``fields`` before it; its formula passed ``_checks.expression_reasons``."""
        scope = {"__builtins__": {}, **{name: spec.call for name, spec in self._operators.items()}, **fields}
        return eval(compile(f.formula, f"<field {f.name}>", "eval"), scope)

    def _bar_mask(self, fields: dict):
        if not self._bar_columns:
            return contextlib.nullcontext()
        return kunquant_nan.missing_bars_only([fields[self._column_fields[c]] for c in self._bar_columns])

    def _parameter_reasons(self, graph: Graph) -> list[str]:
        reasons = []
        for node in graph.nodes:
            params = self.spec(node.op).params
            for key, value in node.attrs:
                kind = params.get(key)
                if kind is None:
                    reasons.append(f"{node.op} has an attribute {key!r} the gate does not know")
                elif kind != FLAG and not self.allowed(kind, value):
                    allowed = ", ".join(f"{v:g}" for v in sorted(set(self.parameters[kind])))
                    reasons.append(f"{node.op}: {kind} {value!r} is not one of the allowed values ({allowed})")
        return reasons

    def _effective_reasons(self, graph: Graph) -> list[str]:
        """Refuse values reached by chaining allowed ones, such as ``x * 0.25 * 1.5`` or a lag of 5 plus 10.

        A merged lag is checked against the ``lag`` set and a merged offset
        or scale against the ``constant`` set, when the operator file has them.
        """
        reasons = []
        for op, kind, value in effective_parameters(graph):
            checked = "constant" if kind == "scale" else kind
            if checked not in self.parameters:
                continue
            if kind == "scale":
                ok = value == 0 or self.allowed("constant", value) or self.allowed("constant", 1 / value)
            else:
                ok = self.allowed(kind, value) or (kind == "constant" and math.isnan(value))
            if not ok:
                reasons.append(
                    f"the constants combine to {op} {kind} {value:g}, which is not one of the allowed values; "
                    "use one allowed value instead of chaining several"
                )
        return list(dict.fromkeys(reasons))

    # Reading the files.

    def _problem(self, where: str, text: str) -> None:
        self._problems.append(f"{where}: {text}")

    def _yaml(self, document: dict) -> dict | None:
        try:
            value = yaml.safe_load(document["text"])
        except yaml.YAMLError as exc:
            self._problem(document["path"], f"not YAML: {exc}")
            return None
        if not isinstance(value, dict):
            self._problem(document["path"], "must be a mapping")
            return None
        return value

    def _unknown_keys(self, where: str, entry: dict, known: set) -> None:
        unknown = set(entry) - known
        if unknown:
            self._problem(where, f"unknown key(s) {sorted(unknown)}")

    def _phrases(self, where: str, raw) -> dict[str, re.Pattern]:
        phrases = {}
        for name, pattern in (raw or {}).items():
            try:
                phrases[str(name)] = re.compile(str(pattern))
            except re.error as exc:
                self._problem(where, f"phrase {name!r} is not a regular expression: {exc}")
        return phrases

    def _resolve(self, where: str, path, what: str) -> object | None:
        """Import ``path`` from an allowed module; None, with the problem recorded, if it cannot be."""
        if not isinstance(path, str) or not path:
            self._problem(where, f"{what} must be a dotted path, got {path!r}")
            return None
        if not any(path.startswith(module + ".") for module in self._modules):
            self._problem(
                where, f"{what} {path} is not from KunQuant, quantlab.factor or a module listed in operator_modules"
            )
            return None
        try:
            return resolve(path)
        except (ImportError, AttributeError) as exc:
            self._problem(where, f"{what} {path} cannot be imported: {exc}")
            return None

    def _name_ok(self, where: str, name) -> bool:
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
            self._problem(where, f"name {name!r} is not a Python identifier")
            return False
        if name in BUILTINS or name in _RESERVED:
            self._problem(where, f"the name {name} is a builtin of candidate code")
            return False
        return True

    def _read_operators(self, path: str, document: dict | None) -> None:
        if document is None:
            return
        self._unknown_keys(path, document, _OPERATOR_FILE_KEYS)
        for kind, values in (document.get("parameters") or {}).items():
            if not isinstance(values, list) or not all(_number(v) for v in values):
                self._problem(path, f"parameter set {kind!r} must be a list of numbers")
            else:
                self.parameters[str(kind)] = tuple(values)
        self._operator_phrases = self._phrases(path, document.get("phrases"))
        for i, entry in enumerate(document.get("operators") or []):
            self._read_operator(f"{path}: operator {i + 1}", entry)
        for class_path, reason in (document.get("excluded") or {}).items():
            where = f"{path}: excluded {class_path}"
            cls = self._resolve(where, class_path, "class")
            if cls is None:
                continue
            if not _is_operator(cls):
                self._problem(where, "is not a KunQuant operator class (an OpBase subclass)")
            elif cls in self.operator_classes:
                self._problem(where, f"is also registered as {self.operator_classes[cls]}")
            else:
                self.excluded[cls] = str(reason)

    def _read_operator(self, where: str, entry) -> None:
        if not isinstance(entry, dict):
            self._problem(where, "must be a mapping")
            return
        name = entry.get("name")
        where = f"{where} ({name})"
        self._unknown_keys(where, entry, _OPERATOR_KEYS)
        ok = self._name_ok(where, name)
        if ok and name in self._operators:
            self._problem(where, f"the name {name} is registered twice")
            ok = False
        cls = self._resolve(where, entry.get("class"), "class")
        if cls is not None and not _is_operator(cls):
            self._problem(where, f"class {entry['class']} is not a KunQuant operator class (an OpBase subclass)")
            cls = None
        call = cls
        if "call" in entry:
            call = self._resolve(where, entry["call"], "callable")
            if call is not None and not callable(call):
                self._problem(where, f"callable {entry['call']} cannot be called")
                call = None
        inputs = tuple(entry.get("inputs", ["series"]))
        for kind in inputs:
            if kind not in INPUT_KINDS:
                self._problem(where, f"input kind {kind!r} is not one of {list(INPUT_KINDS)}")
        params = dict(entry.get("params") or {})
        for attribute, kind in params.items():
            if kind != FLAG and kind not in self.parameters:
                self._problem(where, f"parameter {attribute!r} names the set {kind!r}, which the file does not define")
        dimension = entry.get("dimension")
        if dimension not in _dimension.OPERATOR_RULES:
            self._problem(where, f"dimension rule {dimension!r} is not one of {list(_dimension.OPERATOR_RULES)}")
        binding = tuple(entry.get("binding") or ())
        for phrase in binding:
            if phrase not in self._operator_phrases:
                self._problem(where, f"binding phrase {phrase!r} is not under the file's phrases")
        if ok and cls is not None and call is not None:
            self._operators[name] = OpSpec(
                name, cls, entry["class"], call, str(entry.get("signature", name)), inputs, params, dimension,
                str(entry.get("description", "")), binding,
            )

    def _read_fields(self, path: str, document: dict | None) -> None:
        if document is None:
            return
        self._unknown_keys(path, document, _FIELD_FILE_KEYS)
        family = str(document.get("family", Path(path).stem))
        self.families[family] = str(document.get("description") or "")
        for name, pattern in self._phrases(path, document.get("phrases")).items():
            if name in self._field_phrases and self._field_phrases[name].pattern != pattern.pattern:
                self._problem(path, f"phrase {name!r} is defined differently by another field file")
            self._field_phrases[name] = pattern
        mine: list[str] = []
        declared: list[str] = []  # every name the file gives, so a formula over a refused field adds no problem
        for i, entry in enumerate(document.get("fields") or []):
            name = self._read_field(f"{path}: field {i + 1}", family, entry, declared)
            if name is not None:
                mine.append(name)
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                declared.append(entry["name"])
        if document.get("bar"):
            self._bar_columns += [self.fields[n].column for n in mine if self.fields[n].column is not None]
        if document.get("period_lag") is not None:
            self._read_period_lag(f"{path}: period_lag", family, document["period_lag"])
        for i, rule in enumerate(document.get("rules") or []):
            where = f"{path}: rule {i + 1}"
            if not isinstance(rule, dict) or set(rule) != {"phrase", "needs"}:
                self._problem(where, "must be a mapping of phrase and needs")
                continue
            if rule["phrase"] not in self._field_phrases:
                self._problem(where, f"phrase {rule['phrase']!r} is not under the file's phrases")
            unknown = [f for f in rule["needs"] if f not in mine]
            if unknown:
                self._problem(where, f"needs field(s) {unknown} the family does not have")
            self._rules.append((rule["phrase"], tuple(rule["needs"])))

    def _read_field(self, where: str, family: str, entry, earlier: list[str]) -> str | None:
        if not isinstance(entry, dict):
            self._problem(where, "must be a mapping")
            return None
        name = entry.get("name")
        where = f"{where} ({name})"
        self._unknown_keys(where, entry, _FIELD_KEYS)
        ok = self._name_ok(where, name)
        if ok and name in self._operators:
            self._problem(where, f"the name {name} is also an operator")
            ok = False
        if ok and name in self.fields:
            self._problem(where, f"the name {name} is also a field of {self.fields[name].family}")
            ok = False
        column, formula, dimension = entry.get("column"), entry.get("formula"), None
        if (column is None) == (formula is None):
            self._problem(where, "needs exactly one of column and formula")
            ok = False
        elif column is not None:
            reader = next((f for f in self.fields.values() if f.column == column), None)
            if reader is not None:
                self._problem(where, f"column {column!r} is also read by field {reader.name}")
                ok = False
            dimension = _unit(entry.get("unit"))
            if dimension is None:  # still a field, so formulas over it report nothing more
                self._problem(where, f"unit {entry.get('unit')!r} is not a product of powers of dollars and shares")
        else:
            if "unit" in entry:
                self._problem(where, "a derived field's unit follows from its formula; give none")
            for reason in _checks.expression_reasons(str(formula), frozenset(earlier) | frozenset(self._operators)):
                self._problem(where, f"formula {formula!r}: {reason}")
                ok = False
        for phrase in entry.get("synonyms") or ():
            if phrase not in self._field_phrases:
                self._problem(where, f"synonym {phrase!r} is not under the file's phrases")
            elif ok:
                self._synonyms.setdefault(phrase, []).append(name)
        if not ok:
            return None
        self.fields[name] = Field(name, family, column, formula, dimension, str(entry.get("description", "")))
        return name

    def _read_period_lag(self, where: str, family: str, entry) -> None:
        if not isinstance(entry, dict):
            self._problem(where, "must be a mapping of column, k and period")
            return
        self._unknown_keys(where, entry, _PERIOD_LAG_KEYS)
        ok = True
        column = entry.get("column")
        placeholders = set()
        if isinstance(column, str):
            with contextlib.suppress(ValueError):
                placeholders = {f for _, f, _, _ in string.Formatter().parse(column) if f is not None}
        if placeholders != {"column", "k"}:
            self._problem(where, f"column {column!r} must be a pattern with exactly {{column}} and {{k}}")
            ok = False
        k = entry.get("k")
        if not isinstance(k, list) or not k or not all(_number(v) and v == int(v) and v > 0 for v in k):
            self._problem(where, f"k {k!r} must be a list of positive whole numbers")
            ok = False
        period = entry.get("period")
        if not isinstance(period, dict):
            self._problem(where, "period must be a mapping of description, key, days, tolerance_days and bars")
            return
        self._unknown_keys(f"{where}: period", period, _PERIOD_KEYS)
        for name in ("description", "key"):
            if not isinstance(period.get(name), str) or not period[name].strip():
                self._problem(where, f"period {name} must be a non-empty text, got {period.get(name)!r}")
                ok = False
        for name in ("days", "tolerance_days", "bars"):
            value = period.get(name)
            if not _number(value) or value <= 0 or (name == "bars" and value != int(value)):
                self._problem(where, f"period {name} must be a positive number, got {value!r}")
                ok = False
        if ok:
            self.period_lags[family] = PeriodLag(
                family, column, tuple(int(v) for v in k), period["description"], period["key"],
                float(period["days"]), float(period["tolerance_days"]), int(period["bars"]),
            )

    def _check_lag_columns(self) -> None:
        """A lag column may not be a field's column or a declared key."""
        keys = {lag.key for lag in self.period_lags.values()}
        for column, (name, k) in self._lag_columns.items():
            if column in self._column_fields or column in keys:
                self._problem(
                    f"period lag of {self.fields[name].family}",
                    f"PeriodLag({name}, {k}) reads {column!r}, which is also a column of the price dataset",
                )

    def _build_derived(self) -> None:
        """Build every derived field once, so a formula that cannot be built is a start-up error."""
        fields: dict[str, OpBase] = {}
        for f in self.fields.values():
            if f.column is not None:
                fields[f.name] = Input(f.column)
                continue
            where = f"field {f.name} ({f.family})"
            try:
                fields[f.name] = self._derive(f, fields)
            except Exception as exc:  # the formula's own error, reported with its field
                self._problem(where, f"formula {f.formula!r} cannot be built: {type(exc).__name__}: {exc}")
                return
            if not isinstance(fields[f.name], OpBase):
                self._problem(where, f"formula {f.formula!r} is not an operator")
                return


def resolve(path: str) -> object:
    """Import the object at ``path``: the longest importable module prefix, then attributes.

    Raises
    ------
    ImportError, AttributeError
        If no prefix imports or an attribute is missing.
    """
    parts = path.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            value = importlib.import_module(".".join(parts[:cut]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[cut:]:
            value = getattr(value, attribute)
        return value
    raise ImportError(f"no module of {path} can be imported")


def resolve_classes(paths: Mapping[str, str]) -> dict[str, type]:
    """The operator class at each path, by name, as ``Vocabulary.class_paths`` gave them.

    Raises
    ------
    TypeError
        If a path is not a KunQuant operator class.
    """
    classes = {name: resolve(path) for name, path in paths.items()}
    wrong = [paths[name] for name, cls in classes.items() if not _is_operator(cls)]
    if wrong:
        raise TypeError(f"not KunQuant operator classes: {wrong}")
    return classes


def _unit(text) -> tuple | None:
    """The ``(dollars, shares)`` exponents of a unit such as ``dollars/shares`` or ``1``; None if malformed."""
    if text == 1:
        text = "1"
    if not isinstance(text, str):
        return None
    exponents = [Fraction(0), Fraction(0)]
    sign = 1
    for i, token in enumerate(re.split(r"\s*([*/])\s*", text.strip())):
        if i % 2:
            sign = -1 if token == "/" else 1
            continue
        match = _UNIT_FACTOR.fullmatch(token)
        if match is None:
            return None
        if match.group(1) != "1":
            exponents[_UNIT_BASES.index(match.group(1))] += sign * int(match.group(2) or 1)
    return tuple(exponents)


def _is_operator(cls) -> bool:
    return isinstance(cls, type) and issubclass(cls, OpBase)


def _declared(klass: type) -> bool:
    """Whether ``klass`` is its module's attribute of its name: declared, not made at run time."""
    try:
        module = importlib.import_module(klass.__module__)
    except ImportError:
        return False
    return getattr(module, klass.__name__, None) is klass


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
