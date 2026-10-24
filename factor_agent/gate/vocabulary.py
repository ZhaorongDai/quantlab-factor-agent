"""The operator vocabulary, shared by the Gate and the proposer's prompt: the operator registry, the parameter sets and the namespace candidate code runs in.

Every whitelisted KunQuant and quantlab operator is an ``OpSpec`` in
``OPERATORS``: the class a graph records, what candidate code calls under
that name, its series inputs, its parameters and its dimension rule. Every
other ``OpBase`` subclass of KunQuant and quantlab is in ``EXCLUDED`` with
the reason; a guard test keeps the two lists complete.

KunQuant's ``SetInfOrNanToValue``, ``Clip``, ``Max``, ``Min`` and ``Select``
turn a bar with no data into a number. Candidate code calls quantlab's
stand-ins from ``quantlab.factor.kunquant_nan`` under those names, and the
sandbox runs it inside ``missing_bars_only``, so a symbol without a bar stays
NaN as it does in ``Alpha158Stock``.

The inputs follow ``Alpha158Stock``: the adjusted OHLCV columns, the adjusted
typical price as ``vwap`` and ``vwap * volume`` as ``amount``.
"""

import builtins
from dataclasses import dataclass, field

import KunQuant.ops as kq
from KunQuant.Op import (
    BinaryElementwiseOp, Builder, CompositiveOp, ConstantOp, CrossSectionalOp, ForeachBackWindow, Input,
    IterValue, Output, Rank, ReductionOp, Scale, SimpleCrossSectionalOp, UnaryElementwiseOp,
    WindowedDataSourceOp, WindowedTempOutput, WindowLoopIndex,
)
from KunQuant.ops import CompOp, ElewiseOp, MiscOp
from KunQuant.predefined import talib
from quantlab.factor import kunquant_cs, kunquant_nan, kunquant_ts

#: Candidate input name -> the dataset column it reads.
INPUT_COLUMNS = {
    "open": "adjOpen",
    "high": "adjHigh",
    "low": "adjLow",
    "close": "adjClose",
    "volume": "adjVolume",
}

#: Names of the derived inputs, built from the columns in ``inputs``.
DERIVED_INPUTS = ("vwap", "amount")

#: Dimension of each dataset column as ``(price exponent, volume exponent)``.
COLUMN_DIMENSIONS = {
    "adjOpen": (1, 0),
    "adjHigh": (1, 0),
    "adjLow": (1, 0),
    "adjClose": (1, 0),
    "adjVolume": (0, 1),
}

#: Window lengths, in bars, a time-series operator or loop may use.
WINDOWS = (5, 10, 20, 40, 60, 120, 250)

#: Values each kind of operator parameter may take; a constant may also be
#: negative or ``"nan"``.
PARAMETER_VALUES = {
    "window": WINDOWS,
    "lag": (1, *WINDOWS),
    "half_life": WINDOWS,
    "quantile": (0.1, 0.25, 0.5, 0.75, 0.9),
    "lower": (0.01, 0.025, 0.05, 0.1),
    "upper": (0.9, 0.95, 0.975, 0.99),
    "months": (3, 6, 12),
    "month_length": (21,),
    "constant": (0, 1e-12, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 10, 100, *WINDOWS),
}

#: Attribute kinds that are bookkeeping, not values, and are never checked against a set.
FLAG_PARAMETERS = frozenset({"swap", "column"})


@dataclass(frozen=True)
class OpSpec:
    """One whitelisted operator.

    ``cls`` is the class a graph records; ``call`` is what candidate code
    calls under the operator's name: ``cls`` itself, or quantlab's
    NaN-preserving stand-in. ``inputs`` gives the kind of each series
    argument: ``"series"``, ``"condition"`` (a comparison) or ``"loop"`` (a
    ``ForeachBackWindow``). ``params`` maps each attribute to its kind in
    ``PARAMETER_VALUES`` or ``FLAG_PARAMETERS``. ``dimension`` names the rule
    by which the output's dimension follows from the inputs'. ``signature``
    is how the prompt shows the call.
    """

    cls: type
    signature: str
    dimension: str
    inputs: tuple[str, ...] = ("series",)
    params: dict = field(default_factory=dict)
    call: object = None
    callable: bool = True


def _spec(cls, signature, dimension, inputs=("series",), params=None, call=None, callable=True):
    return cls.__name__, OpSpec(cls, signature, dimension, tuple(inputs), dict(params or {}), call or cls, callable)


_SERIES2 = ("series", "series")
_CONST = {"value": "constant", "swap": "swap"}
_WINDOW = {"window": "window"}
_EW = {"window": "window", "half_life": "half_life"}
_BOUNDS = {"lower": "lower", "upper": "upper"}

#: Every whitelisted operator by the name candidate code calls it, plus
#: ``Input``, which graphs record but candidate code cannot call.
OPERATORS: dict[str, OpSpec] = dict([
    # Elementwise.
    _spec(kq.Add, "Add(x, y)", "match", _SERIES2),
    _spec(kq.Sub, "Sub(x, y)", "match", _SERIES2),
    _spec(kq.Mul, "Mul(x, y)", "product", _SERIES2),
    _spec(kq.Div, "Div(x, y)", "quotient", _SERIES2),
    _spec(kq.AddConst, "AddConst(x, c)", "same", params=_CONST),
    _spec(kq.SubConst, "SubConst(x, c)", "same", params=_CONST),
    _spec(kq.MulConst, "MulConst(x, c)", "scale", params=_CONST),
    _spec(kq.DivConst, "DivConst(x, c)", "scale", params=_CONST),
    _spec(ConstantOp, "ConstantOp(c)", "constant", (), {"value": "constant"}),
    _spec(kq.Abs, "Abs(x)", "same"),
    _spec(kq.Sign, "Sign(x)", "pure"),
    _spec(kq.Exp, "Exp(x)", "transcendental"),
    _spec(kq.Log, "Log(x)", "transcendental"),
    _spec(kq.Sqrt, "Sqrt(x)", "sqrt"),
    _spec(kq.Pow, "Pow(x, y)", "power", _SERIES2),
    _spec(kq.Max, "Max(x, y)", "match", _SERIES2, call=kunquant_nan.Max),
    _spec(kq.Min, "Min(x, y)", "match", _SERIES2, call=kunquant_nan.Min),
    _spec(kq.Clip, "Clip(x, c)", "same", params={"value": "constant"}, call=kunquant_nan.Clip),
    _spec(kq.SetInfOrNanToValue, "SetInfOrNanToValue(x, c=0.0)", "same", params={"value": "constant"},
          call=kunquant_nan.SetInfOrNanToValue),
    _spec(kq.Select, "Select(condition, x, y)", "select", ("condition", "series", "series"),
          call=kunquant_nan.Select),
    # Comparisons and logic; their result is only a Select condition.
    _spec(kq.GreaterThan, "GreaterThan(x, y)", "compare", _SERIES2),
    _spec(kq.GreaterEqual, "GreaterEqual(x, y)", "compare", _SERIES2),
    _spec(kq.LessThan, "LessThan(x, y)", "compare", _SERIES2),
    _spec(kq.LessEqual, "LessEqual(x, y)", "compare", _SERIES2),
    _spec(kq.Equals, "Equals(x, y)", "compare", _SERIES2),
    _spec(kq.GreaterThanConst, "GreaterThanConst(x, c)", "compare_constant", params=_CONST),
    _spec(kq.LessThanConst, "LessThanConst(x, c)", "compare_constant", params=_CONST),
    _spec(kq.And, "And(a, b)", "logic", ("condition", "condition")),
    _spec(kq.Or, "Or(a, b)", "logic", ("condition", "condition")),
    _spec(kq.Not, "Not(a)", "logic", ("condition",)),
    # Time series.
    _spec(kq.BackRef, "BackRef(x, lag)", "same", params={"window": "lag"}),
    _spec(kq.WindowedAvg, "WindowedAvg(x, window)", "same", params=_WINDOW),
    _spec(kq.WindowedSum, "WindowedSum(x, window)", "same", params=_WINDOW),
    _spec(kq.WindowedStddev, "WindowedStddev(x, window)", "same", params=_WINDOW),
    _spec(kq.WindowedVar, "WindowedVar(x, window)", "square", params=_WINDOW),
    _spec(kq.WindowedMax, "WindowedMax(x, window)", "same", params=_WINDOW),
    _spec(kq.WindowedMin, "WindowedMin(x, window)", "same", params=_WINDOW),
    _spec(kq.WindowedCorrelation, "WindowedCorrelation(x, window, y)", "pure", _SERIES2, _WINDOW),
    _spec(kq.WindowedCovariance, "WindowedCovariance(x, window, y)", "product", _SERIES2, _WINDOW),
    _spec(kq.WindowedSkew, "WindowedSkew(x, window)", "pure", params=_WINDOW),
    _spec(kq.WindowedKurt, "WindowedKurt(x, window)", "pure", params=_WINDOW),
    _spec(kq.WindowedQuantile, "WindowedQuantile(x, window, q)", "same",
          params={"window": "window", "q": "quantile"}),
    _spec(kq.WindowedProduct, "WindowedProduct(x, window)", "repeated_product", params=_WINDOW),
    _spec(kq.WindowedMaxDrawdown, "WindowedMaxDrawdown(x, window)", "pure", params=_WINDOW),
    _spec(kq.WindowedLinearRegressionSlope, "WindowedLinearRegressionSlope(x, window)", "same", params=_WINDOW),
    _spec(kq.WindowedLinearRegressionRSqaure, "WindowedLinearRegressionRSqaure(x, window)", "pure", params=_WINDOW),
    _spec(kq.WindowedLinearRegressionResi, "WindowedLinearRegressionResi(x, window)", "same", params=_WINDOW),
    _spec(kq.TsRank, "TsRank(x, window)", "pure", params=_WINDOW),
    _spec(kq.TsArgMax, "TsArgMax(x, window)", "pure", params=_WINDOW),
    _spec(kq.TsArgMin, "TsArgMin(x, window)", "pure", params=_WINDOW),
    _spec(kq.DecayLinear, "DecayLinear(x, window)", "same", params=_WINDOW),
    _spec(kq.ExpMovingAvg, "ExpMovingAvg(x, window)", "same", params=_WINDOW),
    _spec(kunquant_ts.WindowedZScore, "WindowedZScore(x, window)", "pure", params=_WINDOW),
    _spec(kunquant_ts.EWSum, "EWSum(x, window, half_life)", "same", params=_EW),
    _spec(kunquant_ts.EWMean, "EWMean(x, window, half_life)", "same", params=_EW),
    _spec(kunquant_ts.EWVar, "EWVar(x, window, half_life)", "square", params=_EW),
    _spec(kunquant_ts.EWCov, "EWCov(y, x, window, half_life)", "product", _SERIES2, _EW),
    _spec(kunquant_ts.EWBeta, "EWBeta(y, x, window, half_life)", "quotient", _SERIES2, _EW),
    _spec(kunquant_ts.EWAlpha, "EWAlpha(y, x, window, half_life)", "first", _SERIES2, _EW),
    _spec(kunquant_ts.EWResidualStd, "EWResidualStd(y, x, window, half_life)", "first", _SERIES2, _EW),
    _spec(kunquant_ts.CMRA, "CMRA(x, months=12, month_length=21)", "same",
          params={"months": "months", "month_length": "month_length"}),
    # Window loops: `loop = ForeachBackWindow(x, window)`, then inside
    # `with Builder(loop):` `item = IterValue(loop, x)`, then a Reduce* outside the block.
    _spec(ForeachBackWindow, "ForeachBackWindow(x, window, *more_series)", "loop", ("series",), _WINDOW),
    _spec(IterValue, "IterValue(loop, x)", "second", ("loop", "series")),
    _spec(WindowLoopIndex, "WindowLoopIndex(loop)", "pure", ("loop",)),
    _spec(kq.ReduceAdd, "ReduceAdd(item)", "same"),
    _spec(kq.ReduceMul, "ReduceMul(item)", "repeated_product"),
    _spec(kq.ReduceMax, "ReduceMax(item)", "same"),
    _spec(kq.ReduceMin, "ReduceMin(item)", "same"),
    _spec(kq.ReduceArgMax, "ReduceArgMax(item)", "pure"),
    _spec(kq.ReduceArgMin, "ReduceArgMin(item)", "pure"),
    _spec(kq.ReduceRank, "ReduceRank(item)", "pure"),
    _spec(kq.ReduceDecayLinear, "ReduceDecayLinear(item)", "same"),
    # Cross-sectional, across the whole market on each bar.
    _spec(Rank, "Rank(x)", "pure"),
    _spec(Scale, "Scale(x)", "pure"),
    _spec(kunquant_cs.CrossSectionalZScore, "CrossSectionalZScore(x)", "pure"),
    _spec(kunquant_cs.CrossSectionalWinsorize, "CrossSectionalWinsorize(x, lower=0.01, upper=0.99)", "same",
          params=_BOUNDS),
    _spec(kunquant_cs.CrossSectionalTrim, "CrossSectionalTrim(x, lower=0.01, upper=0.99)", "same",
          params=_BOUNDS),
    # Recorded only: a dataset column.
    _spec(Input, "Input(column)", "input", (), {"name": "column"}, callable=False),
])

_BASE = "an abstract base class, never in a graph"
_INTERNAL = "a KunQuant internal made by its lowering passes"
_EXTRA_INPUTS = "needs inputs beyond price and volume; the Evaluator applies it"
_TALIB = "a KunQuant predefined composite; candidates write the formula themselves"

#: Every other public ``OpBase`` subclass of KunQuant and quantlab, with why it is
#: excluded. A class whose name starts with an underscore is a private base class
#: of its library and is excluded without being named.
EXCLUDED: dict[type, str] = {
    UnaryElementwiseOp: _BASE,
    BinaryElementwiseOp: _BASE,
    ElewiseOp.BinaryConstOp: _BASE,
    ElewiseOp.CmpOp: _BASE,
    CompositiveOp: _BASE,
    CompOp.WindowedCompositiveOp: _BASE,
    CompOp.WindowedReduce: _BASE,
    CompOp.WindowedLinearRegressionBase: _BASE,
    CrossSectionalOp: _BASE,
    SimpleCrossSectionalOp: _BASE,
    MiscOp.GenericCrossSectionalOp: _BASE,
    ReductionOp: _BASE,
    WindowedDataSourceOp: _BASE,
    MiscOp.WindowedLinearRegressionImplUnaryBase: _BASE,
    MiscOp.WindowedLinearRegressionImplBinaryBase: _BASE,
    Output: "the graph's sink, added by the Gate and the Evaluator",
    WindowedTempOutput: _INTERNAL,
    MiscOp.WindowedLinearRegressionRSqaureImpl: _INTERNAL,
    MiscOp.WindowedLinearRegressionSlopeImpl: _INTERNAL,
    MiscOp.WindowedLinearRegressionResiImpl: _INTERNAL,
    MiscOp.WindowedLinearRegression: _INTERNAL,
    MiscOp.SkipListState: _INTERNAL,
    MiscOp.SkipListMin: _INTERNAL,
    MiscOp.SkipListMax: _INTERNAL,
    MiscOp.SkipListRank: _INTERNAL,
    MiscOp.SkipListQuantile: _INTERNAL,
    MiscOp.SkipListArgMin: _INTERNAL,
    MiscOp.FastWindowedSum: _INTERNAL,
    MiscOp.Accumulator: _INTERNAL,
    MiscOp.SetAccumulator: _INTERNAL,
    MiscOp.ReturnFirstValue: _INTERNAL,
    MiscOp.GenericPartition: _INTERNAL,
    MiscOp.DiffWithWeightedSum: _EXTRA_INPUTS,
    kunquant_cs.CrossSectionalWeightedMean: _EXTRA_INPUTS,
    kunquant_cs.CapWeightedStandardize: _EXTRA_INPUTS,
    kunquant_cs.CrossSectionalSigmaClip: _EXTRA_INPUTS,
    kunquant_cs.CrossSectionalWLSResidual: _EXTRA_INPUTS,
    kunquant_cs.CrossSectionalIndustrySizeFill: _EXTRA_INPUTS,
    kunquant_cs.CrossSectionalNeutralize: _EXTRA_INPUTS,
    kunquant_cs.RenormalizedCombine: "combines several factors with weights, which stacks signals",
    kunquant_cs.CrossSectionalTopN: "selects a portfolio, not a factor value",
    talib.TRANGE: _TALIB,
    talib.ATR: _TALIB,
    talib.SAR: _TALIB,
}

#: Every class a recorded graph may hold, by the name it is recorded under.
GRAPH_CLASSES = {name: spec.cls for name, spec in OPERATORS.items()}

#: The operators candidate code may call, by name.
CALLABLE = {name: spec.call for name, spec in OPERATORS.items() if spec.callable}

#: The builtins candidate code may call.
BUILTINS = {
    name: getattr(builtins, name)
    for name in ("abs", "bool", "dict", "enumerate", "float", "int", "len", "list", "range",
                 "round", "tuple", "zip")
}

#: Every name candidate code may read without binding it first.
ALLOWED_NAMES = (
    frozenset(INPUT_COLUMNS) | frozenset(DERIVED_INPUTS) | frozenset(CALLABLE) | frozenset(BUILTINS) | {"Builder"}
)

_NAMES = {spec.cls: name for name, spec in OPERATORS.items()}


def recorded_name(op_class: type) -> str | None:
    """Return the name a graph records ``op_class`` under, or None when it is not whitelisted.

    quantlab's quantile-bound operators construct a subclass per bound pair
    (``CrossSectionalWinsorize_0p05_0p95``); such a subclass is recorded
    under its public class's name.
    """
    for klass in op_class.__mro__:
        if klass in _NAMES:
            return _NAMES[klass]
        if klass in EXCLUDED or klass.__name__.startswith("_"):
            return None
    return None


def inputs() -> dict:
    """Return fresh input operators by candidate name: the five columns, ``vwap`` and ``amount``."""
    columns = {name: Input(column) for name, column in INPUT_COLUMNS.items()}
    vwap = (columns["high"] + columns["low"] + columns["close"]) / 3.0
    return {**columns, "vwap": vwap, "amount": vwap * columns["volume"]}


def bar_mask_inputs(fields: dict) -> list:
    """Return the inputs whose missing values make a bar missing, for ``missing_bars_only``."""
    return [fields[name] for name in INPUT_COLUMNS]


def namespace(fields: dict) -> dict:
    """Return fresh globals for one execution of candidate code over the input operators ``fields``."""
    return {"__builtins__": dict(BUILTINS), **CALLABLE, "Builder": Builder, **fields}


def describe() -> dict[str, str]:
    """Return the prompt's description of the vocabulary: inputs, operators, windows, constants, builtins."""
    return {
        "inputs": ", ".join([*INPUT_COLUMNS, *DERIVED_INPUTS]),
        "operators": "\n".join(f"- {spec.signature}" for spec in OPERATORS.values() if spec.callable),
        "parameters": "\n".join(
            f"- {kind}: {', '.join(str(v) for v in values)}" for kind, values in PARAMETER_VALUES.items()
        ),
        "builtins": ", ".join(sorted(BUILTINS)),
    }
