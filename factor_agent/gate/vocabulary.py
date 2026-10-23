"""The operator vocabulary, shared by the Gate and the proposer's prompt: the names candidate code may use, and the classes a graph may hold.

Candidate code sees the price inputs, the operators below and a few harmless
builtins, nothing else. The inputs follow ``Alpha158Stock``: the adjusted
OHLCV columns, the adjusted typical price as ``vwap`` and ``vwap * volume``
as ``amount``.
"""

import builtins

import KunQuant.ops as kq
from KunQuant.Op import Builder, ConstantOp, ForeachBackWindow, Input, IterValue, Rank, Scale, WindowLoopIndex
from quantlab.factor import kunquant_cs, kunquant_ts

#: Candidate input name -> the dataset column it reads.
INPUT_COLUMNS = {
    "open": "adjOpen",
    "high": "adjHigh",
    "low": "adjLow",
    "close": "adjClose",
    "volume": "adjVolume",
}

#: Names of the derived inputs, built from the columns in ``namespace``.
DERIVED_INPUTS = ("vwap", "amount")

#: Operator classes candidate code may construct, by name.
OPERATORS = {
    cls.__name__: cls
    for cls in (
        # Elementwise.
        kq.Abs, kq.Add, kq.AddConst, kq.Sub, kq.SubConst, kq.Mul, kq.MulConst,
        kq.Div, kq.DivConst, kq.Exp, kq.Log, kq.Sqrt, kq.Sign, kq.Pow, ConstantOp,
        # Time series.
        kq.BackRef, kq.WindowedAvg, kq.WindowedSum, kq.WindowedStddev, kq.WindowedVar,
        kq.WindowedMax, kq.WindowedMin, kq.WindowedCorrelation, kq.WindowedCovariance,
        kq.WindowedSkew, kq.WindowedKurt, kq.WindowedQuantile, kq.WindowedProduct,
        kq.WindowedMaxDrawdown, kq.WindowedLinearRegressionSlope,
        kq.WindowedLinearRegressionRSqaure, kq.WindowedLinearRegressionResi,
        kq.TsRank, kq.TsArgMax, kq.TsArgMin, kq.DecayLinear, kq.ExpMovingAvg,
        kunquant_ts.WindowedZScore, kunquant_ts.EWSum, kunquant_ts.EWMean, kunquant_ts.EWVar,
        kunquant_ts.EWCov, kunquant_ts.EWBeta, kunquant_ts.EWAlpha, kunquant_ts.EWResidualStd,
        kunquant_ts.CMRA,
        # Window loops.
        ForeachBackWindow, IterValue, WindowLoopIndex, kq.ReduceAdd, kq.ReduceMul,
        kq.ReduceMax, kq.ReduceMin, kq.ReduceArgMax, kq.ReduceArgMin, kq.ReduceRank,
        kq.ReduceDecayLinear,
        # Cross-sectional.
        Rank, Scale, kunquant_cs.CrossSectionalZScore, kunquant_cs.CrossSectionalWinsorize,
        kunquant_cs.CrossSectionalTrim,
    )
}

#: Every class a recorded graph may hold, by the name it is recorded under.
GRAPH_CLASSES = {**OPERATORS, "Input": Input}

#: The builtins candidate code may call.
BUILTINS = {
    name: getattr(builtins, name)
    for name in ("abs", "bool", "dict", "enumerate", "float", "int", "len", "list", "range",
                 "round", "tuple", "zip")
}

#: Every name candidate code may read without binding it first.
ALLOWED_NAMES = frozenset(INPUT_COLUMNS) | frozenset(DERIVED_INPUTS) | frozenset(OPERATORS) | frozenset(BUILTINS) | {"Builder"}


def namespace() -> dict:
    """Return fresh globals for one execution of candidate code."""
    inputs = {name: Input(column) for name, column in INPUT_COLUMNS.items()}
    vwap = (inputs["high"] + inputs["low"] + inputs["close"]) / 3.0
    return {
        "__builtins__": dict(BUILTINS),
        **OPERATORS,
        "Builder": Builder,
        **inputs,
        "vwap": vwap,
        "amount": vwap * inputs["volume"],
    }
