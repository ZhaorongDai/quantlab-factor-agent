"""Mechanism binding: every input and transformation the hypothesis names must be in the graph.

A deterministic check of the hypothesis text against the graph (after
FactorMiner's ``mechanism_binding``, MIT): it does not judge whether the
story is right, only that the code uses what the story says it uses. Each
rule is a phrase pattern and what the graph must hold for it: one input
field out of each group, or one operator out of a set. Phrases that are
also ordinary words ("high price", "the low end", "open") are not rules,
and negations are not understood: the check errs toward passing.
"""

import re

_VOLUME = ({"volume", "amount"},)
_EW = {"EWSum", "EWMean", "EWVar", "EWCov", "EWBeta", "EWAlpha", "EWResidualStd", "ExpMovingAvg"}

#: (pattern, what the reason calls it, groups of fields; one field of each group must be read).
FIELD_RULES = [
    (r"\bvolumes?\b(?![- ]weighted)", "volume", _VOLUME),
    (r"\bturnover\b", "turnover", _VOLUME),
    (r"\btraded value\b", "traded value", _VOLUME),
    (r"\bvwap\b|\bvolume[- ]weighted average price\b", "VWAP", ({"vwap", "amount"},)),
    (r"\bopen(?:ing)? prices?\b|\bopening\b|\bthe open\b", "opening price", ({"open"},)),
    (r"\bclos(?:e|ing) prices?\b|\bclosing\b|\bthe close\b", "closing price", ({"close"},)),
    (r"\b(?:intraday|daily|session) highs?\b", "the intraday high", ({"high"},)),
    (r"\b(?:intraday|daily|session) lows?\b", "the intraday low", ({"low"},)),
    (r"\bhigh[- ]low\b|\b(?:intraday|daily|trading) range\b", "the high-low range", ({"high"}, {"low"})),
    (r"\bovernight\b", "overnight", ({"open"}, {"close"})),
]

#: (pattern, what the reason calls it, operators of which the graph must hold one).
OPERATOR_RULES = [
    (r"\bvolatilit(?:y|ies)\b|\bstandard deviations?\b|\bvariances?\b", "volatility",
     {"WindowedStddev", "WindowedVar", "EWVar", "EWResidualStd", "WindowedZScore"}),
    (r"\bcorrelat(?:ion|ions|ed)\b", "correlation", {"WindowedCorrelation"}),
    (r"\bcovariances?\b", "covariance", {"WindowedCovariance", "EWCov"}),
    (r"\bbetas?\b", "beta", {"EWBeta", "WindowedLinearRegressionSlope", "WindowedCovariance", "EWCov"}),
    (r"\bregressions?\b|\bresiduals?\b", "regression",
     {"EWBeta", "EWAlpha", "EWResidualStd", "WindowedLinearRegressionSlope", "WindowedLinearRegressionRSqaure",
      "WindowedLinearRegressionResi"}),
    (r"\bskew(?:ness)?\b", "skewness", {"WindowedSkew"}),
    (r"\bkurtosis\b", "kurtosis", {"WindowedKurt"}),
    (r"\bdrawdowns?\b", "drawdown", {"WindowedMaxDrawdown", "WindowedMax", "ReduceMax"}),
    (r"\bmoving averages?\b", "moving average",
     {"WindowedAvg", "WindowedSum", "DecayLinear", "ReduceAdd", "ReduceDecayLinear"} | _EW),
    (r"\bexponential(?:ly)?\b", "exponential weighting", _EW),
]


def reasons(hypothesis: str, fields: list[str], operators: set[str]) -> list[str]:
    """Return the binding failures of ``hypothesis`` against the graph's input ``fields`` and ``operators``."""
    text = hypothesis.lower()
    found = []
    for pattern, name, groups in FIELD_RULES:
        if re.search(pattern, text):
            missing = [group for group in groups if not group & set(fields)]
            if missing:
                wanted = " and ".join(" or ".join(sorted(group)) for group in missing)
                found.append(f"mechanism binding: the hypothesis names {name}, but the graph never reads {wanted}")
    for pattern, name, ops in OPERATOR_RULES:
        if re.search(pattern, text) and not ops & operators:
            found.append(
                f"mechanism binding: the hypothesis names {name}, but the graph has none of {', '.join(sorted(ops))}"
            )
    return found
