"""The Evaluator: the only module that reads data.

It is fixed to one segment when it is made. ``evaluate`` compiles all of a
round's graphs as one KunQuant graph and computes it on the whole-market
dataset (the segment plus the warm-up bars before it). Each candidate, and
each reference factor, is then scored by the campaign's ``Scoring``: ranked
across the whole market (KunQuant ``Rank``), neutralised when the scoring
neutralises, and masked to the members of day t, or kept on every symbol
without a membership. A batch that does
not compile or run is bisected until the failing graphs are found; they
become ``CompileFailure`` and the rest are evaluated.

Only dates whose label lies entirely inside the segment are scored (the
purge of ``segments``), and on each date only the members with a label.
The statistics of a candidate:

- IC: the daily rank IC against the forward label, its mean and the
  Newey-West t of the mean.
- Correlation: the daily correlation of normal scores (below) with every
  factor of the reference pool (the reference set and the factors accepted
  so far), averaged over dates; the pool factor with the largest absolute
  mean is the nearest. It is the van der Waerden rank correlation, within a
  few hundredths of Spearman's.
- Partial IC: the rank IC of the residual of the candidate's normal scores
  from a daily least-squares regression on the normal scores of every pool
  factor (with an intercept), and its Newey-West t.
- Deciles: the mean label of each tenth of the candidate's daily cross
  section, lowest values first, averaged over dates.
- Coverage: the share of the day's scored members on which the graph
  itself (before the rank, which ranks a missing value too) has a value,
  averaged over the dates with any scored member.

Normal scores: on each date a factor's values are ranked among its own
finite symbols and mapped to the standard normal quantiles of their ranks;
a missing value scores 0, the middle rank. A candidate that smooths or
rescales a pool factor is then a linear function of it plus noise, so the
regression removes what the pool factor explains, which ranks would not.

The reference pool grows as candidates are admitted: ``evaluate`` hands each
candidate's metrics to an ``admit`` callback in order, and a candidate it
admits joins the pool before the next one is scored, so two candidates of
one round are never both accepted when the second says nothing new.
"""

import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from KunQuant.Stage import Function
from quantlab.analysis.factor_report import newey_west_lags, newey_west_t_stat
from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.model.evaluation import cross_sectional_rank_ic_series
from scipy import stats

from factor_agent.exported import ranked_function
from factor_agent.gate.graph import Graph
from factor_agent.scoring import Scoring
from factor_agent.segments import PURGE_BARS, Segment, purge
from factor_agent.zoo import owner

#: Groups the daily cross section is cut into for the decile returns.
DECILES = 10

#: Singular values below this fraction of the largest are dropped from a
#: day's pool regression, so collinear pool factors do not blow it up.
_RANK_TOLERANCE = 1e-6

#: An admitted candidate whose residual norm on a day is below this adds no
#: direction to that day's pool (it is already spanned).
_SPANNED_NORM = 1e-9


@dataclass(frozen=True)
class Metrics:
    """A candidate's training-segment statistics; see the module docstring.

    ``nearest`` names the pool factor most correlated with the candidate
    (empty, with a NaN correlation, when the pool is empty) and
    ``nearest_correlation`` is that mean correlation, with its sign.
    ``decile_returns`` holds the mean label of each decile, lowest
    candidate values first. ``coverage`` is the mean daily share of the
    scored members with a value.
    """

    ic_mean: float
    ic_t: float
    partial_ic_mean: float
    partial_ic_t: float
    nearest: str
    nearest_correlation: float
    decile_returns: tuple[float, ...]
    coverage: float
    n_days: int
    start: str
    end: str

    def to_dict(self) -> dict:
        """Return the metrics as the ledger stores them."""
        return {**vars(self), "decile_returns": list(self.decile_returns)}


@dataclass(frozen=True)
class Contribution:
    """A pool factor's part in a group of pool factors.

    ``weight`` is its mean daily coefficient in the least-squares regression
    of the label's normal scores on the group's normal scores (with an
    intercept): a unitless marginal weight. ``residual_ic`` is the mean
    daily rank IC of what is left of it after regressing it on the rest of
    the group.
    """

    weight: float
    residual_ic: float


@dataclass(frozen=True)
class CompileFailure:
    """The candidate's graph did not compile or run; the error KunQuant gave."""

    error: str


#: Called with a candidate's position among the graphs, its metrics and the
#: number of the batch's graphs that ran, in order; returns the name under
#: which the candidate joins the reference pool, or None to leave it out.
Admit = Callable[[int, Metrics, int], str | None]


class _RoundGraph(FactorKunQuant):
    """A batch of graphs as one factor, one whole-market rank per candidate.

    ``compute`` keeps its last result, so the scored factor that wraps it
    (a ``NeutralizedFactor`` when the scoring neutralises) reads the panel
    the bisection already computed instead of computing it again. Only the
    candidate outputs are neutralised; the presence outputs are read raw.
    """

    def __init__(self, config: FactorConfig, graphs: list[Graph]):
        self._graphs = graphs
        self._computed: tuple[tuple, xr.Dataset] | None = None
        super().__init__(config)

    def _get_factor_names(self) -> tuple[str, ...]:
        n = len(self._graphs)
        return tuple(_output_name(i) for i in range(n)) + tuple(_presence_name(i) for i in range(n))

    def _get_factor_func(self) -> Function:
        return ranked_function(
            {_output_name(i): graph for i, graph in enumerate(self._graphs)},
            presence={_output_name(i): _presence_name(i) for i in range(len(self._graphs))},
        )

    def compute(self, start, end) -> xr.Dataset:
        """Compute the batch from ``start`` to ``end``, reusing the last result for the same range."""
        if self._computed is None or self._computed[0] != (start, end):
            self._computed = ((start, end), super().compute(start, end))
        return self._computed[1]


def _missing(name: str):
    raise KeyError(f"{name!r} is not in the reference pool")


def _output_name(i: int) -> str:
    return f"candidate_{i}"


def _presence_name(i: int) -> str:
    return f"candidate_{i}_present"


class Evaluator:
    """Score graphs on one segment; see the module docstring.

    Make one with ``for_segment``; nothing changes its dates afterwards.
    """

    def __init__(
        self, segment: Segment, prices, label, scoring: Scoring, reference: Sequence,
        warmup_bars: int, njobs: int, purge_bars: int = PURGE_BARS,
    ):
        self._segment = segment
        self._purge_bars = purge_bars
        self._prices = prices
        self._label = label
        self._scoring = scoring
        self._reference = tuple(reference)
        self._warmup_bars = warmup_bars
        self._njobs = njobs
        self._frame: _Frame | None = None

    @classmethod
    def for_segment(
        cls, segment: Segment, *, prices, label, scoring: Scoring, reference: Sequence = (),
        warmup_bars: int, njobs: int, purge_bars: int = PURGE_BARS,
    ) -> "Evaluator":
        """Return an evaluator of ``segment``.

        Parameters
        ----------
        segment : Segment
            The only dates it reads, besides ``warmup_bars`` bars before.
        prices : quantlab dataset
            The whole-market adjusted bars the graphs run on.
        label : quantlab label
            The forward return, read from its store; only the symbols it
            has a value for on a date are scored on that date.
        scoring : Scoring
            How candidates are scored and reference factors read.
        reference : sequence of quantlab factors
            The reference set: every output enters the pool. A factor with
            a store is read from it, any other is computed (``Scoring.read``).
        warmup_bars, njobs : int
            Warm-up bars and KunQuant executor threads.
        purge_bars : int
            The fewest bars dropped at the end of the segment (``segments.purge``).
        """
        return cls(segment, prices, label, scoring, reference, warmup_bars, njobs, purge_bars)

    @property
    def segment(self) -> Segment:
        """The segment this evaluator reads."""
        return self._segment

    def evaluate(self, graphs: list[Graph], admit: Admit | None = None) -> list[Metrics | CompileFailure]:
        """Return one entry per graph, in order; see the module docstring.

        Parameters
        ----------
        graphs : list of Graph
            The batch, compiled together.
        admit : callable, optional
            ``admit(position, metrics, ran)`` is called for every graph that
            ran, in order, before the next one is scored; ``ran`` is how many
            of the batch's graphs ran. When it returns a name, the candidate
            joins the reference pool under it. Without it no candidate joins.
        """
        if not graphs:
            return []
        frame = self._scored_frame()
        results: list[Metrics | CompileFailure] = list(self._values(graphs, frame))
        computed = [i for i, v in enumerate(results) if not isinstance(v, CompileFailure)]
        if not computed:
            return results
        scores = _Scores(frame, [results[i][0] for i in computed])
        for k, i in enumerate(computed):
            results[i] = scores.metrics(k, self._label.span_bars(), coverage(results[i][1], frame.rows))
            name = admit(i, results[i], len(computed)) if admit is not None else None
            if name is not None:
                scores.admit(k, name)
        return results

    def contributions(self, names: Sequence[str]) -> dict[str, Contribution]:
        """Each named pool factor's ``Contribution`` within the group of ``names``.

        Raises
        ------
        KeyError
            If a name is not in the pool.
        """
        frame = self._scored_frame()
        columns = [frame.pool_names.index(name) if name in frame.pool_names else _missing(name) for name in names]
        if not columns:
            return {}
        weights = np.full((len(frame.rows), len(columns)), np.nan)
        residual_ic = np.full((len(frame.rows), len(columns)), np.nan)
        for d, rows in enumerate(frame.rows):
            if len(rows) <= len(columns) + 1:
                continue
            target = frame.target[d, rows]
            design = np.column_stack([np.ones(len(rows)), frame.pool[d][:, columns].astype(np.float64)])
            # Column j + 1 of design @ pinv(design' design) is, up to a positive scale, the
            # residual of factor j regressed on the intercept and the other factors.
            dual = design @ np.linalg.pinv(design.T @ design)
            weights[d] = (dual.T @ _normal_scores(target, axis=0))[1:]
            ranked_target = stats.rankdata(target)
            for j in range(len(columns)):
                residual = dual[:, j + 1]
                if np.ptp(residual) > _SPANNED_NORM:
                    residual_ic[d, j] = np.corrcoef(stats.rankdata(residual), ranked_target)[0, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a factor spanned on every day stays NaN
            weight, ic = np.nanmean(weights, axis=0), np.nanmean(residual_ic, axis=0)
        return {name: Contribution(float(weight[j]), float(ic[j])) for j, name in enumerate(names)}

    def _scored_frame(self) -> "_Frame":
        """The scored dates, the label and the reference pool, read once."""
        if self._frame is None:
            name = self._label.get_factor_names()[0]
            label = self._label.read(self._segment.start, self._segment.end)[name]
            keep = label.sizes["timestamp"] - purge(self._label.lookahead_bars(), self._purge_bars)
            self._frame = _Frame(label.isel(timestamp=slice(0, max(keep, 0))))
            self._frame.add_to_pool(dict(item for factor in self._reference for item in self._reference_scores(factor)))
        return self._frame

    def _reference_scores(self, factor) -> Iterator[tuple[str, list[np.ndarray]]]:
        """Yield ``(name, daily normal scores)`` of every output of a reference factor, as ``Scoring.read`` reads it."""
        source = type(owner(factor)).__name__
        for name, values in self._scoring.read(factor, self._segment.start, self._segment.end):
            yield f"{name} ({source})", self._frame.daily_scores(self._frame.on_axes(values))

    def _values(self, graphs: list[Graph], frame: "_Frame") -> list["tuple[np.ndarray, np.ndarray] | CompileFailure"]:
        """Each graph's scored values and its own (unranked) presence, bisecting a batch that fails."""
        columns = sorted({name for graph in graphs for name in graph.input_names})
        factor = _RoundGraph(
            FactorConfig(
                warmup_bars=self._warmup_bars, dataset=self._prices, mode="batch",
                data_columns=tuple(columns), njobs=self._njobs,
            ),
            graphs,
        )
        start, end = self._segment.start, self._segment.end
        try:
            raw = factor.compute(start, end)
        except Exception as exc:  # KunQuant reports a bad graph as a plain exception
            if len(graphs) == 1:
                return [CompileFailure(f"{type(exc).__name__}: {exc}")]
            half = len(graphs) // 2
            return self._values(graphs[:half], frame) + self._values(graphs[half:], frame)
        names = tuple(_output_name(i) for i in range(len(graphs)))
        panel = self._scoring.mask(self._scoring.apply(factor, names=names).compute(start, end))
        return [
            (frame.on_axes(panel[_output_name(i)]), frame.on_axes(raw[_presence_name(i)])) for i in range(len(graphs))
        ]


class _Frame:
    """The scored dates and symbols, the label on them and the reference pool.

    The symbols are those with a label on some scored date. The pool is
    held per date on the symbols with a label that date (``rows``), as
    each factor's normal scores (float32), one column per factor.
    """

    def __init__(self, label: xr.DataArray):
        self.label = label.isel(symbol=np.flatnonzero(np.isfinite(label.values).any(axis=0)))
        self.target = self.label.values.astype(np.float64)
        self.in_rows = np.isfinite(self.target)
        self.rows = [np.flatnonzero(row) for row in self.in_rows]
        self.pool_names: list[str] = []
        self.pool = [np.zeros((len(rows), 0), dtype=np.float32) for rows in self.rows]

    def on_axes(self, values: xr.DataArray) -> np.ndarray:
        """``values`` on the frame's dates and symbols, NaN where it has none."""
        aligned = values.reindex(timestamp=self.label["timestamp"], symbol=self.label["symbol"])
        return aligned.values.astype(np.float64)

    def daily_scores(self, values: np.ndarray) -> list[np.ndarray]:
        """The normal scores of ``(date, symbol)`` values among each date's rows, one array per date."""
        scores = _normal_scores(np.where(self.in_rows, values, np.nan), axis=1)
        return [scores[d, rows] for d, rows in enumerate(self.rows)]

    def add_to_pool(self, factors: dict[str, np.ndarray | list[np.ndarray]]) -> None:
        """Add factors to the pool: each name maps to its ``(date, symbol)`` values or its ``daily_scores``."""
        columns = [v if isinstance(v, list) else self.daily_scores(v) for v in factors.values()]
        self.pool_names += list(factors)
        for d in range(len(self.rows)):
            self.pool[d] = np.column_stack([self.pool[d], *(c[d] for c in columns)]).astype(np.float32)


class _Scores:
    """A batch's statistics, with the residuals that admitting a candidate updates."""

    def __init__(self, frame: _Frame, values: list[np.ndarray]):
        self.frame = frame
        self.values = values
        self.admitted: list[tuple[int, str]] = []
        self.residuals = []
        daily_pool, daily_batch = [], []
        candidates = [frame.daily_scores(v) for v in values]
        for d in range(len(frame.rows)):
            scores = np.column_stack([c[d] for c in candidates])
            daily_pool.append(_correlations(scores, frame.pool[d]))
            daily_batch.append(_correlations(scores, scores))
            basis = _basis(frame.pool[d])
            self.residuals.append(scores - basis @ (basis.T @ scores))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a pair with no usable date stays NaN
            self.correlations = np.nanmean(np.stack(daily_pool), axis=0)
            self.batch_correlations = np.nanmean(np.stack(daily_batch), axis=0)

    def metrics(self, k: int, horizon: int, coverage: float) -> Metrics:
        """Candidate ``k``'s metrics against the pool and the batch candidates admitted before it."""
        frame, values = self.frame, self.values[k]
        ic = cross_sectional_rank_ic_series(values, frame.target)
        residual = np.full_like(values, np.nan)
        for d, rows in enumerate(frame.rows):
            residual[d, rows] = np.where(np.isfinite(values[d, rows]), self.residuals[d][:, k], np.nan)
        partial = cross_sectional_rank_ic_series(residual, frame.target)
        names = frame.pool_names + [name for _, name in self.admitted]
        row = np.concatenate([self.correlations[k], [self.batch_correlations[k, j] for j, _ in self.admitted]])
        nearest, correlation = "", float("nan")
        if np.isfinite(row).any():
            best = int(np.nanargmax(np.abs(row)))
            nearest, correlation = names[best], float(row[best])
        scored = np.isfinite(ic)
        days = pd.DatetimeIndex(frame.label["timestamp"].values)[scored]
        ic_mean, ic_t = _mean_and_t(ic, horizon)
        partial_mean, partial_t = _mean_and_t(partial, horizon)
        return Metrics(
            ic_mean=ic_mean, ic_t=ic_t, partial_ic_mean=partial_mean, partial_ic_t=partial_t,
            nearest=nearest, nearest_correlation=correlation,
            decile_returns=decile_returns(values, frame.target),
            coverage=coverage,
            n_days=int(scored.sum()),
            start=days[0].strftime("%Y-%m-%d") if len(days) else "",
            end=days[-1].strftime("%Y-%m-%d") if len(days) else "",
        )

    def admit(self, k: int, name: str) -> None:
        """Add candidate ``k`` to the pool; the later candidates are residualised on it too."""
        self.admitted.append((k, name))
        for residuals in self.residuals:
            direction = residuals[:, k]
            norm = np.linalg.norm(direction)
            if norm <= _SPANNED_NORM:
                continue
            direction = direction / norm
            residuals[:, k + 1:] -= np.outer(direction, direction @ residuals[:, k + 1:])
        self.frame.add_to_pool({name: self.values[k]})


def _normal_scores(values: np.ndarray, axis: int) -> np.ndarray:
    """Normal scores along ``axis`` among the finite values; a missing value scores 0, the middle rank."""
    if not values.size:
        return np.zeros(values.shape)
    finite = np.isfinite(values)
    ranks = stats.rankdata(values, axis=axis, nan_policy="omit")
    count = np.maximum(finite.sum(axis=axis, keepdims=True), 1)
    return np.where(finite, stats.norm.ppf((ranks - 0.5) / count), 0.0)


def _correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The correlation of every column of ``a`` with every column of ``b``, both centred."""
    norms = np.outer(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norms > 0, (a.T @ b) / norms, np.nan)


def _basis(pool: np.ndarray) -> np.ndarray:
    """An orthonormal basis of an intercept and the pool's columns."""
    design = np.column_stack([np.ones(len(pool)), pool])
    if not len(design):
        return design
    u, s, _ = np.linalg.svd(design, full_matrices=False)
    return u[:, s > s[0] * _RANK_TOLERANCE]


def _mean_and_t(ic: np.ndarray, horizon: int) -> tuple[float, float]:
    """The mean of the finite daily ICs and its Newey-West t."""
    series = pd.Series(ic[np.isfinite(ic)])
    if not len(series):
        return float("nan"), float("nan")
    t_stat, _ = newey_west_t_stat(series, newey_west_lags(len(series), horizon))
    return float(series.mean()), float(t_stat)


def coverage(values: np.ndarray, rows: list[np.ndarray]) -> float:
    """The share of each date's ``rows`` where ``values`` (before ranking) is finite, averaged over the dates with rows."""
    shares = [np.isfinite(values[d, r]).mean() for d, r in enumerate(rows) if len(r)]
    return float(np.mean(shares)) if shares else float("nan")


def decile_returns(values: np.ndarray, target: np.ndarray) -> tuple[float, ...]:
    """The mean label of each decile of the daily cross section, averaged over dates."""
    daily = np.full((len(values), DECILES), np.nan)
    for d in range(len(values)):
        ok = np.isfinite(values[d]) & np.isfinite(target[d])
        n = int(ok.sum())
        if n < DECILES:
            continue
        group = ((stats.rankdata(values[d, ok]) - 1) * DECILES // n).astype(int)
        daily[d] = np.bincount(group, weights=target[d, ok], minlength=DECILES) / np.bincount(group, minlength=DECILES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a segment too short for any decile stays NaN
        return tuple(float(x) for x in np.nanmean(daily, axis=0))
