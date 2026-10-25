"""Acceptance: the accepted factors' incremental test in the acceptance model, the one-time hold-out, and their export.

``validate`` trains the acceptance model (``AcceptanceConfig.model``, a
template that ``factor_agent._acceptance_model`` fills in) twice per seed: on
its baseline (its own factors, or the reference set) and on that plus a
selection of the factors the campaign accepted (every one by default, else
named as ``export`` names them), all read through the campaign's
``Scoring`` (on the members of day t, or on every symbol without a
membership). It trains on the training segment and predicts the validation
segment, both purged by ``Segments``. Each accepted factor enters the model
as the Evaluator scored it: its graph ranked across the whole market and
neutralised when the scoring neutralises (a store written under the run
directory), read as the scoring reads.

With ``acceptance.contribution``, ``validate`` also measures each factor of
the selection's contribution, with the same seeds: drop-one trains the model
on the baseline plus every factor of the selection but it (the whole
selection against that), add-one on the baseline plus only it (against the
baseline). A feature set equal to another (a one-factor selection's drop-one
is the baseline) is trained once.

``open_holdout`` makes the same comparison on the hold-out for a selection
that ``validate`` reported on, trained on the training and validation
segments. It first records the open in the lock
(``Segments.open_holdout``): a second open is refused unless forced, and a
forced one is stated in its report, and it never measures contribution. Reports of every accepted factor are
``acceptance/<segment>.json|md``, of a selection of fewer
``acceptance/<segment>_<digest>.json|md``.

Every trained model's result is cached (``factor_agent._model_cache``), so a
re-run trains only the models not finished; each run first prints how many
it will train and how many are cached.

The comparison, on the members of each day of the predicted segment (every
symbol without a membership) and
against the label the model trains on, per model and averaged over seeds:

- ``rank_ic``, ``ic``: the mean daily rank and Pearson IC of the prediction;
- ``decile_returns``, ``monotonicity``, ``long_short``: the mean label of
  each tenth of the daily cross section (lowest predictions first), their
  Spearman correlation with the decile number and the top decile's minus
  the bottom one's;
- ``top_k_return``, ``top_k_excess``: the mean label of the plain TopK book
  (the ``top_k`` members with the highest prediction, equal weights, no
  smoothing) and its excess over all members, fee-free, over the label's
  horizon; ``top_k_annualised`` compounds ``top_k_return`` to a year of 252
  bars. These read the label as a raw forward return.

``export`` writes each named accepted factor as a module of
``AcceptanceConfig.package`` holding a ``GraphFactor`` subclass, and its
whole-market store, the ranked graph scored as above (neutralised or not), under
``factor_root/<name>/`` with ``README.md`` and ``component.json``.
"""

import dataclasses
import hashlib
import importlib
import json
import keyword
import pprint
import shutil
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr
from quantlab.core.store_folder import save_component
from quantlab.factor.config import FactorConfig
from quantlab.model.base import BaseModel
from quantlab.model.evaluation import cross_sectional_ic_series, cross_sectional_rank_ic_series
from quantlab.runs.trained_run import TrainedRun

from factor_agent import _acceptance_model, _admission, _feedback, _model_cache
from factor_agent.config import CampaignConfig, load
from factor_agent.evaluator import decile_returns
from factor_agent.exported import GraphFactor
from factor_agent.ledger import Ledger, TrialRecord
from factor_agent.scoring import Scoring
from factor_agent.segments import HoldoutLock, Segment, code_version
from factor_agent.snapshot import RunSnapshot

#: Bars a year, for ``top_k_annualised``.
BARS_PER_YEAR = 252
#: The statistics averaged over seeds and compared.
COMPARED = ("rank_ic", "ic", "monotonicity", "long_short", "top_k_return", "top_k_excess", "top_k_annualised")


@dataclass(frozen=True)
class ModelResult:
    """One model's statistics on the predicted segment; see the module docstring."""

    seed: int
    rank_ic: float
    ic: float
    monotonicity: float
    long_short: float
    top_k_return: float
    top_k_excess: float
    top_k_annualised: float
    decile_returns: tuple[float, ...]
    n_days: int
    run: str


def _mean(results: tuple[ModelResult, ...]) -> dict[str, float]:
    """Each compared statistic averaged over the seeds."""
    return {name: float(np.mean([getattr(r, name) for r in results])) for name in COMPARED}


@dataclass(frozen=True)
class Contribution:
    """One accepted factor's contribution in one mode: the models with it against the same models without it.

    ``mode`` is ``drop_one`` (``without``: the selection without the factor,
    ``with_factor``: the whole selection) or ``add_one`` (``without``: the
    baseline, ``with_factor``: the baseline with only the factor); both
    sides hold the same seeds in the same order.
    """

    factor: str
    mode: str
    without: tuple[ModelResult, ...]
    with_factor: tuple[ModelResult, ...]

    @property
    def difference(self) -> dict[str, float]:
        """With the factor minus without it, of each seed-averaged statistic."""
        base, new = _mean(self.without), _mean(self.with_factor)
        return {name: new[name] - base[name] for name in COMPARED}

    @property
    def agreeing(self) -> dict[str, int]:
        """Per statistic, the seeds whose difference has the sign of the seed-averaged difference."""
        mean = self.difference
        return {
            name: sum(
                int(np.sign(getattr(w, name) - getattr(o, name)) == np.sign(mean[name]))
                for o, w in zip(self.without, self.with_factor)
            )
            for name in COMPARED
        }


@dataclass(frozen=True)
class AcceptanceReport:
    """The baseline against the baseline plus the accepted factors, seed by seed and on average.

    ``accepted`` maps each factor added to the baseline, by label, to its
    output name in the model: every accepted factor, or the members of a
    selection when ``selection`` is true. ``forced`` is true when the hold-out was opened past its lock;
    ``holdout_opens`` lists every open of the hold-out so far (empty for a
    validation report). ``contributions`` holds each accepted factor's
    contribution per mode (``acceptance.contribution``), empty when none was
    asked for; there is no threshold.
    """

    segment: str
    train_window: tuple[str, str]
    test_window: tuple[str, str]
    seeds: tuple[int, ...]
    top_k: int
    accepted: dict
    baseline: tuple[ModelResult, ...]
    with_accepted: tuple[ModelResult, ...]
    selection: bool = False
    forced: bool = False
    holdout_opens: tuple[dict, ...] = ()
    code_version: dict = field(default_factory=dict)
    contributions: tuple[Contribution, ...] = ()

    def mean(self, results: tuple[ModelResult, ...]) -> dict[str, float]:
        """Each compared statistic averaged over the seeds."""
        return _mean(results)

    @property
    def difference(self) -> dict[str, float]:
        """With the accepted factors minus the baseline, of each seed-averaged statistic."""
        base, new = self.mean(self.baseline), self.mean(self.with_accepted)
        return {name: new[name] - base[name] for name in COMPARED}

    def contribution_table(self) -> list[dict]:
        """One row per factor and mode: ``factor``, ``mode``, ``difference`` and ``agreeing`` of every statistic.

        Sorted by the rank IC difference, the largest first; a NaN difference sorts last.
        """
        rows = [
            {"factor": c.factor, "mode": c.mode, "difference": c.difference, "agreeing": c.agreeing}
            for c in self.contributions
        ]
        return sorted(rows, key=lambda row: -row["difference"]["rank_ic"] if np.isfinite(row["difference"]["rank_ic"]) else np.inf)

    def to_dict(self) -> dict:
        """The report as ``<segment>.json`` holds it, with the seed means, the difference and the contribution table."""
        out = dataclasses.asdict(self)
        out["mean"] = {"baseline": self.mean(self.baseline), "with_accepted": self.mean(self.with_accepted)}
        out["difference"] = self.difference
        out["contribution"] = self.contribution_table()
        return out

    def markdown(self) -> str:
        """The report as a short Markdown page."""
        base, new, diff = self.mean(self.baseline), self.mean(self.with_accepted), self.difference
        lines = [f"# Acceptance: {self.segment}", ""]
        if self.forced:
            lines += [
                f"**The hold-out was opened again with force: this is open {len(self.holdout_opens)}, the first "
                f"was on {self.holdout_opens[0]['opened_at']}. It is no longer an untouched hold-out.**", "",
            ]
        lines += [
            f"- Trained {self.train_window[0]} to {self.train_window[1]}, predicted "
            f"{self.test_window[0]} to {self.test_window[1]} (both purged); seeds {list(self.seeds)}.",
            f"- {'Selection of accepted factors' if self.selection else 'Accepted factors'} ({len(self.accepted)}): "
            + ", ".join(self.accepted) + ".",
            f"- Plain TopK of {self.top_k}, equal weights; returns are fee-free, over the label's horizon.", "",
            "| statistic | baseline | with accepted | difference |", "|---|---|---|---|",
            *(f"| {name} | {base[name]:.4f} | {new[name]:.4f} | {diff[name]:+.4f} |" for name in COMPARED), "",
            "| seed | baseline rank IC | with accepted rank IC | baseline TopK excess | with accepted TopK excess |",
            "|---|---|---|---|---|",
            *(
                f"| {b.seed} | {b.rank_ic:.4f} | {a.rank_ic:.4f} | {b.top_k_excess:.4f} | {a.top_k_excess:.4f} |"
                for b, a in zip(self.baseline, self.with_accepted)
            ),
        ]
        if self.contributions:
            lines += [
                "", "## Contribution", "",
                "Per accepted factor: drop_one is the whole selection against the selection without it, add_one the "
                "baseline with only it against the baseline; seed-averaged differences, and the seeds whose rank IC "
                "difference has the mean's sign. No threshold.", "",
                "| factor | mode | rank IC difference | seeds agreeing | TopK excess difference |", "|---|---|---|---|---|",
                *(
                    f"| {row['factor']} | {row['mode']} | {row['difference']['rank_ic']:+.4f} | "
                    f"{row['agreeing']['rank_ic']}/{len(self.seeds)} | {row['difference']['top_k_excess']:+.4f} |"
                    for row in self.contribution_table()
                ),
            ]
        return "\n".join(lines) + "\n"


class Acceptance:
    """Validate, open the hold-out of, and export a campaign's accepted factors; see the module docstring.

    Parameters
    ----------
    config : CampaignConfig
        The campaign's config; its ``segments`` need ``validation`` (and
        ``holdout`` for ``open_holdout``).
    run_dir : str or Path, optional
        The campaign's run directory; ``config.run_dir`` by default.

    The accepted graphs are read with the vocabulary of the run's snapshot
    (``RunSnapshot``), not of the config's field and operator files.

    Raises
    ------
    FileNotFoundError
        If the run directory has no snapshot.
    """

    def __init__(self, config: CampaignConfig, run_dir: str | Path | None = None):
        self.config = config
        self.run_dir = Path(run_dir if run_dir is not None else config.run_dir)
        #: The campaign's vocabulary, from its run's snapshot: the operator classes the accepted graphs are built from.
        self.vocabulary = config.vocabulary(files=RunSnapshot.read(self.run_dir).search_files)
        self._loaded: dict = {}

    @property
    def lock_path(self) -> Path:
        """The hold-out lock."""
        return self.run_dir / "holdout.lock"

    def accepted(self) -> list[TrialRecord]:
        """The ledger's accepted factors, in the order they were accepted."""
        return [r for r in Ledger(self.run_dir / "ledger.jsonl").records() if r.status == "accepted"]

    def validate(self, selection: list[str] | None = None) -> AcceptanceReport:
        """Compare the baseline with the baseline plus a selection of the accepted factors on the validation segment.

        Parameters
        ----------
        selection : list of str, optional
            The accepted factors to add, named as ``export`` names them (by
            description or label); every accepted factor by default.

        Writes ``acceptance/validation.json`` and ``acceptance/validation.md``
        under the run directory for every accepted factor, and
        ``acceptance/validation_<digest>.json|md`` for a selection of fewer,
        the digest over its members' labels.

        Raises
        ------
        KeyError
            If a name names no accepted factor, or a description several.
        ValueError
            If there is no validation segment or no accepted factor, the
            selection is empty or names a factor twice.
        """
        segments = self.config.segments
        if segments.validation is None:
            raise ValueError("acceptance: the config has no validation segment")
        members, suffix = self._selected(selection)
        return self._compare(
            segments.validation, segments.training, members, suffix, contribution=self.config.acceptance.contribution,
        )

    def open_holdout(self, selection: list[str], force: bool = False) -> AcceptanceReport:
        """Open the hold-out, once, and compare on it with a selection, trained on the training and validation segments.

        Parameters
        ----------
        selection : list of str
            The accepted factors to add, named as in ``validate``; ``validate``
            must have written the report of exactly this selection first.
        force : bool, default False
            Open a hold-out that was opened before.

        The open is recorded in ``holdout.lock`` before any hold-out data is
        read, so an open that fails midway still counts (the hold-out may
        have been seen) and running it again needs ``force``; the lock counts
        opens of any selection. Writes ``acceptance/holdout.json|md``, or
        ``acceptance/holdout_<digest>.json|md`` for a selection of fewer than
        every accepted factor.

        Raises
        ------
        HoldoutLocked
            If the hold-out was opened before and ``force`` is false.
        KeyError
            If a name names no accepted factor, or a description several.
        ValueError
            If there is no validation or hold-out segment, no accepted factor,
            the selection is empty or names a factor twice, or there is no
            validation report of the selection.
        """
        segments = self.config.segments
        if segments.validation is None or segments.holdout is None:
            raise ValueError("acceptance: the config needs validation and hold-out segments to open the hold-out")
        members, suffix = self._selected(selection)
        validation = self.run_dir / "acceptance" / f"{segments.validation.name}{suffix}.json"
        if not validation.exists() or set(json.loads(validation.read_text())["accepted"]) != set(members):
            raise ValueError(f"acceptance: run validate({sorted(members)}) before opening the hold-out on that selection")
        lock = segments.open_holdout(self.lock_path, force=force)
        return self._compare(segments.holdout, segments.validation, members, suffix, contribution="none", lock=lock)

    def export(self, names: dict[str, str]) -> list[Path]:
        """Export accepted factors as ``GraphFactor`` subclasses and whole-market stores.

        Parameters
        ----------
        names : dict
            An accepted factor's description (as the ledger holds it) or
            label (``description (accepted in round n)``) to the name of its
            module, output and store: a lowercase identifier.

        Returns
        -------
        list of Path
            The store folders written, in the order of ``names``.

        Raises
        ------
        KeyError
            If a key names no accepted factor, or a description several.
        ValueError
            If the config sets no ``acceptance.factor_root``, a name is not a lowercase
            identifier, or its module or store already exists.
        """
        if self.config.acceptance.factor_root is None:
            raise ValueError("export: the config sets no acceptance.factor_root, the folder the stores are written into")
        records = self._named(names)
        package = importlib.import_module(self.config.acceptance.package)
        package_dir = Path(package.__path__[0])
        root = Path(self.config.acceptance.factor_root)
        if len(set(names.values())) != len(names):
            raise ValueError(f"export: the names {list(names.values())} repeat")
        for name in names.values():
            if not (name.isidentifier() and name == name.lower() and not keyword.iskeyword(name)):
                raise ValueError(f"export: {name!r} is not a lowercase identifier")
            for path in (package_dir / f"{name}.py", root / name):
                if path.exists():
                    raise ValueError(f"export: {path} already exists")
        start, end = self._export_window()
        folders = []
        for (key, name), record in zip(names.items(), records):
            module, folder = package_dir / f"{name}.py", root / name
            module.write_text(_module_source(
                name, record, self.run_dir, self.vocabulary.class_paths([record.graph]),
                GraphFactor.period_lags({name: record.graph}, self.vocabulary.lag_columns),
                _admission.direction(record.metrics["ic_mean"]),
            ))
            try:
                importlib.invalidate_caches()
                factor_class = getattr(importlib.import_module(f"{package.__name__}.{name}"), _class_name(name))
                factor = factor_class(self._factor_config(factor_class.GRAPHS, factor_class.PERIOD_LAGS, kwargs={
                    "directions": dict(factor_class.DIRECTIONS), "training_ic": {name: record.metrics["ic_mean"]},
                }))
                scored = self._scoring().apply(factor, folder / f"{name}.zarr")
                scored.build(start, end)
                save_component(scored, folder, readme=self._readme(name, record, factor_class, start, end))
            except BaseException:  # leave nothing half written, so the export can be run again
                module.unlink(missing_ok=True)
                shutil.rmtree(folder, ignore_errors=True)
                raise
            folders.append(folder)
        return folders

    # The comparison.

    def _compare(
        self, segment: Segment, trained_to: Segment, members: list[str], suffix: str, contribution: str,
        lock: HoldoutLock | None = None,
    ) -> AcceptanceReport:
        """Compare on ``segment`` the baseline with it plus the accepted factors labelled ``members``.

        With ``contribution`` (``AcceptanceConfig.contribution``), also each member's contribution.
        Reports and models go under names ending in ``suffix``.
        """
        settings, segments = self.config.acceptance, self.config.segments
        label = self._data("label")
        calendar = self._data("prices").calendar(segments.training.start, segment.end)
        lookahead = label.lookahead_bars()
        train_window = (segments.training.start, segments.scored(trained_to, calendar, lookahead)[1])
        test_window = segments.scored(segment, calendar, lookahead)
        # An accepted factor keeps its output name, its place among all accepted, in every selection; every
        # selection reads its outputs from the one store of all accepted, so equal feature sets share their models.
        records = self.accepted()
        everything = {_feedback.accepted_label(r): f"accepted_{k}" for k, r in enumerate(records)}
        outputs = {label: output for label, output in everything.items() if label in members}
        accepted, build = self._accepted_feature(records, list(everything.values()), test_window[1])
        labels = list(outputs)
        planned, contributions = _plan(list(outputs.values()), contribution)
        scoring = self._scoring()
        feature_sets = {
            kind: [scoring.on_members(type(accepted)(dataclasses.replace(accepted.config, factor_names=names)))] if names else []
            for kind, names in planned.items()
        }
        results = self._train(segment, suffix, feature_sets, build, {
            "labels": [label], "factor_data_strategy": "read", "label_data_strategy": "read",
            "start_date": train_window[0], "end_date": test_window[1],
            "train_start": train_window[0], "train_end": train_window[1],
            "test_start": test_window[0], "test_end": test_window[1],
            "val_size": settings.val_size, "metric_splits": ("test",),
        })
        report = AcceptanceReport(
            segment=segment.name, train_window=train_window, test_window=test_window, seeds=settings.seeds,
            top_k=settings.top_k, accepted=outputs, selection=bool(suffix),
            baseline=tuple(results["baseline"]), with_accepted=tuple(results["with_accepted"]),
            contributions=tuple(
                Contribution(labels[k], mode, tuple(results[without]), tuple(results[with_factor]))
                for k, mode, without, with_factor in contributions
            ),
            forced=lock.last.forced if lock else False,
            holdout_opens=tuple(dataclasses.asdict(o) for o in lock.opens) if lock else (),
            code_version=code_version(),
        )
        out = self.run_dir / "acceptance"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{segment.name}{suffix}.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        (out / f"{segment.name}{suffix}.md").write_text(report.markdown())
        return report

    def _train(
        self, segment: Segment, suffix: str, feature_sets: dict[str, list], build: Callable[[], None], filled: dict,
    ) -> dict[str, list[ModelResult]]:
        """Each feature set's results on ``segment``, one per seed, read from the cache or trained and cached.

        ``feature_sets`` maps a name (its models are saved under
        ``acceptance/<segment><suffix>/<name>_s<seed>``) to the features added to the
        baseline; the cache, ``acceptance/<segment>/cache``, is shared by every selection; ``build`` writes the
        accepted factors' store, called before the first model is trained; ``filled`` is every
        ``_acceptance_model.FILLED`` value but the seed and the save directory. Before any training it prints how many
        models it will train, how many are cached and, from the elapsed times
        cached with finished models, an estimate of the time it will take.
        """
        settings, scoring, label = self.config.acceptance, self._scoring(), filled["labels"][0]
        directory = self.run_dir / "acceptance" / f"{segment.name}{suffix}"
        cache = _model_cache.ModelCache(self.run_dir / "acceptance" / segment.name / "cache")
        models, code = {}, code_version()
        for kind, added in feature_sets.items():
            for seed in settings.seeds:
                model = _acceptance_model.build(settings.model, self._baseline, added, {
                    **filled, "random_seed": seed, "model_save_dir": str(directory / f"{kind}_s{seed}"),
                })
                models[kind, seed] = model, _model_cache.key(
                    model.get_config(), self.run_dir, code=code, top_k=settings.top_k,
                    scoring=dataclasses.asdict(self.config.scoring),
                )
        print(f"acceptance ({segment.name}): " + cache.summary([k for _, k in models.values()]), flush=True)
        name = label.get_factor_names()[0]
        truth = None
        results = {kind: [] for kind in feature_sets}
        for kind in feature_sets:
            panel = None
            for seed in settings.seeds:
                model, key = models[kind, seed]
                cached = cache.get(key)
                if cached is None:
                    started = time.monotonic()
                    if truth is None:
                        build()
                        truth = label.read(filled["test_start"], filled["test_end"])[name]
                    panel = _collect(model, panel)
                    run = TrainedRun.open(model.train())
                    predictions = scoring.mask(xr.open_zarr(run.test_predictions)[[name]].load())[name]
                    result = self._result(seed, predictions, truth, label.span_bars(), str(run.path))
                    cache.put(key, dataclasses.asdict(result), time.monotonic() - started)
                else:
                    result = ModelResult(**{**cached, "decile_returns": tuple(cached["decile_returns"])})
                results[kind].append(result)
        return results

    def _baseline(self, entries: tuple) -> list:
        """A model's baseline: its factor entries, or the reference set, read as the scoring reads; built once."""
        key = ("baseline", json.dumps(entries, sort_keys=True))
        if key not in self._loaded:
            features = [self._scoring().on_members(load(c)) for c in (entries or self.config.reference)]
            if not features:
                raise ValueError("acceptance: a baseline has no feature (the model names none and there is no reference)")
            self._loaded[key] = features
        return self._loaded[key]

    def _result(self, seed: int, predictions: xr.DataArray, truth: xr.DataArray, horizon: int, run: str) -> ModelResult:
        """One model's statistics on the dates and members both the prediction and the label have."""
        truth = truth.reindex_like(predictions)
        pred, real = predictions.values.astype(np.float64), truth.values.astype(np.float64)
        top_k = self.config.acceptance.top_k
        book, members = np.full(len(pred), np.nan), np.full(len(pred), np.nan)
        for d in range(len(pred)):
            ok = np.isfinite(pred[d]) & np.isfinite(real[d])
            if ok.sum() < top_k:
                continue
            ranked = real[d, ok][np.argsort(-pred[d, ok], kind="stable")]
            book[d], members[d] = ranked[:top_k].mean(), ranked.mean()
        deciles = decile_returns(pred, real)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a segment with no usable date stays NaN
            top_return = float(np.nanmean(book))
            return ModelResult(
                seed=seed,
                rank_ic=float(np.nanmean(cross_sectional_rank_ic_series(pred, real))),
                ic=float(np.nanmean(cross_sectional_ic_series(pred, real))),
                monotonicity=_admission.monotonicity(deciles, 1),
                long_short=deciles[-1] - deciles[0],
                top_k_return=top_return,
                top_k_excess=float(np.nanmean(book - members)),
                top_k_annualised=float((1 + top_return) ** (BARS_PER_YEAR / horizon) - 1),
                decile_returns=deciles, n_days=int(np.isfinite(book).sum()), run=run,
            )

    def _accepted_feature(self, records: list[TrialRecord], outputs: list[str], end: str):
        """Every accepted factor ``records``, scored, its outputs named ``outputs``, and the function writing its store.

        The store, under ``acceptance/accepted_<end>``, runs from the training segment's start to ``end``, so a
        validation never computes a hold-out date; it is written again only when its build record differs.
        """
        graphs = dict(zip(outputs, (r.graph for r in records)))
        classes = self.vocabulary.class_paths(graphs.values())
        lags = GraphFactor.period_lags(graphs, self.vocabulary.lag_columns)
        directions = {name: _admission.direction(r.metrics["ic_mean"]) for name, r in zip(outputs, records)}
        factor = GraphFactor(self._factor_config(graphs, lags, kwargs={
            "graphs": graphs, "classes": classes, "directions": directions, **({"period_lags": lags} if lags else {}),
        }))
        folder = self.run_dir / "acceptance" / f"accepted_{end}"
        scored = self._scoring().apply(factor, folder / "accepted.zarr")
        start = self.config.segments.training.start

        def build() -> None:
            built = {"start": start, "end": end, "config": _model_cache.portable(scored.get_config(), self.run_dir)}
            record = folder / "accepted.json"
            if not record.exists() or json.loads(record.read_text()) != json.loads(json.dumps(built)):
                shutil.rmtree(folder, ignore_errors=True)
                scored.build(start, end)
                record.write_text(json.dumps(built) + "\n")

        return scored, build

    # Shared.

    def _factor_config(self, graphs: dict, lags: dict, kwargs: dict | None = None) -> FactorConfig:
        """The config of a factor of the recorded ``graphs``, which read the period ``lags``."""
        return FactorConfig(
            warmup_bars=self.config.evaluator.warmup_bars, dataset=self._data("prices"), mode="batch",
            data_columns=GraphFactor.data_columns(graphs, lags),
            njobs=self.config.evaluator.njobs, kwargs=kwargs,
        )

    def _scoring(self) -> Scoring:
        """The campaign's scoring, the one its Evaluator admitted with, built once."""
        if "scoring" not in self._loaded:
            self._loaded["scoring"] = Scoring.from_config(
                self.config.scoring, self.config.data, njobs=self.config.evaluator.njobs,
            )
        return self._loaded["scoring"]

    def _data(self, name: str):
        """A data component of the config, rebuilt once."""
        if name not in self._loaded:
            self._loaded[name] = self.config.data.load(name)
        return self._loaded[name]

    def _export_window(self) -> tuple[str, str]:
        settings, segments = self.config.acceptance, self.config.segments
        last = segments.holdout or segments.validation or segments.training
        return settings.export_start or segments.training.start, settings.export_end or last.end

    def _selected(self, selection: list[str] | None) -> tuple[list[str], str]:
        """The labels of the accepted factors ``selection`` names (every one for None) and the suffix of their reports.

        The suffix is empty for every accepted factor, else ``_`` and a digest of the sorted labels.
        """
        accepted = [_feedback.accepted_label(r) for r in self.accepted()]
        if not accepted:
            raise ValueError(f"acceptance: {self.run_dir} has no accepted factor")
        if len(set(accepted)) != len(accepted):
            raise ValueError("acceptance: two accepted factors share a label (description and round)")
        if selection is None:
            return accepted, ""
        if not selection:
            raise ValueError("acceptance: the selection is empty")
        members = [_feedback.accepted_label(r) for r in self._named(selection, "acceptance")]
        if len(set(members)) == len(accepted):
            return accepted, ""
        return members, "_" + hashlib.sha256(json.dumps(sorted(members)).encode()).hexdigest()[:12]

    def _named(self, names, caller: str = "export") -> list[TrialRecord]:
        """The accepted record each of ``names`` (or each key) names, by label or description.

        Raises ``KeyError`` for a name of no accepted factor or a description of several, and
        ``ValueError`` for two names of one factor.
        """
        accepted = self.accepted()
        out = []
        for key in names:
            matches = [r for r in accepted if _feedback.accepted_label(r) == key] or [
                r for r in accepted if r.candidate["description"] == key
            ]
            if len(matches) != 1:
                raise KeyError(
                    f"{caller}: {key!r} names {len(matches)} accepted factors; "
                    f"accepted: {[_feedback.accepted_label(r) for r in accepted]}"
                )
            if any(r is matches[0] for r in out):
                raise ValueError(f"{caller}: {_feedback.accepted_label(matches[0])!r} is named twice")
            out.append(matches[0])
        return out

    def _readme(self, name: str, record: TrialRecord, factor_class: type, start: str, end: str) -> str:
        candidate, ic = record.candidate, record.metrics["ic_mean"]
        validation = self.run_dir / "acceptance" / "validation.json"
        lines = [
            f"# {name}", "",
            f"{candidate['description']}", "",
            "## What it is", "",
            f"- Hypothesis: {candidate['hypothesis']}",
            f"- Mechanism type: {candidate['mechanism_type']}; {_feedback.direction_text(ic)}"
            + ("; the factor is turned to it (one minus its rank) before scoring, so a higher value means a higher return."
               if _admission.direction(ic) < 0 else "; a higher value means a higher return.")
            + f" Training IC {ic:+.4f}.",
            f"- Horizon: {candidate['horizon_rationale']}",
            f"- Output `{name}`: the graph's whole-market values as the campaign scored them "
            f"({self._scoring().describe()}); the store is not masked to members.", "",
            "```python", candidate["code"], "```", "",
            "## How it was made", "",
            f"- Proposed and accepted in round {record.round} of the factor-agent campaign `{self.run_dir}` "
            "(quantlab-factor-agent), on the training segment "
            f"{self.config.segments.training.start} to {self.config.segments.training.end}.",
            f"- Computed by `{factor_class.__module__}.{factor_class.__name__}` (a `FactorKunQuant`), "
            f"code {code_version()}.",
        ]
        if validation.exists():
            difference = json.loads(validation.read_text())["difference"]
            lines.append(
                "- Validation (with the campaign's accepted factors, against the baseline, seed mean): "
                + ", ".join(f"{k} {v:+.4f}" for k, v in difference.items()) + "."
            )
        lines += [
            "", "## Coverage", "",
            f"- {start} to {end}, every symbol of the price dataset (the whole market).", "",
            "## Who uses it", "",
            "- Read it on a universe with `MembershipMaskedFactor(RosterFactor(load_component(<this folder>)))`.",
        ]
        return "\n".join(lines) + "\n"


def _plan(outputs: list[str], contribution: str) -> tuple[dict[str, tuple[str, ...]], list[tuple]]:
    """The feature sets to train and the contributions they make; the comparison plan.

    The feature sets map a name to the accepted ``outputs`` added to the
    baseline: ``baseline`` (none), ``with_accepted`` (all), and per output
    ``drop_<output>`` (all but it) for drop-one and ``add_<output>`` (only
    it) for add-one; a set equal to an earlier one is that one, so a
    one-factor selection trains nothing more. The contributions are
    ``(index of the output, mode, set without it, set with it)``.
    """
    named = {(): "baseline", tuple(outputs): "with_accepted"}
    contributions = []
    for k, output in enumerate(outputs):
        if contribution in ("drop_one", "both"):
            without = named.setdefault(tuple(o for o in outputs if o != output), f"drop_{output}")
            contributions.append((k, "drop_one", without, "with_accepted"))
        if contribution in ("add_one", "both"):
            contributions.append((k, "add_one", "baseline", named.setdefault((output,), f"add_{output}")))
    return {name: members for members, name in named.items()}, contributions


def _collect(model, panel):
    """Collect ``model``'s data, or hand it ``panel``, collected by a model of the same features; the panel to share.

    A single model shares a panel across seeds; an ensemble's members collect their own each time (None).
    """
    if not isinstance(model, BaseModel):
        model.collect()
        return None
    if panel is None:
        model.collect()
    else:
        model.use_panel(panel)
    return model.collected_panel


def _class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part) or "Factor"


def _module_source(
    name: str, record: TrialRecord, run_dir: Path, classes: dict, period_lags: dict, direction: int,
) -> str:
    """The module of an exported factor: the candidate as data and a ``GraphFactor`` subclass fixing its graph.

    ``classes`` are the class paths of the graph's operators and ``period_lags`` how its lag columns are
    computed, both from the campaign's vocabulary; ``direction`` is the sign of its training IC.
    """
    code = "\n".join(f"#     {line}" for line in record.candidate["code"].splitlines())
    return (
        f'"""{name}: a factor accepted by quantlab-factor-agent, exported by ``Acceptance.export``.\n\n'
        f"Accepted in round {record.round} of the campaign run directory {str(run_dir)!r}. The\n"
        "graph is the one the Evaluator scored; ``CANDIDATE`` is the proposal it came from.\n"
        '"""\n\n'
        "# The candidate's code:\n#\n"
        f"{code}\n\n"
        "from factor_agent.exported import GraphFactor\n\n"
        f"CANDIDATE = {pprint.pformat(record.candidate, width=100, sort_dicts=False)}\n\n\n"
        f"class {_class_name(name)}(GraphFactor):\n"
        f'    """The accepted factor ``{name}``, ranked across the market; see ``CANDIDATE``."""\n\n'
        f"    GRAPHS = {{{name!r}: {pprint.pformat(record.graph, width=100)}}}\n"
        f"    CLASSES = {pprint.pformat(classes, width=100)}\n"
        + (f"    PERIOD_LAGS = {pprint.pformat(period_lags, width=100)}\n" if period_lags else "")
        + f"    DIRECTIONS = {{{name!r}: {direction}}}\n"
    )
