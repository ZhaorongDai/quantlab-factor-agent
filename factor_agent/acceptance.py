"""Acceptance: the accepted factors' incremental test in SEL0 xgb, the one-time hold-out, and their export.

``validate`` trains the xgb of ``AcceptanceConfig`` twice per seed: on the
baseline features (``features``, or the reference set) and on those plus
every factor the campaign accepted, all read through the campaign's
``Scoring`` (on the members of day t, or on every symbol without a
membership). It trains on the training segment and predicts the validation
segment, both purged by ``Segments``. Each accepted factor enters the model
as the Evaluator scored it: its graph ranked across the whole market and
neutralised when the scoring neutralises (a store written under the run
directory), read as the scoring reads.

``open_holdout`` makes the same comparison on the hold-out, trained on the
training and validation segments. It first records the open in the lock
(``Segments.open_holdout``): a second open is refused unless forced, and a
forced one is stated in its report.

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
import importlib
import json
import keyword
import pprint
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr
from quantlab.config import get_data_root
from quantlab.core.store_folder import save_component
from quantlab.factor.config import FactorConfig
from quantlab.model.config import ModelConfig
from quantlab.model.evaluation import cross_sectional_ic_series, cross_sectional_rank_ic_series
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.runs.trained_run import TrainedRun

from factor_agent import _admission, _feedback
from factor_agent.config import CampaignConfig, load
from factor_agent.evaluator import decile_returns
from factor_agent.exported import GraphFactor
from factor_agent.ledger import Ledger, TrialRecord
from factor_agent.scoring import Scoring
from factor_agent.segments import HoldoutLock, Segment, code_version

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


@dataclass(frozen=True)
class AcceptanceReport:
    """The baseline against the baseline plus the accepted factors, seed by seed and on average.

    ``accepted`` maps each accepted factor's label to its output name in the
    model. ``forced`` is true when the hold-out was opened past its lock;
    ``holdout_opens`` lists every open of the hold-out so far (empty for a
    validation report).
    """

    segment: str
    train_window: tuple[str, str]
    test_window: tuple[str, str]
    seeds: tuple[int, ...]
    top_k: int
    accepted: dict
    baseline: tuple[ModelResult, ...]
    with_accepted: tuple[ModelResult, ...]
    forced: bool = False
    holdout_opens: tuple[dict, ...] = ()
    code_version: dict = field(default_factory=dict)

    def mean(self, results: tuple[ModelResult, ...]) -> dict[str, float]:
        """Each compared statistic averaged over the seeds."""
        return {name: float(np.mean([getattr(r, name) for r in results])) for name in COMPARED}

    @property
    def difference(self) -> dict[str, float]:
        """With the accepted factors minus the baseline, of each seed-averaged statistic."""
        base, new = self.mean(self.baseline), self.mean(self.with_accepted)
        return {name: new[name] - base[name] for name in COMPARED}

    def to_dict(self) -> dict:
        """The report as ``<segment>.json`` holds it, with the seed means and the difference."""
        out = dataclasses.asdict(self)
        out["mean"] = {"baseline": self.mean(self.baseline), "with_accepted": self.mean(self.with_accepted)}
        out["difference"] = self.difference
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
            f"- Accepted factors ({len(self.accepted)}): " + ", ".join(self.accepted) + ".",
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
    """

    def __init__(self, config: CampaignConfig, run_dir: str | Path | None = None):
        self.config = config
        self.run_dir = Path(run_dir if run_dir is not None else config.run_dir)
        self._loaded: dict = {}

    @property
    def lock_path(self) -> Path:
        """The hold-out lock."""
        return self.run_dir / "holdout.lock"

    def accepted(self) -> list[TrialRecord]:
        """The ledger's accepted factors, in the order they were accepted."""
        return [r for r in Ledger(self.run_dir / "ledger.jsonl").records() if r.status == "accepted"]

    def validate(self) -> AcceptanceReport:
        """Compare the baseline with the baseline plus the accepted factors on the validation segment.

        Writes ``acceptance/validation.json`` and ``acceptance/validation.md``
        under the run directory.

        Raises
        ------
        ValueError
            If there is no validation segment or no accepted factor.
        """
        segments = self.config.segments
        if segments.validation is None:
            raise ValueError("acceptance: the config has no validation segment")
        return self._compare(segments.validation, trained_to=segments.training)

    def open_holdout(self, force: bool = False) -> AcceptanceReport:
        """Open the hold-out, once, and compare on it, trained on the training and validation segments.

        The open is recorded in ``holdout.lock`` before any hold-out data is
        read, so an open that fails midway still counts (the hold-out may
        have been seen) and running it again needs ``force``. ``validate``
        must have written its report first. Writes ``acceptance/holdout.json`` and ``acceptance/holdout.md``.

        Raises
        ------
        HoldoutLocked
            If the hold-out was opened before and ``force`` is false.
        ValueError
            If there is no validation or hold-out segment, no accepted factor,
            or no validation report.
        """
        segments = self.config.segments
        if segments.validation is None or segments.holdout is None:
            raise ValueError("acceptance: the config needs validation and hold-out segments to open the hold-out")
        if not self.accepted():
            raise ValueError(f"acceptance: {self.run_dir} has no accepted factor")
        if not (self.run_dir / "acceptance" / "validation.json").exists():
            raise ValueError("acceptance: run validate() before opening the hold-out")
        lock = segments.open_holdout(self.lock_path, force=force)
        return self._compare(segments.holdout, trained_to=segments.validation, lock=lock)

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
            If a name is not a lowercase identifier, or its module or store already exists.
        """
        records = self._named(names)
        package = importlib.import_module(self.config.acceptance.package)
        package_dir = Path(package.__path__[0])
        root = Path(self.config.acceptance.factor_root or get_data_root() / "factors" / "market")
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
            module.write_text(_module_source(name, record, self.run_dir))
            try:
                importlib.invalidate_caches()
                factor_class = getattr(importlib.import_module(f"{package.__name__}.{name}"), _class_name(name))
                factor = factor_class(self._factor_config(factor_class.GRAPHS))
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

    def _compare(self, segment: Segment, trained_to: Segment, lock: HoldoutLock | None = None) -> AcceptanceReport:
        records = self.accepted()
        if not records:
            raise ValueError(f"acceptance: {self.run_dir} has no accepted factor")
        settings, segments = self.config.acceptance, self.config.segments
        label, scoring = self._data("label"), self._scoring()
        calendar = self._data("prices").calendar(segments.training.start, segment.end)
        lookahead = label.lookahead_bars()
        train_window = (segments.training.start, segments.scored(trained_to, calendar, lookahead)[1])
        test_window = segments.scored(segment, calendar, lookahead)
        outputs = {_feedback.accepted_label(r): f"accepted_{k}" for k, r in enumerate(records)}
        if len(outputs) != len(records):
            raise ValueError("acceptance: two accepted factors share a label (description and round)")
        accepted = self._accepted_feature(records, list(outputs.values()), test_window[1])
        baseline_features = [scoring.on_members(load(c)) for c in (settings.features or self.config.reference)]
        if not baseline_features:
            raise ValueError("acceptance: the baseline has no feature (neither acceptance.features nor reference)")
        name = label.get_factor_names()[0]
        truth = label.read(*test_window)[name]
        results = {}
        for kind, features in (("baseline", baseline_features), ("with_accepted", [*baseline_features, accepted])):
            panel, results[kind] = None, []
            for seed in settings.seeds:
                head = XGBoostRegressor(ModelConfig(
                    factors=features, labels=[label],
                    model_save_dir=str(self.run_dir / "acceptance" / segment.name / f"{kind}_s{seed}"),
                    factor_data_strategy="read", label_data_strategy="read",
                    start_date=train_window[0], end_date=test_window[1],
                    train_start=train_window[0], train_end=train_window[1],
                    test_start=test_window[0], test_end=test_window[1],
                    val_size=settings.val_size, metric_splits=("test",), random_seed=seed,
                    hyperparameters=dict(settings.hyperparameters),
                ))
                if panel is None:
                    panel = head.collect().collected_panel
                else:
                    head.use_panel(panel)
                run = TrainedRun.open(head.train())
                predictions = scoring.mask(xr.open_zarr(run.test_predictions)[[name]].load())[name]
                results[kind].append(self._result(seed, predictions, truth, label.span_bars(), str(run.path)))
        report = AcceptanceReport(
            segment=segment.name, train_window=train_window, test_window=test_window, seeds=settings.seeds,
            top_k=settings.top_k, accepted=outputs,
            baseline=tuple(results["baseline"]), with_accepted=tuple(results["with_accepted"]),
            forced=lock.last.forced if lock else False,
            holdout_opens=tuple(dataclasses.asdict(o) for o in lock.opens) if lock else (),
            code_version=code_version(),
        )
        out = self.run_dir / "acceptance"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{segment.name}.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        (out / f"{segment.name}.md").write_text(report.markdown())
        return report

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
        """The accepted factors, scored, from a store under the run directory, read as the scoring reads."""
        graphs = dict(zip(outputs, (r.graph for r in records)))
        factor = GraphFactor(self._factor_config(graphs, kwargs={"graphs": graphs}))
        folder = self.run_dir / "acceptance" / "accepted"
        scored = self._scoring().apply(factor, folder / "accepted.zarr")
        start = self.config.segments.training.start
        built = {"start": start, "end": end, "config": scored.get_config()}
        record = folder / "accepted.json"
        if not record.exists() or json.loads(record.read_text()) != json.loads(json.dumps(built)):
            shutil.rmtree(folder, ignore_errors=True)
            scored.build(start, end)
            record.write_text(json.dumps(built) + "\n")
        return self._scoring().on_members(scored)

    # Shared.

    def _factor_config(self, graphs: dict, kwargs: dict | None = None) -> FactorConfig:
        return FactorConfig(
            warmup_bars=self.config.evaluator.warmup_bars, dataset=self._data("prices"), mode="batch",
            data_columns=GraphFactor.data_columns(graphs), njobs=self.config.evaluator.njobs, kwargs=kwargs,
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

    def _named(self, names: dict[str, str]) -> list[TrialRecord]:
        """The accepted record each key of ``names`` names, by label or description."""
        accepted = self.accepted()
        out = []
        for key in names:
            matches = [r for r in accepted if _feedback.accepted_label(r) == key] or [
                r for r in accepted if r.candidate["description"] == key
            ]
            if len(matches) != 1:
                raise KeyError(
                    f"export: {key!r} names {len(matches)} accepted factors; "
                    f"accepted: {[_feedback.accepted_label(r) for r in accepted]}"
                )
            out.append(matches[0])
        return out

    def _readme(self, name: str, record: TrialRecord, factor_class: type, start: str, end: str) -> str:
        candidate = record.candidate
        validation = self.run_dir / "acceptance" / "validation.json"
        lines = [
            f"# {name}", "",
            f"{candidate['description']}", "",
            "## What it is", "",
            f"- Hypothesis: {candidate['hypothesis']}",
            f"- Mechanism type: {candidate['mechanism_type']}; expected sign: {candidate['expected_sign']:+d}.",
            f"- Horizon: {candidate['horizon_rationale']}",
            f"- Reference: {candidate['reference']}",
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


def _class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part) or "Factor"


def _module_source(name: str, record: TrialRecord, run_dir: Path) -> str:
    """The module of an exported factor: the candidate as data and a ``GraphFactor`` subclass fixing its graph."""
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
    )
