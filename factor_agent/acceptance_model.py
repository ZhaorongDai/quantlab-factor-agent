"""The acceptance model: a quantlab model template that Acceptance fills in and trains.

``acceptance.model`` is the component config (a ``get_config()`` dict) of a
quantlab ``BaseModel``, or of a ``ModelEnsemble`` whose ``members`` are such
configs. A model's ``factors`` are the baseline: entries as the campaign
config's ``reference`` lists them, or empty (or absent) for the reference set
itself. Everything else of the template, its ``hyperparameters`` above all,
is the model's own.

``build`` fills the template in: each model (each member of an ensemble) gets
its baseline followed by the factors added, and the fields Acceptance owns
(``FILLED``: the label, the save directory, the read strategies, the
windows, ``val_size``, the scored split and the seed) replace whatever the
template says of them, so a config copied from a trained run's
``config.json`` is a template as it is. A ``SeedEnsemble`` template is
refused: Acceptance trains every model once per ``acceptance.seeds``.
"""

import dataclasses
from collections.abc import Callable

from quantlab.core.component import get_cls_from_path, rebuild
from quantlab.model.base import BaseModel
from quantlab.model.config import ModelConfig
from quantlab.model.predefined.model_ensemble import ModelEnsemble
from quantlab.model.predefined.seed_ensemble import SeedEnsemble

#: The ``ModelConfig`` fields Acceptance sets in every model, whatever the template says.
FILLED = (
    "labels", "model_save_dir", "factor_data_strategy", "label_data_strategy", "start_date", "end_date",
    "train_start", "train_end", "test_start", "test_end", "val_size", "metric_splits", "random_seed",
)
#: The ``ModelConfig`` fields a template may give.
_FIELDS = {f.name for f in dataclasses.fields(ModelConfig)}


def check(template: dict) -> None:
    """Refuse a template that ``build`` cannot fill in, before anything is read or trained.

    Raises
    ------
    ValueError
        If the template is a ``SeedEnsemble`` (seeds are ``acceptance.seeds``).
    TypeError
        If it names no class, a class that is neither a ``BaseModel`` nor a
        ``ModelEnsemble``, or a model with a key ``ModelConfig`` lacks.
    """
    if not isinstance(template, dict) or "name" not in template:
        raise TypeError("acceptance.model: the model's component config needs its class path as name")
    cls = get_cls_from_path(template["name"])
    if not isinstance(cls, type):
        raise TypeError(f"acceptance.model: {template['name']} is not a BaseModel or a ModelEnsemble")
    if issubclass(cls, SeedEnsemble):
        raise ValueError(
            "acceptance.model: a SeedEnsemble is not a template; give its model and set the seeds as acceptance.seeds"
        )
    if issubclass(cls, ModelEnsemble):
        unknown = set(template) - {"name", "members"}
        if unknown:
            raise TypeError(f"acceptance.model: the ModelEnsemble has unknown key(s) {sorted(unknown)}")
        for member in template.get("members") or ():
            check(member)
        return
    if not issubclass(cls, BaseModel):
        raise TypeError(f"acceptance.model: {template['name']} is not a BaseModel or a ModelEnsemble")
    unknown = set(template) - _FIELDS
    if unknown:
        raise TypeError(f"acceptance.model: {template['name']} has key(s) {sorted(unknown)} ModelConfig lacks")


def build(template: dict, baseline: Callable[[tuple], list], added: list, filled: dict):
    """The model (or ``ModelEnsemble``) of ``template``, filled in and untrained.

    Parameters
    ----------
    template : dict
        ``acceptance.model``; see the module docstring.
    baseline : callable
        A model's ``factors`` entries (empty for the reference set) to its baseline features.
    added : list
        The features added to every model after its baseline.
    filled : dict
        A value for each of ``FILLED``, set in every model.
    """
    check(template)
    cls = get_cls_from_path(template["name"])
    if issubclass(cls, ModelEnsemble):
        return cls([build(member, baseline, added, filled) for member in template.get("members") or ()])
    fields = {key: value for key, value in template.items() if key not in ("name", *FILLED)}
    if "tracker" in fields:
        fields["tracker"] = rebuild(fields["tracker"])
    fields["factors"] = [*baseline(tuple(fields.get("factors") or ())), *added]
    return cls(ModelConfig(**fields, **filled))
