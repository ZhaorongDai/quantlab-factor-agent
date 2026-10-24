"""The campaign config: one YAML file holds every setting of a run.

``data`` holds quantlab components: each is either a ``get_config()`` dict
inline or the path of a store folder whose ``component.json`` describes it
(``quantlab.core.store_folder.save_component``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from quantlab.core.component import rebuild
from quantlab.core.store_folder import load_component

from factor_agent.segments import Segment


@dataclass(frozen=True)
class DataConfig:
    """The whole-market price dataset and the forward label masked by day-t membership."""

    prices: dict | str
    label: dict | str

    def load(self, name: str) -> Any:
        """Rebuild the component ``name`` (``"prices"`` or ``"label"``)."""
        value = getattr(self, name)
        return load_component(value) if isinstance(value, str) else rebuild(value)


@dataclass(frozen=True)
class EvaluatorConfig:
    """How the Evaluator computes the round's graph."""

    #: Bars read before the segment start so every window is filled.
    warmup_bars: int = 260
    #: KunQuant executor threads.
    njobs: int = 51


@dataclass(frozen=True)
class GateConfig:
    """The Gate's sandbox limits and graph constraints."""

    #: Seconds the sandboxed graph build may take.
    timeout_seconds: float = 10.0
    #: Memory, in MB, the sandboxed graph build may use.
    memory_mb: int = 1024
    #: Nodes a candidate's graph may have.
    max_nodes: int = 200
    #: Hand-set constants (other than 0 and +-1) per operator.
    max_free_parameter_ratio: float = 0.5
    #: Distinct input fields (open, high, low, close, volume, vwap, amount) a candidate may read.
    max_base_fields: int = 5


@dataclass(frozen=True)
class AdmissionConfig:
    """What an evaluated candidate needs to be accepted."""

    #: Newey-West t of the mean daily rank IC a candidate needs.
    t_threshold: float = 3.0


@dataclass(frozen=True)
class RoundConfig:
    """The size of a round."""

    #: Candidates asked for per round.
    candidates: int = 20
    #: Times a candidate the parser or the Gate refused is sent back to the proposer to fix.
    retries: int = 2


@dataclass(frozen=True)
class RoleConfig:
    """One role's LLM: the backend name, the model and the backend's own settings."""

    backend: str
    model: str
    options: dict

    @classmethod
    def from_dict(cls, data: dict) -> "RoleConfig":
        """Split a role's YAML mapping into the backend, the model and the rest."""
        data = dict(data)
        return cls(backend=data.pop("backend"), model=data.pop("model"), options=data)

    def spec(self) -> dict:
        """The dict ``make_backend`` takes."""
        return {"backend": self.backend, "model": self.model, **self.options}


@dataclass(frozen=True)
class CampaignConfig:
    """Every setting of a campaign, read from one YAML file with ``from_file``."""

    run_dir: str
    training: Segment
    data: DataConfig
    proposer: RoleConfig
    evaluator: EvaluatorConfig = EvaluatorConfig()
    gate: GateConfig = GateConfig()
    admission: AdmissionConfig = AdmissionConfig()
    round: RoundConfig = RoundConfig()

    @classmethod
    def from_file(cls, path: str | Path) -> "CampaignConfig":
        """Read a campaign config from YAML.

        Raises
        ------
        TypeError
            If a section holds a key its config class does not have.
        KeyError
            If a required section is missing.
        """
        raw = yaml.safe_load(Path(path).read_text())
        sections = {
            "evaluator": EvaluatorConfig, "gate": GateConfig,
            "admission": AdmissionConfig, "round": RoundConfig,
        }
        unknown = set(raw) - {"run_dir", "segments", "data", "llm", *sections}
        if unknown:
            raise TypeError(f"campaign config: unknown section(s) {sorted(unknown)}")
        training = raw["segments"]["training"]
        return cls(
            run_dir=raw["run_dir"],
            training=Segment("training", str(training["start"]), str(training["end"])),
            data=DataConfig(**raw["data"]),
            proposer=RoleConfig.from_dict(raw["llm"]["proposer"]),
            **{name: klass(**raw[name]) for name, klass in sections.items() if name in raw},
        )
