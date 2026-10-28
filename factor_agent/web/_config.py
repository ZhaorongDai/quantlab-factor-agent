"""The files a run was set up with: the config of each opening and the snapshot's files."""

from dataclasses import dataclass
from pathlib import Path

from factor_agent.web._files import read_json


@dataclass(frozen=True)
class ConfigCopy:
    """The config text the run's ``opening``-th start or resume was given, opened at ``opened_at`` (ISO, UTC)."""

    opening: int
    opened_at: str | None
    text: str


@dataclass(frozen=True)
class SnapshotFile:
    """A file of the run's snapshot by its name there (``fields/<file>``, ``operators/<file>``, ``prompts/<file>``)."""

    name: str
    text: str


@dataclass(frozen=True)
class RunFiles:
    """The configs of a run, the latest opening first, empty for a run written before configs were kept, and its snapshot's files."""

    configs: tuple[ConfigCopy, ...]
    snapshot: tuple[SnapshotFile, ...]


def read_files(run_dir: str | Path) -> RunFiles:
    """The files ``run_dir`` was set up with, as its config page shows them."""
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "snapshot.json") or {}
    versions = manifest.get("versions") or []
    configs = []
    for path in (run_dir / "config").glob("open-*.yaml"):
        opening = int(path.stem.removeprefix("open-"))
        opened_at = versions[opening - 1].get("opened_at") if opening <= len(versions) else None
        configs.append(ConfigCopy(opening, opened_at, path.read_text()))
    snapshot = tuple(SnapshotFile(name, (run_dir / "snapshot" / name).read_text()) for name in manifest.get("files", {}))
    return RunFiles(tuple(sorted(configs, key=lambda c: c.opening, reverse=True)), snapshot)
