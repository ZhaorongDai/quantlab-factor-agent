"""The RunSnapshot: the files a run searched with and the code it ran, kept in its run directory.

A run's search space is what its files say (``SearchFiles``): the field
files and the operator file and the replaced prompt templates. On a first
start ``open`` copies each file into ``<run_dir>/snapshot/<name>`` and records
its hash and the code versions (the commits of this package and of
quantlab) in ``<run_dir>/snapshot.json``. On a resume a file whose name or
content differs from the snapshot is an error, since a new search space
needs a new run directory; a code version that differs only warns, so a
bug-fix resume is allowed but visible, and every code version a run was
opened with is kept in order.

The run reads the snapshot's copies (``search_files``), not the files it was given,
and ``read`` opens a run's snapshot without recording anything, for
Acceptance and for a replay of the run.
"""

import hashlib
import json
import shutil
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

#: The code versions compared on a resume (``segments.code_version`` holds them and quantlab's version).
COMPARED_VERSIONS = ("factor_agent", "quantlab")


class SnapshotMismatch(ValueError):
    """A resume was given files that differ from the run's snapshot."""


class CodeVersionChanged(UserWarning):
    """A resume runs other code than the run was last opened with."""


@dataclass(frozen=True)
class SearchFiles:
    """The files a run searches with: the field files, the operator file and the replaced templates.

    ``prompts`` maps a template's name to the file that replaces it. In a
    snapshot a field file is named ``fields/<file name>``, the operator file
    ``operators/<file name>`` and a replaced template
    ``prompts/<template>/<file name>`` (``names``, ``from_names``).
    """

    fields: tuple
    operators: str | Path
    prompts: Mapping[str, str | Path] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(self.fields))

    def names(self) -> dict[str, str | Path]:
        """Every file by its name in a snapshot, field files first, then the operator file and the templates.

        Raises
        ------
        ValueError
            If two field files have the same file name.
        """
        files = {f"fields/{Path(path).name}": path for path in self.fields}
        if len(files) != len(self.fields):
            raise ValueError(f"run snapshot: the field files {[str(f) for f in self.fields]} repeat a file name")
        templates = {f"prompts/{name}/{Path(path).name}": path for name, path in self.prompts.items()}
        return {**files, f"operators/{Path(self.operators).name}": self.operators, **templates}

    @classmethod
    def from_names(cls, files: Mapping[str, str | Path]) -> "SearchFiles":
        """The files of a snapshot's ``files`` (name to path), as ``names`` names them."""
        [operators] = [path for name, path in files.items() if name.startswith("operators/")]
        return cls(
            fields=tuple(path for name, path in files.items() if name.startswith("fields/")),
            operators=operators,
            prompts={name.split("/")[1]: path for name, path in files.items() if name.startswith("prompts/")},
        )


class RunSnapshot:
    """The snapshot of a run directory; build it with ``open`` or ``read``.

    ``files`` maps each name to its copy, in the order the run was first
    given them; ``versions`` lists the code versions of every opening.
    """

    def __init__(self, run_dir: Path, manifest: dict):
        self.run_dir = run_dir
        self.files = {name: run_dir / "snapshot" / name for name in manifest["files"]}
        self.versions: list[dict] = manifest["versions"]
        self.settings: dict = manifest.get("settings", {})

    @classmethod
    def open(
        cls, run_dir: str | Path, search: SearchFiles, versions: Mapping[str, str], settings: Mapping | None = None,
    ) -> "RunSnapshot":
        """Open ``run_dir`` with the files of ``search``, the code ``versions`` it runs on and its ``settings``.

        A first start copies the files, under their ``SearchFiles.names``, and
        records ``settings`` (by section, the config settings the prompts
        depend on, such as ``{"context": {"recent_rounds": 5}}``); a resume
        checks the files against the copies and the settings against the
        recorded ones. Either way ``versions`` is appended to the recorded versions.

        Raises
        ------
        SnapshotMismatch
            On a resume, naming every file added, removed or changed and every setting changed since the first start.
        ValueError
            If two field files have the same file name.

        Warns
        -----
        CodeVersionChanged
            On a resume whose ``versions`` differ from the last recorded ones.
        """
        run_dir = Path(run_dir)
        files = search.names()
        manifest_path = run_dir / "snapshot.json"
        hashes = {name: _hash(Path(path)) for name, path in files.items()}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            _check(run_dir, manifest["files"], hashes)
            _check_settings(run_dir, manifest.get("settings", {}), dict(settings or {}))
            _warn(manifest["versions"][-1], versions)
        else:
            manifest = {"files": hashes, "settings": dict(settings or {}), "versions": []}
            for name, path in files.items():
                copy = run_dir / "snapshot" / name
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, copy)
        manifest["versions"].append({**versions, "opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.replace(manifest_path)
        return cls(run_dir, manifest)

    @classmethod
    def read(cls, run_dir: str | Path) -> "RunSnapshot":
        """The snapshot of ``run_dir`` as it is, recording nothing.

        Raises
        ------
        FileNotFoundError
            If the run directory has no snapshot.
        """
        run_dir = Path(run_dir)
        manifest_path = run_dir / "snapshot.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"run snapshot: {run_dir} has no snapshot ({manifest_path})")
        return cls(run_dir, json.loads(manifest_path.read_text()))

    @staticmethod
    def exists(run_dir: str | Path) -> bool:
        """Whether ``run_dir`` has a snapshot."""
        return (Path(run_dir) / "snapshot.json").exists()

    def path(self, name: str) -> Path:
        """The snapshot's copy of the file ``name``.

        Raises
        ------
        KeyError
            If the snapshot holds no file of that name.
        """
        if name not in self.files:
            raise KeyError(f"run snapshot of {self.run_dir}: no file {name!r}; it holds {list(self.files)}")
        return self.files[name]

    @property
    def search_files(self) -> SearchFiles:
        """The snapshot's copies, as the files the run searches with."""
        return SearchFiles.from_names(self.files)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(run_dir: Path, recorded: dict, given: dict) -> None:
    """Raise ``SnapshotMismatch`` naming every file of ``given`` that is not the one ``recorded``."""
    problems = [f"{name} was added" for name in given if name not in recorded]
    problems += [f"{name} was removed" for name in recorded if name not in given]
    problems += [f"{name} changed" for name in given if name in recorded and given[name] != recorded[name]]
    if problems:
        raise SnapshotMismatch(
            f"run snapshot of {run_dir}: the run's files differ from the ones it started with: "
            f"{'; '.join(problems)}. A changed search space needs a new run directory."
        )


def _check_settings(run_dir: Path, recorded: dict, given: dict) -> None:
    """Raise ``SnapshotMismatch`` naming every setting of ``given`` that is not the one ``recorded``."""
    before, now = _flat(recorded), _flat(given)
    changed = [
        f"{name} was {before.get(name, 'not recorded')}, now {now.get(name, 'not given')}"
        for name in sorted(before.keys() | now.keys()) if before.get(name) != now.get(name)
    ]
    if changed:
        raise SnapshotMismatch(
            f"run snapshot of {run_dir}: the run's settings differ from the ones it started with: "
            f"{'; '.join(changed)}. A changed search space needs a new run directory."
        )


def _flat(settings: dict) -> dict:
    """``settings`` by section as one mapping from ``"section.key"`` to the value."""
    return {f"{section}.{key}": value for section, values in settings.items() for key, value in values.items()}


def _warn(last: dict, versions: Mapping[str, str]) -> None:
    changed = [f"{key} {last.get(key)} -> {versions.get(key)}" for key in COMPARED_VERSIONS if last.get(key) != versions.get(key)]
    if changed:
        warnings.warn(
            f"run snapshot: the run resumes on other code than it was last opened with: {'; '.join(changed)}",
            CodeVersionChanged, stacklevel=3,
        )
