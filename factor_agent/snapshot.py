"""The RunSnapshot: the files a run searched with and the code it ran, kept in its run directory.

A run's search space is what its files say: the field files and the
operator file (and, later, the prompt templates). On a first start
``open`` copies each file into ``<run_dir>/snapshot/<name>`` and records
its hash and the code versions (the commits of this package and of
quantlab) in ``<run_dir>/snapshot.json``. On a resume a file whose name or
content differs from the snapshot is an error, since a new search space
needs a new run directory; a code version that differs only warns, so a
bug-fix resume is allowed but visible, and every code version a run was
opened with is kept in order.

The run reads the snapshot's copies (``path``), not the files it was given,
and ``read`` opens a run's snapshot without recording anything, for
Acceptance and for a replay of the run.
"""

import hashlib
import json
import shutil
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

#: The code versions compared on a resume (``segments.code_version`` holds them and quantlab's version).
COMPARED_VERSIONS = ("factor_agent", "quantlab")


class SnapshotMismatch(ValueError):
    """A resume was given files that differ from the run's snapshot."""


class CodeVersionChanged(UserWarning):
    """A resume runs other code than the run was last opened with."""


class RunSnapshot:
    """The snapshot of a run directory; build it with ``open`` or ``read``.

    ``files`` maps each name to its copy, in the order the run was first
    given them; ``versions`` lists the code versions of every opening.
    """

    def __init__(self, run_dir: Path, manifest: dict):
        self.run_dir = run_dir
        self.files = {name: run_dir / "snapshot" / name for name in manifest["files"]}
        self.versions: list[dict] = manifest["versions"]

    @classmethod
    def open(cls, run_dir: str | Path, files: Mapping[str, str | Path], versions: Mapping[str, str]) -> "RunSnapshot":
        """Open ``run_dir`` with ``files`` (name to path) and the code ``versions`` it runs on.

        A name is a relative path inside the snapshot, such as
        ``fields/price_volume.yaml``. A first start copies the files; a
        resume checks them against the copies. Either way ``versions`` is
        appended to the recorded versions.

        Raises
        ------
        SnapshotMismatch
            On a resume, naming every file added, removed or changed since the first start.

        Warns
        -----
        CodeVersionChanged
            On a resume whose ``versions`` differ from the last recorded ones.
        """
        run_dir = Path(run_dir)
        manifest_path = run_dir / "snapshot.json"
        hashes = {name: _hash(Path(path)) for name, path in files.items()}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            _check(run_dir, manifest["files"], hashes)
            _warn(manifest["versions"][-1], versions)
        else:
            manifest = {"files": hashes, "versions": []}
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

    def names(self, prefix: str = "") -> list[str]:
        """The names of the snapshot's files that start with ``prefix``, in their recorded order."""
        return [name for name in self.files if name.startswith(prefix)]


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


def _warn(last: dict, versions: Mapping[str, str]) -> None:
    changed = [f"{key} {last.get(key)} -> {versions.get(key)}" for key in COMPARED_VERSIONS if last.get(key) != versions.get(key)]
    if changed:
        warnings.warn(
            f"run snapshot: the run resumes on other code than it was last opened with: {'; '.join(changed)}",
            CodeVersionChanged, stacklevel=3,
        )
