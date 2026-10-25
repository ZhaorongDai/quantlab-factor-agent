"""The trained models' results of one acceptance segment, cached so a re-run trains only what is missing.

A model's key is the sha256 of its filled-in component config (the template,
its features' component configs, the label, the windows and the seed), less
its save directory and with every path under the run directory relative to
it (``portable``), so a moved run directory keeps its cache, and of whatever
else its result depends on (the code version, the TopK size, the scoring).
The digest is of plain JSON: a value JSON cannot hold raises rather than
hashing its ``repr``. ``ModelCache`` keeps one JSON file per key under the
segment's acceptance directory: the result as a plain dict, the seconds its
training and scoring took and the digest of its template, the record the
duration estimate reads.
"""

import hashlib
import json
from pathlib import Path


def key(config: dict, root: Path, **extra) -> str:
    """The cache key of a model of component ``config`` in the run directory ``root``, whose result also depends on ``extra``.

    Raises
    ------
    TypeError
        If ``config`` or ``extra`` holds a value JSON cannot hold.
    """
    return digest({"model": portable(_without_save_dir(config), root), **extra})


def digest(value) -> str:
    """The sha256 of ``value`` as sorted JSON; a value JSON cannot hold raises ``TypeError``."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def portable(value, root: Path):
    """``value`` with every string path under ``root`` written relative to it, as ``<run_dir>/...``."""
    if isinstance(value, dict):
        return {k: portable(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v, root) for v in value]
    if isinstance(value, str):
        for prefix in {str(Path(root)), str(Path(root).resolve())}:
            if value == prefix or value.startswith(prefix + "/"):
                return "<run_dir>" + value[len(prefix):]
    return value


class ModelCache:
    """Cached model results in ``directory``, by ``key``."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def get(self, key: str) -> dict | None:
        """The result cached for ``key``, or None."""
        path = self.directory / f"{key}.json"
        return json.loads(path.read_text())["result"] if path.exists() else None

    def put(self, key: str, result: dict, elapsed: float, template: str) -> None:
        """Cache ``result`` of a model of the template digested ``template``, trained and scored in ``elapsed`` seconds.

        Written whole or not at all.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps({"result": result, "elapsed": elapsed, "template": template}) + "\n")
        partial.replace(path)

    def summary(self, keys: list[str], template: str) -> str:
        """How many of ``keys`` are to train and cached, and the time to train them.

        The estimate averages the elapsed times of the finished models of the template digested ``template``.
        """
        missing = sum(not (self.directory / f"{k}.json").exists() for k in keys)
        line = f"{missing} models to train, {len(keys) - missing} cached"
        records = [json.loads(p.read_text()) for p in sorted(self.directory.glob("*.json"))]
        elapsed = [r["elapsed"] for r in records if r["template"] == template]
        if missing and elapsed:
            mean = sum(elapsed) / len(elapsed)
            line += f", about {_duration(missing * mean)} ({_duration(mean)} a model over {len(elapsed)} finished)"
        return line


def _without_save_dir(config):
    """``config`` without any ``model_save_dir``, an ensemble's members' included."""
    if isinstance(config, dict):
        return {k: _without_save_dir(v) for k, v in config.items() if k != "model_save_dir"}
    if isinstance(config, (list, tuple)):
        return [_without_save_dir(v) for v in config]
    return config


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"
