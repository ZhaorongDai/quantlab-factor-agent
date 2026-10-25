"""The trained models' results of one acceptance segment, cached so a re-run trains only what is missing.

A model's key is the sha256 of its filled-in component config (the template,
its features' component configs, the label, the windows and the seed), less
its save directory, and of whatever else its result depends on (the TopK
size). ``ModelCache`` keeps one JSON file per key under the segment's
acceptance directory: the result as a plain dict and the seconds its
training and scoring took, the record the duration estimate reads.
"""

import hashlib
import json
from pathlib import Path


def key(config: dict, **extra) -> str:
    """The cache key of a model of component ``config``, whose result also depends on ``extra``."""
    payload = json.dumps({"model": _without_save_dir(config), **extra}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class ModelCache:
    """Cached model results in ``directory``, by ``key``."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def get(self, key: str) -> dict | None:
        """The result cached for ``key``, or None."""
        path = self.directory / f"{key}.json"
        return json.loads(path.read_text())["result"] if path.exists() else None

    def put(self, key: str, result: dict, elapsed: float) -> None:
        """Cache ``result``, trained and scored in ``elapsed`` seconds; written whole or not at all."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps({"result": result, "elapsed": elapsed}) + "\n")
        partial.replace(path)

    def summary(self, keys: list[str]) -> str:
        """How many of ``keys`` are to train and cached, and the time to train them from the cached elapsed times."""
        missing = sum(not (self.directory / f"{k}.json").exists() for k in keys)
        line = f"{missing} models to train, {len(keys) - missing} cached"
        elapsed = [json.loads(p.read_text())["elapsed"] for p in sorted(self.directory.glob("*.json"))]
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
