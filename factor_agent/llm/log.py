"""The LLM log: every call of a run, in full, one JSON file per call under ``llm_log/``.

A file ``<number>-<role>.json`` holds the request (role, model, prompt and
the parameters sent) and the reply (text, usage, the model that
answered and the provider's raw reply); the numbers count the calls of the
run from 1, across every role, in the order they finish (calls sent side by
side are written one at a time); the calls of a round a resume dropped are
kept in ``dropped/``, out of the sequence. No key is ever part of a request
or a reply, so none is written.
"""

import json
import threading
from dataclasses import asdict
from pathlib import Path

from factor_agent.llm.base import LLMBackend, Reply, Request


class LLMLog:
    """The call files of one run directory's ``llm_log``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._writing = threading.Lock()

    def files(self) -> list[Path]:
        """The call files, in call order."""
        return sorted(self.path.glob("*.json"))

    def drop_after(self, count: int) -> None:
        """Move every call after the first ``count`` to ``dropped/``: calls of a round a crash cut off."""
        for file in self.files()[count:]:
            (self.path / "dropped").mkdir(exist_ok=True)
            file.replace(self.path / "dropped" / file.name)

    def calls(self) -> list[dict]:
        """Every recorded call, in call order, as ``{"request": ..., "reply": ...}``."""
        return [json.loads(file.read_text()) for file in self.files()]

    def write(self, request: Request, reply: Reply) -> Path:
        """Write the next call file and return its path; the file appears whole or not at all.

        Safe to call from several threads: one call is written at a time, each under the next number.
        """
        call = {
            "request": {**asdict(request), "params": {**request.params, **reply.params}},
            "reply": {k: v for k, v in asdict(reply).items() if k != "params"},
        }
        with self._writing:
            self.path.mkdir(parents=True, exist_ok=True)
            number = len(self.files()) + 1
            path = self.path / f"{number:06d}-{request.role}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(call, indent=2, ensure_ascii=False) + "\n")
            tmp.replace(path)
        return path


class RecordingBackend:
    """A backend that writes every call it passes on to ``log``."""

    def __init__(self, backend: LLMBackend, log: LLMLog):
        self.backend = backend
        self.log = log

    def complete(self, request: Request) -> Reply:
        """Pass ``request`` on and write the call once its reply is back."""
        reply = self.backend.complete(request)
        self.log.write(request, reply)
        return reply
