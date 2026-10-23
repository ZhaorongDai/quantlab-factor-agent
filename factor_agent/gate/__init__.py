"""The Gate: deterministic checks that every candidate passes before it is evaluated.

``Gate.check`` never reads data. A bad candidate comes back ``Rejected``
with readable reasons; only an infrastructure failure raises.
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

from factor_agent.candidate import Candidate
from factor_agent.gate._checks import static_reasons
from factor_agent.gate.graph import Graph

#: Characters of the subprocess's standard error quoted when it crashes.
_STDERR_TAIL = 2000


@dataclass(frozen=True)
class Passed:
    """A candidate whose graph was built: the graph, its fingerprint and its statistics."""

    graph: Graph
    fingerprint: str
    stats: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Rejected:
    """A candidate refused by the gate, with the reasons sent back to the proposer."""

    reasons: tuple[str, ...]


class Gate:
    """Check candidates: static AST checks, then a sandboxed graph build.

    Parameters
    ----------
    timeout_seconds : float
        How long the sandboxed build may run before it is killed.
    """

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds

    def check(self, candidate: Candidate) -> Passed | Rejected:
        """Return ``Passed`` with the candidate's graph, or ``Rejected`` with why not.

        Raises
        ------
        RuntimeError
            If the sandbox cannot be started or replies with something
            that is not its protocol.
        """
        reasons = static_reasons(candidate.code)
        if reasons:
            return Rejected(tuple(reasons))
        reply = self._build(candidate.code)
        if isinstance(reply, Rejected):
            return reply
        graph = Graph.from_json(reply)
        return Passed(graph, graph.fingerprint(), {"nodes": len(graph.nodes)})

    def _build(self, code: str) -> list | Rejected:
        """Build the graph in the sandbox subprocess; return its nodes or the rejection."""
        with tempfile.TemporaryDirectory(prefix="factor-agent-gate-") as scratch:
            try:
                done = subprocess.run(
                    [sys.executable, "-I", "-m", "factor_agent.gate._sandbox"],
                    input=json.dumps({"code": code}),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=scratch,
                    env={"PATH": os.defpath},
                )
            except subprocess.TimeoutExpired:
                return Rejected((f"graph build timed out after {self.timeout_seconds:g} s",))
        if done.returncode != 0:
            tail = done.stderr[-_STDERR_TAIL:].strip()
            if "ModuleNotFoundError" in tail and "factor_agent" in tail:
                raise RuntimeError(f"Gate: the sandbox cannot import factor_agent:\n{tail}")
            return Rejected((f"graph build crashed (exit {done.returncode}): {tail}",))
        try:
            message = json.loads(done.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gate: the sandbox replied {done.stdout[:200]!r}") from exc
        if "reasons" in message:
            return Rejected(tuple(message["reasons"]))
        return message["graph"]
