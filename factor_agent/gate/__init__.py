"""The Gate: deterministic checks that every candidate passes before it is evaluated.

``Gate.check`` never reads data; what candidates may write with is its
``Vocabulary``, from the campaign's field and operator files. A bad candidate comes back ``Rejected``
with readable reasons; only an infrastructure failure raises. The checks
run in this order:

1. static checks on the code (syntax allow-list, names in the vocabulary, ``alpha`` assigned);
2. the graph build in a sandboxed subprocess (timeout, memory limit,
   bubblewrap filesystem isolation where it is installed);
3. the stage rules of an intraday family's candidate (``stages.split``):
   an intraday field reaches the daily stage only through a session
   aggregation, the intraday stage reads no daily field, each operator is
   used only in its stages;
4. the graph's size, parameter values, dimensions and complexity, all reported together;
5. originality against the zoo: not a rewrite of a zoo factor, not one with
   other windows or parameters only, and no common subtree with one larger
   than the limit. The reasons name the zoo factor and quote the subtree;
6. not a duplicate of a candidate this gate passed before: the same canonical
   graph up to what keeps or reverses its ranking (``Graph.ranking``): a
   factor and its negation predict alike, in opposite directions.
"""

import ctypes
import ctypes.util
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from factor_agent.candidate import Candidate
from factor_agent.gate import _constraints, _originality, stages
from factor_agent.gate._checks import static_reasons
from factor_agent.gate.graph import Graph

if TYPE_CHECKING:  # the sandbox imports this package; it must not load pandas through the config
    from factor_agent.config import GateConfig
    from factor_agent.gate.vocabulary import Vocabulary
    from factor_agent.zoo import Zoo

#: Characters of the subprocess's standard error quoted when it crashes.
_STDERR_TAIL = 2000

#: Seconds between two looks at the sandbox's memory.
_POLL_SECONDS = 0.02

#: The reason given when a graph build runs out of memory.
MEMORY_REASON = "graph build used more than the {memory_mb} MB memory limit"


@dataclass(frozen=True)
class Passed:
    """A candidate whose graph was built: the graph, its fingerprint and its statistics."""

    graph: Graph
    fingerprint: str
    stats: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Rejected:
    """A candidate refused by the gate, with the reasons sent back to the proposer.

    ``hints`` are the vocabulary's fixes for a writing error (a static check
    or the graph build), how to write it instead; they go with the reasons
    into the rewrite request only.
    """

    reasons: tuple[str, ...]
    hints: tuple[str, ...] = ()


class Gate:
    """Check candidates; see the module docstring for the order of the checks.

    A gate remembers the ranking of every candidate it passed and rejects
    a later candidate that only keeps or reverses one.

    Parameters
    ----------
    config : GateConfig
        The sandbox limits and the graph constraints.
    vocabulary : Vocabulary
        The fields, operators and parameter values candidates may use;
        the sandbox rebuilds the same one from its sources.
    seen : iterable of Graph, optional
        The graphs of candidates passed before, such as those of a campaign
        that is resumed; a candidate that keeps or reverses one's ranking
        (``Graph.ranking``) is a duplicate.
    """

    def __init__(self, config: "GateConfig", vocabulary: "Vocabulary", seen=()):
        self.config = config
        self.vocabulary = vocabulary
        self._seen: set[str] = {graph.ranking() for graph in seen}

    def check(self, candidate: Candidate, zoo: "Zoo") -> Passed | Rejected:
        """Return ``Passed`` with the candidate's graph, or ``Rejected`` with why not.

        ``zoo`` holds the factors the candidate must be original against.
        A writing error carries the hints of the vocabulary's fixes it
        matches (``Rejected.hints``).

        Raises
        ------
        RuntimeError
            If the sandbox cannot be started or replies with something
            that is not its protocol.
        """
        reasons = static_reasons(candidate.code, self.vocabulary.names)
        if reasons:
            return self._written_badly(reasons)
        reply = self._build(candidate.code)
        if isinstance(reply, Rejected):
            return self._written_badly(reply.reasons)
        graph = Graph.from_json(reply, self.vocabulary.classes)
        reasons = stages.split(graph, self.vocabulary).reasons
        if reasons:
            return Rejected(reasons)
        reasons, stats = _constraints.check(graph, self.config, self.vocabulary)
        if reasons:
            return Rejected(tuple(reasons))
        reasons = _originality.reasons(graph, zoo, self.config.max_common_subtree_nodes, self.vocabulary)
        if reasons:
            return Rejected(tuple(reasons))
        ranking = graph.ranking()
        if ranking in self._seen:
            return Rejected((
                "the graph is a duplicate: it keeps or reverses the ranking of an earlier candidate (the same "
                "canonical graph up to a rank, a constant offset or a scale, a sign flip included); "
                "propose a different mechanism",
            ))
        self._seen.add(ranking)
        return Passed(graph, graph.fingerprint(), stats)

    def withdraw(self, graph: Graph) -> None:
        """Forget that ``graph`` passed, so a rewrite of its candidate is not refused as its duplicate.

        A candidate the reviewer sends back is rewritten, and its rewrite
        often keeps the graph: the rewrite continues the same candidate, so
        the version it replaces gives up its ranking. A candidate the
        reviewer rejects for good keeps it, as a resume restores it.
        """
        self._seen.discard(graph.ranking())

    def _written_badly(self, reasons) -> Rejected:
        """A writing error's refusal, with the hints of the fixes its reasons match."""
        return Rejected(tuple(reasons), self.vocabulary.fix_hints(reasons))

    def _build(self, code: str) -> list | Rejected:
        """Build the graph in the sandbox subprocess; return its nodes or the rejection."""
        request = json.dumps({"code": code, "memory_mb": self.config.memory_mb, "vocabulary": self.vocabulary.sources})
        # The request (the vocabulary's files included) is larger than a pipe holds, so it is read from a file.
        with tempfile.TemporaryDirectory(prefix="factor-agent-gate-") as scratch, tempfile.TemporaryFile("w+") as stdin:
            stdin.write(request)
            stdin.flush()
            stdin.seek(0)
            command = [sys.executable, "-I", "-m", "factor_agent.gate._sandbox"]
            if shutil.which("bwrap") and _bubblewrap_works():
                command = _bubblewrap(command, self.vocabulary.sources["operator_modules"])
            try:
                process = subprocess.Popen(
                    command, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, cwd=scratch, env={"PATH": os.defpath},
                )
            except OSError as exc:
                raise RuntimeError(f"Gate: cannot start the sandbox: {exc}") from exc
            outcome = self._watch(process)
        if isinstance(outcome, Rejected):
            return outcome
        stdout, stderr = outcome
        if process.returncode != 0:
            tail = stderr[-_STDERR_TAIL:].strip()
            if "ModuleNotFoundError" in tail and "factor_agent" in tail:
                raise RuntimeError(f"Gate: the sandbox cannot import factor_agent:\n{tail}")
            if "MemoryError" in tail:
                return Rejected((MEMORY_REASON.format(memory_mb=self.config.memory_mb),))
            return Rejected((f"graph build crashed (exit {process.returncode}): {_last_line(tail)}",))
        try:
            message = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gate: the sandbox replied {stdout[:200]!r}") from exc
        if "reasons" in message:
            return Rejected(tuple(message["reasons"]))
        return message["graph"]

    def _watch(self, process: subprocess.Popen) -> tuple[str, str] | Rejected:
        """Wait for the reply, killing the sandbox past the time or memory limit."""
        deadline = time.monotonic() + self.config.timeout_seconds
        limit = self.config.memory_mb * 2**20
        watched = psutil.Process(process.pid)
        while True:
            try:
                return process.communicate(timeout=_POLL_SECONDS)
            except subprocess.TimeoutExpired:
                pass
            try:
                used = sum(_memory(p) for p in [watched, *watched.children(recursive=True)])
            except psutil.Error:
                continue
            if used > limit:
                reason = MEMORY_REASON.format(memory_mb=self.config.memory_mb)
            elif time.monotonic() > deadline:
                reason = f"graph build timed out after {self.config.timeout_seconds:g} s"
            else:
                continue
            process.kill()
            process.communicate()
            return Rejected((reason,))


def _memory(process: psutil.Process) -> int:
    """Return the bytes ``process`` uses.

    On macOS this is its physical footprint, which counts the pages the
    memory compressor holds; the resident set alone falls as the compressor works.
    """
    if sys.platform != "darwin":
        return process.memory_info().rss
    usage = (ctypes.c_uint64 * 40)()  # struct rusage_info_v2
    if _libproc().proc_pid_rusage(process.pid, 2, ctypes.byref(usage)) != 0:
        raise psutil.NoSuchProcess(process.pid)
    return usage[9]  # ri_phys_footprint, after the 16-byte uuid and seven counters


@cache
def _libproc() -> ctypes.CDLL:
    """macOS's libproc, loaded once."""
    return ctypes.CDLL(ctypes.util.find_library("proc"))


def _last_line(stderr: str) -> str:
    """The last line of a crash's stderr, its exception, with no file path, so a replay elsewhere reads the same."""
    lines = [line for line in stderr.splitlines() if line.strip()]
    return re.sub(r"(?<![\w.])(?:/[\w.-]+){2,}/", "", lines[-1].strip()) if lines else "no output"


@cache
def _bubblewrap_works() -> bool:
    """Check once that the installed bubblewrap can run the sandbox's imports.

    Raises
    ------
    RuntimeError
        If it cannot (for example where unprivileged user namespaces are
        disabled): every build would crash, which is a broken sandbox, not
        a bad candidate.
    """
    probe = _bubblewrap([sys.executable, "-I", "-c", "import factor_agent.gate.graph"])
    done = subprocess.run(probe, capture_output=True, text=True, timeout=60, env={"PATH": os.defpath})
    if done.returncode != 0:
        raise RuntimeError(f"Gate: bubblewrap is installed but cannot run the sandbox:\n{done.stderr[-_STDERR_TAIL:]}")
    return True


def _bubblewrap(command: list[str], modules=()) -> list[str]:
    """Wrap ``command`` in bubblewrap.

    The sandbox gets no network, a private /tmp, and read-only access to the
    system directories, the Python installation and the packages it imports
    (the vocabulary's ``modules`` among them), nothing else. A package
    imported from outside the Python installation (an editable checkout) is
    bound alone, so the rest of its checkout, credentials included, stays
    out of view.
    """
    prefixes = {Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
    visible = set(prefixes) | {Path(p) for p in ("/usr", "/lib", "/lib64", "/bin", "/etc/ld.so.cache")}
    for name in dict.fromkeys(["factor_agent", "quantlab", "KunQuant", *(m.split(".")[0] for m in modules)]):
        source = Path(importlib.import_module(name).__file__).resolve()
        package = source.parent if source.name == "__init__.py" else source
        if not any(package.is_relative_to(prefix) for prefix in prefixes):
            visible.add(package)
    binds = []
    for path in sorted(visible):
        binds += ["--ro-bind-try", str(path), str(path)]
    return [
        "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
        *binds, "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/tmp",
        *command,
    ]
