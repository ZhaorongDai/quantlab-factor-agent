"""The report of ``Campaign.check``: what a run would refuse before its first LLM call.

A ``CheckReport`` holds one ``Check`` per part of a campaign checked on its
own, in the order they ran: each data component, the scoring, each role's
backend, the reference and known factors, the run snapshot, the vocabulary
and the templates, then with a ping each distinct backend's answer to one
short request. A check holds the problems it found (any one stops the
run), the warnings it gives (the run starts, but should be started
knowingly) or, when a part it needs failed, why it was skipped.
"""

import time
from dataclasses import dataclass

from factor_agent.llm.base import LLMBackend, Request

#: What a ping asks: an answer of one word, so that the request is short.
PING_PROMPT = "Reply with the single word OK."


@dataclass(frozen=True)
class Check:
    """One part of a campaign checked: its ``name``, its problems and warnings, or why it was ``skipped``."""

    name: str
    problems: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    #: Why the check did not run (a part it needs failed); None if it ran.
    skipped: str | None = None
    #: What a check that ran found besides its problems, such as how long a ping took.
    detail: str | None = None

    @property
    def status(self) -> str:
        """``"problem"``, ``"skipped"``, ``"warning"`` or ``"ok"``."""
        if self.problems:
            return "problem"
        if self.skipped is not None:
            return "skipped"
        return "warning" if self.warnings else "ok"


@dataclass(frozen=True)
class CheckReport:
    """Every ``Check`` of a campaign, in order; the run can start when ``problems`` is empty."""

    checks: tuple[Check, ...]

    @property
    def problems(self) -> list[str]:
        """Every problem, as ``"<check>: <problem>"``."""
        return [f"{c.name}: {problem}" for c in self.checks for problem in c.problems]

    @property
    def warnings(self) -> list[str]:
        """Every warning, as ``"<check>: <warning>"``."""
        return [f"{c.name}: {warning}" for c in self.checks for warning in c.warnings]

    @property
    def skipped(self) -> list[tuple[str, str]]:
        """``(check, reason)`` of every check that did not run."""
        return [(c.name, c.skipped) for c in self.checks if c.skipped is not None]

    @property
    def ok(self) -> bool:
        """Whether no check found a problem."""
        return not self.problems

    def text(self) -> str:
        """The report as a reader sees it: one line per check, its problems and warnings under it, and a summary."""
        lines = []
        for c in self.checks:
            said = c.skipped if c.skipped is not None else c.detail
            lines.append(f"{c.status:<8} {c.name}" + (f": {said}" if said is not None else ""))
            lines += [f"         - {problem}" for problem in c.problems]
            lines += [f"         ! {warning}" for warning in c.warnings]
        summary = (
            f"{len(self.problems)} problem(s), {len(self.warnings)} warning(s), {len(self.skipped)} check(s) skipped"
        )
        return "\n".join([*lines, summary if self.ok else summary + "; the run would be refused"])


def ping(name: str, backend: LLMBackend, role: str, model: str) -> Check:
    """``name``'s check of ``backend``: one short request as ``role`` to ``model``, its duration or its error.

    The request goes to the backend itself, outside any LLM log, and is not
    asked again.
    """
    started = time.monotonic()
    try:
        backend.complete(Request(role=role, model=model, prompt=PING_PROMPT))
    except Exception as exc:  # the error is the problem, never raised
        return Check(name, problems=(f"{type(exc).__name__}: {exc}",))
    return Check(name, detail=f"answered in {time.monotonic() - started:.2f} s")
