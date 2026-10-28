"""The memory's versions, as the distiller wrote them in ``memory/round_NN.md``, each compared with the one before."""

import difflib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryLine:
    """One line of a memory version compared with the version before it.

    ``change`` is ``"kept"``, ``"added"`` (not in the version before) or
    ``"removed"`` (in the version before, not in this one).
    """

    change: str
    line: str


@dataclass(frozen=True)
class MemoryVersion:
    """One version of the memory the distiller wrote.

    ``round`` is the round after which it was written. ``changes`` holds the
    lines of this version and of the one before, in order, each marked kept,
    added or removed; it is ``None`` for the run's first version.
    """

    round: int
    text: str
    changes: tuple[MemoryLine, ...] | None


def memory(folder: Path) -> tuple[MemoryVersion, ...]:
    """The memory's versions, the earliest first, each but the first compared with the one before.

    ``round_NN.md`` is written before round ``NN``, so after round ``NN - 1``.
    """
    files = {}
    for path in folder.glob("round_*.md"):
        if match := re.fullmatch(r"round_(\d+)\.md", path.name):
            files[int(match[1]) - 1] = path.read_text().rstrip("\n")
    versions, before = [], None
    for number, text in sorted(files.items()):
        versions.append(MemoryVersion(round=number, text=text, changes=None if before is None else _changes(before, text)))
        before = text
    return tuple(versions)


def _changes(before: str, after: str) -> tuple[MemoryLine, ...]:
    """The lines of ``before`` and ``after`` in order, each marked kept, added or removed."""
    old, new = before.splitlines(), after.splitlines()
    lines = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            lines += [MemoryLine("kept", line) for line in new[j1:j2]]
            continue
        lines += [MemoryLine("removed", line) for line in old[i1:i2]]
        lines += [MemoryLine("added", line) for line in new[j1:j2]]
    return tuple(lines)
