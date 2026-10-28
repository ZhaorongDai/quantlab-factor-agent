"""The memory's versions, as the distiller wrote them in ``memory/round_NN.md``, each compared with the one before."""

import re
from dataclasses import dataclass
from pathlib import Path

from factor_agent.web._files import LineChange, line_changes


#: One line of a memory version compared with the version before it; see ``LineChange``.
MemoryLine = LineChange


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
        versions.append(MemoryVersion(round=number, text=text, changes=None if before is None else line_changes(before, text)))
        before = text
    return tuple(versions)

