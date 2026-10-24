"""The sandbox subprocess: run candidate code once and send back its graph.

Reads ``{"code": ..., "memory_mb": ..., "vocabulary": ...}`` on standard input and writes one
JSON object on standard output: ``{"graph": [...]}`` or ``{"reasons": [...]}``.
The parent runs it with ``python -I``, an empty environment, a scratch
working directory, a timeout, a memory watchdog and, where bubblewrap is
installed, its own filesystem view; the code sees only the namespace of
the vocabulary, rebuilt here from the parent's ``Vocabulary.sources``.
Where the platform allows it (Linux), the address space is also capped
here, so an allocation past the limit fails at once.
"""

import json
import resource
import sys

from KunQuant.Op import OpBase, Output
from KunQuant.Stage import Function

from factor_agent.gate import MEMORY_REASON
from factor_agent.gate.graph import Graph
from factor_agent.gate.vocabulary import Vocabulary


def build(code: str, memory_mb: int, vocabulary: Vocabulary) -> dict:
    """Run ``code`` over ``vocabulary`` and return the message for the parent."""
    try:
        with vocabulary.namespace() as scope:
            exec(compile(code, "<candidate>", "exec"), scope)
    except MemoryError:
        return {"reasons": [MEMORY_REASON.format(memory_mb=memory_mb)]}
    except Exception as exc:  # the candidate's own error, reported back to it
        return {"reasons": [f"graph build failed: {type(exc).__name__}: {exc}"]}
    if "alpha" not in scope:
        return {"reasons": ["the code never assigns alpha"]}
    alpha = scope["alpha"]
    if isinstance(alpha, (tuple, list)):
        return {"reasons": [f"alpha holds {len(alpha)} values; a candidate has exactly one output"]}
    if not isinstance(alpha, OpBase):
        return {"reasons": [f"alpha must be an operator, got {type(alpha).__name__}"]}
    try:
        graph = Graph.record(alpha, vocabulary)
    except ValueError as exc:
        return {"reasons": [str(exc)]}
    try:
        output, ops = graph.build({})
        Function(ops + [Output(output, "alpha")])
    except Exception as exc:
        return {"reasons": [f"graph build failed: {type(exc).__name__}: {exc}"]}
    return {"graph": graph.to_json()}


def _cap_address_space(memory_mb: int) -> None:
    """Cap the address space at what is mapped now plus ``memory_mb``; a no-op where unsupported."""
    try:
        with open("/proc/self/statm") as statm:
            mapped = int(statm.read().split()[0]) * resource.getpagesize()
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (mapped + memory_mb * 2**20, hard))
    except (OSError, ValueError):
        pass  # macOS has no /proc and does not enforce RLIMIT_AS; the parent's watchdog applies


def main() -> None:
    """Read the request, build, write the reply."""
    request = json.load(sys.stdin)
    out = sys.stdout
    sys.stdout = sys.stderr  # nothing the code does can corrupt the reply
    vocabulary = Vocabulary.from_sources(request["vocabulary"])
    _cap_address_space(request["memory_mb"])
    json.dump(build(request["code"], request["memory_mb"], vocabulary), out)
    out.flush()


if __name__ == "__main__":
    main()
