"""The sandbox subprocess: run candidate code once and send back its graph.

Reads ``{"code": ..., "memory_mb": ...}`` on standard input and writes one
JSON object on standard output: ``{"graph": [...]}`` or ``{"reasons": [...]}``.
The parent runs it with ``python -I``, an empty environment, a scratch
working directory, a timeout, a memory watchdog and, where bubblewrap is
installed, its own filesystem view; the code sees only the whitelisted
namespace. Where the platform allows it (Linux), the address space is also
capped here, so an allocation past the limit fails at once.
"""

import json
import resource
import sys

from KunQuant.Op import OpBase, Output
from KunQuant.Stage import Function
from quantlab.factor.kunquant_nan import missing_bars_only

from factor_agent.gate import MEMORY_REASON
from factor_agent.gate.graph import Graph
from factor_agent.gate.vocabulary import bar_mask_inputs, inputs, namespace


def build(code: str, memory_mb: int) -> dict:
    """Run ``code`` and return the message for the parent."""
    fields = inputs()
    scope = namespace(fields)
    try:
        with missing_bars_only(bar_mask_inputs(fields)):
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
        graph = Graph.record(alpha)
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
    _cap_address_space(request["memory_mb"])
    json.dump(build(request["code"], request["memory_mb"]), out)
    out.flush()


if __name__ == "__main__":
    main()
