"""The sandbox subprocess: run candidate code once and send back its graph.

Reads ``{"code": ...}`` on standard input and writes one JSON object on
standard output: ``{"graph": [...]}`` or ``{"reasons": [...]}``. The parent
runs it with ``python -I``, an empty environment, a scratch working
directory and a timeout; the code sees only the whitelisted namespace.
"""

import json
import sys

from KunQuant.Op import OpBase, Output
from KunQuant.Stage import Function

from factor_agent.gate.vocabulary import namespace
from factor_agent.gate.graph import Graph


def build(code: str) -> dict:
    """Run ``code`` and return the message for the parent."""
    scope = namespace()
    try:
        exec(compile(code, "<candidate>", "exec"), scope)
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


def main() -> None:
    """Read the request, build, write the reply."""
    request = json.load(sys.stdin)
    out = sys.stdout
    sys.stdout = sys.stderr  # nothing the code does can corrupt the reply
    json.dump(build(request["code"]), out)
    out.flush()


if __name__ == "__main__":
    main()
