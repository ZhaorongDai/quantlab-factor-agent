"""Static checks on candidate code, before anything runs.

The syntax is an allow-list: assignments, arithmetic and comparisons, calls,
``def``/``return``, ``for``/``while``/``if`` and ``with`` blocks over plain
names and constants. Anything else, attribute access included, is refused.
"""

import ast

#: The syntax candidate code may use.
_ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.Tuple, ast.List, ast.Subscript, ast.Slice,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.keyword,
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
    ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
    ast.For, ast.While, ast.If, ast.Break, ast.Continue, ast.Pass, ast.With, ast.withitem,
)

#: Clearer reasons for the refusals a proposer is most likely to hit.
_REASONS = {
    ast.Import: "import is not allowed",
    ast.ImportFrom: "import is not allowed",
    ast.ClassDef: "class definitions are not allowed",
}


def static_reasons(code: str, names: frozenset[str]) -> list[str]:
    """Return why ``code`` is refused before it runs; empty when it may run.

    ``names`` are the names the code may read without binding them (the vocabulary's).
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"]
    reasons = _reasons(tree, names, _bound_names(tree))
    if "alpha" not in _bound_names(tree):
        reasons.append("the code never assigns alpha; it must end by assigning the factor to alpha")
    return list(dict.fromkeys(reasons))


def expression_reasons(expression: str, names: frozenset[str]) -> list[str]:
    """Return why ``expression`` (a derived field's formula) is refused: one expression over ``names`` only."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg}; a formula is one expression"]
    return list(dict.fromkeys(_reasons(tree, names, set())))


def _reasons(tree: ast.AST, names: frozenset[str], bound: set[str]) -> list[str]:
    reasons: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        where = f"line {line}: " if line else ""
        if type(node) in _REASONS:
            reasons.append(where + _REASONS[type(node)])
        elif isinstance(node, ast.Attribute):
            reasons.append(f"{where}attribute access (.{node.attr}) is not allowed")
        elif not isinstance(node, _ALLOWED_NODES + (ast.Expression,)):
            reasons.append(f"{where}{type(node).__name__} is not allowed in candidate code")
        elif (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in names
            and node.id not in bound
        ):
            reasons.append(f"{where}name {node.id} is not in the vocabulary (its fields, operators and builtins)")
    return reasons


def _bound_names(tree: ast.AST) -> set[str]:
    """Return every name the code binds: assignments, loop targets, functions and their arguments."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names
