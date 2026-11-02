"""KunQuant's expression folding, with operators compared as graphs.

KunQuant's ``expr_fold`` pass merges equal operators. Two whose hashes match
and whose inputs are not the same objects it tells apart by walking both as
trees (``OpBase.items``), which visits a value once for every path to it: a
graph that reuses its values, as ``x / Scale(x)`` reuses ``x``, nested a few
times under an EW beta, takes longer to compare than any round can wait (an
evaluation of arm A2's round 6 ran for hours). ``patch`` makes the comparison
the same walk over the graph, side by side and with a stack, visiting each pair
of operators once; it is the fix proposed to KunQuant
(https://github.com/Menooker/KunQuant/pull/83), and a KunQuant that has it is
left as it is.
"""

from KunQuant.passes import ExprFold


def _same(x, y) -> bool:
    """Whether ``x.items()`` and ``y.items()`` are equal."""
    stack = [(x, {}, y, {})]
    visited = set()
    while stack:
        a, a_args, b, b_args = stack.pop()
        if a is b and a_args.keys() == b_args.keys() and all(a_args[k] is b_args[k] for k in a_args):
            continue
        key = (id(a), id(b), *((k, id(v)) for k, v in a_args.items()), *((k, id(v)) for k, v in b_args.items()))
        if key in visited:
            continue
        visited.add(key)
        if a.__class__ is not b.__class__ or a.attrs != b.attrs:
            return False
        a_inputs, b_inputs = a.get_args(True, **a_args), b.get_args(True, **b_args)
        if len(a_inputs) != len(b_inputs):
            return False
        stack += [(c, c_args, d, d_args) for (c, c_args), (d, d_args) in zip(a_inputs, b_inputs)]
    return True


def _equal(self, other) -> bool:
    if self.parent != other.parent or self.op.__class__ != other.op.__class__ or self.op.attrs != other.op.attrs:
        return False
    return self.op.inputs == other.op.inputs or _same(self.op, other.op)


def patch() -> None:
    """Make KunQuant's expression folding compare operators as graphs, unless it already does; calling it again does nothing."""
    if not hasattr(ExprFold, "_is_same"):
        ExprFold._OpProxy.__eq__ = _equal
