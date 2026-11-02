"""KunQuant's expression folding, with operators compared as graphs.

KunQuant's ``expr_fold`` pass merges equal operators. Two whose hashes match
and whose inputs are not the same objects it tells apart by walking both as
trees (``OpBase.items``), which visits a value once for every path to it: a
graph that reuses its values, as ``x / Scale(x)`` reuses ``x``, nested a few
times under an EW beta, takes longer to compare than any round can wait (an
evaluation of arm A2's round 6 ran for hours). ``patch`` makes the comparison
the same walk over the graph, remembering each pair of operators it has
compared, so each pair is compared once.
"""

from KunQuant.passes import ExprFold


def _same(x, x_args: dict, y, y_args: dict, compared: dict) -> bool:
    """Whether ``x`` and ``y`` would yield the same ``OpBase.items``, given the keyword arguments of each."""
    if x is y and x_args.keys() == y_args.keys() and all(x_args[k] is y_args[k] for k in x_args):
        return True
    key = (id(x), id(y), *map(id, x_args.values()), *map(id, y_args.values()))
    if key not in compared:
        compared[key] = False  # an operator does not depend on itself
        xs, ys = x.get_args(True, **x_args), y.get_args(True, **y_args)
        compared[key] = (
            x.__class__ is y.__class__ and x.attrs == y.attrs and len(xs) == len(ys)
            and all(_same(a, a_args, b, b_args, compared) for (a, a_args), (b, b_args) in zip(xs, ys))
        )
    return compared[key]


def _equal(self, other) -> bool:
    if self.parent != other.parent or self.op.__class__ != other.op.__class__ or self.op.attrs != other.op.attrs:
        return False
    return self.op.inputs == other.op.inputs or _same(self.op, {}, other.op, {}, {})


def patch() -> None:
    """Make KunQuant's expression folding compare operators as graphs; calling it again does nothing."""
    ExprFold._OpProxy.__eq__ = _equal
