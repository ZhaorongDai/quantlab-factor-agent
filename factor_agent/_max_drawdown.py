"""KunQuant's max drawdown, over a value computed from the fields.

KunQuant's ``WindowedMaxDrawdown`` loops over the values of its input without
putting them in a windowed buffer first: the ``decompose`` pass buffers the
loop's input, but the loop's ``IterValue`` still reads the input, and
compiling fails with "Bad buffer" on any input that is not a data column. The
price-volume fields vwap and amount are computed, so in the daily learning
test on the local model two candidates of ``WindowedMaxDrawdown(amount, ...)``
failed to evaluate. ``patch`` makes the operator loop over a buffer of its
input, as KunQuant's other windowed operators do; it is the fix proposed to
KunQuant (https://github.com/Menooker/KunQuant/pull/85), and a KunQuant that
has it is left as it is.
"""

from KunQuant.Op import Builder, ConstantOp, ForeachBackWindow, Input, IterValue, WindowedDataSourceOp, WindowedTempOutput, WindowLoopIndex
from KunQuant.ops import Mul, ReduceMin, Select, TsArgMax, WindowedMax, WindowedMaxDrawdown


def _decompose(self, options: dict) -> list:
    builder = Builder(self.get_parent())
    window = self.attrs["window"]
    v = self.inputs[0]
    with builder:
        peak = WindowedMax(v, window)
        max_bar_index = TsArgMax(v, window) - 1  # TsArgMax starts from 1
        inf = ConstantOp(float("inf"))
        buffered = WindowedTempOutput(v, window)
        with ForeachBackWindow(buffered, window) as each:
            index = WindowLoopIndex(each)
            filtered = Select(index >= max_bar_index, IterValue(each, buffered), inf)
        trough = ReduceMin(filtered)
        (peak - trough) / peak
    return builder.ops


def _loops_over_a_buffer() -> bool:
    """Whether KunQuant's max drawdown of a computed value reads the loop's values from a windowed buffer."""
    with Builder():
        x = Input("x")
        drawdown = WindowedMaxDrawdown(Mul(x, x), 2)
    return all(isinstance(op.inputs[1], WindowedDataSourceOp) for op in drawdown.decompose({}) if isinstance(op, IterValue))


def patch() -> None:
    """Make KunQuant's max drawdown loop over a buffer of its input, unless it already does; calling it again does nothing."""
    if not _loops_over_a_buffer():
        WindowedMaxDrawdown.decompose = _decompose
