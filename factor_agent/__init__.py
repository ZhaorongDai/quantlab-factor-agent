"""LLM-driven, mechanism-first formulaic factor research on quantlab."""

from factor_agent import _expression_fold, _max_drawdown

__version__ = "0.2.0"

_expression_fold.patch()
_max_drawdown.patch()
