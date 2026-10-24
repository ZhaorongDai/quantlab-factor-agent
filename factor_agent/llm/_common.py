"""What the provider adapters share: the key from its environment variable and the price of a reply."""

import os
from dataclasses import dataclass


def key_from_environment(name: str) -> str:
    """The value of the environment variable ``name``.

    Raises
    ------
    ValueError
        If the variable is unset or empty.
    """
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"the environment variable {name} holding the API key is not set")
    return value


@dataclass(frozen=True)
class Prices:
    """US dollars per million input and output tokens, from a role's config."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float

    @classmethod
    def from_spec(cls, spec: dict, backend: str) -> "Prices":
        """Read ``input_usd_per_mtok`` and ``output_usd_per_mtok``.

        Raises
        ------
        ValueError
            If either is missing.
        """
        missing = [k for k in ("input_usd_per_mtok", "output_usd_per_mtok") if k not in spec]
        if missing:
            raise ValueError(f"the {backend} backend needs {missing} in its config to count the cost")
        return cls(float(spec["input_usd_per_mtok"]), float(spec["output_usd_per_mtok"]))

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """The cost of a call with these token counts."""
        return (input_tokens * self.input_usd_per_mtok + output_tokens * self.output_usd_per_mtok) / 1e6
