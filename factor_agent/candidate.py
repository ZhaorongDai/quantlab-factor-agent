"""A factor candidate as the proposer writes it: a stated mechanism and the KunQuant code."""

import dataclasses
from collections.abc import Collection
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """One proposed factor.

    ``code`` is a Python body over the vocabulary's operators that ends by
    assigning the factor to ``alpha``. ``expected_sign`` is +1 when a higher
    value is expected to predict a higher forward return, -1 when lower.
    """

    hypothesis: str
    mechanism_type: str
    expected_sign: int
    horizon_rationale: str
    description: str
    code: str

    @classmethod
    def from_dict(cls, item: dict, mechanism_types: Collection[str]) -> "Candidate":
        """Build a candidate from the proposer's JSON object, whose type must be one of ``mechanism_types``.

        Raises
        ------
        ValueError
            If a field is missing or extra, ``mechanism_type`` is not one of
            ``mechanism_types``, ``expected_sign`` is not +1 or -1, or a text
            field is not a string.
        """
        if not isinstance(item, dict):
            raise ValueError(f"a candidate must be a JSON object, got {type(item).__name__}")
        names = [field.name for field in dataclasses.fields(cls)]
        missing = [name for name in names if name not in item]
        extra = sorted(set(item) - set(names))
        if missing or extra:
            raise ValueError(f"candidate fields: missing {missing}, unexpected {extra}")
        if item["mechanism_type"] not in mechanism_types:
            raise ValueError(
                f"mechanism_type {item['mechanism_type']!r} is not one of {list(mechanism_types)}"
            )
        if item["expected_sign"] not in (1, -1) or isinstance(item["expected_sign"], bool):
            raise ValueError(f"expected_sign must be 1 or -1, got {item['expected_sign']!r}")
        for name in names:
            if name != "expected_sign" and not isinstance(item[name], str):
                raise ValueError(f"{name} must be a string, got {type(item[name]).__name__}")
        return cls(**{name: item[name] for name in names})

    def to_dict(self) -> dict:
        """Return the candidate as the JSON object it was parsed from."""
        return dataclasses.asdict(self)
