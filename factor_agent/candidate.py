"""A factor candidate as the proposer writes it: a stated mechanism and the KunQuant code."""

import dataclasses
from collections.abc import Collection
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """One proposed factor.

    ``code`` is a Python body over the vocabulary's operators that ends by
    assigning the factor to ``alpha``. A candidate states no direction:
    which way it predicts is read from its training IC.
    """

    hypothesis: str
    mechanism_type: str
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
            ``mechanism_types``, or a field is not a string.
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
        for name in names:
            if not isinstance(item[name], str):
                raise ValueError(f"{name} must be a string, got {type(item[name]).__name__}")
        return cls(**{name: item[name] for name in names})

    def to_dict(self) -> dict:
        """Return the candidate as the JSON object it was parsed from."""
        return dataclasses.asdict(self)
