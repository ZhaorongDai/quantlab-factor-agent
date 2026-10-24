"""The round's quota of candidates per mechanism type (after RD-Agent(Q)'s bandit).

Every type gets ``min_per_type`` slots. Each of the other slots goes to the
type with the largest draw from the Beta posterior of its admission rate:
``Beta(1 + accepted, 1 + trials - accepted)`` over every earlier trial
written for a direction of that type, failed ones included, so a type whose
candidates keep being accepted draws the extra slots. Round n draws from
the generator seeded ``(seed, n)``: a resumed campaign plans the round as
the uninterrupted one would.
"""

import numpy as np

from factor_agent.candidate import MECHANISM_TYPES
from factor_agent.config import RoundConfig
from factor_agent.ledger import TrialRecord


def quota(number: int, records: list[TrialRecord], config: RoundConfig) -> dict[str, int]:
    """The slots of each mechanism type in round ``number``, given the records of the earlier rounds."""
    trials = dict.fromkeys(MECHANISM_TYPES, 0)
    accepted = dict.fromkeys(MECHANISM_TYPES, 0)
    for record in records:
        kind = record.direction["mechanism_type"]
        trials[kind] += 1
        accepted[kind] += record.status == "accepted"
    slots = dict.fromkeys(MECHANISM_TYPES, config.min_per_type)
    rng = np.random.default_rng([config.seed, number])
    alpha = np.array([1 + accepted[kind] for kind in MECHANISM_TYPES])
    beta = np.array([1 + trials[kind] - accepted[kind] for kind in MECHANISM_TYPES])
    for _ in range(config.candidates - config.min_per_type * len(MECHANISM_TYPES)):
        slots[MECHANISM_TYPES[int(np.argmax(rng.beta(alpha, beta)))]] += 1
    return slots
