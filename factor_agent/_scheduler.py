"""The round's quota of candidates per mechanism type (after RD-Agent(Q)'s bandit).

Every type gets ``min_per_type`` slots. Each of the other slots goes to the
type with the largest draw from the Beta posterior of its admission rate:
``Beta(1 + accepted, 1 + trials - accepted)`` over every earlier trial
written for a direction of that type, failed ones included, so a type whose
candidates keep being accepted draws the extra slots. The types are the
config's, in its order; a record of another type counts toward none. Round n draws from
the generator seeded ``(seed, n)``: a resumed campaign plans the round as
the uninterrupted one would.
"""

import numpy as np

from factor_agent.config import RoundConfig
from factor_agent.ledger import TrialRecord


def quota(number: int, records: list[TrialRecord], config: RoundConfig, types: tuple[str, ...]) -> dict[str, int]:
    """The slots of each of ``types`` in round ``number``, given the records of the earlier rounds."""
    trials = dict.fromkeys(types, 0)
    accepted = dict.fromkeys(types, 0)
    for record in records:
        kind = record.direction["mechanism_type"]
        if kind not in trials:
            continue
        trials[kind] += 1
        accepted[kind] += record.status == "accepted"
    slots = dict.fromkeys(types, config.min_per_type)
    rng = np.random.default_rng([config.seed, number])
    alpha = np.array([1 + accepted[kind] for kind in types])
    beta = np.array([1 + trials[kind] - accepted[kind] for kind in types])
    for _ in range(config.candidates - config.min_per_type * len(types)):
        slots[types[int(np.argmax(rng.beta(alpha, beta)))]] += 1
    return slots
