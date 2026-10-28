"""The repository's price-volume and minute-bar field files and example operator file, and the Vocabularies they make."""

from functools import cache
from pathlib import Path

from factor_agent.gate.vocabulary import Vocabulary

CONFIGS = Path(__file__).parent.parent / "configs"
PRICE_VOLUME = CONFIGS / "fields" / "sharadar_price_volume.yaml"
EXAMPLE_OPERATORS = CONFIGS / "operators" / "kunquant_example.yaml"
#: The example intraday family: one-minute trade bars.
MINUTE_BARS = CONFIGS / "fields" / "massive_minute_bars.yaml"


@cache
def price_volume() -> Vocabulary:
    """The Vocabulary of the price-volume field file and the example operator file."""
    return Vocabulary.load([PRICE_VOLUME], EXAMPLE_OPERATORS)


@cache
def intraday() -> Vocabulary:
    """The Vocabulary of the price-volume and minute-bar field files and the example operator file."""
    return Vocabulary.load([PRICE_VOLUME, MINUTE_BARS], EXAMPLE_OPERATORS)
