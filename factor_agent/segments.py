"""Evaluation segments: the date ranges the campaign and the acceptance read."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Segment:
    """A named, inclusive date range.

    A segment's last bars are purged by the reader of a forward label: a
    label at t reads prices up to t + lookahead, so only the dates whose
    whole label lies inside the segment are used (``Evaluator`` does this).
    """

    name: str
    start: str
    end: str

    def __post_init__(self):
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError(f"segment {self.name}: start {self.start} is after end {self.end}")
