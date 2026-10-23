"""The Campaign: the research loop, from one config to accepted and rejected candidates.

One round: the proposer writes candidates, the Gate checks each, the
Evaluator scores the survivors on the training segment in one compile,
admission decides, and every candidate goes to the ledger.
"""

from dataclasses import dataclass
from pathlib import Path

from factor_agent.candidate import Candidate
from factor_agent.config import CampaignConfig
from factor_agent.evaluator import CompileFailure, Evaluator, Metrics
from factor_agent.gate import Gate, Passed
from factor_agent.ledger import Ledger, TrialRecord
from factor_agent.llm.base import make_backend
from factor_agent.researcher import Researcher, Unparsed


@dataclass(frozen=True)
class CampaignResult:
    """The trial records of the run, split by outcome, and where the ledger is."""

    accepted: list[TrialRecord]
    rejected: list[TrialRecord]
    ledger_path: Path


class Campaign:
    """Run a mining campaign from its config.

    Parameters
    ----------
    config : CampaignConfig
        Every setting of the run.
    """

    def __init__(self, config: CampaignConfig):
        self.config = config

    def run(self) -> CampaignResult:
        """Run the campaign and return every candidate's outcome."""
        config = self.config
        ledger = Ledger(Path(config.run_dir) / "ledger.jsonl")
        researcher = Researcher(make_backend(config.proposer.spec()), config.proposer.model)
        gate = Gate(config.gate.timeout_seconds)
        evaluator = Evaluator.for_segment(
            config.training,
            prices=config.data.load("prices"),
            label=config.data.load("label"),
            warmup_bars=config.evaluator.warmup_bars,
            njobs=config.evaluator.njobs,
        )
        records = self._round(1, researcher, gate, evaluator)
        ledger.append(records)
        return CampaignResult(
            accepted=[r for r in records if r.status == "accepted"],
            rejected=[r for r in records if r.status == "rejected"],
            ledger_path=ledger.path,
        )

    def _round(self, number: int, researcher: Researcher, gate: Gate, evaluator: Evaluator) -> list[TrialRecord]:
        """Propose, gate, evaluate and admit one round; return its records in proposal order."""
        proposals = researcher.propose(self.config.round.candidates)
        records: list[TrialRecord | None] = [None] * len(proposals)
        passed: list[tuple[int, Candidate, Passed]] = []
        for i, proposal in enumerate(proposals):
            if isinstance(proposal, Unparsed):
                item = proposal.item if isinstance(proposal.item, dict) else {"raw": proposal.item}
                records[i] = TrialRecord(
                    round=number, candidate=item, status="rejected", stage="parse", reasons=[proposal.reason]
                )
                continue
            checked = gate.check(proposal)
            if isinstance(checked, Passed):
                passed.append((i, proposal, checked))
            else:
                records[i] = TrialRecord(
                    round=number, candidate=proposal.to_dict(), status="rejected", stage="gate",
                    reasons=list(checked.reasons),
                )
        results = evaluator.evaluate([checked.graph for _, _, checked in passed])
        for (i, candidate, checked), result in zip(passed, results):
            records[i] = self._admit(number, candidate, checked, result)
        return records

    def _admit(self, number: int, candidate: Candidate, checked: Passed, result: Metrics | CompileFailure) -> TrialRecord:
        """Decide one evaluated candidate: its NW t must reach the threshold."""
        if isinstance(result, CompileFailure):
            return TrialRecord(
                round=number, candidate=candidate.to_dict(), status="rejected", stage="evaluate",
                reasons=[result.error], fingerprint=checked.fingerprint,
            )
        threshold = self.config.admission.t_threshold
        accepted = result.ic_t >= threshold
        reasons = [] if accepted else [f"Newey-West t {result.ic_t:.2f} is below the threshold {threshold:g}"]
        return TrialRecord(
            round=number, candidate=candidate.to_dict(), status="accepted" if accepted else "rejected",
            stage="admission", reasons=reasons, fingerprint=checked.fingerprint, metrics=result.to_dict(),
        )
