"""The Campaign: the research loop, from one config to accepted and rejected candidates.

One round: the proposer writes candidates, the Gate checks each (against
the zoo of reference factors and the factors accepted in earlier rounds) and
the refused ones go back to the proposer to be fixed, the Evaluator scores
the survivors on the training segment in one compile, admission decides
(``_admission``: sign, correlation, partial IC against a threshold that
rises with the number of evaluated trials, decile monotonicity), every
candidate goes to the ledger and the accepted ones join the zoo and the
Evaluator's reference pool, before the next candidate of the round is
admitted.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from factor_agent import _admission
from factor_agent.config import CampaignConfig, load
from factor_agent.evaluator import CompileFailure, Evaluator, Metrics
from factor_agent.gate import Gate
from factor_agent.ledger import Ledger, TrialRecord
from factor_agent.llm.base import make_backend
from factor_agent.researcher import Proposal, Researcher
from factor_agent.zoo import Zoo


@dataclass(frozen=True)
class CampaignResult:
    """The trial records of the run, split by outcome, and where the ledger is.

    ``evaluated_trials`` counts the candidates that were evaluated, accepted
    or rejected; failed trials are not among them. ``not_covered`` names the
    reference factors without a KunQuant graph, which the Gate's originality
    check cannot compare a candidate with.
    """

    accepted: list[TrialRecord]
    rejected: list[TrialRecord]
    failed: list[TrialRecord]
    ledger_path: Path
    not_covered: tuple[str, ...] = ()

    @property
    def evaluated_trials(self) -> int:
        """The number of evaluated candidates."""
        return len(self.accepted) + len(self.rejected)


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
        gate = Gate(config.gate)
        reference = [load(component) for component in config.reference]
        evaluator = Evaluator.for_segment(
            config.training,
            prices=config.data.load("prices"),
            label=config.data.load("label"),
            membership=config.data.load("membership"),
            exposures=config.data.load("exposures"),
            reference=reference,
            warmup_bars=config.evaluator.warmup_bars,
            njobs=config.evaluator.njobs,
        )
        zoo = Zoo.from_reference(reference)
        _write_zoo(Path(config.run_dir) / "zoo.json", zoo)
        records = []
        for number in range(1, config.round.max_rounds + 1):
            evaluated = sum(r.status != "failed" for r in records)
            round_records = self._round(number, researcher, gate, zoo, evaluator, evaluated)
            ledger.append(round_records)
            records += round_records
        return CampaignResult(
            accepted=[r for r in records if r.status == "accepted"],
            rejected=[r for r in records if r.status == "rejected"],
            failed=[r for r in records if r.status == "failed"],
            ledger_path=ledger.path,
            not_covered=zoo.not_covered,
        )

    def _round(
        self, number: int, researcher: Researcher, gate: Gate, zoo: Zoo, evaluator: Evaluator, evaluated: int,
    ) -> list[TrialRecord]:
        """Propose, gate, evaluate and admit one round; return its records in proposal order.

        ``evaluated`` counts the trials evaluated in earlier rounds. A round's
        candidates are tested together, so each is judged at the count after
        the round: ``evaluated`` plus every candidate of the round that ran,
        whatever its place in the proposal order. They are admitted in that
        order, and the accepted ones are added to ``zoo``.
        """
        proposals = researcher.propose(self.config.round.candidates, gate, zoo, self.config.round.retries)
        passed = [i for i, p in enumerate(proposals) if p.checked is not None]
        records = [self._failed(number, p) if p.checked is None else None for p in proposals]

        def admit(position: int, metrics: Metrics, ran: int) -> str | None:
            proposal = proposals[passed[position]]
            record = self._admit(number, proposal, metrics, evaluated + ran)
            records[passed[position]] = record
            if record.status != "accepted":
                return None
            return zoo.add(proposal.candidate.description, f"accepted in round {number}", proposal.checked.graph).label

        results = evaluator.evaluate([proposals[i].checked.graph for i in passed], admit)
        for i, result in zip(passed, results):
            if isinstance(result, CompileFailure):
                records[i] = TrialRecord(
                    status="failed", stage="evaluate", reasons=[result.error], **self._common(number, proposals[i]),
                )
        return records

    @staticmethod
    def _failed(number: int, proposal: Proposal) -> TrialRecord:
        """The record of a candidate the parser or the Gate still refused after its retries."""
        last = proposal.attempts[-1]
        item = proposal.item
        return TrialRecord(
            round=number, candidate=item if isinstance(item, dict) else {"raw": item}, status="failed",
            stage=last.stage, reasons=list(last.reasons), attempts=_attempts(proposal),
        )

    def _admit(self, number: int, proposal: Proposal, metrics: Metrics, trials: int) -> TrialRecord:
        """Decide one evaluated candidate, the ``trials``-th evaluated trial of the campaign."""
        reasons = _admission.reasons(proposal.candidate, metrics, trials, self.config.admission)
        return TrialRecord(
            status="rejected" if reasons else "accepted", stage="admission", reasons=reasons,
            metrics=metrics.to_dict(), **self._common(number, proposal),
        )

    @staticmethod
    def _common(number: int, proposal: Proposal) -> dict:
        """The record fields of a candidate the Gate passed."""
        checked = proposal.checked
        return dict(
            round=number, candidate=proposal.candidate.to_dict(), fingerprint=checked.fingerprint,
            stats=checked.stats, attempts=_attempts(proposal),
        )


def _write_zoo(path: Path, zoo: Zoo) -> None:
    """Write which reference factors the originality check covers and which it cannot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"covered": list(zoo.covered), "not_covered": list(zoo.not_covered)}, indent=2) + "\n")


def _attempts(proposal: Proposal) -> list[dict]:
    """The refused versions of a proposal as the ledger stores them."""
    return [{**asdict(a), "reasons": list(a.reasons)} for a in proposal.attempts]
