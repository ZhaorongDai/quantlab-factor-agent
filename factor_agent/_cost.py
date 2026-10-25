"""The cost estimate of a campaign before it runs, from its config.

The estimate assumes the run goes the full number of rounds (``max_rounds``,
or fewer if ``max_trials`` comes first) and that every round makes a plan,
a proposal, ``retries`` rewrite requests, a review, and the rewrite of the
candidates the reviewer finds inconsistent and their second review (each
sized here as a whole round's candidates), with a distillation every
``memory_every`` rounds. Prompt sizes come from the real templates
with the data description, the seed directions and the reference families;
the planner's recent history holds a fixed number of characters per trial
of the last ``recent_rounds`` rounds, its earlier summary a fixed number per
mechanism type, and the distiller reads the memory and the trials since it.
Replies are sized per candidate, plus an allowance for thinking. It is a rough figure for deciding whether to start, not a
bound: the budget is what stops a run.
"""

from dataclasses import dataclass

from factor_agent.config import ROLES, RoundConfig
from factor_agent.prompts import Prompts
from factor_agent.researcher import _PLANNER_SECTIONS, _PROPOSER_SECTIONS, _render

#: Characters per token of English prose and code.
_CHARS_PER_TOKEN = 4
#: Characters a trial adds to the history the planner and the distiller read.
_HISTORY_CHARS = 160
#: Characters of a mechanism type's line in the earlier summary, and of a memory.
_SUMMARY_CHARS = 160
_MEMORY_CHARS = 2400
#: Characters a candidate adds to the last round's feedback.
_FEEDBACK_CHARS = 600
#: Characters of one direction in a proposal request, and of one candidate in a review or rewrite request.
_DIRECTION_CHARS = 300
_CANDIDATE_CHARS = 900
#: Reply tokens per candidate of each role, and per distillation; thinking tokens per call on top.
_REPLY_TOKENS = {"planner": 60, "proposer": 350, "reviewer": 50}
_MEMORY_TOKENS = 600
_THINKING_TOKENS = 2000


@dataclass(frozen=True)
class CostEstimate:
    """Calls per role and their estimated cost in US dollars (None for a role whose backend cannot price them)."""

    calls: dict[str, int]
    usd: dict[str, float | None]

    @property
    def total(self) -> float | None:
        """The estimated cost of every call, or None if a role's is unknown."""
        values = list(self.usd.values())
        return None if None in values else sum(values)

    def summary(self, budget: float | None, spent: float = 0.0) -> str:
        """One line for the user: the calls, the cost per role and the total, against the budget.

        ``spent`` is what a resumed run's earlier rounds cost; the estimate is for the whole run.
        """
        roles = ", ".join(
            f"{role} {self.calls[role]} calls {'unknown' if self.usd[role] is None else f'${self.usd[role]:.2f}'}"
            for role in ROLES
        )
        total = "unknown" if self.total is None else f"${self.total:.2f}"
        line = f"Estimated LLM cost: {total} for {sum(self.calls.values())} calls ({roles})"
        if budget is None:
            return line + "; no budget"
        line += f"; budget ${budget:.2f}" + (f", ${spent:.2f} spent so far" if spent else "")
        if self.total is not None and self.total > budget:
            line += "; the estimate is above the budget, so the run will stop early"
        return line


def estimate(
    settings: RoundConfig, backends: dict, context: dict[str, str], prompts: Prompts, recent_rounds: int = 5,
) -> CostEstimate:
    """Estimate a campaign's calls and cost.

    ``backends`` maps each role to its backend; a backend prices a call
    with ``estimate(input_tokens, output_tokens)``, and one without that
    method makes its role's cost unknown. ``context`` holds the first
    round's sections (the seed directions and the reference families);
    ``prompts`` are the campaign's templates and what fills them.
    """
    rounds = min(settings.max_rounds, settings.max_trials // settings.candidates)
    n = settings.candidates
    planner_base = len(prompts.render("plan", round=1, context=_render(context, _PLANNER_SECTIONS), quota="{}"))
    proposer_base = len(prompts.render(
        "propose", round=1, context=_render(context, _PROPOSER_SECTIONS), n=n, directions="",
    ))
    retry_base = len(prompts.render("retry", round=1, failures="", n=n))
    review_base = len(prompts.render("review", round=1, candidates=""))
    distill_base = len(prompts.render("distill", round=1, memory="", history=""))
    summary = len(prompts.mechanism_types) * _SUMMARY_CHARS

    calls: dict[str, list[tuple[int, int]]] = {role: [] for role in ROLES}  # (input chars, reply tokens)
    distilled = 0
    for number in range(1, rounds + 1):
        feedback = n * _FEEDBACK_CHARS if number > 1 else 0
        if number - 1 - distilled >= settings.memory_every:
            memory = _MEMORY_CHARS if distilled else 0
            calls["distiller"].append((distill_base + memory + (number - 1 - distilled) * n * _HISTORY_CHARS, _MEMORY_TOKENS))
            distilled = number - 1
        recent = min(number - 1, recent_rounds) * n * _HISTORY_CHARS + (summary if number - 1 > recent_rounds else 0)
        memory = _MEMORY_CHARS if distilled else 0
        calls["planner"].append((planner_base + memory + recent + feedback, n * _REPLY_TOKENS["planner"]))
        calls["proposer"].append((proposer_base + n * _DIRECTION_CHARS + feedback, n * _REPLY_TOKENS["proposer"]))
        calls["proposer"] += [(retry_base + n * _CANDIDATE_CHARS, n * _REPLY_TOKENS["proposer"])] * settings.retries
        calls["proposer"].append((retry_base + n * _CANDIDATE_CHARS, n * _REPLY_TOKENS["proposer"]))
        calls["reviewer"] += [(review_base + n * _CANDIDATE_CHARS, n * _REPLY_TOKENS["reviewer"])] * 2

    usd = {}
    for role in ROLES:
        price = getattr(backends[role], "estimate", None)
        costs = [
            price(chars // _CHARS_PER_TOKEN, tokens + _THINKING_TOKENS) if price else None
            for chars, tokens in calls[role]
        ]
        usd[role] = None if None in costs else sum(costs)
    return CostEstimate(calls={role: len(calls[role]) for role in ROLES}, usd=usd)
