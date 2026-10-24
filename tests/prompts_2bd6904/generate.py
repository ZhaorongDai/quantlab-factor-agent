"""Write the first-round planner and proposer prompts of commit 2bd6904 into a folder.

Run on a checkout of 2bd6904 (before spec #10), whose prompts had all their
wording in code::

    git worktree add --detach /tmp/old 2bd6904
    python -I -c "import sys; sys.path.insert(0, '/tmp/old'); sys.argv = ['generate', 'OUT']; \
exec(open('tests/prompts_2bd6904/generate.py').read())"

``tests/test_prompt_migration.py`` makes the same calls on the current code.
"""

import sys
from pathlib import Path

from factor_agent.researcher import Direction, Researcher


class Stop(Exception):
    pass


class Recorder:
    """A backend of every role that keeps the prompt and stops the call."""

    def __init__(self):
        self.prompts = {}

    def complete(self, request):
        self.prompts[request.role] = request.prompt
        raise Stop


def main(out: Path) -> None:
    recorder = Recorder()
    researcher = Researcher({role: (recorder, "m") for role in ("planner", "proposer", "reviewer", "distiller")})
    context = {"seed": "SEED", "families": "FAMILIES"}
    quota = {"behavioural_bias": 4, "risk_compensation": 4, "liquidity": 4, "microstructure": 4, "information_diffusion": 4}
    planned = [Direction("liquidity", "new", "", "DIRECTION A"), Direction("microstructure", "new", "", "DIRECTION B")]
    for ask in (lambda: researcher.plan(1, quota, context, []), lambda: researcher.propose(1, planned, context, None, None, 0)):
        try:
            ask()
        except Stop:
            pass
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.txt").write_text(recorder.prompts["planner"])
    (out / "propose.txt").write_text(recorder.prompts["proposer"])


main(Path(sys.argv[1]))
