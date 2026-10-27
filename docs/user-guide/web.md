# The web frontend

The frontend is a page on your own machine that lists the runs under a folder and opens
each of them, so that you can look at a run without reading its files. It only reads: it
never writes to a run directory, starts or stops a campaign, or runs Acceptance.

## Install

The frontend's web service (FastAPI and uvicorn) comes with the optional `web` extra; a
campaign-only install does without it:

```bash
uv sync --extra web
```

The page itself is built and kept in the package, so no Node is needed to serve it. Only
changing the page needs Node 18 or newer: edit `frontend/`, then run
`npm ci && npm run build` there and commit the rebuilt `factor_agent/web/static/`.

## Start it

Point `factor-agent-web` at the folder that holds your run directories (the `run_dir` of
your configs), at any depth, or at a single run directory:

```bash
uv run factor-agent-web runs/
```

It serves the page on <http://127.0.0.1:8000>; `--port` picks another port. It listens on
localhost only and has no option to listen on another interface: the pages show your
prompts, code and statistics, and there is no login. Stop it with Ctrl-C.

The run list shows every run directory under the folder, the latest active first,
including one whose first round has not been written yet:

| Column | Meaning |
|--------|---------|
| run | the run directory's path under the folder |
| code version | the commit of quantlab-factor-agent the run was last started or resumed with |
| rounds | the rounds written so far |
| trials, accepted | the trials of those rounds, and how many were accepted |
| state | "stopped" with the campaign's reason, or "not stopped" while it runs or after a crash |
| last activity | when the run last wrote a file |

A round counts once the campaign has written it at the round's end; the round in progress
is not counted yet. Click a run to open it: a run opens on its results.

## The results page

A line at the top says whether the run has stopped, with the campaign's reason, and how
many rounds, trials and accepted factors it has. Below it is one card per accepted factor,
with what decides how far to trust it (see {doc}`evaluation` for each statistic):

| On the card | Meaning |
|-------------|---------|
| round, mechanism type | the round that accepted it and its research direction's mechanism type |
| +1 or −1 | its factor direction: +1 when a higher value went with a higher training return, −1 when with a lower one |
| decile returns | each decile's mean return over the deciles' mean, from decile 1 (the lowest values) to decile 10 |
| partial-IC t | its partial IC's Newey-West t against the threshold it had to clear, and by how much it cleared it |
| monotonicity | how steadily its decile returns rise with the decile, in its direction (1 is a steady rise) |
| closest, closest known | the reference or accepted factor, and the known factor, most correlated with it, with that correlation |
| ρ limit | how far its largest correlation is below the correlation limit |
| coverage | the mean daily share of the members it has a value for |

Every IC and t is given in the factor's direction; hover one for the raw value the ledger
records. A factor accepted earlier in the run is named by its round and the first words of
its description; hover the name for the whole label.

The run directory does not record the admission settings, so the page takes them from the
refusals that quote them: a round's threshold is the one its refusals below the threshold
quote, carried over from the round before when the round has none (a round's candidates
are all judged at the same threshold), and the correlation limit is the one a refusal for
correlation quotes. Until a refusal quotes them, the page says they are unknown.

The cards sort by round, by partial-IC t, by distance from the correlation limit or by
coverage. Click a factor's description to open it written up: its hypothesis, description
and code, and its trust facts in sentences. A run that has accepted nothing says so.

### The diagnosis

Between the health line and the cards, the page lists at most three findings, each a
thing seen in the run's trials and what to do about it, the one concerning the most
trials first. While fewer than three factors are accepted the diagnosis stands out; a run
where no rule holds shows none. The rules and their thresholds:

| Finding | When | What to do |
|---------|------|------------|
| the LLM's replies could not be read | more than 20% of the trials met a reply the parser could not read as candidates | change the model, or raise its output limit |
| candidates keep colliding with one of your factors | more than 5 trials collided with the same reference or known factor, named with the count | steer the research directions away from it, or check your known factors |
| candidates collide with factors the run accepted | more than 5 trials collided with factors accepted earlier in the run, all counted together | nothing: the novelty checks are at work |
| candidates cover too few members | more than 15% of the trials were refused for coverage | check your data's fields and dates |
| candidate code fails to build | more than 15% of the trials met a graph build failure | check the operator file, or the model |
| nothing accepted, below the t threshold | no factor accepted, and most evaluated trials were refused below the partial-IC t threshold | nothing: this is the statistical bar, not an error |

Shares count trials, not the versions within them: a trial is counted once, whichever of
its versions met the problem. A collision is a trial refused for being too like an
existing factor (see {doc}`concepts`): by the Gate as a rewrite, the same graph with
other parameters, or a too large shared subtree, or by admission as a correlation above
the limit. A repeat of one of the run's own earlier candidates is not a collision.

## The memory page

Every `round.memory_every` rounds the distiller folds the trials since its last version
into the memory that every later prompt carries (see {doc}`campaign`). The memory page
lists every version it wrote in the run, named by the round after which it was written,
the latest first and open. From the second version on, a version shows its lines beside
those of the version before: added lines are highlighted, removed ones struck through,
and "Text alone" shows the version as the prompts carry it. A version the distiller left
as it was says it is unchanged. A run whose distiller has not written a version yet says
so; an empty or unreadable distiller reply writes no version, and the memory before it is
kept.

## On a server

When the campaign runs on a server, start the frontend there and forward its port to your
laptop over SSH:

```bash
# on the server
uv run factor-agent-web /path/to/runs

# on your laptop
ssh -N -L 8000:localhost:8000 user@server.example.com
```

Then open <http://localhost:8000> on your laptop. The page is reachable only through your
SSH connection; nothing is served to the server's network.
