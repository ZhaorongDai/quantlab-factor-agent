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

## The trials page

A bar at the top counts the run's rounds, trials, evaluated trials (those the Evaluator
scored), accepted and failed trials, and says whether the run has stopped.

Below it, the admission map places every evaluated trial by two facts admission judges
(see {doc}`evaluation`): across, its largest absolute correlation with an existing factor
(the nearest reference or accepted factor, or the nearest known one); up, its partial
IC's Newey-West t in the trial's direction. A dashed upright line marks the correlation
limit and a dashed level line the latest round's t threshold, both as the run's refusals
quote them; the shaded region between them is where admission accepts a trial, when its
coverage and decile monotonicity pass too. Accepted trials are green, those admission
refused red. Each round had its own threshold, which rises with the trials evaluated, so
an early accepted trial can sit below the final line. A trial with nothing to compare
with sits at a correlation of 0. Until a refusal quotes the threshold or the limit, its
line is not drawn.

Under the map, the trials that never reached the Evaluator are listed by the stage that
refused them: the parser (the reply could not be read as candidates), the Gate, or the
reviewer, which still found the rewrite inconsistent; code the Evaluator could not build
comes last. Trials refused for the same reason, apart from the numbers, reply or code the
reason quotes, are one row with their count and rounds, so a round lost to one problem is
one line; click the row to list its trials.

Click a point or a trial to read it in the side panel: its round, mechanism type and
research direction, the hypothesis, description and code of its final version, the
reasons it was refused, the versions refused before it with their reasons and hints, and,
for an evaluated trial, its statistics in its direction (hover one for the raw value) with
its round's threshold, nearest factors and coverage. Trials that arrived since the page
was opened stand out on the map for fifteen seconds.

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
