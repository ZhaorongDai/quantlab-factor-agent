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
changing the page needs Node: edit `frontend/`, then run `npm ci && npm run build` there
with Node 22 and commit the rebuilt `factor_agent/web/static/`. CI builds the page with
Node 22 and fails when the build differs from the committed one, and another major version
of Node may build a different bundle. Its tests, `npm test`, need Node 22.6 or newer.

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
is not counted yet. Click a run to open it: a run opens on its results. A run whose files
cannot be read (a state file that is not JSON, say) is listed as unreadable, with the file
and the reason, and does not open.

## The health line

Every page of a run starts with its health line, read from the run's events file (see
{doc}`campaign`, "The events file"):

| It says | Meaning |
|---------|---------|
| Stopped | the run stopped, with the campaign's reason |
| Round N of M: planning the next round, asking the LLM, evaluating | what the running campaign is doing in its round in progress, against the round limit |
| Last LLM call back … ago | how long ago the latest call to the LLM came back |
| About … left | the time the rounds the round and trial limits still allow would take, at the pace of the last three rounds |
| Tokens | the tokens of every call so far, as the providers report them: input (of which read from the provider's cache) and output (of which spent reasoning); "Tokens by round" opens them round by round. No prices: spending limits belong to your provider |
| Crashed | the campaign's process has exited without recording a stop; running the campaign again resumes it |
| may be stuck | no LLM call was sent or came back for 15 minutes while asking the LLM, or the evaluation has lasted 30 minutes |

The campaign's process is looked for only on the machine the campaign runs on: when the
frontend runs on another machine, the process is reported as unknown, never as gone. A
run written by a campaign of an older version has no events file: its line says only which
LLM roles have answered how many times in the round in progress.

The page reads the run again by itself whenever the run writes, checking every 5 seconds,
so it can stay open while the campaign runs. The browser tab's title counts the factors
accepted since the page was opened and says when the run has stopped.

While the run has neither stopped nor crashed, **Pause live updates** beside the page names holds the page
still, to read or click a running round without anything moving. Paused, the page stands
at the moment of the pause: the round's timeline, the times since the last call and the
stall warnings do not move, and the run is not read again. A bar says since when the page
is paused and, as the page still checks every 5 seconds, whether the run has changed since.
**Resume** reads the run again at once.

## The results page

Below the health line, one sentence says whether the accepted factors improve your model, from
{doc}`acceptance`'s validation report on every accepted factor (or, when there is none, on
the latest selection): they improve it when the seed-averaged rank IC and top-k excess
return both rise with them, do not when neither does, and the sentence says which one rose
when only one did. Beside it are the validation figures with and without them (rank IC,
top-k excess and annualised return, long-short return, decile monotonicity), and, once the
hold-out has been opened and its report written, the same comparison on the hold-out with
every open the hold-out lock records. A hold-out opened again with force is marked in red:
it has been seen before and no longer tests the factors out of sample. A lock with no
hold-out report (an open that stopped before writing it) is listed as such. When the run
accepted more factors after the report, the page says how many the report judges. Before
Acceptance has run, the page says the factors are not judged against your model yet and
shows the call that judges them.

Below that is one card per accepted factor, with what decides how far to trust it (see
{doc}`evaluation` for each statistic):

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
| drop-one, add-one | when the validation report has a contribution table (`acceptance.contribution`): the rank IC the model loses without it, and gains from it alone on your model, with how many seeds agree on the sign |

Every IC and t is given in the factor's direction; hover one for the raw value the ledger
records. A factor accepted earlier in the run is named by its round and the first words of
its description; hover the name for the whole label.

A round's threshold is the one admission judged the round's candidates by (they are all
judged at the same one), and the correlation limit the one the campaign was last started
with, both from the events file. A run without an events file does not record them, so the
page takes them from the refusals that quote them: a round's threshold is the one its
refusals below the threshold quote, carried over from the round before when the round has
none, and the correlation limit is the one a refusal for correlation quotes. A card whose
threshold was quoted by its round's refusals, or carried over from an earlier round, says
so. Until a refusal quotes them, the page says they are unknown.

The cards rank by their drop-one contribution when there is one. A report written by an
older version, without the contribution table or the selection flag, is shown with what it
holds. The cards also sort by round, by partial-IC t, by distance from the correlation limit or by
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
| candidates keep colliding with one of your known factors | more than 5 trials collided with the same known factor, named with the count | steer the research directions away from it, or check your known factors |
| candidates collide with factors the run accepted | more than 5 trials collided with factors accepted earlier in the run, all counted together | nothing: the novelty checks are at work |
| candidates cover too few members | more than 15% of the trials were refused for coverage | check your data's fields and dates |
| candidate code fails to build | more than 15% of the trials met a graph build failure | check the operator file, or the model |
| nothing accepted, below the t threshold | no factor accepted, and most evaluated trials were refused below the partial-IC t threshold | nothing: this is the statistical bar, not an error |

Shares count trials, not the versions within them: a trial is counted once, whichever of
its versions met the problem. A collision is a trial refused for being too like an
existing factor (see {doc}`concepts`): by the Gate as a rewrite, the same graph with
other parameters, or a too large shared subtree, or by admission as a correlation above
the limit. A repeat of one of the run's own earlier candidates is not a collision, and
collisions with the reference set are no finding: the reference set is there to be
compared with.

## The selection view

The results page's *select factors for Acceptance* opens a table of the accepted factors,
one row each with a tick box: its round, factor direction, partial-IC t and how far over
its threshold, decile monotonicity, closest reference or accepted factor and closest known
factor with their correlations, coverage, and, when validation measured them, its drop-one
and add-one rank IC contributions with the seeds agreeing on their sign. A row opens to the
factor's hypothesis and code. Every accepted factor starts ticked.

Beside the table are the {doc}`acceptance` calls for the ticked factors, to copy into Python
on the machine that holds the run: validate the selection, open the hold-out on it, and
export it. Factors are named by their labels; each export name is the description's first
words, to replace with your own. The config path is a placeholder, since the run directory
does not record it. When the hold-out lock records an open, the view says the hold-out is no
longer untouched, so a selection judged on it now was chosen knowing an earlier result; the
call to open it again with `force=True` is written commented out, with that warning, so a
pasted block never reopens it. The page runs none of these calls.

## The trials page

A bar at the top counts the run's rounds, trials, evaluated trials (those the Evaluator
scored), accepted and failed trials.

Below it, the admission map places every evaluated trial by two facts admission judges
(see {doc}`evaluation`): across, its largest absolute correlation with an existing factor
(the nearest reference or accepted factor, or the nearest known one); up, its partial
IC's Newey-West t in the trial's direction. A dashed upright line marks the correlation
limit and a dashed level line the latest round's t threshold (see the results page for
where they come from); the shaded region between them is where admission accepts a trial, when its
coverage and decile monotonicity pass too. Accepted trials are green, those admission
refused red. Each round had its own threshold, which rises with the trials evaluated, so
an early accepted trial can sit below the final line. A trial with nothing to compare
with sits at a correlation of 0. While the threshold or the limit is unknown, its
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

### The round in progress

While the campaign runs, the round in progress sits between the bar and the map, as a
timeline of its LLM calls in one lane per role (planner, proposer, reviewer) and a lane
for the evaluation, from the end of the round before to now. A call still waiting for its
reply is drawn open; a call that failed is marked; calls asked side by side, as proposal
groups do, are drawn on rows of their own. After a crash the round is marked as cut short,
its timeline ends when the round was last heard of, and resuming the campaign drops it and
runs it again. Click a call to read it in the side
panel: its role, model, the time it came back and how long it took, its tokens (input,
output and of those spent reasoning), the reply (JSON laid out to read), the model's
reasoning when the provider returns it (open when the reply is empty) and the prompt
(folded). A flag says when the reply was cut off at the output limit, a sign to raise
the backend's output token limit, or when it was empty. The slots the call concerned are
listed; click one to read its pending trial.

"Pending trials" opens one card per slot of the round's plan (see
[Concepts](concepts.md)) with the stage it reached: planned, proposed, fixed after the
Gate, rewritten after review, found consistent or found inconsistent by the reviewer.
Click a card to read its research direction, its latest version, and the Gate's and the
reviewer's outcomes so far, with the Gate's reasons and hints for a refused version and
the reviewer's reason either way. A pending trial becomes a trial on the map or in the
list below when the round is written. When a resume drops a round a crash left half
written, its pending trials disappear; the round runs again under the same number.

A run written by a campaign of an older version has no events file: it shows no
timeline and no pending trials, and the health line counts the calls of the round in
progress by role.

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
