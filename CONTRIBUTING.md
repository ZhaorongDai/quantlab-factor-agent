# Contributing to quantlab-factor-agent

Thank you for your interest in quantlab-factor-agent. Bug reports, questions, documentation
fixes and code changes are all welcome. This page explains how to set up a development
environment and what we look for in a pull request.

## Reporting a bug or asking a question

Open an issue at https://github.com/ZhaorongDai/quantlab-factor-agent/issues. For a bug,
include the config (with its data paths if you like, never with a key), the command or code
you ran, the full error message, your operating system and your Python version. The run
directory's `llm_log/` lets anyone replay a campaign without a provider (the `replay`
backend, see [LLM backends](docs/user-guide/llm-backends.md)); share it when it reproduces
the problem and holds nothing you would not publish. Never paste API keys or other
credentials into an issue.

To report a security problem, follow [SECURITY.md](SECURITY.md) instead of opening an issue.

## Setting up a development environment

quantlab-factor-agent tracks quantlab's `main` and installs it in editable mode from a
checkout beside its own:

```bash
git clone https://github.com/ZhaorongDai/quantlab.git
git clone https://github.com/ZhaorongDai/quantlab-factor-agent.git
cd quantlab-factor-agent
uv sync --extra web
uv run pytest -n auto
```

KunQuant compiles factor graphs to native code, so a C++ compiler must be on the `PATH`; on
macOS, xgboost also needs Homebrew's `libomp` (`brew install libomp`). The test suite runs
offline and needs no credentials: every LLM backend is faked or replayed and the market is
synthetic.

The web frontend's source is in `frontend/`, and its built page is committed under
`factor_agent/web/static/` so that an install needs no Node. If you change the frontend,
rebuild and commit the page with it:

```bash
cd frontend
npm ci
npm test
npm run build
```

CI fails when the committed page differs from what `frontend/` builds.

## Making a change

Create a branch for your change and keep each pull request focused on one topic. Before you
open the pull request, run the tests and make sure they pass. If you fix a bug, add a test
that fails without your fix. If you add a feature, add tests for it, update the relevant
page under `docs/` and add a line to [CHANGELOG.md](CHANGELOG.md) under *Unreleased*.

Keep the search free of anything market-specific in code: fields, operators, prompt
templates, mechanism types and scoring come from the config's files, so a new input family
is a new field file rather than a new code path. Read credentials only from environment
variables, never from code or configuration files; a config that holds a key is refused.
Nothing a prompt carries may come from outside the training segment.

## Docstrings and comments

Every module, class and function has a docstring in the
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format, as in quantlab.
Write for a reader who knows Python and basic finance but not this project, use the terms
[concepts](docs/user-guide/concepts.md) defines, and describe what the code does now rather
than how it got there. Comments should explain why the code does something, not repeat what
it does. [docs/README.md](docs/README.md) explains how to build and preview the
documentation site.

## Pull requests

`main` is protected: a change reaches it through a pull request whose checks pass (the
test suite on Linux and macOS, the frontend's page, the documentation build, the package
build and the changelog check), and it cannot be force-pushed or deleted. A pull request
that changes `factor_agent/` needs an entry in `CHANGELOG.md` under *Unreleased*, or the
`no-changelog` label when the change is not one a user would notice. A merged branch is
deleted.

## Releasing

A release is built and published by `.github/workflows/release.yml`, never from a working
tree. The package depends on `quantlab-py` from PyPI, so release quantlab first when this
package needs a quantlab change, and raise the bound in `pyproject.toml`; CI's package job
imports every module from the built wheel with its dependencies from PyPI and fails when
the quantlab there is too old. To release, set `version` in `pyproject.toml` and
`__version__` in `factor_agent/__init__.py`, set `version` and `date-released` in
`CITATION.cff`, move the *Unreleased* entries of the changelog
under a `## <version> (<date>)` heading, merge that, then tag `main` and push the tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow checks that the tag is on `main`, that it names the version, that the
changelog has a section for it and that the tagged commit passed CI; it then builds from
the tag's checkout, publishes to PyPI through trusted publishing (the `pypi` environment,
which only `v*` tags may deploy to) and creates the GitHub Release from the version's
changelog section. A version on PyPI cannot be replaced, so a manual run
(`workflow_dispatch`) builds and checks without publishing.

Every Monday, `latest-deps.yml` runs the suite on the newest dependencies PyPI offers,
quantlab-py among them, and opens an issue when it fails.
