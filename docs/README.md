# quantlab-factor-agent documentation

The site uses Sphinx, MyST Markdown, numpydoc and the PyData Sphinx Theme, as
[quantlab's](https://zhaorongdai.github.io/quantlab/) does. CI builds it on every push and
pull request (`.github/workflows/ci.yml`).

## Build and preview

```bash
uv sync --group docs
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open <http://localhost:8000>.

## Where to edit

- `index.md`: short introduction and task-based entry points.
- `getting-started/`: installation and a first campaign.
- `user-guide/`: explanations of the research loop, the config and its sections.
- `reference/*.rst`: curated API topics and the complete module index.
- Python docstrings: object descriptions, parameters, returns and examples.
- `_templates/autosummary/`: generated class and module page layouts.
- `design.md`: the design record (decisions Q1 onward, in Chinese). It is not part of the
  site; the code and the site describe what the decisions produced.

`reference/generated/` and `_build/` are generated and ignored by Git.

## Writing documentation

Follow quantlab's conventions: explain the operation from the user's side, keep decision
numbers and implementation history out of the user guide and the docstrings, use
NumPy-style docstrings (`Parameters`, `Returns`, `Raises`, `Notes`, `Examples`), and state
units, defaults and missing-value behaviour where they matter. Use the terms of
[`GLOSSARY.md`](../GLOSSARY.md).
