"""The frontend's web service: JSON routes over ``run_view`` and the built page, on localhost only.

``factor-agent-web <folder>`` serves the run directories under a folder on
``http://127.0.0.1:<port>``. It binds the loopback interface and has no
option to bind another: the pages show prompts, code and statistics and
there is no login. On a server, forward the port over SSH.

Routes:

- ``GET /api/runs``: every run under the folder (``list_runs``), the latest written first;
- ``GET /api/run?name=<name>``: one run (``RunView.read``) by its name in the list, 404 for any other name;
- ``GET /``: the page, built from ``frontend/`` into ``factor_agent/web/static``.

FastAPI and uvicorn come with the ``web`` extra and are imported only when
the app is made, so the module imports without them.
"""

import argparse
from pathlib import Path

from factor_agent.web.run_view import RunView, list_runs

#: The built page: ``npm run build`` in ``frontend/`` writes it, and it is committed.
STATIC = Path(__file__).parent / "static"

#: The interface the service binds: the loopback interface only.
HOST = "127.0.0.1"


def create_app(root: str | Path):
    """The web service over the run directories under ``root``.

    Parameters
    ----------
    root : str or Path
        A folder of run directories, at any depth, or a run directory itself.

    Returns
    -------
    fastapi.FastAPI

    Raises
    ------
    ImportError
        When the ``web`` extra is not installed.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles

    root = Path(root).resolve()
    app = FastAPI(title="factor-agent", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/api/runs")
    def runs():
        return list_runs(root)

    @app.get("/api/run")
    def run(name: str):
        # Only a listed name is read: a name never reaches the file system unchecked.
        if name not in {summary.name for summary in list_runs(root)}:
            raise HTTPException(404, f"no run named {name!r} under {root}")
        return RunView.read(root if (root / "snapshot.json").is_file() else root / name, name=name)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="page")
    return app


def main(argv: list[str] | None = None) -> None:
    """Serve the run directories under a folder on localhost: the ``factor-agent-web`` console script."""
    parser = argparse.ArgumentParser(
        prog="factor-agent-web",
        description="Serve the run directories under a folder on http://127.0.0.1:<port> (localhost only).",
    )
    parser.add_argument("root", type=Path, help="a folder of run directories, or a run directory")
    parser.add_argument("--port", type=int, default=8000, help="the port to listen on (default: 8000)")
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError:
        parser.exit(1, "factor-agent-web: the frontend needs the web extra: uv sync --extra web\n")
    if not args.root.is_dir():
        parser.exit(1, f"factor-agent-web: {args.root} is not a folder\n")
    print(f"factor-agent-web: serving the runs under {args.root.resolve()} on http://{HOST}:{args.port}", flush=True)
    uvicorn.run(create_app(args.root), host=HOST, port=args.port)
