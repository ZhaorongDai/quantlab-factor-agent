"""The web frontend: a read-only view of the run directories under a folder, served on localhost.

``run_view`` reads run directories (``list_runs``, ``RunView.read``) and needs
nothing beyond the campaign's own dependencies; ``app`` serves what it reads
as JSON routes and the page, and needs the ``web`` extra (FastAPI and uvicorn).
The frontend never writes to a run directory.
"""
