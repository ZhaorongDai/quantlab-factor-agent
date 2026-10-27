Web frontend
============

Read run directories as the frontend shows them, and serve them on localhost.
See :doc:`../user-guide/web` to start the frontend.

Reading runs
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   factor_agent.web.run_view.list_runs
   factor_agent.web.run_view.RunView
   factor_agent.web.run_view.RunSummary
   factor_agent.web.run_view.Trial
   factor_agent.web.run_view.Collision
   factor_agent.web.run_view.Factor
   factor_agent.web.run_view.Neighbour
   factor_agent.web.run_view.AcceptanceReports
   factor_agent.web.run_view.Comparison
   factor_agent.web.run_view.Contribution
   factor_agent.web.run_view.HoldoutOpen
   factor_agent.web.run_view.Placed
   factor_agent.web.run_view.Refusal
   factor_agent.web.run_view.TrialRef
   factor_agent.web.run_view.Finding
   factor_agent.web.run_view.Stop
   factor_agent.web.run_view.MemoryVersion
   factor_agent.web.run_view.MemoryLine

Serving
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   factor_agent.web.app.create_app
   factor_agent.web.app.main
