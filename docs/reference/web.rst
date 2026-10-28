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
   factor_agent.web.run_view.find_run
   factor_agent.web.run_view.run_version
   factor_agent.web.run_view.read_call
   factor_agent.web.run_view.read_files
   factor_agent.web.run_view.read_ic
   factor_agent.web.run_view.RunView
   factor_agent.web.run_view.RunSummary
   factor_agent.web.run_view.Health
   factor_agent.web.run_view.Tokens
   factor_agent.web.run_view.RoundTokens
   factor_agent.web.run_view.RunningRound
   factor_agent.web.run_view.RoundCall
   factor_agent.web.run_view.PendingTrial
   factor_agent.web.run_view.Check
   factor_agent.web.run_view.LLMCall
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
   factor_agent.web.run_view.LineChange
   factor_agent.web.run_view.RunFiles
   factor_agent.web.run_view.ConfigCopy
   factor_agent.web.run_view.SnapshotFile
   factor_agent.web.run_view.TrialIC
   factor_agent.web.run_view.AdmissionLimits

Serving
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   factor_agent.web.app.create_app
   factor_agent.web.app.main
