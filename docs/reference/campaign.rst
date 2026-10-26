Campaign
========

Read a campaign config, run the research loop, and read its records.
See :doc:`../user-guide/campaign` for the config's sections.

Running a campaign
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   factor_agent.campaign.Campaign
   factor_agent.campaign.CampaignResult
   factor_agent.campaign.Stop

Configuration
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   factor_agent.config.CampaignConfig
   factor_agent.config.DataConfig
   factor_agent.config.ScoringConfig
   factor_agent.config.NeutralizeConfig
   factor_agent.config.EvaluatorConfig
   factor_agent.config.GateConfig
   factor_agent.config.AdmissionConfig
   factor_agent.config.RoundConfig
   factor_agent.config.ContextConfig
   factor_agent.config.AcceptanceConfig
   factor_agent.config.RoleConfig
   factor_agent.config.Mechanism

Records and run state
---------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   factor_agent.ledger.Ledger
   factor_agent.ledger.TrialRecord
   factor_agent.snapshot.RunSnapshot
   factor_agent.snapshot.SnapshotMismatch
   factor_agent.snapshot.CodeVersionChanged
   factor_agent.segments.Segments
   factor_agent.segments.Segment
   factor_agent.prompts.Templates
   factor_agent.prompts.PromptError
