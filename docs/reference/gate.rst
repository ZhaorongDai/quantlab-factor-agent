Gate and vocabulary
===================

Check a candidate's code and graph before it is evaluated.
See :doc:`../user-guide/vocabulary` and :doc:`../user-guide/evaluation`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   factor_agent.gate.Gate
   factor_agent.gate.Passed
   factor_agent.gate.Rejected
   factor_agent.gate.graph.Graph
   factor_agent.gate.vocabulary.Vocabulary
   factor_agent.gate.vocabulary.VocabularyError
   factor_agent.zoo.Zoo
   factor_agent.candidate.Candidate

Intraday stages
---------------

Split an intraday family's candidate into its intraday and daily stages, and the session
aggregations it may reduce a session with. See :doc:`../user-guide/intraday`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   factor_agent.gate.stages.split
   factor_agent.gate.stages.Staged
   factor_agent.gate.stages.session_column
   factor_agent.aggregations.SessionAggregation
   factor_agent.aggregations.SessionMean
   factor_agent.aggregations.SessionSum
   factor_agent.aggregations.SessionCount
   factor_agent.aggregations.SessionStd
   factor_agent.aggregations.SessionSkew
   factor_agent.aggregations.SessionMin
   factor_agent.aggregations.SessionMax
   factor_agent.aggregations.SessionFirst
   factor_agent.aggregations.SessionLast
   factor_agent.aggregations.SessionWeightedMean
   factor_agent.aggregations.SessionCorr
   factor_agent.aggregations.WithinSession
