Evaluation
==========

Score candidates on the training segment against the label and the reference pool.
See :doc:`../user-guide/evaluation`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   factor_agent.evaluator.Evaluator
   factor_agent.evaluator.Metrics
   factor_agent.evaluator.CompileFailure
   factor_agent.scoring.Scoring
   factor_agent.scoring.Neutralization

Session columns
---------------

Compute the intraday stage of a round's candidates on the intraday bars, one value per stock
and session. See :doc:`../user-guide/intraday`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   factor_agent.session.SessionColumns
   factor_agent.session.SessionFactor
   factor_agent.session.session_function
   factor_agent.derived.Provider
   factor_agent.derived.DerivedColumnsFactor
