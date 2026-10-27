.. _reference:

API reference
=============

Look up classes, functions, parameters and return values here. To learn the workflow,
see the :doc:`../user-guide/index`; to run a campaign, see the
:doc:`../getting-started/quickstart`.

.. list-table:: Find an API by task
   :header-rows: 1
   :widths: 28 42 30

   * - Task
     - Main entry points
     - Reference
   * - Configure and run a campaign
     - ``CampaignConfig``, ``Campaign``, ``CampaignResult``
     - :doc:`campaign`
   * - Check candidate code
     - ``Gate``, ``Graph``, ``Vocabulary``, ``Zoo``
     - :doc:`gate`
   * - Score candidates
     - ``Evaluator``, ``Metrics``, ``Scoring``
     - :doc:`evaluation`
   * - Judge and export accepted factors
     - ``Acceptance``, ``AcceptanceReport``, ``GraphFactor``
     - :doc:`acceptance`
   * - Connect a language model
     - ``LLMBackend``, ``make_backend``, ``OpenAICompatibleBackend``
     - :doc:`llm`
   * - Read run directories, serve the frontend
     - ``list_runs``, ``RunView``, ``factor-agent-web``
     - :doc:`web`

Each object page gives its signature and description, followed by parameters, return
values, exceptions and examples where documented. The :doc:`modules` lists every module.
The :ref:`genindex` finds objects by name.

.. toctree::
   :hidden:

   campaign
   gate
   evaluation
   acceptance
   llm
   web
   modules
