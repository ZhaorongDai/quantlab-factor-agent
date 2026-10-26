LLM backends
============

Turn a request into a reply: the one external seam of a campaign.
See :doc:`../user-guide/llm-backends`.

Interface
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   factor_agent.llm.base.LLMBackend
   factor_agent.llm.base.Request
   factor_agent.llm.base.Reply
   factor_agent.llm.base.ProviderQuotaExhausted
   factor_agent.llm.base.make_backend

Backends
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   factor_agent.llm.openai_compatible.OpenAICompatibleBackend
   factor_agent.llm.anthropic.AnthropicBackend
   factor_agent.llm.claude_code.ClaudeCodeBackend
   factor_agent.llm.replay.ReplayBackend
   factor_agent.llm.replay.ReplayExhausted
   factor_agent.llm.log.LLMLog
