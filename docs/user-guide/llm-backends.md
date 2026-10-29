# LLM backends

Each role under `llm` picks its backend and model; the other keys are the backend's
settings.

```yaml
llm:
  planner:   {backend: openai_compatible, model: deepseek-v4-pro, base_url: https://api.deepseek.com,
              api_key_env: DEEPSEEK_API_KEY, max_tokens: 32000, timeout_seconds: 900}
  proposer:  {backend: anthropic, model: claude-opus-5-5, max_tokens: 16000}
  reviewer:  {backend: claude_code, model: claude-opus-5-5, effort: high}
  distiller: {backend: openai_compatible, model: deepseek-v4-pro, base_url: https://api.deepseek.com,
              api_key_env: DEEPSEEK_API_KEY}
```

| Backend | Calls | Credentials |
|---|---|---|
| `openai_compatible` | any OpenAI-compatible endpoint: OpenAI, DeepSeek, DashScope, Gemini's compatible endpoint, vLLM, Ollama | the variable `api_key_env` names, or none for a local server |
| `anthropic` | the Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `claude_code` | `claude -p --output-format json`, with every tool, MCP server, skill and settings file off, from an empty directory | the local Claude Code login |
| `replay` | the replies of a recorded LLM log, matched on role and prompt | none |

For a reasoning model on an OpenAI-compatible endpoint give `max_completion_tokens` in
place of `max_tokens`, and `reasoning_effort`. A key written in the config is refused.

## The LLM log

Every call is written in full to `<run_dir>/llm_log/`: the prompt, the model, the
parameters, the reply, the token usage as the provider reports it, and the raw reply.
Calls are numbered in the order they finish, which with proposal groups need not be the
order they were sent.

## Replaying a run

Point every role at the `replay` backend and the run reproduces byte for byte without
calling a provider:

```yaml
llm:
  planner:   {backend: replay, model: deepseek-v4-pro, log: runs/my_campaign/llm_log}
  proposer:  {backend: replay, model: deepseek-v4-pro, log: runs/my_campaign/llm_log}
  reviewer:  {backend: replay, model: deepseek-v4-pro, log: runs/my_campaign/llm_log}
  distiller: {backend: replay, model: deepseek-v4-pro, log: runs/my_campaign/llm_log}
```

A replay searches with the snapshot of the run whose log it replays, not the config's
files. A request the log has no reply for fails (`ReplayExhausted`). A log written before
a prompt change cannot be replayed with the new prompts.

## Cost and quota

The library counts no money. Spending limits belong to the provider: a prepaid balance or
an account limit. The token usage in the log is there for working out a cost; a config
that names `round.budget_usd` or per-token prices is refused.

When the provider refuses a call because the balance or quota is exhausted (HTTP 402 from
an OpenAI-compatible provider; Anthropic's billing error or too-low credit balance), the
backend raises `ProviderQuotaExhausted`. The campaign drops the round in progress, records
the stop `provider_quota` with the provider's message and returns; `run()` after topping
up runs that round again.

A provider sometimes answers a call as a success with something that is not a reply: a body
of `null` (DeepSeek, after the blank lines it sends while a call waits), a reply without
its text or its token usage, or Claude Code output that is not its JSON. The SDKs do not
retry it. A campaign asks any role's backend again after 5 s, then after 30 s, and then
raises `ProviderReplyMalformed`; a backend of your own raises it for such an answer to
be asked again too. Every other provider error is raised.

## Your own backend

A backend is anything with `complete(request) -> reply` (`LLMBackend`). Pass backends by
role to `Campaign(config, backends={...})` to replace the ones the config names; the
model still comes from the config.
