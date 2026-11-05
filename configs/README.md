# Example configs

Complete campaign configs for US equities: Sharadar daily data, the 5-day open-to-open
return as the label, scored on the members of a 3000-stock universe. Every path is a
quantlab store under one data root, `/data/quantlab/`; point them at your own stores and set
`run_dir` before you run one. The roles use DeepSeek through its OpenAI-compatible API and
read the key from `DEEPSEEK_API_KEY`; see
[LLM backends](../docs/user-guide/llm-backends.md) for other providers.

| Config | Opens |
|--------|-------|
| `us3000_ret5.yaml` | daily price and volume |
| `us3000_fund_ret5.yaml` | price and volume, and Sharadar fundamentals |
| `us3000_hf_ret5.yaml` | price and volume, and daily statistics of one-minute trade bars |
| `us3000_intraday_ret5.yaml` | price and volume, and the one-minute trade bars themselves |

`fields/` holds the field files the configs name, one per input family, and `operators/`
the operator file: the KunQuant operators candidate code may call and the values their
parameters may take. [The vocabulary](../docs/user-guide/vocabulary.md) explains both, and
[the campaign config](../docs/user-guide/campaign.md) every section of a config.
