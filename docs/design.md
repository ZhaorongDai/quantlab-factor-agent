# quantlab-factor-agent 设计

2026-10-23 拷问（grilling）结论，全部决定经用户确认。调研依据见 quantlab 本地笔记
`docs/research/llm-alpha-mining.md`（RD-Agent(Q)、AlphaAgent、QuantaAlpha、FactorMiner、alphagen_llm 等）。

## 1. 目标与边界

用 LLM 按**经济机制假设**生成 KunQuant 因子代码，在 quantlab 的数据契约上评估，产出对现有 us3000 模型
**有增量**的日频因子。不是暴力搜索：测试次数在几百量级，每个因子都带可校验的机制假设。

- 独立项目，依赖 quantlab 公开 API（`quantlab-py`，本地 editable 指向 `../quantlab`）。
- 库里缺的通用能力补进 quantlab 的公开 API，不在本项目里私自实现或调用 quantlab 私有方法。
- 通过验收的因子写成 `FactorKunQuant` 子类，store 进 `/data/quantlab/factors/market/<名字>/`（附 README.md、component.json）。
- 运行产物进 `/data/quantlab/runs/<实验>/`；实验记录按 quantlab CLAUDE.md 写进 quantlab-experiment 并推送。

## 2. 决策记录

| # | 决定 |
|---|---|
| Q1 | 成功标准两级：初筛 = 相对参考集残差化后偏 IC 的 NW t ≥ 门槛；最终 = 加进 SEL0 xgb 后在验证段按普通 TopK、IC、10 分组单调性、不扣费收益优于基线 |
| Q2 | LLM 按机制假设生成（假设 → 代码 → 评估 → 反馈），不做 GP/RL 暴力搜索 |
| Q3 | 产物位置见 §1（原"实验目录"决定被 Q22 新项目取代，store 与实验记录规则不变） |
| Q4 | 全市场 `market/sharadar/sharadar_sep_1d`（`BadPrintMaskedDataset`）上求值，截面算子按全市场；IC 按 t 当天 `universe/us3000/membership_estu` 成员屏蔽；目标 `labels/market/ret_5`（`MembershipMaskedLabel`） |
| Q5 | 训练段 2000–2014（生成与反馈只用它）；验证段 2015–2019（增量验收）；保留期 2020-01 至今只看一次；每个测试过的候选都进台账，报告写总试验数 |
| Q6 | 自建薄循环，不采用框架；MIT 代码（AlphaAgent legacy-main、FactorMiner）按许可证借鉴并注明出处；无 LICENSE 的项目只借思路 |
| Q7 | 默认 LLM：Claude 最新旗舰；见 Q25 多后端 |
| Q8' | 不另造 DSL，直接用 KunQuant / quantlab 算子；白名单注册表写明类型签名与量纲规则；守护测试保证每个 `OpBase` 子类在注册表或排除表；NaN 敏感算子用 quantlab NaN 保留版；整数窗口 ∈ {5,10,20,40,60,120,250}，浮点参数取离散集合；输入字段沿用 `Alpha158Stock` 口径（复权 OHLCV + 典型价 vwap），v1 只用日频价量 |
| Q9 | 原创性：最大公共子树 + 截面相关 > 0.7 拒绝；反馈只含训练段 IC、NW t、残差 IC、最相近已有因子及相关、拒绝原因；不给回测收益、不提示调参；验证段与保留期代码层面不可见 |
| Q10 | 候选 JSON：`hypothesis`、`mechanism_type`（行为偏差/风险补偿/流动性/微观结构/信息扩散）、`expected_sign`、`horizon_rationale`、`reference`（可空但须说明）、`code`（Q28）、`description` |
| Q11 | 训练段 IC 方向与 `expected_sign` 相反 → 拒绝，不翻转 |
| Q12' | 增量参考集 = SEL0 的 191 个因子 + 本次已收下因子；公共子树比对覆盖其中所有 KunQuant 实现的因子（在图上比），其余只靠截面相关与残差化 |
| Q13 | 准入对象 = 截面 rank 后再行业+市值中性化的版本 |
| Q14 | 每轮 20 个候选，≤ 30 轮，≤ 600 次试验；NW t 门槛 3.0 起，按累计试验数上调（Harvey–Liu–Zhu 思路，公式为常量）；连续 5 轮无收获即停 |
| Q15 | 首轮上下文：数据说明、已有因子族名称与一句话描述、已收下/已拒绝因子及原因；不给具体 IC 数值 |
| Q16 | 验收：收下的因子加进 SEL0 xgb，训练 ≤ 2014、验证 2015–2019、3 个种子对比基线；通过后看一次保留期；服务器 Node 1，KunQuant ≤ 51 核，xgb 用 GPU，工作日美东 06:00 前停 |
| Q17 | 实验记录进 quantlab-experiment 并推送；quantlab 库改动只在服务器本地提交 |
| Q18 | 角色：方向规划 → 提案 → 审查（独立调用，假设与代码一致性，只判定）；每轮各机制类至少 3 个 |
| Q19 | 构图/检查失败带具体错误回传，最多重试 2 次；仍失败记一次失败试验，不计入 NW t 门槛的累计数 |
| Q20 | 每次 LLM 调用完整落盘（prompt、模型 id、参数、原始回复）到 `runs/<实验>/llm_log/`；`replay` 回放逐字节复现；模型与参数为配置常量 |
| Q21 | token / 成本预算上限，超出即停并记录；开跑前估算总成本报用户确认 |
| Q22 | 项目名 `quantlab-factor-agent`，包名 `factor_agent`，本地与服务器均在 `~/projects/quantlab-factor-agent` |
| Q23 | GitHub 私有仓库 `ZhaorongDai/quantlab-factor-agent`，每个提交即推送 |
| Q24 | `[tool.uv.sources]` 指向 `../quantlab`；服务器需软链接 `~/projects/quantlab -> quantlab2`（改服务器前单独确认） |
| Q25 | `LLMBackend` 后端：`claude_code`（`claude -p --output-format json`，走 Claude Code 登录）、`anthropic`（`ANTHROPIC_API_KEY`）、`openai_compatible`（OpenAI/DeepSeek/DashScope/Gemini 兼容端点/vLLM/Ollama，`base_url` 与密钥变量名在配置）、`replay`；各角色在配置里独立选后端和模型，可交叉审查 |
| Q26 | `claude_code` 后端关闭全部工具，纯文本生成（否则可读 `/data/quantlab` 破坏数据隔离） |
| Q27 | 产物路径见 §1 |
| Q28 | 候选写 `code`：Python 函数体，最后赋值 `alpha`；一行公式是特例。允许赋值、`with` 循环块、`for/while/if`、`def`；禁止 `class`、`import`、文件与网络 |
| Q29 | 构图在隔离子进程：无数据、无网络、超时、内存上限、白名单命名空间、AST 检查；有 bubblewrap 则加文件系统隔离（不擅自安装） |
| Q30 | 图约束：单输出；节点 ≤ 200；窗口与浮点参数在离散集合；量纲推不出标"未知"交审查判断 |
| Q31 | 可复用写法经人工审查进 quantlab `kunquant_ts.py`/`kunquant_cs.py` 再进白名单；循环不自动扩展白名单 |
| Q32 | 每类机制保底 3 个，余下 5 个按准入率 Thompson 采样（RD-Agent bandit 思路） |
| Q33 | 每 5 轮 LLM 将台账蒸馏为 `memory.md`（有效机制模式 / 失败约束，不含数值），每轮带入（FactorMiner） |
| Q34 | "父因子改进"一类：同一机制下改表达；图比对判定只改窗口/浮点参数 → 拒绝（AlphaPROBE 思路） |
| Q35 | 各段边界留 5 个交易日清洗空隙（`ret_5` 跨界） |
| Q36 | 保留期首次读取写锁文件（时间、代码版本）；再读须显式 `force` 并写进报告 |
| Q37 | v1 种子方向由 agent 按已有因子族空白起草约 10 条（附文献），用户审后写进配置 |
| Q38 | 采用 §4 的深 module 设计 |

## 3. 一轮的流程

```
种子方向 + memory.md + 三段历史（全部 / 上一轮 / 当前最好机制）
 ① 方向规划（LLM）  机制类保底 + bandit；新方向 / 正交变异 / 父因子改进
 ② 提案（LLM）      假设 JSON + KunQuant 代码
 ③ Gate（确定性）   隔离构图 → 规范化+哈希 → 单输出/节点/离散参数/复杂度
                     → 量纲 → 机制绑定 → 与比对库最大公共子树；失败带原因回 ②（≤ 2 次）
 ④ 审查（LLM）      假设 → 描述 → 代码 一致性，严格，只判定
 ⑤ Evaluator        一批一图编译 → 全市场求值 → rank → 中性化 → 成员屏蔽 → 对 ret_5（仅训练段）
 ⑥ 准入（纯函数）   符号一致 → 相关 ≤ 0.7 → 偏 IC NW t ≥ 门槛(试验数) → 10 分组单调
 ⑦ 反馈与记录       台账 · 本轮反馈 · 池内边际贡献 · 每 5 轮蒸馏记忆 · 停止条件
循环后：Acceptance.validate（xgb 3 种子）→ open_holdout（一次）→ export（FactorKunQuant + store）→ 实验记录
```

### 借鉴来源

| 做法 | 来源 |
|---|---|
| 角色分离、初始方向形式化 | AlphaAgent |
| 三段历史上下文、动作 bandit | RD-Agent(Q) |
| 最大公共子树原创性、复杂度比例、重试时回传重复子树 | AlphaAgent（MIT）/ QuantaAlpha（阈值） |
| 规范化后去重、确定性机制绑定、增量准入、经验记忆、试验台账、从代码重算、切分清洗、无凭证冒烟 | FactorMiner（MIT） |
| 假设与代码一致性审查（只判定）、正交方向变异 | QuantaAlpha（只借思路） |
| 父因子改进 | AlphaPROBE（只借思路） |
| 池内边际贡献反馈 | alphagen_llm（只借思路） |

刻意不采用：用回测段指标替换 SOTA（RD-Agent/AlphaAgent/QuantaAlpha）、反馈中鼓励换窗口调参（AlphaAgent）、
空比对库（AlphaAgent 开源版）、0.99 相关才去重、假设拼合 crossover（QuantaAlpha）。

## 4. 模块设计

对外 interface 只有两个：

- `Campaign(config).run() -> CampaignResult`：可从台账断点续跑，每轮落盘后才推进。
- `Acceptance(config, run_dir)`：`.validate()`、`.open_holdout()`、`.export(names)`。

内部深 module：

| Module | Interface | 藏在里面 | 不变量 / 依赖类别 |
|---|---|---|---|
| `Gate` | `check(candidate, zoo) -> Passed(graph, fingerprint, stats) \| Rejected(reasons)` | 隔离子进程、AST 白名单、算子注册表、规范化、各项图约束、量纲、机制绑定、公共子树、检查顺序 | 不读数据、确定性、坏候选返回 `Rejected` 不抛异常；local-substitutable（测试真起子进程） |
| `Evaluator` | `for_segment(segment, reference)`；`.evaluate(graphs) -> list[Metrics \| CompileFailure]` | 批量编译与二分定位、全市场读取与坏报价屏蔽、rank、中性化、成员屏蔽、标签对齐、IC/NW t/偏 IC/10 分组 | 唯一碰数据；segment 构造时固定、无换日期方法；测试用小型合成 Zarr store |
| `Segments` | 日期、清洗、保留期锁 | Q35、Q36 | `Evaluator` 与 `Acceptance` 共用 |
| `Researcher` | `plan`、`propose(plan, gate)`、`review`、`distill` | prompt 模板、三段历史、JSON 解析与重试、Gate 失败重试、调用落盘 | 各角色后端由配置注入 |
| `LLMBackend` | `complete(request) -> Reply(text, usage, cost, model_id)` | — | 真实 seam：claude_code / anthropic / openai_compatible / replay + 测试 fake |
| `Ledger` | 追加与读取试验记录、轮次状态 | 文件格式 | 测试用临时目录 |

`Campaign` 注入 `Researcher`、`Gate`、`Evaluator`、`Ledger`；`Evaluator` 有真实与 fake 两个 adapter，
所以 `Campaign` 的测试不编译、不读数据。bandit、反馈拼装、记忆节奏、准入策略、停止条件都是 `Campaign` 的内部细节。

```
factor_agent/
  campaign.py      （内部 _scheduler.py、_feedback.py、_admission.py、_memory.py）
  researcher.py    （内部 prompts/、_parse.py、_recorder.py）
  llm/             base.py、claude_code.py、anthropic.py、openai_compatible.py、replay.py
  gate/            只暴露 Gate；内部 _sandbox.py、_registry.py、_graph.py、_checks.py
  evaluator.py     （内部 _batch.py）
  segments.py · ledger.py · zoo.py · acceptance.py · config.py
tests/  只测对外与深 module 的 interface：test_gate、test_evaluator、test_campaign、test_acceptance、test_llm_backends
```

## 5. 运行约束

- 凭证只从环境变量读；仓库不存任何密钥。
- 服务器：CPU 与内存绑 Node 1；KunQuant 并行 ≤ 51 核；不设 OMP/MKL 线程环境变量；只杀自己的进程。
- 超过约 1 分钟的命令放后台（nohup + 日志 + pid）。
