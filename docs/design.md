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
| Q10 | 候选 JSON：`hypothesis`、`mechanism_type`（行为偏差/风险补偿/流动性/微观结构/信息扩散）、`expected_sign`、`horizon_rationale`、`code`（Q28）、`description`（`reference` 已由 Q75 取消） |
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
| Q24 | `[tool.uv.sources]` 指向 `../quantlab`；服务器仓库本就在 `~/projects/quantlab`（与 quantlab-ibkr 同一写法），直接 clone 即可，无需软链接 |
| Q25 | `LLMBackend` 后端：`claude_code`（`claude -p --output-format json`，走 Claude Code 登录）、`anthropic`（`ANTHROPIC_API_KEY`）、`openai_compatible`（OpenAI/DeepSeek/DashScope/Gemini 兼容端点/vLLM/Ollama，`base_url` 与密钥变量名在配置）、`replay`；各角色在配置里独立选后端和模型，可交叉审查 |
| Q26 | `claude_code` 后端关闭全部工具，纯文本生成（否则可读 `/data/quantlab` 破坏数据隔离） |
| Q27 | 产物路径见 §1 |
| Q28 | 候选写 `code`：Python 函数体，最后赋值 `alpha`；一行公式是特例。允许赋值、`with` 循环块、`for/while/if`、`def`；禁止 `class`、`import`、文件与网络 |
| Q29 | 构图在隔离子进程：无数据、无网络、超时、内存上限、白名单命名空间、AST 检查；有 bubblewrap 则加文件系统隔离（不擅自安装） |
| Q30 | 图约束：单输出；节点 ≤ 200；窗口与浮点参数在离散集合；量纲推不出标"未知"交审查判断 |
| Q31 | 可复用写法经人工审查进 quantlab `kunquant_ts.py`/`kunquant_cs.py` 再进白名单；循环不自动扩展白名单 |
| Q32 | 每类机制保底 3 个，余下 5 个按准入率 Thompson 采样（RD-Agent bandit 思路） |
| Q33 | 每 5 轮 LLM 将台账蒸馏为 `memory.md`（有效机制模式 / 失败约束，不含数值），每轮带入（FactorMiner） |
| Q34 | "父因子改进"一类：同一机制下改表达；图比对判定只改窗口/浮点参数 → 拒绝（AlphaPROBE 思路）；比对对象是整个比对库（参考集 + 已收下），不只已收下（2026-10-23 用户确认，例：Alpha158 STD20 换成窗口 40 也拒） |
| Q35 | 各段边界留 5 个交易日清洗空隙（`ret_5` 跨界） |
| Q36 | 保留期首次读取写锁文件（时间、代码版本）；再读须显式 `force` 并写进报告 |
| Q37 | v1 种子方向由 agent 按已有因子族空白起草约 10 条（附文献），用户审后写进配置 |
| Q38 | 采用 §4 的深 module 设计 |
| Q39 | （2026-10-24 v2 基本面拷问）候选输入在量价之外开放 SF1 ART 财报项目与 DAILY 市值估值；SF3A 持仓、SF2 内部人、另类数据暂不开放 |
| Q40 | 基本面另开独立 campaign（自己的配置与 `run_dir`）；配置新增 `inputs` 选输入族（默认只 `price_volume`，现有配置行为不变）；运行中的 us3000_ret5 不动 |
| Q41 | 一个候选可同时用量价与基本面字段；机制绑定照旧（假设点名的字段须在图里出现） |
| Q42 | 基本面 campaign 仍用 `labels/market/ret_5`，与量价 campaign、SEL0 验收口径一致 |
| Q43 | 机制类保持五类，不新增；价值/盈利/投资归风险补偿或行为偏差，盈余漂移归信息扩散 |
| Q44 | 开放字段为精选清单（约 40 个：`FundamentalAlpha` 用到的 + 常用三表项目 + DAILY `marketcap`/`ev`/`pe`/`pb`/`ps`/`evebit`/`evebitda`），每个附单位与一句话说明进提示词；清单由 agent 起草、用户审 |
| Q45 | 季度滞后用新算子 `Quarter(x, k)`，k ∈ {1,2,3,4,8}，读数据集的 `x__qk` 列；日频窗口算子作用于基本面字段照样允许，数据说明写明其为阶梯函数 |
| Q46 | 量纲改为（美元，股数）基：价格 (1,−1)、成交量 (0,1)、成交额与总额项目 (1,0)、股数 (0,1)、比率 (0,0)；对现有量价规则只是线性换基 |
| Q47 | 不开放每股项目（`eps`、`bvps` 等）与未复权价：复权价与原样每股数字相除会因拆股历史出错；估值用总额或 DAILY 现成比率 |
| Q48 | `max_base_fields` 仍为 5；同一字段的季度滞后不重复计数 |
| Q49 | 准入加覆盖率门槛：训练段每日有值成员占比平均 ≥ 50%（配置常量） |
| Q50 | 基本面 campaign 参考集仍为 SEL0 的 191 个；种子方向按已有基本面族空白由 agent 起草约 10 条附文献、用户审（同 Q37） |
| Q51 | 候选中的基本面字段名直接用 SF1 / DAILY 列名（`netinc`、`marketcap` …），量价仍用短名（`close`、`volume`），使 Zoo 能在图上认出对 `FundamentalAlpha` 的重写 |
| Q52 | 每个开放字段配一组同义词（机制绑定用），与 Q44 字段清单同表、同一来源，一起交用户审 |
| Q53 | v2 本地开发照常推送；服务器上运行中的 us3000_ret5 工作树钉在 `6298c2a` 不动，v2 测试另开 worktree |
| Q54 | campaign 开跑与每次续跑把 factor_agent 与 quantlab 提交号写进 `state.json`；续跑发现提交号变了只警告、不拦停 |
| Q55 | （2026-10-24 框架可配置拷问）开放字段不写进代码：每个输入族一个字段文件（YAML），每个字段写 `name`、`column`、`unit`、`period_lag` 可用否、`description`、`synonyms`；campaign 配置 `fields` 列字段文件路径（取代 Q40 的 `inputs` 名字列表）；现有量价输入也迁进 `price_volume.yaml` |
| Q56 | 派生字段在字段文件里写 `formula`（只引用同族已有字段，走候选代码同一套 AST 白名单），如 `vwap`、`amount`、换手率 |
| Q57 | 启动即校验：字段 `column` 在 `data.prices` 中、单位合法、名字不与算子/内置/其他字段冲突；不过即报错，不等到 Evaluator |
| Q58 | 用到的字段、算子、模板文件开跑时原样快照进 `run_dir`、哈希记入状态；续跑时文件与快照不符即报错（改搜索空间须另开 `run_dir`）；replay 读快照 |
| Q59 | 核心框架只含 agent 系统（循环、调度、角色与提示词骨架、LLM 后端、台账续跑回放、Gate 沙箱与图约束、Evaluator 统计、准入、验收流程）；数据组件、字段、参考集、标签、成员、中性化、市场描述、机制类、验收模型全来自配置；框架代码不出现数据集名、列名、路径 |
| Q60 | 配置 `market` 写市场描述；预测周期由标签 `span_bars()` 推出；数据说明段由框架按配置（字段说明、中性化方式）拼出 |
| Q61 | 配置 `scoring`：`neutralize: {exposures, size_column, industry_column}` 或 null（只 rank）；`membership` 可为 null（全体标的）；截面 rank 始终做；Evaluator 与 Acceptance 共用 |
| Q62 | 机制类为配置列表，每类附一句说明进提示词；默认现有五类；`min_per_type` 按配置类数校验（Q43 的五类成为默认值） |
| Q63 | 字段的绑定短语进字段文件（每字段 `synonyms`，族级 `rules` 表达多字段要求，如 overnight → open 与 close）；算子的绑定短语进算子文件 |
| Q64 | `Quarter(x, k)` 改为通用 `PeriodLag(x, k)`：字段文件族级声明 `period_lag: {column 模式, k 取值, period 说明}`，未声明的族不可用（取代 Q45） |
| Q65 | 框架不放用户默认值：删 `XGB_DEEP` 默认（不写用 quantlab XGBoost 默认，us3000 配置显式写 xgb_deep）；导出 `factor_root` 必填 |
| Q66 | 框架不自带输入族；仓库 `configs/fields/*.yaml` 只是示例与用户自己的配置，文件名标数据源（如 `sharadar_`） |
| Q67 | 自定义数据集与因子：任何遵守 quantlab 组件契约、可 import、能被 `rebuild` 重建的类都可写进 `data`/`reference`，不另设插件注册表 |
| Q68 | 算子白名单为算子文件（YAML），配置 `operators` 指向它；每条写 `name`、`class`（KunQuant `OpBase` 子类）、`call`（可为 NaN 保留版）、`signature`、`inputs`、`params`、`dimension`（从代码固定的量纲规则集中选）、`description`、`binding`；代码只留类路径解析、量纲规则、循环语言构造（取代 Q8' 的代码注册表与 Q31 的"只经 quantlab 进白名单"） |
| Q69 | 参数集合（窗口、常量、分位数、上下界等）进算子文件 `parameters` 段，同 Q58 快照与哈希（推翻本轮早先"参数集合留在代码"的推荐） |
| Q70 | 解析时安全校验：`class` 须为 KunQuant `OpBase` 子类；`call` 只能来自 `KunQuant`、`quantlab.factor` 或配置 `operator_modules` 显式列出的模块；沙箱按同一解析结果建命名空间；构图后 Gate 仍查每个节点类已注册 |
| Q71 | 全覆盖守护测试只针对仓库示例算子文件的 `excluded:` 段（类 → 理由），提醒 KunQuant 升级新增算子；用户算子文件不受约束 |
| Q72 | 四个角色的提示词模板与数据说明段可由配置 `prompts` 替换；构造时校验必需占位符；模板同 Q58 快照 |
| Q73 | 图约束常量（节点数、自由参数比例、字段数、公共子树）维持现状在配置 `gate` 段，不增约束种类 |
| Q74 | 模块划分（2026-10-24 codebase-design）：新建深 module `Vocabulary`（`load`、`namespace`、`check`、`bind`、`columns`、`prompt`；只按数据不同、不设可替换实现）、`Scoring`（`from_config`、`apply`、`read`、`describe`；收走 Evaluator 与 Acceptance 重复的打分代码）、`RunSnapshot`（`open`、`path`；Campaign 与 Acceptance 共用）；`Researcher` 接口不变，模板与机制类由配置注入；`vocabulary.py` 不再存常量 |
| Q75 | （2026-10-24 用户）框架不要求文献支持：候选去掉 `reference` 字段，提示词不再要求写文献；种子方向是否附文献由写配置的人决定，框架不检查 |
| Q76 | （2026-10-24，#20）被数据证伪的假设可以跟进：planner 新增 `reversal` 动作，parent 为方向相反被拒的试验（"…(refuted in round N)"），须给出预测观察到的方向的机制，不要求文献；Gate 去重按"保持或反转排名"（`Graph.ranking`）比较，只取负、缩放、rank 的写法算重复；全部历史保留押的方向和"opposite sign"类别；distiller 记忆新增"Directional findings"（仍不写数字）；拒绝理由改为提示可另提反方向机制。Q11 不翻转不变 |
| Q77 | （2026-10-25 拷问，取代 Q10 的 `expected_sign`、Q11、Q76 的 reversal 部分）去掉方向要求：候选不再声明方向（删 `expected_sign`）；因子的方向 = 训练段 IC 的符号，偏 IC 的 NW t 与十分组单调性都按这个方向算；多重检验门槛把每个候选算两次检验（Bonferroni 分母 n → 2n，起点 3.0 不变：1 次 3.21、100 次 4.35、600 次 4.73）；撤掉 reversal 动作、被证伪假设列表、历史里的方向类别、distiller 的 Directional findings；保留 `Graph.ranking` 去重；导出的因子乘以方向（值越大训练段收益越高），README 与 component.json 记方向和训练段 IC；reviewer 只查代码与描述、描述与机制是否一致，不查方向；已接受因子与上一轮反馈标出数据显示的方向，distiller 可用文字记机制的实际方向（不写数字） |
| Q78 | （2026-10-25）回放可复现与构图错误：沙箱崩溃的拒绝理由只留最后一行异常且不含绝对路径（不同目录回放也逐字节一致）；数值参数位置放了算子（如 `SubConst(x, ConstantOp(1))`）在 Gate 里按普通理由提前拒绝，并提示写法；`Sub(x, 1)` 这类整数输入的构图失败给出同样的写法提示 |
| Q79 | （2026-10-25 拷问，RD-Agent CoSTEER 的错误→修法思路）写法错误的修法提示（Fix）：只管构图与静态检查类写法错误，不管机制绑定、重复、原创性、准入；手写清单、不从 ledger 自动学；算子文件新增可省略的 `fixes` 一节（每条 `match` 正则 + `hint`，启动时与其他词表错误一起校验，多条命中按文件顺序全附），随算子文件进快照；提示与被拒理由分开（`Rejected.hints`），只随重写请求发出，不进 ledger 的理由、也不进常驻提示；以两个下划线开头的名字候选代码不得读写；示例算子文件先放两条（整数当算子输入，原 Q78 写死的提示挪入；不支持的运算符，如 `%`、`//`）；沙箱改写语法树，`-x` 直接得 `MulConst(x, -1)`、`a ** b` 在登记了 `Pow` 时得 `Pow(a, ConstantOp(b))`，否则照常报错；"算子填进数字参数"与"比较当数用"仍是框架自己的检查与提示 |
| Q80 | （2026-10-24 提示词工程复审，依据 quantlab 本地笔记 `docs/research/prompt-engineering.md`）五个角色模板重写，占位符与必需/可选集合不变（旧的替换模板照常加载）：开头一两句说角色、任务和产出给谁怎么用，长数据放中间，结尾重申现在要做什么并给输出骨架（只有字段名和类型，不给完整示例），写明"第 i 个元素对应第 i 条输入""不要多余字段"；输入一律包进 XML 标签（`<quota>`、`<directions>`、`<refused_candidates>`、`<candidates>`、`<history>`、`<data>`、`<vocabulary>`，上下文各段按键名打标签、保留原标题），```` ```json ```` 围栏只留给结尾的输出骨架；RETRY 明说返回修正后的候选对象本身（不带 number/direction/refused_because/how_to_write_it、不嵌套），输入条目的 `candidate` 键改名 `refused_version`；解析兜底：数组元素缺候选字段、但在 "candidate" 或 "refused_version" 下有对象时取该对象，别的形状照旧报错；一词一义：candidate、factor、research direction（规划的槽位）与 factor direction（训练段 IC 的符号，反馈里写作 "factor direction ±1"）、mechanism；规则用正面表述并说明原因，不用全大写；REVIEW 写成编号的不一致判据、代码长短不作理由、输出字段顺序 number → reason → consistent（解析不看顺序）；DISTILL 写明读者与用法、每条附原因、每部分最多 6 条、合并重复并删掉被后来推翻的结论，仍是三部分；删除对照 2bd6904 旧提示词的迁移测试（这次是有意改写）；历史压缩另议 |
| Q81 | （2026-10-25 拷问，取代 Q41 与 #3 的确定性机制绑定）机制绑定交给 reviewer：Gate 删掉 `Vocabulary.bind`，字段文件删 `phrases`/`synonyms`/`rules`、算子文件删 `phrases`/`binding`（不兼容旧键，仓库三份文件同改）；reviewer 的编号标准新增"假设说因子由哪些输入、变换或量构成，代码须真的用到；作为背景、后果或对比提到的不算"并附两例对照；reviewer 提示加数据段（字段名与说明）；reviewer 判为不一致的候选退回一次重写，用 RETRY 模板、理由为 reviewer 原话，改好的版本只过一次 Gate（再拒即 Gate 失败）、再审一次（再不一致即淘汰）；第一次 reviewer 判定作为 `stage: "review"` 的 attempt 记入 ledger；两次审核各为一轮一次调用；被 reviewer 拒绝仍不计入试验数 |

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
| `Evaluator` | `for_segment(segment, reference)`；`.evaluate(graphs, admit) -> list[Metrics \| CompileFailure]`（`admit` 是 `Campaign` 的准入回调，按提案顺序调用，收下的候选当轮即进参考池） | 批量编译与二分定位、全市场读取与坏报价屏蔽、rank、中性化、成员屏蔽、标签对齐、IC/NW t/偏 IC/10 分组 | 唯一碰数据；segment 构造时固定、无换日期方法；测试用小型合成 Zarr store |
| `Segments` | 日期、清洗（`scored`：至少 5 个交易日，标签读得更远则按 `lookahead_bars`）、保留期锁（`open_holdout`） | Q35、Q36 | `Evaluator` 与 `Acceptance` 共用 |
| `Researcher` | `plan`、`propose(plan, gate)`、`review`、`distill` | prompt 模板、三段历史、JSON 解析与重试、Gate 失败重试、调用落盘 | 各角色后端由配置注入 |
| `LLMBackend` | `complete(request) -> Reply(text, usage, cost, model_id)` | — | 真实 seam：claude_code / anthropic / openai_compatible / replay + 测试 fake |
| `Ledger` | 追加与读取试验记录、轮次状态 | 文件格式 | 测试用临时目录 |
| `Vocabulary` | `load(fields, operators)`、`namespace()`、`check(graph)`、`bind(hypothesis, graph)`、`columns(graph)`、`prompt()` | 类路径解析与模块白名单、单位与量纲规则、派生字段、周期滞后、绑定短语、启动校验 | 由字段文件与算子文件构建；沙箱子进程从快照重建同一份；`test_vocabulary` |
| `Scoring` | `from_config(scoring, data)`、`apply(factor)`、`read(factor, start, end)`、`describe()` | 截面 rank、可选中性化、可选成员屏蔽、从 store 首日读 | Evaluator 与 Acceptance 共用，口径唯一 |
| `RunSnapshot` | `open(run_dir, files, versions)`、`path(name)` | 文件复制、哈希、提交号；续跑时文件变了报错、提交号变了警告 | Campaign 与 Acceptance 共用 |

`Campaign` 由配置构造 `Researcher`、`Gate`、`Evaluator`、`Ledger`；唯一的外部 seam 是 `LLMBackend`
（2026-10-23 to-spec 时收紧：取消 `Evaluator` 的 fake，`Campaign` 测试用真 `Gate`、真 `Evaluator` 加小型合成 Zarr store）。
bandit、反馈拼装、记忆节奏、准入策略、停止条件都是 `Campaign` 的内部细节。

测试面：`CampaignConfig.from_file()`（配置文件即运行的入口）、`Campaign.run()`（主测试面，脚本化 fake `LLMBackend`）、`Gate.check()`（表驱动拒绝规则）、
`Acceptance`（验证对比、保留期锁、导出逐位一致）、`LLMBackend` adapter（replay/fake 离线；真实提供商仅带凭证冒烟）。

```
factor_agent/
  campaign.py      （内部 _scheduler.py、_feedback.py、_admission.py、_memory.py）
  researcher.py    （内部 prompts/、_parse.py、_recorder.py）
  llm/             base.py、claude_code.py、anthropic.py、openai_compatible.py、replay.py
  gate/            Gate；与 Evaluator / 提示词共用的 graph.py（图的纯数据形式）、vocabulary.py（算子白名单）；内部 _sandbox.py、_checks.py
  evaluator.py     （内部 _batch.py）
  segments.py · ledger.py · zoo.py · acceptance.py · config.py
  exported.py      GraphFactor（FactorKunQuant 子类，Evaluator 与导出共用的建图）；accepted/ 放 export 生成的模块
tests/  只测对外与深 module 的 interface：test_gate、test_evaluator、test_campaign、test_acceptance、test_llm_backends，
        以及运行的入口配置文件 test_config（参考集写法、真实配置 configs/us3000_ret5.yaml 可解析）
```

## 5. 运行约束

- 凭证只从环境变量读；仓库不存任何密钥。
- 服务器：CPU 与内存绑 Node 1；KunQuant 并行 ≤ 51 核；不设 OMP/MKL 线程环境变量；只杀自己的进程。
- 超过约 1 分钟的命令放后台（nohup + 日志 + pid）。
