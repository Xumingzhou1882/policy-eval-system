# 修改日志

## 2026-06-04 — Stage 6 自动化

### Stage 6: 从手动验证到确定性脚本

Stage 6 是 pipeline 中最后一个未自动化的方法论阶段。现在有了专用脚本 `stage6_confirm.py`（约 700 行），实现了与 Stage 3 同水准的确定性自动化。

**新增脚本：**

1. **`stage6_confirm.py`** — 最终方法确认引擎：
   - 双层数据结构：`AssumptionVerdict`（假设判定）+ `DiagnosticResult`（诊断结果）+ `FallbackAttempt`（备选尝试）+ `MethodConfirmation`（完整确认）
   - 假设→检验映射注册表：覆盖 8 种机制，20+ 个可检验假设，每个都有对应诊断函数
   - 内联诊断函数（无需子进程）：倾向得分重叠检查、协变量平衡、合规率检查、分组覆盖率、剂量-反应事件研究
   - 子进程诊断包装器（调用现有脚本）：事件研究、Bacon 分解、McCrary、IV 第一阶段、SCM、Synthetic DID、DML
   - 确定性降级引擎：当主方法失败时，按 Stage 3 的 fallbacks 排序依次尝试备选方法
   - 为 Stage 7 生成标准化规范字典（含 entity_col、time_col、covariates 等所有派发字段）
   - CLI 独立使用或通过 `run_pipeline.py` 调用

**修改文件：**

2. **`run_pipeline.py`**：
   - 新增 `run_stage6()` 函数：调用 `stage6_confirm.py`，加载其输出至状态文件
   - Stage 7 和 Stage 8 现优先从 `stage6.specification` 读取参数，向后兼容旧的手动 stage6 键
   - 主循环：原先打印"需用户检查"的 3 行代码替换为 `run_stage6()` 调用
   - 新增 `_resolve_data_path()` 辅助函数

3. **`bacon_decomp.py`**：
   - 新增 `import json`
   - 新增 `--output` 参数：结果可保存为 JSON（后续 Stage 6 的程序化消费需要）
   - JSON 输出包含 `negative_weight_pct`、`n_comparisons` 及完整的比较列表

**文档更新：**

4. **`SKILL.md`**：
   - Stage 6 章节：113 行叙述性假设检验指南替换为脚本使用文档、诊断覆盖表、JSON 模式及确定性语义
   - 流水线编排：描述从"需用户交互"更新为"自动化"
   - Stage 8 章节新增说明：当 Stage 6 确认依赖 CIA 的方法时，Oster 边界至为重要

### 架构说明

Stage 6 与 Stage 3 共享相同模式：
- **确定性**：相同输入 → 相同输出（诊断函数为纯规则或子进程调用）
- **结构化输出**：JSON 模式（`stage6_confirmation.json`）供下游消费
- **子进程隔离**：诊断通过调用 `/scripts/` 下的同级脚本运行，而非直接导入（故障隔离、无循环导入）
- **语言为主**：假设名称与 Stage 3 输出精确匹配，确保所有假设均有注册表条目对应

## 2026-06-04 — Stage 9 报告学术标准化

### 数据与渲染分离 + MD/XeLaTeX 双输出

Stage 9 拆为两步流水线：`output_report.py` 提取并结构化所有阶段数据 → `render_report.py` 渲染为学术格式。

**新增脚本：**

1. **`render_report.py`** — 双格式渲染引擎（约 400 行）：
   - `render_markdown(data) → str`：pipe 表格 Markdown，GitHub/VS Code 可直接预览，可通过 pandoc 转 PDF/Word
   - `render_latex(data) → str`：XeLaTeX 完整文档（`booktabs` 三线表 + `siunitx` 数字对齐 + `xeCJK` 中文支持 + `fontspec` 系统字体），`xelatex` 一键编译出 PDF
   - 共享格式化函数：`_stars()`/`_fmt_coef()`/`_fmt_se()`/`_fmt_pval()` 保证两种格式的数值渲染一致
   - 9 个标准章节：主回归表、假设检验表、事件研究表、方法降级链、稳健性检验表、数据质量、研究局限、警告、因果强度
   - 章节按条件渲染（无常出现数据时自动跳过）

**重构文件：**

2. **`output_report.py`** — 从文本生成器重构为结构化数据提取器：
   - 产出 `report_data.json`：所有渲染所需的键值（`method_chain`、`main_result`、`assumptions`、`event_study`、`fallback_attempts`、`robustness`、`data_quality`、`limitations`、`warnings`、`causal_claim_strength`）
   - `build_report_data()` 为核心入口函数
   - 保留 `generate_text_report()` 用于终端预览（`--text` 参数）
   - Stage 7 wrapper dict 解包 + key 搜索列表不变
   - 事件研究数据从 Stage 7 event_study 输出文件自动加载

**修改文件：**

3. **`run_pipeline.py`** — `run_stage9()` 两步执行：
   - Step 1：调用 `output_report.py` → `report_data.json`
   - Step 2：调用 `render_report.py` → `final_report.md` + `final_report.tex`
   - 输出文件追踪至 state：`report_data`、`report_md`、`report_tex`

4. **`SKILL.md`** Stage 9 章节：替换为架构说明 + 报告内容表 + XeLaTeX 编译说明
5. **`CHANGELOG.md`**：新增本条

## 2026-06-04 — Stage 9 报告撰写重构

### `output_report.py` 重写

旧版与改造后的 Stage 6 输出字段不兼容（读的是 `assumptions`/`holds`/`method_switch`，实际是 `assumption_verdicts`/`verdict`/`method_changed`），且 Stage 7 wrapper dict 未处理、Stage 8 checks 嵌套层级错误。重写后：

- **字段对齐**：所有 Stage 6 引用改为实际字段名。假设从 `assumption_verdicts` 读取，因果强度直接复用 Stage 6 的 `causal_claim_strength`（删除了重复的 `_assess_strength()` 函数）。
- **Stage 7 格式统一**：新增 `_unwrap_stage7()` 自动解包 wrapper dict（`callaway_santanna`/`2sls`/`liml`），扩展 key 搜索列表覆盖所有脚本的输出字段名（`coefficient`/`att`/`overall_att`/`late`/`ate`/`aggregate_att`）。SCM 无 SE 时从 `placebo_std` 推断。
- **Stage 8 整合**：`_normalize_stage8_checks()` 从 sensitivity 的 `summary.checks` 提取 checks，并自动将 placebo test 结果合并为独立的 check 条目。
- **报告内容丰富**：新增假设检验表（含诊断值、阈值、解读）、降级链展示、数据质量摘要、warnings 章节。
- **新增 `--stage8-placebo` CLI 参数**。

### `run_pipeline.py` 更新

- `run_stage9()` 修复：新增传递 `--stage6`（之前从未传递）、`--stage8-placebo`、`--data-source`

### 文档更新

- `SKILL.md` Stage 9 章节：ASCII 模板替换为实际脚本用法、报告各章节说明、格式差异处理机制
- `CHANGELOG.md`：新增本次条目

## 2026-06-03 (晚间 - 决策树完善)

### 新 Rule 3：工具变量自动路由

`classify_mechanism()` 新增 Rule 3：当 `q7.has_plausible_instrument == True` 时，自动路由到 `time_varying_unobservables`（IV/SCM 族），优先于 DID 族规则。IV 提供比平行趋势更强的识别力度。

同时：
- `has_instrument` 不再硬编码为 False，改为从 `q7` 读取
- `unobservables_risk` 标志：DID 族命中但无 IV 时，警告"识别完全依赖平行趋势"
- 所有 8 个分支函数的 fallback 列表都加入了 IV/2SLS 备选

### 混合机制检测

新增 `_detect_secondary_features()`：在 Level 1 分类完成后，扫描 q2-q7，收集存在但被主规则覆盖的特征（如阈值规则下的交错时间维度），转化为备选策略自动追加到 Level 2 输出。

- 8 个 `return` 统一通过 `_finalize_classification()` 包装，自动调用特征检测
- `_secondary_features_to_fallbacks()` 将次要特征转为 Fallback 对象
- 所有 8 个分支函数在返回前同步次要特征到 fallback 列表
- 主推荐方法不受影响（确定性保证），但备选列表自动扩充

示例：有阈值+交错时间+强度差异的政策，主推荐 Sharp RDD，备选自动包含 C&S staggered DID + Intensity DID + DDD + never-treated control。

### 新增 Synthetic DID

- 新脚本 `run_synthetic_did.py`：实现 Arkhangelsky et al. (2021)
  - 最优控制单元权重（约束优化）
  - 时间权重平衡处理前趋势
  - 安慰剂推断（逐单元施加处理，构建零分布）
  - 支持单或多处理单元
- `_branch_single_policy_shock` 无对照组路径：主推荐从 SCM 改为 Synthetic DID
- `run_pipeline.py` 新增 `run_synthetic_did.py` 路由
- `method_guide.md` 新增 Synthetic DID 条目；SKILL.md 新增 Stage 7 用法

## 2026-06-03 (下午 - 决策树重构)

### 双层决策树：从扁平 if-else 到真正的两级决策

**之前的架构**：`decide_method()` 是一个扁平的大函数，LLM 在 Stage 2 读完报告后手工把政策特征翻译成 CLI flag 传给 Stage 3。`mechanism` 和 `high_dimensional` 是两个独立入口，但产生相同推荐。

**现在的架构**：

```
Stage 2 (LLM)                        Stage 3 (确定性脚本)
─────────────                        ─────────────────────
叙事报告                               Level 1: classify_mechanism()
  ↓                                    ├── Rule 1: 随机分配？
结构化事实 JSON                           ├── Rule 2: 阈值规则？
  {                                     ├── Rule 3: 无时间维度？
    q1_assignment: ...                  ├── Rule 4: 多政策叠加？
    q2_threshold: ...                   ├── Rule 5: 连续强度？
    q3_timing: ...                      ├── Rule 6: 交错时间？
    ...                                 └── Rule 7: 单次冲击？
  }                                       ↓
  ↓                                    Level 2: decide_method()
  ────────────→ --from-facts ───────→   ├── _branch_threshold()
                                         ├── _branch_selection_on_observables()
                                         ├── _branch_single_policy_shock()
                                         ├── _branch_staggered_policy_shock()
                                         └── ... (8 个分支函数)
```

**关键改动**：

1. **Level 1 现在是代码而非人工判断**：`classify_mechanism()` 读入 Stage 2 的 7 个事实问题，用优先级规则链自动判定机制类型。7 条规则按识别力度排序（随机 > 阈值 > 无时间维度 > 多政策 > 连续强度 > 交错 > 单次）。

2. **Stage 2 输出结构化事实而非方法论标签**：新的 `stage2_facts.json` 模板包含 7 个问题，每个问题只要求观察政策设计（"是抽签决定的吗？""有分数线吗？""什么时候开始的？"），不要求方法论知识。

3. **删除 `high_dimensional` 伪机制**：高维控制变量不是分配机制，是数据特征。现在统一为 `selection_on_observables` 下的 `high_dimensional_controls` flag。

4. **两个 CLI 入口**：
   - `--from-facts stage2_facts.json` → 运行完整的 Level 1 + Level 2
   - `--mechanism X` → 只运行 Level 2（向后兼容，或 LLM 已知机制类型时使用）

5. **每个分支附带数据兼容性警告**：如果用户选了 `selection_on_observables` 但有面板数据，系统会提示"为什么不考虑 DID？"

6. **Stage 3 的 SKILL.md 文档大幅精简**：旧的 170 行手动决策树被移除，替换为两级架构说明和脚本用法。决策逻辑的权威来源是脚本代码本身。

## 2026-06-03 (下午 - ML 方法)

### 新增因果机器学习方法

**新增脚本：**

1. **`run_dml.py`** — Double/Debiased Machine Learning (Chernozhukov et al. 2018)
   - 双引擎：econml.LinearDML（首选）+ 手动 sklearn 实现（fallback）
   - 支持多种 ML 后端：RandomForest / GradientBoosting / Lasso / Linear
   - Neyman 正交得分 + K-fold cross-fitting 消除过拟合偏差
   - 有效推断：ML 模型不需要正确设定，只需 n^(-1/4) 收敛速度
   - 自动检测处理变量类型（二值 / 连续）
   - CATE 估计 + 分组异质性分析
   - 干扰模型 CV R² 诊断
   - 适用于高维控制变量场景（控制变量 > 15-20 个）

2. **`run_causal_forest.py`** — Causal Forest (Wager & Athey 2018, Athey et al. 2019)
   - 双引擎：econml.CausalForestDML + 手动 SimpleCausalForest
   - Honest estimation：一半样本建树，一半样本估计叶节点效应
   - CATE 分布：均值/SD/分位数/正效应占比
   - 变量重要性：哪些特征驱动处理效应异质性
   - Best Linear Projection：哪些变量系统性预测更大的处理效应
   - 分组 CATE：按特征四分位数/类别分组
   - 可视化：CATE 直方图 + 变量重要性 + BLP 系数图

### Stage 3 决策树更新

- 新增 `high_dimensional` assignment mechanism：高维控制变量下推荐 DML
- `selection_on_observables` 机制新增 `--high-dimensional-controls` 标志
- DML 作为 PSM/IPW 的首选替代方案（当控制变量多时）
- Causal Forest 作为各方法的 Stage 7+ 异质性分析推荐

### 文档更新

- `SKILL.md` Stage 7 新增 ML Methods 章节（DML + Causal Forest 用法）
- `method_guide.md` 新增 DML 和 Causal Forest 方法条目
- folder structure 新增两个脚本

## 2026-06-03 (上午 - 紧急修复和重要新增)

### 紧急修复：`run_staggered_did.py` 方法论重写

原脚本用简单 OLS + Entity/Time FE 近似 C&S 估计，存在三个方法论问题：
1. C&S (2021) 的核心贡献是**双重稳健估计**（倾向得分 IPW + 结果回归），不是简单 OLS
2. 标准误未考虑多阶段估计的不确定性
3. 未估计倾向得分，未做协变量调整

重写后的实现：
- **倾向得分估计**：sklearn LogisticRegression 估计 P(cohort_g | X)
- **双重稳健 ATT(g,t)**：IPW 加权 + 结果回归调整 + 影响函数标准误
- **Bootstrap 选项**：entity-level cluster bootstrap 用于置信区间
- **事件研究聚合**：按相对时间合并 cohort-specific ATT
- **Sun & Abraham (2021)**：交互加权估计，各 cohort 独立事件研究后加权平均

### 重要新增

**新增脚本：**

1. **`sensitivity_analysis.py`** — 五项敏感性检验：
   - Oster (2019) bounds：δ 值，不可观测因素需要多强才能解释掉处理效应
   - 系数稳定性：逐步加入控制变量，追踪系数变化
   - Rosenbaum bounds：Γ 值，匹配设计中隐藏偏差的敏感度
   - Placebo-in-time：将处理时间前移，检验是否有虚假"效应"
   - Leave-one-out influence：逐单位剔除，识别影响点

2. **`run_pipeline.py`** — 流水线编排器：
   - 自动检测当前阶段并继续执行
   - 支持从任意阶段重启（`--from-stage N`）
   - JSON 状态文件追踪所有阶段输出
   - Dry-run 模式预览命令
   - `--status` 查看完成进度

3. **`validate_data.py`** — 数据验证（Stage 5→6 之间）：
   - 面板结构检查（平衡性、重复行、时间间隔）
   - 缺失值分析（单变量 + 联合缺失 + 时间维度）
   - 异常值检测（IQR / z-score）
   - 变量类型和范围验证
   - 处理变量逻辑一致性（treated 不随时间变化、post 不在 treated 前出现等）
   - 处理前数据充分性（至少 2 期处理前结果数据）

### SKILL.md 更新

- Stage 5 新增数据验证步骤
- Stage 7 示例改为新的 DR 估计器
- Stage 8 新增敏感性分析（含 Oster/Rosenbaum/placebo-in-time）
- 文件夹结构中新增 4 个脚本
- 新增 Pipeline orchestration 章节

## 2026-05-29

### 流程重构：8 阶段 → 9 阶段

原来的方法选择（Stage 5）被拆成两个节点：

- **Stage 3: 理论方法分析** — 只看政策分配机制，不看数据可得性。回答"理论上应该用什么方法"
- **Stage 6: 最终方法确认** — 面对真实数据，检验理论方法的假设是否成立。不行就降级

核心原则：理论先于数据。先定基准方法，再记录妥协。

### Stage 3 决策树

从 7 种分配机制出发，覆盖 10 种因果识别方法：

| 分配机制 | 理论方法 |
|---|---|
| 随机分配 | Randomization inference |
| 阈值规则 (sharp) | Sharp RDD |
| 阈值规则 (fuzzy) | Fuzzy RDD |
| 可观测选择 | PSM / IPW / AIPW |
| 单次政策冲击 | 标准 DID |
| 交错政策冲击 | C&S / S&A (异质性稳健) |
| 时变不可观测 + 有 IV | 2SLS / LIML |
| 时变不可观测 + 无 IV | SCM / 交互固定效应 |
| 连续处理强度 | 强度 DID |
| 多政策叠加 | DDD |

`stage3_analyze.py` 将决策树编码为确定性脚本，保证跨会话一致性。

### 脚本

- 删除了 3 个数据抓取脚本（`fetch_wb.py`, `fetch_akshare.py`, `fetch_cn_stats.py`）
- 保留 11 个方法论脚本（决策树、清洗、DID、交错DID、事件研究、SCM、RDD、IV、安慰剂、Bacon分解、报告生成）
- 数据获取改为 LLM 现场写代码，不再用固定脚本

### 参考文件

- 新增 `references/data_sources.md`：数据源手册（含各 API 对应的 Python 包）
- 新增 `references/method_guide.md`：方法选择指南（含假设检验清单）

### 清理硬编码

- SKILL.md 和 data_sources.md 中不再出现 CFPS/CHARLS/CHFS 等特定数据源名称
- 改为通用表述（"微观调查数据""专有数据库"），适用于非中国、非家庭领域的政策评估
- data_sources.md 加了说明：所列数据源为中国示例，其他国家/领域会匹配对应数据源

### Stage 4 格式改进

- 输出从树状图改为结构化表格（Essential / Optional 两张表）
- 增加了确认步骤：必须等用户回复哪些数据可得，才进入 Stage 5
