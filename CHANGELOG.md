# 修改日志

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
