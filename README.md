# Policy Eval — 政策评估因果推断系统

端到端的政策评估因果推断管线：从政策研究、方法选择、数据获取、估计到学术报告生成，9 个阶段全自动化。

## 快速开始

```bash
# 一条命令启动全流程
python scripts/run_pipeline.py \
  --policy "长期护理保险试点" \
  --outcome "城市生育率" \
  --state my_analysis.json \
  --data data/merged/panel.dta
```

系统会自动完成 Stage 3-9。Stage 1-2（问题定义和政策研究）需要交互式输入，Stage 5（数据获取）需要手动准备数据。

## 9 阶段管线

```
Stage 1: 问题定义           →  什么政策？什么结果？
Stage 2: 政策研究           →  自动搜索政策细节、时间线、覆盖范围
Stage 3: 理论方法分析       →  根据分配机制确定性推荐识别策略
Stage 4: 数据需求           →  自动匹配变量到数据源
Stage 5: 数据获取           →  Tier A 自动拉取 / Tier B/C 生成模板
Stage 6: 方法确认           →  用实际数据检验所有可检验假设
Stage 7: 估计               →  运行选定的估计方法
Stage 8: 稳健性检验         →  安慰剂、替代窗口、留一法、敏感性分析
Stage 9: 结果报告           →  生成 MD + XeLaTeX PDF 学术论文
```

## 支持的识别方法

| 分配机制 | 方法 |
|---|---|
| 随机分配 | Randomization Inference |
| 断点/阈值规则 | Sharp / Fuzzy RDD |
| 工具变量 | IV (2SLS / LIML) |
| 选择依赖可观测变量 | DML / PSM / IPW / Causal Forest |
| 已知政策时点（单一） | Standard TWFE DID |
| 已知政策时点（交错） | Callaway & Sant'Anna (2021) Staggered DID |
| 连续处理强度 | Intensity DID |
| 多重政策重叠 | Triple Difference (DDD) |
| 无时间维度 | Synthetic Control / Synthetic DID |

## 输出

每次运行在桌面生成完整报告：

```
Desktop/policy_eval_output/<政策名>/<时间戳>/
├── markdown/<政策名>_report.md    # Markdown 格式
├── latex/<政策名>_report.tex      # XeLaTeX 源文件
└── latex/<政策名>_report.pdf      # 编译好的 PDF
```

报告包含：摘要、引言（研究背景+文献综述）、制度背景、理论机制、研究设计（数据来源+变量定义+描述性统计+识别策略）、实证结果（基准回归+事件研究+平行趋势检验）、稳健性检验（含图）、因果推断可信度评估、结论与政策建议、参考文献。

## 目录结构

```
policy-eval/
├── SKILL.md                 # 完整系统文档
├── README.md                # 本文件
├── scripts/                 # 所有脚本
│   ├── run_pipeline.py      # 管线编排器
│   ├── stage3_analyze.py    # 确定性方法决策引擎
│   ├── stage4_requirements.py
│   ├── stage6_confirm.py    # 方法确认引擎
│   ├── fetch_data.py        # 数据获取 (Tier A)
│   ├── run_staggered_did.py # C&S 交错 DID
│   ├── run_did.py           # 标准 TWFE DID
│   ├── run_rdd.py           # 断点回归
│   ├── run_iv.py            # 工具变量
│   ├── run_scm.py           # 合成控制法
│   ├── run_synthetic_did.py # 合成 DID
│   ├── run_dml.py           # 双/去偏机器学习
│   ├── run_causal_forest.py # 因果森林
│   ├── run_event_study.py   # 事件研究
│   ├── placebo_test.py      # 安慰剂置换检验
│   ├── bacon_decomp.py      # Goodman-Bacon 分解
│   ├── sensitivity_analysis.py
│   ├── output_report.py     # 报告数据提取
│   ├── render_report.py     # MD/XeLaTeX 渲染
│   └── validate_data.py     # 数据验证
├── references/
│   ├── variable_map.json    # 变量→数据源映射 (52+ 条目)
│   ├── method_guide.md      # 方法选择指南
│   └── data_sources.md      # 数据源文档
└── data/
    ├── auto/                # 自动生成的数据和阶段输出
    ├── manual/              # 用户填写的模板
    ├── raw/                 # 原始数据文件
    └── merged/              # 最终分析用面板数据
```

## 依赖

```bash
pip install pandas numpy statsmodels scipy matplotlib openpyxl
# 可选：世界银行数据
pip install wbgapi
# 可选：中国宏观数据
pip install akshare
# LaTeX 编译（需单独安装）
# MiKTeX 或 TeX Live，含 xelatex
```

## 许可

MIT
