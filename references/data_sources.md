# Data Source Reference

This guide lists common data sources for policy evaluation. Tier A sources have pre-written, tested fetch functions in `scripts/fetch_data.py`. The LLM does NOT write ad-hoc fetch code — it looks up variables in `references/variable_map.json` and calls the corresponding function.

For sources or variables not yet in the variable map, the LLM can either extend `fetch_data.py` with a new function, or escalate to Tier B/C.

**Note**: The sources listed below are China-focused examples. For policy evaluations in other countries or domains (trade, environment, health, education, etc.), add new entries to `variable_map.json` with the appropriate World Bank indicator codes or local API packages. This list is a starting point, not a constraint.

## Tier A — Public APIs (use fetch_data.py)

| Source | Coverage | Common variables | Access | Module |
|---|---|---|---|---|
| World Bank API | 1960–present, 200+ countries | GDP, fertility (TFR), population, trade, CO2 | Free, no key | `wbgapi` or `requests` |
| akshare | China city/province/national | GDP, population, CPI, PPI, fiscal revenue, housing | Free, no key | `akshare` |
| CNBS (国家统计局) | China province/city | GDP components, industrial output, retail sales | Free, via akshare | `akshare` |
| OECD API | OECD member countries | Employment, education, health expenditure | Free, API key optional | `pandas_datareader` |
| FRED (St. Louis Fed) | US and international | Interest rates, exchange rates, CPI, employment | Free, API key required | `pandas_datareader` |
| Wind (万得) | China macro, industry, firm | Comprehensive financial and macro data | Subscription | `WindPy` |
| CSMAR (国泰安) | China firm-level | Financial statements, governance, patents | Subscription | `requests` (API) |
| CEIC | Global + China macro | GDP, trade, industry by sector | Subscription | `requests` (API) |

## Tier B — Requires registration

| Source | Coverage | Variables | Access |
|---|---|---|---|
| CFPS (中国家庭追踪调查) | 2010–2020, 25 provinces, ~14,000 households | Income, education, health, fertility, migration | cfps.pku.edu.cn, application |
| CHARLS (中国健康与养老追踪调查) | 2011–2020, 45+ age group | Health, retirement, insurance, pensions | charls.pku.edu.cn, application |
| CHFS (中国家庭金融调查) | 2011–2019, ~40,000 households | Assets, debt, income, insurance | chfs.swufe.edu.cn, application |
| CMDS (中国流动人口动态监测) | Annual, 300,000+ migrants | Migration, employment, health, fertility | Request from 国家卫健委 |
| CNRDS (中国研究数据服务平台) | Various | Firm-level financial, patent, ESG | cnrds.com, subscription |

## Tier C — Manual collection

| Source | What to collect |
|---|---|
| Provincial/city statistical yearbooks | City-level GDP, population, fertility, fiscal |
| Policy documents (国务院/各部委) | Exact pilot city lists, dates, eligibility rules |
| Local government websites | Policy implementation details, subsidy amounts |
| 中国城市统计年鉴 | City-level employment, wages, education, health |

## Tier D — Typically inaccessible

- Individual-level tax records (unless collaborating with tax authority)
- Social security admin data (requires government partnership)
- Real-time economic indicators at city-day level

## Data format conventions

All fetched data saved to `data/auto/` as JSON:

```json
[
  {
    "entity_id": "110100",
    "year": 2020,
    "variable_name": 123.45
  }
]
```

Merged analysis-ready data saved to `data/merged/` as `.dta` or `.csv`.

## Common issues

- **City code mapping**: CNBS city codes change over time. Use 6-digit administrative codes with year suffix.
- **Province vs. city**: Fertility rates often only at province level. City estimates may require census microdata.
- **Pre-2010 data**: Chinese statistical yearbooks before 2010 have inconsistent variable definitions.
- **API rate limits**: World Bank allows ~50 requests/second. akshare may throttle on rapid queries — add `time.sleep(1)` between calls.
