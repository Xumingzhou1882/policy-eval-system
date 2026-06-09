"""
Data fetch module for policy evaluation — configuration-driven.

Architecture:
  - Generic engine `fetch_akshare(entry)` handles all simple akshare calls
    (function call → column rename → transform → standardize → output).
    Config lives in references/variable_map.json.
  - Custom functions only for non-trivial cases: World Bank API, city/province
    year-loop, yfinance, AQI city-loop.
  - All functions return standardized DataFrames.
  - Retry with exponential backoff for transient failures.
  - Fallback chains: try alternative akshare functions when primary fails.
  - Incremental save: year-loop functions save partial results per iteration.

Usage:
    from fetch_data import fetch_akshare, fetch_from_variable_map

    # Single akshare call driven by variable_map config
    entry = {"akshare": {"func": "stock_zh_index_daily", ...}}
    df = fetch_akshare(entry)

    # Batch from variable map
    results = fetch_from_variable_map(["gdp", "shanghai_index", "cny_usd"])
"""

import functools
import json
import re
import sys
import time
import traceback
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════

SKILL_ROOT = Path(__file__).resolve().parent.parent
DATA_AUTO = SKILL_ROOT / "data" / "auto"
DATA_RAW = SKILL_ROOT / "data" / "raw"
DATA_PARTIAL = SKILL_ROOT / "data" / "auto" / ".partial"


# ═══════════════════════════════════════════════════════════════════════
# Structured result type
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FetchResult:
    """Structured result from a fetch operation."""
    variable: str               # Variable name
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    path: Optional[str] = None  # Path to saved file
    status: str = "success"     # success | empty | error | partial | cached
    rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempts: int = 0           # Number of attempts (including retries)
    fallback_used: Optional[str] = None  # Name of fallback that succeeded
    source_label: str = ""      # Human-readable source (e.g., "世界银行API", "中国城市统计年鉴 via akshare")
    description: str = ""       # Variable description from variable_map.json
    tier: str = ""              # Data tier: A (public API), B (registration), C (manual), D (unavailable)

    def is_ok(self) -> bool:
        return self.status in ("success", "cached")

    def describe(self) -> str:
        parts = [f"{self.variable}: {self.status}"]
        if self.rows:
            parts.append(f"{self.rows} rows")
        if self.path:
            parts.append(str(self.path))
        if self.fallback_used:
            parts.append(f"(fallback: {self.fallback_used})")
        if self.attempts > 1:
            parts.append(f"({self.attempts} attempts)")
        return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Retry with exponential backoff
# ═══════════════════════════════════════════════════════════════════════

def with_retry(max_retries: int = 3, base_delay: float = 1.0,
               backoff: float = 2.0, max_delay: float = 30.0):
    """Decorator: retry a function with exponential backoff on exception."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = min(base_delay * (backoff ** attempt), max_delay)
                        print(f"    Retry {attempt+1}/{max_retries} in {delay:.1f}s: "
                              f"{type(e).__name__}: {str(e)[:120]}")
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator


def retry_call(func, *args, max_retries: int = 3, base_delay: float = 1.0,
               backoff: float = 2.0, max_delay: float = 30.0, **kwargs):
    """Call a function with exponential backoff retry. Returns (result, attempts)."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs), attempt + 1
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = min(base_delay * (backoff ** attempt), max_delay)
                print(f"    Retry {attempt+1}/{max_retries} in {delay:.1f}s: "
                      f"{type(e).__name__}: {str(e)[:120]}")
                time.sleep(delay)
    raise last_error


# ═══════════════════════════════════════════════════════════════════════
# Incremental save for year-loop functions
# ═══════════════════════════════════════════════════════════════════════

def _partial_path(variable_name: str) -> Path:
    """Path for incremental save checkpoint file."""
    DATA_PARTIAL.mkdir(parents=True, exist_ok=True)
    return DATA_PARTIAL / f"{variable_name}.json"


def _load_partial(variable_name: str) -> list[dict]:
    """Load previously saved partial rows."""
    pp = _partial_path(variable_name)
    if pp.exists():
        try:
            with open(pp, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_partial(variable_name: str, rows: list[dict]):
    """Save partial rows to checkpoint file."""
    pp = _partial_path(variable_name)
    with open(pp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


def _clear_partial(variable_name: str):
    """Remove partial checkpoint after successful full fetch."""
    pp = _partial_path(variable_name)
    if pp.exists():
        pp.unlink()


def _collect_partial_errors(variable_name: str) -> list[str]:
    """Return errors stored in partial state for this variable."""
    pp = _partial_path(variable_name)
    if not pp.exists():
        return []
    try:
        with open(pp, encoding="utf-8") as f:
            data = json.load(f)
        # Last entry may be an error marker
        if data and isinstance(data[-1], dict) and data[-1].get("_error"):
            return [data[-1]["_error"]]
    except (json.JSONDecodeError, OSError):
        pass
    return []


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def save_to_auto(df: pd.DataFrame, variable_name: str,
                 force: bool = False) -> str:
    """Save DataFrame to data/auto/{variable_name}.json. Skips if cached."""
    DATA_AUTO.mkdir(parents=True, exist_ok=True)
    output_path = DATA_AUTO / f"{variable_name}.json"

    if output_path.exists() and not force:
        print(f"  ✓ {variable_name}: already cached ({output_path})")
        return str(output_path)

    df.to_json(output_path, orient="records", force_ascii=False)
    action = "Overwrote" if output_path.exists() else "Saved"
    print(f"  {action} {len(df)} rows to {output_path}")
    return str(output_path)


def save_to_auto_with_meta(df: pd.DataFrame, variable_name: str,
                            force: bool = False,
                            meta: dict = None) -> str:
    """Save DataFrame with metadata (fetch timestamp, source version)."""
    DATA_AUTO.mkdir(parents=True, exist_ok=True)
    output_path = DATA_AUTO / f"{variable_name}.json"

    if output_path.exists() and not force:
        print(f"  ✓ {variable_name}: already cached ({output_path})")
        return str(output_path)

    # Write metadata alongside data
    meta = meta or {}
    meta["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["n_rows"] = len(df)
    meta["columns"] = list(df.columns)

    # Save data
    df.to_json(output_path, orient="records", force_ascii=False)

    # Save metadata
    meta_path = DATA_AUTO / f"{variable_name}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    action = "Overwrote" if output_path.exists() else "Saved"
    print(f"  {action} {len(df)} rows to {output_path}")
    return str(output_path)


def cache_age_hours(variable_name: str) -> Optional[float]:
    """Return age of cache in hours, or None if not cached."""
    meta_path = DATA_AUTO / f"{variable_name}_meta.json"
    if not meta_path.exists():
        # Check data file fallback
        data_path = DATA_AUTO / f"{variable_name}.json"
        if not data_path.exists():
            return None
        mtime = data_path.stat().st_mtime
    else:
        mtime = meta_path.stat().st_mtime
    return (time.time() - mtime) / 3600


def is_cached(variable_name: str, max_age_hours: float = None) -> bool:
    """Check if cached data exists, optionally within a max age."""
    data_path = DATA_AUTO / f"{variable_name}.json"
    if not data_path.exists():
        return False
    if max_age_hours is not None:
        age = cache_age_hours(variable_name)
        if age is not None and age > max_age_hours:
            return False
    return True


def cache_status(variable_names: list[str],
                 max_age_hours: float = None) -> dict[str, bool]:
    return {v: is_cached(v, max_age_hours) for v in variable_names}


def _check_akshare() -> bool:
    try:
        import akshare  # noqa
        return True
    except ImportError:
        print("  akshare not installed. Install: pip install akshare")
        return False


def _std_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize date column to YYYY-MM-DD string."""
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


# ═══════════════════════════════════════════════════════════════════════
# Transforms (applied by fetch_akshare when config specifies "transform")
# ═══════════════════════════════════════════════════════════════════════

def _transform_wide_to_long(df: pd.DataFrame, kwargs: dict) -> pd.DataFrame:
    """
    Convert wide-format table (columns = categories) to long format.
    Used for bond yields, money supply, Shibor, LPR, house prices.

    kwargs:
        value_cols: list of column names to melt
        entity_labels: list of entity_id labels (same order as value_cols)
        date_col: name of the date column
    """
    value_cols = kwargs["value_cols"]
    entity_labels = kwargs["entity_labels"]
    date_col = kwargs["date_col"]

    # Find date column
    if date_col not in df.columns:
        for alt in ["日期", "date", "Date", "月份"]:
            if alt in df.columns:
                date_col = alt
                break

    if date_col not in df.columns:
        date_col = df.columns[0]

    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    rows = []
    for label, col in zip(entity_labels, value_cols):
        if col in df.columns:
            for _, row in df.iterrows():
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(val):
                    rows.append({"entity_id": label, "date": row["date"],
                                 "value": float(val)})
        else:
            # Fuzzy match
            matches = [c for c in df.columns if col[:2] in c or col in c]
            for match in matches:
                for _, row in df.iterrows():
                    val = pd.to_numeric(row[match], errors="coerce")
                    if pd.notna(val):
                        rows.append({"entity_id": label, "date": row["date"],
                                     "value": float(val)})
                break  # only use first match

    return pd.DataFrame(rows).reset_index(drop=True) if rows else df


def _transform_exchange_rate(df: pd.DataFrame, kwargs: dict) -> pd.DataFrame:
    """
    Extract exchange rate from BOC Sina table.

    kwargs:
        value_col_candidates: ordered list of column names to try for the rate
        entity_id: entity ID label
    """
    entity_id = kwargs["entity_id"]
    candidates = kwargs.get("value_col_candidates", ["中间价", "中行折算价"])

    # Find date column
    date_col = None
    for c in ["日期", "date", "Date"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    # Find value column
    value_col = None
    for c in candidates:
        if c in df.columns:
            value_col = c
            break
    if value_col is None and len(df.columns) > 1:
        value_col = df.columns[1]

    if date_col and value_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
        df["value"] = pd.to_numeric(df[value_col], errors="coerce")
        df["entity_id"] = entity_id
        return df[["entity_id", "date", "value"]].dropna(subset=["value"]).reset_index(drop=True)
    return df


def _transform_pmi(df: pd.DataFrame, kwargs: dict) -> pd.DataFrame:
    """
    Extract manufacturing and non-manufacturing PMI.

    kwargs:
        date_col_candidates: ordered list for date column
        col_map: {column_name: entity_id} mapping
    """
    date_candidates = kwargs.get("date_col_candidates", ["日期", "date"])
    col_map = kwargs.get("col_map", {})

    date_col = None
    for c in date_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    rows = []
    for col_name, entity_id in col_map.items():
        if col_name in df.columns:
            for _, row in df.iterrows():
                v = pd.to_numeric(row[col_name], errors="coerce")
                if pd.notna(v):
                    rows.append({"entity_id": entity_id, "date": row["date"],
                                 "value": float(v)})
        else:
            # Fallback for PMI: second column = manufacturing
            fallback_col = df.columns[1] if len(df.columns) > 1 else None
            if fallback_col and entity_id == "pmi_manufacturing":
                for _, row in df.iterrows():
                    v = pd.to_numeric(row[fallback_col], errors="coerce")
                    if pd.notna(v):
                        rows.append({"entity_id": entity_id, "date": row["date"],
                                     "value": float(v)})

    return pd.DataFrame(rows).reset_index(drop=True) if rows else df


# Registry for dispatch
_TRANSFORMS = {
    "wide_to_long": _transform_wide_to_long,
    "exchange_rate": _transform_exchange_rate,
    "pmi": _transform_pmi,
}


# ═══════════════════════════════════════════════════════════════════════
# Source metadata — maps variable_map source keys to human-readable labels
# ═══════════════════════════════════════════════════════════════════════

_SOURCE_LABELS = {
    "world_bank": ("世界银行API (World Bank)", "A"),
    "akshare_city": ("中国城市统计年鉴 (via akshare)", "A"),
    "akshare_province": ("中国省级统计年鉴 (via akshare)", "A"),
    "akshare": ("akshare 公开数据", "A"),
    "akshare_stock_index": ("akshare 股票指数数据", "A"),
    "akshare_stock_individual": ("akshare 个股数据", "A"),
    "akshare_custom": ("akshare 数据", "A"),
    "yfinance": ("Yahoo Finance", "A"),
    "yfinance_global_index": ("Yahoo Finance 全球指数", "A"),
    "census": ("中国人口普查数据 (via akshare)", "A"),
    "nbs": ("国家统计局 API (data.stats.gov.cn)", "A"),
    "fred": ("FRED 美联储经济数据库", "A"),
    "oecd": ("OECD API", "A"),
    "eurostat": ("Eurostat API", "A"),
    "nyc_taxi": ("NYC TLC Trip Data (AWS Open Data)", "A"),
    "nyc_taxi_agg": ("NYC TLC Trip Data (AWS Open Data)", "A"),
    "citi_bike": ("Citi Bike NYC (AWS Open Data)", "A"),
    "epa_aqs": ("EPA AQS 美国空气质量", "A"),
    "noaa_gsod": ("NOAA GSOD 全球气象数据", "A"),
    "bls": ("BLS 美国劳工统计局", "A"),
    "zillow": ("Zillow Home Value Index", "A"),
    "nyc_311": ("NYC 311 公共服务请求", "A"),
    "chicago_crimes": ("Chicago Crime Data", "A"),
    "ipums": ("IPUMS USA 普查微观数据", "B"),
}


def _source_meta(source_key: str, description: str = "") -> tuple[str, str, str]:
    """Return (source_label, description, tier) for a given source key."""
    label, tier = _SOURCE_LABELS.get(source_key, (f"{source_key} 数据", "C"))
    return label, description, tier


def results_to_data_status(results: dict) -> dict:
    """Convert fetch results (dict[str, FetchResult]) to data_status dict
    for pipeline state tracking.

    Usage in Stage 5 (LLM calls this after fetching):
        results = fetch_from_variable_map(["gdp", "population", ...])
        data_status = results_to_data_status(results)
        # Then write data_status into the pipeline state JSON.
    """
    data_status = {}
    for var_name, r in results.items():
        entry = {
            "tier": r.tier or "未知",
            "status": r.status,
            "description": r.description or var_name,
            "source": r.source_label or "",
            "path": r.path or "",
        }
        if r.rows:
            entry["rows"] = r.rows
        data_status[var_name] = entry
    return data_status


# ═══════════════════════════════════════════════════════════════════════
# Generic akshare engine — one function to rule them all
# ═══════════════════════════════════════════════════════════════════════

def _resolve_akshare_func(func_name: str):
    """Resolve an akshare function by name. Returns (func, None) or (None, error_msg)."""
    import akshare as ak
    try:
        func = getattr(ak, func_name)
        return func, None
    except AttributeError:
        # Try fuzzy match — akshare sometimes renames functions
        candidates = [n for n in dir(ak) if func_name.lower() in n.lower()]
        if candidates:
            msg = (f"akshare has no '{func_name}'. "
                   f"Similar functions: {candidates[:5]}")
        else:
            msg = f"akshare has no '{func_name}' and no similar functions found"
        return None, msg


def _try_one_akshare_call(ak_func, func_name: str, cfg: dict,
                           kwargs: dict) -> pd.DataFrame:
    """Execute one akshare call with the given function and kwargs.
    Returns DataFrame or raises exception."""
    import akshare as ak

    # Check if function name changed (cfg may have an updated name)
    actual_func = ak_func
    actual_name = func_name

    try:
        df = actual_func(**kwargs)
    except TypeError as e:
        # Common akshare issue: parameter name changed
        # Try without keyword arguments if function signature changed
        if "unexpected keyword" in str(e) or "got an unexpected" in str(e):
            print(f"    akshare.{actual_name}: parameter mismatch — "
                  f"trying positional args")
            positional = list(kwargs.values())
            try:
                df = actual_func(*positional)
            except Exception:
                raise
        else:
            raise

    if df is None or (hasattr(df, "empty") and df.empty):
        raise ValueError(f"akshare.{actual_name} returned empty DataFrame")

    return df


def fetch_akshare(entry: dict,
                  start_date: str = None,
                  end_date: str = None,
                  indicator_override: str = None) -> pd.DataFrame:
    """
    Generic akshare data fetcher driven by variable_map.json config.

    Supports fallback chains: if the primary akshare config fails, tries
    alternative configs listed in "fallback" (list of akshare sub-configs).

    The entry dict has an "akshare" sub-dict with:
      - func: akshare function name (str)
      - kwargs: keyword arguments for the function (dict)
      - fallback: list of alternative akshare configs to try (optional)
      - rename: column rename mapping {old: new} (dict, optional)
      - entity_id: constant entity_id value (str, optional)
      - entity_id_from_kwarg: use a kwargs value as entity_id (str, optional)
      - transform: post-processing transform name (str, optional)
      - transform_kwargs: arguments for the transform (dict, optional)
      - output_cols: columns to include in output (list[str], optional)

    Parameters
    ----------
    entry : dict
        Variable map entry (the part after the variable name).
    start_date, end_date : str
        Date range filter in YYYY-MM-DD format.
    indicator_override : str
        If set, override the stock code / symbol in kwargs.

    Returns
    -------
    pd.DataFrame with standardized columns.
    """
    if not _check_akshare():
        return pd.DataFrame()

    import akshare as ak

    cfg = entry.get("akshare", {})
    fallbacks = cfg.get("fallback", [])

    errors: list[str] = []
    total_attempts = 0

    # ── Build config chain: primary + fallbacks ──────────────────────
    configs_to_try = [cfg]
    for fb in fallbacks:
        fb_cfg = dict(cfg)       # inherit primary config
        fb_cfg.update(fb)        # override with fallback specifics
        fb_cfg.pop("fallback", None)  # don't recurse
        configs_to_try.append(fb_cfg)

    # ── Try each config in order ─────────────────────────────────────
    for idx, try_cfg in enumerate(configs_to_try):
        func_name = try_cfg["func"]
        kwargs = dict(try_cfg.get("kwargs", {}))
        label = "primary" if idx == 0 else f"fallback #{idx}"

        # Override indicator if specified
        if indicator_override:
            for key in ["symbol", "code", "stock"]:
                if key in kwargs:
                    kwargs[key] = indicator_override
                    break

        # Resolve function
        ak_func, resolve_err = _resolve_akshare_func(func_name)
        if ak_func is None:
            errors.append(f"[{label}] {resolve_err}")
            continue

        # Retry with exponential backoff
        try:
            df, attempts = retry_call(
                _try_one_akshare_call, ak_func, func_name, try_cfg, kwargs,
                max_retries=3, base_delay=1.0, backoff=2.0)
            total_attempts += attempts
        except Exception as e:
            errors.append(f"[{label}] akshare.{func_name}: "
                          f"{type(e).__name__}: {str(e)[:200]}")
            continue

        if df.empty:
            errors.append(f"[{label}] akshare.{func_name}: empty result")
            continue

        # ── Success — process and return ──────────────────────────
        if idx > 0:
            print(f"  ⚠ Fallback #{idx} ({func_name}) succeeded where "
                  f"primary failed")

        # Rename columns
        rename = try_cfg.get("rename", {})
        if rename:
            df = df.rename(columns=rename)

        # Add entity_id
        entity_id = try_cfg.get("entity_id")
        entity_id_key = try_cfg.get("entity_id_from_kwarg")
        if entity_id_key and entity_id_key in kwargs:
            df["entity_id"] = str(kwargs[entity_id_key])
        elif entity_id_key and indicator_override:
            df["entity_id"] = str(indicator_override)
        elif entity_id:
            df["entity_id"] = entity_id

        # Apply transform
        transform_name = try_cfg.get("transform")
        if transform_name and transform_name in _TRANSFORMS:
            transform_kwargs = try_cfg.get("transform_kwargs", {})
            df = _TRANSFORMS[transform_name](df, transform_kwargs)

        # Standardize date
        df = _std_dates(df)

        # Filter date range
        if start_date and end_date and "date" in df.columns:
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            df = df[mask]

        # Select output columns
        out_cols = try_cfg.get("output_cols")
        if out_cols:
            out_cols = [c for c in out_cols if c in df.columns]
            df = df[out_cols]
        elif "value" in df.columns:
            out = ["entity_id"] if "entity_id" in df.columns else []
            out.append("date" if "date" in df.columns else "year")
            out.append("value")
            df = df[[c for c in out if c in df.columns]]

        return df.reset_index(drop=True)

    # All configs failed — print structured error summary
    print(f"  ✗ All {len(configs_to_try)} config(s) failed for "
          f"'{entry.get('description', 'unknown')}':")
    for err in errors:
        print(f"    {err}")

    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
# World Bank API
# ═══════════════════════════════════════════════════════════════════════

WB_INDICATORS = {
    "gdp_current_usd": "NY.GDP.MKTP.CD",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "population": "SP.POP.TOTL",
    "fertility_rate": "SP.DYN.TFRT.IN",
    "life_expectancy": "SP.DYN.LE00.IN",
    "infant_mortality": "SP.DYN.IMRT.IN",
    "urban_population": "SP.URB.TOTL",
    "urban_population_pct": "SP.URB.TOTL.IN.ZS",
    "trade_pct_gdp": "NE.TRD.GNFS.ZS",
    "fdi_pct_gdp": "BX.KLT.DINV.WD.GD.ZS",
    "co2_emissions": "EN.ATM.CO2E.KT",
    "health_expenditure_pct": "SH.XPD.CHEX.GD.ZS",
    "education_expenditure": "SE.XPD.TOTL.GD.ZS",
    "gini_index": "SI.POV.GINI",
    "unemployment_rate": "SL.UEM.TOTL.ZS",
    "inflation_cpi": "FP.CPI.TOTL.ZG",
    "exports_pct_gdp": "NE.EXP.GNFS.ZS",
    "imports_pct_gdp": "NE.IMP.GNFS.ZS",
    "school_enrollment_primary": "SE.PRM.ENRR",
    "school_enrollment_secondary": "SE.SEC.ENRR",
}

WB_COUNTRY_CODES = {
    "china": "CN", "india": "IN", "usa": "US", "indonesia": "ID",
    "brazil": "BR", "japan": "JP", "germany": "DE", "uk": "GB",
    "france": "FR", "russia": "RU", "south_africa": "ZA", "south_korea": "KR",
    "mexico": "MX", "turkey": "TR", "vietnam": "VN", "thailand": "TH",
    "world": "1W", "oecd": "OE", "eu": "EU",
}


def _resolve_wb_country(code_or_name: str) -> str:
    name_lower = code_or_name.lower().strip()
    if name_lower in WB_COUNTRY_CODES:
        return WB_COUNTRY_CODES[name_lower]
    if len(code_or_name) == 2 and code_or_name == code_or_name.upper():
        return code_or_name
    for name, code in WB_COUNTRY_CODES.items():
        if name in name_lower or name_lower in name:
            return code
    return code_or_name.upper()


def fetch_wb_indicator(indicator: str, countries: list[str] = None,
                       start_year: int = 2000, end_year: int = 2025) -> pd.DataFrame:
    """Fetch a World Bank indicator. Returns [entity_id, year, value]."""
    indicator_code = WB_INDICATORS.get(indicator, indicator)
    if countries is None:
        countries = ["CN", "US", "IN"]
    country_codes = [_resolve_wb_country(c) for c in countries]

    # Try wbgapi first
    try:
        import wbgapi as wb
        data = wb.data.DataFrame(series=indicator_code, economy=country_codes,
                                 time=range(start_year, end_year + 1))
        rows = []
        for entity in data.index:
            for yr in data.columns:
                val = data.loc[entity, yr]
                if pd.notna(val):
                    rows.append({"entity_id": str(entity),
                                 "year": int(yr[2:]) if yr.startswith("YR") else int(yr),
                                 "value": float(val)})
        if rows:
            df = pd.DataFrame(rows)
            df["year"] = df["year"].astype(int)
            return df
    except Exception as e:
        print(f"  wbgapi error: {e} — trying HTTP fallback")

    # HTTP fallback
    import urllib.request
    country_str = ";".join(country_codes)
    url = (f"https://api.worldbank.org/v2/country/{country_str}"
           f"/indicator/{indicator_code}?format=json&per_page=5000"
           f"&date={start_year}:{end_year}")
    rows = []
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read())
        if raw and len(raw) > 1 and raw[1]:
            for entry in raw[1]:
                if entry.get("value") is not None:
                    rows.append({"entity_id": entry["country"]["id"],
                                 "year": int(entry["year"]),
                                 "value": float(entry["value"])})
    except Exception as e:
        print(f"  World Bank API error: {e}")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# China city / province macro (year-loop — non-trivial)
# ═══════════════════════════════════════════════════════════════════════

def _try_yearbook_fallback(indicator: str, year: int) -> Optional[pd.DataFrame]:
    """
    Attempt to get data from statistical yearbook as last-resort fallback.

    Returns DataFrame on success, None on failure.
    """
    try:
        from extract_yearbook import extract_yearbook_table
        return extract_yearbook_table(indicator, year)
    except ImportError:
        pass
    except Exception as e:
        print(f"    yearbook fallback error: {type(e).__name__}: "
              f"{str(e)[:100]}")
    return None


def fetch_cn_city_macro(indicator: str = "gdp",
                        start_year: int = 2010, end_year: int = 2025,
                        incremental: bool = True) -> pd.DataFrame:
    """
    Fetch China city-level macro data with incremental save and fallback.

    Tries multiple akshare functions in order per year, saves partial
    results after each successful year so partial data survives failures.

    Returns [entity_id, year, value].
    """
    if not _check_akshare():
        return pd.DataFrame()
    import akshare as ak

    indicator_map = {
        # GDP & economic output
        "gdp": "地区生产总值", "gdp_per_capita": "人均地区生产总值",
        "gdp_growth": "地区生产总值增长率",
        "primary_industry": "第一产业增加值",
        "secondary_industry": "第二产业增加值",
        "tertiary_industry": "第三产业增加值",
        "industrial_output": "规模以上工业总产值",
        "industrial_output_above_designated": "规模以上工业增加值",
        # Population & vital statistics
        "population": "年末总人口",
        "population_registered": "户籍人口",
        "birth_rate": "出生率",
        "death_rate": "死亡率",
        "natural_growth_rate": "自然增长率",
        "urbanization_rate": "城镇化率",
        "employment": "城镇从业人员年平均人数",
        "employment_private": "城镇私营和个体从业人员",
        "avg_wage": "职工平均工资",
        "avg_wage_private": "私营单位从业人员平均工资",
        "unemployment_registered": "城镇登记失业人数",
        # Fiscal
        "fiscal_revenue": "地方财政一般预算收入",
        "fiscal_expenditure": "地方财政一般预算支出",
        "tax_revenue": "税收收入",
        # Household income & consumption
        "urban_disposable_income": "城镇居民人均可支配收入",
        "rural_disposable_income": "农村居民人均可支配收入",
        "urban_consumption": "城镇居民人均消费支出",
        "rural_consumption": "农村居民人均消费支出",
        "retail_sales": "社会消费品零售总额",
        "savings": "城乡居民储蓄存款余额",
        "savings_per_capita": "人均储蓄存款余额",
        # Investment & trade
        "fixed_investment": "固定资产投资总额",
        "real_estate_investment": "房地产开发投资额",
        "foreign_trade": "进出口总额",
        "exports": "出口总额",
        "imports": "进口总额",
        "fdi_actual": "实际利用外资",
        # Infrastructure
        "road_area": "年末实有城市道路面积",
        "road_area_per_capita": "人均城市道路面积",
        "built_up_area": "建成区面积",
        "green_area": "绿地面积",
        "green_coverage_rate": "建成区绿化覆盖率",
        "park_green_area": "公园绿地面积",
        "water_supply": "供水总量",
        "gas_supply": "供气总量",
        "electricity_consumption": "全社会用电量",
        "residential_electricity": "城乡居民生活用电",
        "public_transport_vehicles": "年末公共交通车辆运营数",
        "public_transport_passengers": "公共汽电车客运总量",
        "taxi_count": "年末实有出租汽车数",
        # Education
        "education_expenditure": "教育支出",
        "primary_students": "小学在校学生数",
        "secondary_students": "普通中学在校学生数",
        "college_students": "普通高等学校在校学生数",
        "teachers_primary": "小学专任教师数",
        "teachers_secondary": "普通中学专任教师数",
        "teachers_college": "普通高等学校专任教师数",
        "library_collections": "公共图书馆图书总藏量",
        # Health
        "hospital_beds": "医院卫生院床位数",
        "hospital_beds_per_1000": "每千人口医院卫生院床位数",
        "health_institutions": "卫生机构数",
        "doctors": "执业助理医师数",
        "doctors_per_1000": "每千人口执业助理医师数",
        "nurses": "注册护士数",
        # Social security
        "pension_insurance": "基本养老保险参保人数",
        "medical_insurance": "基本医疗保险参保人数",
        "unemployment_insurance": "失业保险参保人数",
        "social_welfare_homes": "社会福利院数",
        "social_welfare_beds": "社会福利院床位数",
        # Environment
        "wastewater_discharge": "工业废水排放量",
        "so2_emission": "工业二氧化硫排放量",
        "sewage_treatment_rate": "污水处理率",
        "waste_treatment_rate": "生活垃圾无害化处理率",
        "air_quality_good_days": "空气质量优良天数",
        # Innovation
        "patents": "专利授权数",
        "patents_invention": "发明专利授权数",
        "rd_expenditure": "科学技术支出",
        "rd_expenditure_pct_gdp": "研发支出占GDP比重",
        # Communications
        "mobile_phone_users": "移动电话年末用户数",
        "internet_users": "互联网宽带接入用户数",
        "postal_services": "邮政业务总量",
        "telecom_services": "电信业务总量",
        # Car ownership
        "car_ownership": "民用汽车拥有量",
        "car_ownership_private": "私人汽车拥有量",
    }
    cn_name = indicator_map.get(indicator, indicator)

    # Resume from incremental save
    partial_rows = _load_partial(indicator) if incremental else []
    completed_years: set[int] = set()
    if partial_rows:
        completed_years = set(r["year"] for r in partial_rows)
        print(f"  Resuming {indicator}: {len(partial_rows)} partial rows "
              f"({len(completed_years)} years already fetched)")

    errors_by_year: dict[int, list[str]] = {}

    # Function chains to try per year (ordered: most reliable first)
    fn_chains = [
        # Chain 1: city public finance (broadest coverage)
        ["macro_china_city_public_finance"],
        # Chain 2: city GDP specific
        ["macro_china_city_gdp"],
        # Chain 3: newer akshare functions (try variations)
        ["macro_china_city_shoot"],
    ]

    years = [y for y in range(start_year, end_year + 1)
             if y not in completed_years]

    for year in years:
        year_success = False
        year_errors = []

        for chain in fn_chains:
            for fn_name in chain:
                try:
                    func = getattr(ak, fn_name, None)
                    if func is None:
                        year_errors.append(f"{fn_name}: not found in akshare")
                        continue
                    df_raw, _ = retry_call(func, year=year, max_retries=2,
                                           base_delay=0.5)
                except Exception as e:
                    year_errors.append(f"{fn_name}: {type(e).__name__}: {str(e)[:120]}")
                    continue

                if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
                    year_errors.append(f"{fn_name}: empty result for {year}")
                    continue

                # Detect columns
                city_col = None
                for c in ["城市", "地区", "城市名称", "地区名称", "city"]:
                    if c in df_raw.columns:
                        city_col = c
                        break
                if city_col is None:
                    city_col = df_raw.columns[0]

                # Find value column
                value_col = cn_name
                if cn_name not in df_raw.columns:
                    matches = [c for c in df_raw.columns
                               if indicator in c.lower() or cn_name[:2] in c]
                    if matches:
                        value_col = matches[0]
                    else:
                        # Try second numeric column
                        num_cols = [c for c in df_raw.columns[1:]
                                   if pd.api.types.is_numeric_dtype(df_raw[c])]
                        if num_cols:
                            value_col = num_cols[0]
                        else:
                            year_errors.append(
                                f"{fn_name}: cannot find value column "
                                f"'{cn_name}' in {list(df_raw.columns)[:8]}")
                            continue

                # Extract rows
                year_rows = []
                for _, row in df_raw.iterrows():
                    val = pd.to_numeric(row[value_col], errors="coerce")
                    if pd.notna(val) and val > 0:
                        year_rows.append({
                            "entity_id": str(row[city_col]),
                            "year": year,
                            "value": float(val),
                        })

                if year_rows:
                    partial_rows.extend(year_rows)
                    completed_years.add(year)
                    if incremental:
                        _save_partial(indicator, partial_rows)
                    year_success = True
                    break  # chain succeeded

            if year_success:
                break

        if not year_success:
            # ── Last-resort fallback: statistical yearbook PDF/HTML ──
            df_yb = _try_yearbook_fallback(indicator, year)
            if df_yb is not None and not df_yb.empty:
                yb_rows = df_yb.to_dict(orient="records")
                partial_rows.extend(yb_rows)
                completed_years.add(year)
                if incremental:
                    _save_partial(indicator, partial_rows)
                year_success = True

        if not year_success:
            errors_by_year[year] = year_errors
            # Save error marker in partial so resume shows gap
            if incremental:
                partial_rows.append({
                    "_error": f"Year {year} failed: {'; '.join(year_errors[:3])}",
                    "_year": year,
                })
                _save_partial(indicator, partial_rows)

        time.sleep(0.5)  # Rate limit

    # Report
    n_years = len(completed_years)
    n_total = end_year - start_year + 1
    if errors_by_year:
        failed_years = sorted(errors_by_year.keys())
        print(f"  ⚠ {indicator}: {n_years}/{n_total} years fetched. "
              f"Failed: {failed_years}")
        for fy in failed_years[:5]:
            print(f"    Year {fy}: {'; '.join(errors_by_year[fy][:2])}")
    else:
        print(f"  ✓ {indicator}: {n_years}/{n_total} years fetched")

    if not partial_rows:
        return pd.DataFrame()

    # Filter out error markers
    valid_rows = [r for r in partial_rows if "_error" not in r]
    df = pd.DataFrame(valid_rows)

    if df.empty:
        return df

    # Clean up partial on success
    if incremental and not errors_by_year:
        _clear_partial(indicator)

    return df


def fetch_cn_province_macro(indicator: str = "gdp",
                            start_year: int = 2010, end_year: int = 2025,
                            incremental: bool = True) -> pd.DataFrame:
    """
    Fetch China province-level macro data with incremental save.

    Uses multiple akshare functions per year to maximize coverage.
    Returns [entity_id, year, value].
    """
    if not _check_akshare():
        return pd.DataFrame()
    import akshare as ak

    indicator_map = {
        "gdp": "地区生产总值", "gdp_per_capita": "人均地区生产总值",
        "gdp_growth": "地区生产总值增长率",
        "primary_industry": "第一产业增加值",
        "secondary_industry": "第二产业增加值",
        "tertiary_industry": "第三产业增加值",
        "population": "年末常住人口",
        "population_registered": "户籍人口",
        "urbanization_rate": "城镇化率",
        "birth_rate": "出生率", "death_rate": "死亡率",
        "natural_growth_rate": "自然增长率",
        "employment": "从业人员年末人数",
        "avg_wage": "城镇单位就业人员平均工资",
        "urban_disposable_income": "城镇居民人均可支配收入",
        "rural_disposable_income": "农村居民人均可支配收入",
        "urban_consumption": "城镇居民人均消费支出",
        "rural_consumption": "农村居民人均消费支出",
        "fiscal_revenue": "地方一般公共预算收入",
        "fiscal_expenditure": "地方一般公共预算支出",
        "tax_revenue": "税收收入",
        "retail_sales": "社会消费品零售总额",
        "fixed_investment": "固定资产投资总额",
        "real_estate_investment": "房地产开发投资额",
        "foreign_trade": "进出口总额",
        "exports": "出口总额", "imports": "进口总额",
        "fdi_actual": "实际利用外资",
        "cpi": "居民消费价格指数",
        "hospital_beds": "医疗卫生机构床位数",
        "doctors": "执业(助理)医师数",
        "college_students": "普通高等学校在校学生数",
        "patents": "专利授权数",
        "highway_mileage": "公路里程",
        "power_generation": "发电量",
        "steel_output": "粗钢产量",
        "cement_output": "水泥产量",
    }
    cn_name = indicator_map.get(indicator, indicator)

    # Resume from incremental save
    partial_rows = _load_partial(indicator) if incremental else []
    completed_years: set[int] = set()
    if partial_rows:
        completed_years = set(r["year"] for r in partial_rows)
        print(f"  Resuming {indicator}: {len(partial_rows)} partial rows "
              f"({len(completed_years)} years already fetched)")

    errors_by_year: dict[int, list[str]] = {}
    years = [y for y in range(start_year, end_year + 1)
             if y not in completed_years]

    # Function chains ordered by coverage breadth
    fn_chains = [
        ["macro_china_province_gdp"],
        ["macro_china_province_pop"],
    ]

    for year in years:
        year_success = False
        year_errors = []

        for chain in fn_chains:
            for fn_name in chain:
                try:
                    func = getattr(ak, fn_name, None)
                    if func is None:
                        year_errors.append(f"{fn_name}: not found in akshare")
                        continue
                    df_raw, _ = retry_call(func, year=year, max_retries=2,
                                           base_delay=0.5)
                except Exception as e:
                    year_errors.append(f"{fn_name}: {type(e).__name__}: {str(e)[:120]}")
                    continue

                if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
                    year_errors.append(f"{fn_name}: empty result for {year}")
                    continue

                # Detect province column
                province_col = None
                for c in ["地区", "省份", "地区名称", "省份名称", "province"]:
                    if c in df_raw.columns:
                        province_col = c
                        break
                if province_col is None:
                    province_col = df_raw.columns[0]

                # Find value column using indicator_map first, then fuzzy
                value_col = cn_name
                if cn_name not in df_raw.columns:
                    matches = [c for c in df_raw.columns
                               if indicator in c.lower() or cn_name[:2] in c]
                    if matches:
                        value_col = matches[0]
                    else:
                        num_cols = [c for c in df_raw.columns[1:]
                                   if pd.api.types.is_numeric_dtype(df_raw[c])]
                        if num_cols:
                            value_col = num_cols[0]
                        else:
                            year_errors.append(
                                f"{fn_name}: cannot find value column "
                                f"'{cn_name}' in {list(df_raw.columns)[:8]}")
                            continue

                # Extract rows
                year_rows = []
                for _, row in df_raw.iterrows():
                    val = pd.to_numeric(row[value_col], errors="coerce")
                    if pd.notna(val):
                        year_rows.append({
                            "entity_id": str(row[province_col]),
                            "year": year,
                            "value": float(val),
                        })

                if year_rows:
                    partial_rows.extend(year_rows)
                    completed_years.add(year)
                    if incremental:
                        _save_partial(indicator, partial_rows)
                    year_success = True
                    break

            if year_success:
                break

        # Yearbook fallback
        if not year_success:
            df_yb = _try_yearbook_fallback(indicator, year)
            if df_yb is not None and not df_yb.empty:
                yb_rows = df_yb.to_dict(orient="records")
                partial_rows.extend(yb_rows)
                completed_years.add(year)
                if incremental:
                    _save_partial(indicator, partial_rows)
                year_success = True

        if not year_success:
            errors_by_year[year] = year_errors
            if incremental:
                partial_rows.append({
                    "_error": f"Year {year} failed: {'; '.join(year_errors[:3])}",
                    "_year": year,
                })
                _save_partial(indicator, partial_rows)

        time.sleep(0.5)

    n_years = len(completed_years)
    n_total = end_year - start_year + 1
    if errors_by_year:
        failed_years = sorted(errors_by_year.keys())
        print(f"  ⚠ {indicator}: {n_years}/{n_total} years fetched. "
              f"Failed: {failed_years}")
    else:
        print(f"  ✓ {indicator}: {n_years}/{n_total} years fetched")

    valid_rows = [r for r in partial_rows if "_error" not in r]
    df = pd.DataFrame(valid_rows)

    if df.empty:
        return df

    if incremental and not errors_by_year:
        _clear_partial(indicator)

    return df


# ═══════════════════════════════════════════════════════════════════════
# AQI — city-level air quality (city-loop)
# ═══════════════════════════════════════════════════════════════════════

def fetch_cn_aqi(city: str = "北京",
                 start_date: str = "2015-01-01",
                 end_date: str = "2025-12-31") -> pd.DataFrame:
    """
    Fetch daily AQI data for a Chinese city.

    Parameters
    ----------
    city : str
        City name in Chinese (e.g., "北京", "上海").
    start_date, end_date : str
        Date range in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame with columns [entity_id, date, aqi, pm25, pm10, so2, no2, co, o3]
    """
    if not _check_akshare():
        return pd.DataFrame()
    import akshare as ak

    try:
        df = ak.air_city_hist(city=city, start_date=start_date,
                              end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()

        col_map = {
            "日期": "date", "AQI": "aqi", "PM2.5": "pm25", "PM10": "pm10",
            "SO2": "so2", "NO2": "no2", "CO": "co", "O3": "o3",
        }
        df = df.rename(columns={k: v for k, v in col_map.items()
                                 if k in df.columns})
        df["entity_id"] = city
        df = _std_dates(df)

        out_cols = ["entity_id", "date"]
        out_cols += [c for c in ["aqi", "pm25", "pm10", "so2", "no2", "co", "o3"]
                     if c in df.columns]
        return df[out_cols].reset_index(drop=True)
    except Exception as e:
        print(f"  akshare AQI error: {e}")
        return pd.DataFrame()


def fetch_cn_aqi_multi(cities: list[str],
                       start_date: str = "2015-01-01",
                       end_date: str = "2025-12-31",
                       max_workers: int = 8) -> pd.DataFrame:
    """Fetch AQI for multiple cities in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    frames = []
    n = len(cities)

    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as executor:
        future_to_city = {
            executor.submit(fetch_cn_aqi, city, start_date, end_date): city
            for city in cities
        }
        done = 0
        for future in as_completed(future_to_city):
            city = future_to_city[future]
            done += 1
            try:
                df = future.result()
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                print(f"  [{done}/{n}] AQI: {city} — {type(e).__name__}: {str(e)[:100]}")
            if done % 20 == 0 or done == n:
                print(f"  AQI progress: {done}/{n} cities")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
# yfinance — global stock indices
# ═══════════════════════════════════════════════════════════════════════

GLOBAL_STOCK_INDICES = {
    "sp500": "^GSPC", "nasdaq": "^IXIC", "dow_jones": "^DJI",
    "ftse100": "^FTSE", "nikkei225": "^N225", "hsi": "^HSI",
    "dax": "^GDAXI", "kospi": "^KS11", "asx200": "^AXJO",
}


def fetch_global_stock_index(index_name: str = "sp500",
                             start_date: str = "2010-01-01",
                             end_date: str = "2025-12-31") -> pd.DataFrame:
    """Fetch daily global stock index via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed. Install: pip install yfinance")
        return pd.DataFrame()

    ticker = GLOBAL_STOCK_INDICES.get(index_name, index_name)
    try:
        data = yf.download(ticker, start=start_date, end=end_date,
                          progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0].lower() for c in data.columns]
        else:
            data.columns = [c.lower() for c in data.columns]
        data = data.reset_index()
        date_col = "date" if "date" in data.columns else "Date"
        data["date"] = pd.to_datetime(data[date_col]).dt.strftime("%Y-%m-%d")
        data["entity_id"] = index_name

        out_cols = ["entity_id", "date"]
        out_cols += [c for c in ["open", "high", "low", "close", "volume"]
                     if c in data.columns]
        return data[out_cols].reset_index(drop=True)
    except Exception as e:
        print(f"  yfinance error: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
# Batch fetch from variable map
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# China census data (七普/六普 city-level)
# ═══════════════════════════════════════════════════════════════════════

CENSUS_INDICATORS = {
    "total_population": "常住人口",
    "male_population": "男性人口",
    "female_population": "女性人口",
    "sex_ratio": "性别比",
    "age_0_14_pct": "0-14岁占比",
    "age_15_59_pct": "15-59岁占比",
    "age_60_plus_pct": "60岁以上占比",
    "age_65_plus_pct": "65岁以上占比",
    "urban_population": "城镇人口",
    "urbanization_rate": "城镇化率",
    "avg_household_size": "平均家庭户规模",
    "college_edu_pct": "大专及以上占比",
    "illiteracy_rate": "文盲率",
    "avg_education_years": "平均受教育年限",
    "birth_rate": "出生率",
    "death_rate": "死亡率",
    "natural_growth_rate": "自然增长率",
    "migrant_population": "流动人口",
}


def fetch_cn_census(indicator: str = "total_population",
                    census_year: int = 2020,
                    level: str = "city") -> pd.DataFrame:
    """
    Fetch China census data at city or province level.

    Uses akshare census-related functions. The 7th census (2020) and
    6th census (2010) have the best city-level coverage.

    Parameters
    ----------
    indicator : str
        Census indicator key (see CENSUS_INDICATORS).
    census_year : int
        Census year: 2020 (七普) or 2010 (六普).
    level : str
        Geographic level: "city", "province", or "nation".

    Returns
    -------
    pd.DataFrame with [entity_id, year, value].
    """
    if not _check_akshare():
        return pd.DataFrame()
    import akshare as ak

    cn_name = CENSUS_INDICATORS.get(indicator, indicator)
    rows = []

    # Try multiple akshare census functions
    fn_chain = [
        "macro_china_population_census",
        "macro_china_city_population",
        "macro_china_pop_census",
    ]

    for fn_name in fn_chain:
        func = getattr(ak, fn_name, None)
        if func is None:
            continue
        try:
            df_raw, _ = retry_call(func, max_retries=2, base_delay=0.5)
        except Exception:
            continue

        if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
            continue

        # Detect columns
        area_col = None
        for c in ["地区", "城市", "省份", "地区名称", "城市名称"]:
            if c in df_raw.columns:
                area_col = c
                break
        if area_col is None:
            area_col = df_raw.columns[0]

        # Find indicator column
        value_col = None
        if cn_name in df_raw.columns:
            value_col = cn_name
        else:
            matches = [c for c in df_raw.columns
                       if indicator.replace("_", "") in c.replace(" ", "")
                       or cn_name[:2] in c]
            if matches:
                value_col = matches[0]
            else:
                # Try numeric columns
                for c in df_raw.columns[1:]:
                    if pd.api.types.is_numeric_dtype(df_raw[c]):
                        value_col = c
                        break

        if value_col:
            for _, row in df_raw.iterrows():
                val = pd.to_numeric(row[value_col], errors="coerce")
                if pd.notna(val) and val > 0:
                    rows.append({
                        "entity_id": str(row[area_col]),
                        "year": census_year,
                        "value": float(val),
                    })
            break  # success — don't try other functions

    if rows:
        print(f"  ✓ census {indicator} ({census_year}): {len(rows)} rows "
              f"via {fn_name}")
    else:
        print(f"  ⚠ census {indicator} ({census_year}): no data found. "
              f"Census microdata at {level} level may require Tier B access "
              f"(application at stats.gov.cn).")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# National Bureau of Statistics API (data.stats.gov.cn)
# ═══════════════════════════════════════════════════════════════════════

NBS_DBCODE = {
    "national_annual": "fsnd",     # 国家年度数据
    "national_quarterly": "fsnd",  # 国家季度数据
    "national_monthly": "fsnd",    # 国家月度数据
    "province_annual": "fsyd",     # 分省年度数据
    "province_quarterly": "fsyd",  # 分省季度数据
    "city_annual": "csyd",         # 城市年度数据
    "city_monthly": "csyd",        # 城市月度数据
}

NBS_INDICATORS = {
    "gdp": {"wbcode": "zb", "code": "A0201", "name": "地区生产总值"},
    "gdp_per_capita": {"wbcode": "zb", "code": "A0202", "name": "人均地区生产总值"},
    "population": {"wbcode": "zb", "code": "A0301", "name": "年末常住人口"},
    "urban_population": {"wbcode": "zb", "code": "A0302", "name": "城镇人口"},
    "rural_population": {"wbcode": "zb", "code": "A0303", "name": "乡村人口"},
    "employment": {"wbcode": "zb", "code": "A0401", "name": "就业人员"},
    "avg_wage": {"wbcode": "zb", "code": "A0406", "name": "城镇单位就业人员平均工资"},
    "cpi": {"wbcode": "zb", "code": "A0901", "name": "居民消费价格指数"},
    "fiscal_revenue": {"wbcode": "zb", "code": "A0801", "name": "一般公共预算收入"},
    "fiscal_expenditure": {"wbcode": "zb", "code": "A0802", "name": "一般公共预算支出"},
    "retail_sales": {"wbcode": "zb", "code": "A0701", "name": "社会消费品零售总额"},
    "fixed_investment": {"wbcode": "zb", "code": "A0601", "name": "固定资产投资"},
    "import_export": {"wbcode": "zb", "code": "A0706", "name": "进出口总额"},
    "hospital_beds": {"wbcode": "zb", "code": "A0G01", "name": "医疗卫生机构床位数"},
    "college_students": {"wbcode": "zb", "code": "A0M02", "name": "普通高等学校在校学生数"},
}


def fetch_nbs_data(indicator: str = "gdp",
                   region_codes: list[str] = None,
                   start_year: int = 2010, end_year: int = 2025,
                   level: str = "city") -> pd.DataFrame:
    """
    Fetch data from National Bureau of Statistics API.

    Uses the official data.stats.gov.cn easyquery API. More stable and
    authoritative than akshare, but the API response format is complex.

    Parameters
    ----------
    indicator : str
        Indicator key (see NBS_INDICATORS).
    region_codes : list[str]
        GB/T 2260 region codes (e.g., ["110000", "310000"]).
        Defaults to all prefecture-level cities.
    level : str
        "nation", "province", or "city".
    start_year, end_year : int
        Year range.

    Returns
    -------
    pd.DataFrame with [entity_id, year, value].
    """
    import urllib.request

    ind_info = NBS_INDICATORS.get(indicator, {})
    if not ind_info:
        print(f"  NBS indicator '{indicator}' not found in NBS_INDICATORS")
        return pd.DataFrame()

    dbcode = NBS_DBCODE.get(f"{level}_annual", "csyd")

    # Build region filter
    if region_codes is None:
        # Default: all regions at the specified level
        region_codes = []
    region_filter = ",".join(region_codes) if region_codes else ""

    rows = []

    for year in range(start_year, end_year + 1):
        # Build NBS easyquery API request
        url = "https://data.stats.gov.cn/easyquery/api"
        params = {
            "m": "QueryData",
            "dbcode": dbcode,
            "rowcode": "reg",
            "colcode": "sj",
            "wds": "[]",
            "dfwds": json.dumps([
                {"wdcode": "zb", "valuecode": ind_info["code"]},
                {"wdcode": "reg", "valuecode": region_filter},
                {"wdcode": "sj", "valuecode": str(year)},
            ]),
        }
        query_string = urllib.parse.urlencode(params)

        try:
            req = urllib.request.Request(f"{url}?{query_string}")
            req.add_header("User-Agent", "policy-eval/1.0")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            # NBS API often blocks non-browser requests; fall back to akshare
            print(f"  NBS API year {year}: {type(e).__name__} — "
                  f"falling back to akshare")
            # Fall back to city macro fetch
            if level == "city":
                return fetch_cn_city_macro(indicator, start_year, end_year)
            elif level == "province":
                return fetch_cn_province_macro(indicator, start_year, end_year)
            return pd.DataFrame()

        # Parse NBS response
        if data.get("returncode") != 200:
            continue

        datanodes = data.get("returndata", {}).get("datanodes", [])
        wdnodes = data.get("returndata", {}).get("wdnodes", [])

        # Build code → name lookup
        reg_map = {}
        for wd in wdnodes:
            if wd.get("wdcode") == "reg":
                for node in wd.get("nodes", []):
                    reg_map[node["code"]] = node["name"]

        for node in datanodes:
            val = node.get("data", {}).get("data")
            if val is None:
                continue
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue
            # Extract region code from wds
            wds = node.get("wds", [])
            reg_code = None
            for wd in wds:
                if wd.get("wdcode") == "reg":
                    reg_code = wd.get("valuecode", "")
                    break
            entity = reg_map.get(reg_code, reg_code)
            rows.append({"entity_id": str(entity), "year": year,
                        "value": val})

        time.sleep(0.3)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# FRED (Federal Reserve Economic Data)
# ═══════════════════════════════════════════════════════════════════════

FRED_SERIES = {
    "gdp": "GDP",                          # US GDP (billions)
    "real_gdp": "GDPC1",                   # Real GDP
    "unemployment_rate": "UNRATE",         # US unemployment rate
    "cpi": "CPIAUCSL",                     # CPI all items
    "core_cpi": "CPILFESL",                # Core CPI
    "federal_funds_rate": "FEDFUNDS",      # Fed funds rate
    "ten_year_treasury": "DGS10",          # 10-year Treasury yield
    "industrial_production": "INDPRO",     # Industrial production index
    "retail_sales": "RSAFS",               # Retail sales
    "nonfarm_payrolls": "PAYEMS",          # Nonfarm payrolls
    "labor_force_participation": "CIVPART", # Labor force participation
    "housing_starts": "HOUST",             # Housing starts
    "m2_money_supply": "M2SL",             # M2 money supply
    "trade_balance": "BOPGSTB",            # Trade balance
    "consumer_sentiment": "UMCSENT",        # U Michigan consumer sentiment
}


def fetch_fred(series_id: str = "GDP",
               start_date: str = "2010-01-01",
               end_date: str = "2025-12-31",
               api_key: str = None) -> pd.DataFrame:
    """
    Fetch data from FRED (Federal Reserve Economic Data).

    Requires a FRED API key (free from research.stlouisfed.org).
    Set FRED_API_KEY environment variable or pass directly.

    Parameters
    ----------
    series_id : str
        FRED series ID (see FRED_SERIES for common ones).
    start_date, end_date : str
        Date range in YYYY-MM-DD format.
    api_key : str
        FRED API key. If None, reads from FRED_API_KEY env var.

    Returns
    -------
    pd.DataFrame with [entity_id, date, value].
    """
    import os
    import urllib.request

    if api_key is None:
        api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        # Try fredapi package
        try:
            from fredapi import Fred
            fred = Fred(api_key=None)  # will look for FRED_API_KEY env var
            try:
                data = fred.get_series(series_id, observation_start=start_date,
                                       observation_end=end_date)
                df = data.reset_index()
                df.columns = ["date", "value"]
                df["entity_id"] = series_id
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                return df[["entity_id", "date", "value"]].dropna()
            except Exception as e:
                print(f"  FRED (fredapi): {e}")
        except ImportError:
            pass

        # Direct HTTP without key
        print("  FRED: no API key set. Set FRED_API_KEY env var or pass "
              "api_key= parameter. Get a free key at "
              "https://fred.stlouisfed.org/docs/api/api_key.html")
        return pd.DataFrame()

    # Direct HTTP with API key
    series_code = FRED_SERIES.get(series_id, series_id)
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_code}&api_key={api_key}"
        f"&observation_start={start_date}&observation_end={end_date}"
        f"&file_type=json"
    )

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  FRED API error: {e}")
        return pd.DataFrame()

    rows = []
    for obs in data.get("observations", []):
        if obs.get("value") not in (".", None, ""):
            rows.append({
                "entity_id": series_id,
                "date": obs["date"],
                "value": float(obs["value"]),
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# OECD API
# ═══════════════════════════════════════════════════════════════════════

OECD_DATASETS = {
    "gdp_quarterly": {"id": "QNA", "measure": "GDP", "unit": "USD"},
    "cpi": {"id": "PRICES", "measure": "CPI", "unit": "IDX"},
    "unemployment_rate": {"id": "LAB", "measure": "UNRATE", "unit": "PC"},
    "employment_rate": {"id": "LAB", "measure": "EMPRATE", "unit": "PC"},
    "labor_productivity": {"id": "PDBI", "measure": "GDPHR", "unit": "IDX"},
    "government_debt": {"id": "GOV", "measure": "DEBT", "unit": "PC_GDP"},
    "health_spending": {"id": "HEALTH", "measure": "TOT", "unit": "PC_GDP"},
    "education_spending": {"id": "EDU", "measure": "TOT", "unit": "PC_GDP"},
    "co2_emissions": {"id": "AIR", "measure": "CO2", "unit": "TONNE"},
    "gini": {"id": "WISE", "measure": "GINI", "unit": "IDX"},
}

OECD_COUNTRIES = {
    "australia": "AUS", "austria": "AUT", "belgium": "BEL", "canada": "CAN",
    "chile": "CHL", "colombia": "COL", "czech": "CZE", "denmark": "DNK",
    "estonia": "EST", "finland": "FIN", "france": "FRA", "germany": "DEU",
    "greece": "GRC", "hungary": "HUN", "iceland": "ISL", "ireland": "IRL",
    "israel": "ISR", "italy": "ITA", "japan": "JPN", "korea": "KOR",
    "latvia": "LVA", "lithuania": "LTU", "luxembourg": "LUX",
    "mexico": "MEX", "netherlands": "NLD", "new_zealand": "NZL",
    "norway": "NOR", "poland": "POL", "portugal": "PRT", "slovakia": "SVK",
    "slovenia": "SVN", "spain": "ESP", "sweden": "SWE", "switzerland": "CHE",
    "turkey": "TUR", "uk": "GBR", "usa": "USA",
}


def fetch_oecd(dataset: str = "gdp_quarterly",
               countries: list[str] = None,
               start_year: int = 2010, end_year: int = 2025) -> pd.DataFrame:
    """
    Fetch data from OECD API (OECD.Stat SDMX-JSON).

    Parameters
    ----------
    dataset : str
        Dataset key (see OECD_DATASETS).
    countries : list[str]
        Country names or codes. Defaults to ["usa", "uk", "germany", "japan"].
    start_year, end_year : int
        Year range.

    Returns
    -------
    pd.DataFrame with [entity_id, year, value].
    """
    import urllib.request

    ds = OECD_DATASETS.get(dataset, {"id": dataset})
    country_list = countries or ["usa", "uk", "germany", "japan"]
    country_codes = [OECD_COUNTRIES.get(c.lower(), c.upper())
                     for c in country_list]

    rows = []

    # OECD SDMX-JSON API
    for country in country_codes:
        url = (
            f"https://sdmx.oecd.org/public/rest/data/{ds['id']}"
            f"/{ds.get('measure', '')}.{country}"
            f"/all?format=jsondata"
            f"&startPeriod={start_year}&endPeriod={end_year}"
        )
        # Fallback: simpler OECD API endpoint
        alt_url = (
            f"https://stats.oecd.org/SDMX-JSON/data/{ds['id']}"
            f"/{country}.{ds.get('measure', '')}"
            f"/all?startTime={start_year}&endTime={end_year}"
        )

        for url_to_try in [url, alt_url]:
            try:
                req = urllib.request.Request(url_to_try)
                req.add_header("Accept", "application/json")
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except Exception:
                data = None
                continue

        if data is None:
            continue

        # Parse SDMX-JSON structure
        try:
            datasets = data.get("dataSets", [])
            structure = data.get("structure", {})
            obs_dim = structure.get("dimensions", {}).get("observation", [])

            for ds_idx, dataset_obj in enumerate(datasets):
                observations = dataset_obj.get("observations", {})
                for obs_key, obs_val in observations.items():
                    # Extract time period from observation key
                    time_idx = -1
                    for dim in obs_dim:
                        if dim.get("name") == "TIME_PERIOD":
                            time_idx = dim.get("keyPosition", -1)
                            break
                    parts = obs_key.split(":")
                    if time_idx >= 0 and time_idx < len(parts):
                        year_str = parts[time_idx]
                    else:
                        year_str = parts[-1] if parts else ""

                    try:
                        year = int(year_str)
                    except ValueError:
                        continue

                    if isinstance(obs_val, list) and len(obs_val) > 0:
                        val = obs_val[0]
                    else:
                        val = obs_val
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        continue

                    rows.append({
                        "entity_id": country,
                        "year": year,
                        "value": val,
                    })
        except Exception:
            continue

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Eurostat API
# ═══════════════════════════════════════════════════════════════════════

EUROSTAT_DATASETS = {
    "gdp": "nama_10_gdp",              # GDP and main components
    "gdp_per_capita": "nama_10_pc",    # GDP per capita (PPS)
    "cpi_hicp": "prc_hicp_manr",       # Harmonized CPI
    "unemployment": "une_rt_a",        # Unemployment rate
    "employment": "lfsi_emp_a",        # Employment rate
    "population": "demo_pjan",         # Population by age and sex
    "fertility": "demo_frate",        # Fertility rates
    "government_debt": "gov_10dd_edpt1", # Gov debt
    "trade_balance": "ext_st_eu27_2020sitc", # Trade
    "co2_emissions": "env_air_gge",    # Greenhouse gas emissions
    "rd_expenditure": "rd_e_gerdtot",  # R&D expenditure
}

EUROSTAT_COUNTRIES = {
    "austria": "AT", "belgium": "BE", "bulgaria": "BG", "croatia": "HR",
    "cyprus": "CY", "czech": "CZ", "denmark": "DK", "estonia": "EE",
    "finland": "FI", "france": "FR", "germany": "DE", "greece": "EL",
    "hungary": "HU", "ireland": "IE", "italy": "IT", "latvia": "LV",
    "lithuania": "LT", "luxembourg": "LU", "malta": "MT",
    "netherlands": "NL", "poland": "PL", "portugal": "PT",
    "romania": "RO", "slovakia": "SK", "slovenia": "SI", "spain": "ES",
    "sweden": "SE", "eu27": "EU27_2020",
}


def fetch_eurostat(dataset: str = "gdp",
                   countries: list[str] = None,
                   start_year: int = 2010, end_year: int = 2025) -> pd.DataFrame:
    """
    Fetch data from Eurostat API.

    Parameters
    ----------
    dataset : str
        Dataset key (see EUROSTAT_DATASETS).
    countries : list[str]
        Country names or codes. Defaults to ["germany", "france", "italy"].
    start_year, end_year : int
        Year range.

    Returns
    -------
    pd.DataFrame with [entity_id, year, value].
    """
    import urllib.request

    ds_code = EUROSTAT_DATASETS.get(dataset, dataset)
    country_list = countries or ["germany", "france", "italy"]
    country_codes = [EUROSTAT_COUNTRIES.get(c.lower(), c.upper())
                     for c in country_list]

    rows = []

    for country in country_codes:
        url = (
            f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
            f"/{ds_code}?format=JSON"
            f"&geo={country}"
            f"&time={start_year}&to={end_year}"
        )

        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  Eurostat ({country}): {type(e).__name__}")
            continue

        # Parse Eurostat JSON
        try:
            # Find the time → value mapping
            values = data.get("value", {})
            dimensions = data.get("dimension", {})
            time_dim = dimensions.get("time", {})
            time_labels = time_dim.get("category", {}).get("label", {})

            # Build index → year mapping
            time_index = time_dim.get("category", {}).get("index", {})
            idx_to_year = {}
            for year_str, idx in time_index.items():
                idx_to_year[idx] = int(year_str)

            for idx_str, val in values.items():
                idx = int(idx_str)
                year = idx_to_year.get(idx)
                if year is None:
                    continue
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    continue
                rows.append({
                    "entity_id": country,
                    "year": year,
                    "value": val,
                })
        except Exception:
            continue

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# NYC TLC Trip Record Data (taxi + ride-hail, 2009–present)
# ═══════════════════════════════════════════════════════════════════════
# Hosted on AWS Open Data: s3://nyc-tlc/ — completely free, no registration
# Billions of trip records with pickup/dropoff coords, fares, tips, etc.
#
# Available datasets per month:
#   yellow_tripdata_YYYY-MM.parquet  — Yellow taxi (metered street hails)
#   green_tripdata_YYYY-MM.parquet   — Green taxi (street hails, outer boros)
#   fhvhv_tripdata_YYYY-MM.parquet   — High-volume for-hire (Uber/Lyft)
#   fhv_tripdata_YYYY-MM.parquet     — Other for-hire vehicles

def fetch_nyc_taxi(dataset: str = "yellow",
                    year: int = 2024, month: int = 1,
                    sample_rows: int = None) -> pd.DataFrame:
    """
    Fetch NYC TLC trip record data from AWS Open Data.

    Parameters
    ----------
    dataset : str
        "yellow", "green", "fhvhv" (Uber/Lyft), or "fhv".
    year, month : int
        Data period. Available: yellow 2009–, green 2013–, fhvhv 2019–.
    sample_rows : int or None
        If set, return a random sample of N rows (useful for quick analysis).

    Returns
    -------
    pd.DataFrame with trip-level microdata.
    """
    import urllib.request

    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    file_map = {
        "yellow": f"yellow_tripdata_{year}-{month:02d}.parquet",
        "green": f"green_tripdata_{year}-{month:02d}.parquet",
        "fhvhv": f"fhvhv_tripdata_{year}-{month:02d}.parquet",
        "fhv": f"fhv_tripdata_{year}-{month:02d}.parquet",
    }
    filename = file_map.get(dataset)
    if filename is None:
        print(f"  Unknown NYC TLC dataset: {dataset}")
        return pd.DataFrame()

    url = f"{base_url}/{filename}"
    print(f"  Fetching NYC {dataset} taxi: {filename}...")

    try:
        df = pd.read_parquet(url)
    except Exception as e:
        print(f"  NYC TLC fetch error: {type(e).__name__}: {str(e)[:120]}")
        return pd.DataFrame()

    if sample_rows and len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42)

    df["entity_id"] = f"nyc_{dataset}_taxi"
    df["year"] = year
    df["month"] = month
    print(f"  ✓ {len(df):,} rows, columns: {list(df.columns)[:8]}...")
    return df


def fetch_nyc_taxi_aggregate(dataset: str = "yellow",
                              year: int = 2024,
                              months: list[int] = None,
                              agg_col: str = "trip_distance",
                              agg_func: str = "sum",
                              sample_per_month: int = 50000) -> pd.DataFrame:
    """
    Fetch multiple months of NYC taxi data and aggregate.

    Useful for policy evaluation: track trip volume/fares before and after
    a policy change (e.g., congestion pricing, fuel surcharges).

    Parameters
    ----------
    dataset : str
        "yellow", "green", "fhvhv", or "fhv".
    year : int
    months : list[int]
        Months to fetch (default: all 12).
    agg_col : str
        Column to aggregate (e.g., "trip_distance", "total_amount", "tolls").
    agg_func : str
        "sum", "mean", "count", "median".
    sample_per_month : int
        Rows to sample per month (controls memory usage).

    Returns
    -------
    pd.DataFrame with [entity_id, year, month, value].
    """
    rows = []
    months = months or list(range(1, 13))

    for m in months:
        df = fetch_nyc_taxi(dataset, year, m, sample_rows=sample_per_month)
        if df.empty:
            continue
        if agg_col in df.columns:
            val = float(getattr(df[agg_col], agg_func)())
        else:
            val = float(len(df))
        rows.append({
            "entity_id": f"nyc_{dataset}_taxi",
            "year": year,
            "month": m,
            "value": val,
        })
        print(f"  {year}-{m:02d}: {agg_func}({agg_col}) = {val:,.1f}")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Citi Bike (NYC bike share, 2013–present)
# ═══════════════════════════════════════════════════════════════════════

def fetch_citi_bike(year: int = 2024, month: int = 1,
                     sample_rows: int = None) -> pd.DataFrame:
    """
    Fetch Citi Bike trip data (NYC bike share system).

    Public data hosted at: https://s3.amazonaws.com/tripdata/
    Contains: start/end station, duration, user type, bike type.

    Parameters
    ----------
    year, month : int
        Data period. Available: 2013-07 onward.
    sample_rows : int or None

    Returns
    -------
    pd.DataFrame with trip-level microdata.
    """
    url = (f"https://s3.amazonaws.com/tripdata/"
           f"{year}{month:02d}-citibike-tripdata.csv.zip")
    try:
        df = pd.read_csv(url, low_memory=False)
    except Exception as e:
        # Try JC (Jersey City) naming convention
        alt_url = (f"https://s3.amazonaws.com/tripdata/"
                   f"JC-{year}{month:02d}-citibike-tripdata.csv.zip")
        try:
            df = pd.read_csv(alt_url, low_memory=False)
        except Exception:
            print(f"  Citi Bike fetch error: {type(e).__name__}: {str(e)[:100]}")
            return pd.DataFrame()

    if sample_rows and len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42)

    df["entity_id"] = "citi_bike_nyc"
    df["year"] = year
    df["month"] = month
    print(f"  ✓ Citi Bike {year}-{month:02d}: {len(df):,} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════
# EPA Air Quality System (AQS) — US ambient air monitoring
# ═══════════════════════════════════════════════════════════════════════

EPA_AQS_PARAMETERS = {
    "pm25": "88101",    # PM2.5 - Local Conditions
    "pm10": "81102",    # PM10
    "ozone": "44201",   # Ozone
    "so2": "42401",     # SO2
    "no2": "42602",     # NO2
    "co": "42101",       # CO
}


def fetch_epa_aqs(param: str = "pm25",
                   state_code: str = "36",  # 36 = New York
                   start_year: int = 2015, end_year: int = 2025) -> pd.DataFrame:
    """
    Fetch US EPA AQS monitor-level air quality data.

    Free API at: https://aqs.epa.gov/aqsweb/documents/data_api.html
    No API key needed for the daily summary endpoint.

    Parameters
    ----------
    param : str
        Pollutant key (see EPA_AQS_PARAMETERS).
    state_code : str
        FIPS state code (e.g., "36" = NY, "06" = CA, "17" = IL).
    start_year, end_year : int

    Returns
    -------
    pd.DataFrame with [entity_id, date, value] per monitor.
    """
    import urllib.request

    param_code = EPA_AQS_PARAMETERS.get(param, param)
    state_str = f"{int(state_code):02d}"
    rows = []

    for year in range(start_year, end_year + 1):
        url = (
            f"https://aqs.epa.gov/data/api/dailyData/byState"
            f"?email=test@example.com&key=crimsonant57"
            f"&param={param_code}&bdate={year}0101&edate={year}1231"
            f"&state={state_str}"
        )
        # Fallback: use the simpler monitor-level endpoint
        alt_url = (
            f"https://aqs.epa.gov/data/api/sampleData/byState"
            f"?email=test@example.com&key=crimsonant57"
            f"&param={param_code}&bdate={year}0101&edate={year}1231"
            f"&state={state_str}"
        )

        for epa_url in [url, alt_url]:
            try:
                req = urllib.request.Request(epa_url)
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read())
                if data.get("Header", [{}])[0].get("status") == "Success":
                    for record in data.get("Data", []):
                        rows.append({
                            "entity_id": f"{record.get('state_code', '')}-"
                                         f"{record.get('county_code', '')}-"
                                         f"{record.get('site_number', '')}",
                            "date": f"{record.get('date_local', '')}",
                            "value": float(record.get("arithmetic_mean",
                                         record.get("sample_measurement", 0))),
                            "parameter": param,
                        })
                    break
                else:
                    msg = data.get("Header", [{}])[0].get("status", "unknown")
                    if "Invalid key" in str(msg):
                        print(f"  EPA AQS: API key invalid — get a free key at "
                              f"https://aqs.epa.gov/aqsweb/documents/data_api.html")
                        return pd.DataFrame()
            except Exception:
                continue

        print(f"  EPA {param} {year}: {sum(1 for r in rows if str(year) in r.get('date',''))} records")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# NOAA GSOD (Global Surface Summary of the Day) — worldwide weather
# ═══════════════════════════════════════════════════════════════════════

def fetch_noaa_gsod(station_id: str = "725030",  # NYC Central Park
                     start_year: int = 2015, end_year: int = 2025) -> pd.DataFrame:
    """
    Fetch NOAA daily weather data for a station.

    Free, no registration. Data from:
    https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/

    Common US station IDs:
      725030 = NYC Central Park,  722950 = LAX,
      725300 = Chicago O'Hare,    722430 = Houston IAH,
      724940 = San Francisco,     722190 = Atlanta

    Parameters
    ----------
    station_id : str
        NOAA/USAF station ID (6 digits).
    start_year, end_year : int

    Returns
    -------
    pd.DataFrame with [entity_id, date, temp, precip, wind, ...].
    """
    import urllib.request
    import csv

    rows = []
    for year in range(start_year, end_year + 1):
        url = (f"https://www.ncei.noaa.gov/data/global-summary-of-the-day/"
               f"access/{year}/{station_id}.csv")
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "policy-eval/1.0")
            with urllib.request.urlopen(req, timeout=20) as resp:
                text = resp.read().decode("utf-8")
            reader = csv.DictReader(text.splitlines())
            for record in reader:
                rows.append({
                    "entity_id": station_id,
                    "date": record.get("DATE", ""),
                    "temp_c": float(record.get("TEMP", 0) or 0) / 10.0,
                    "temp_max_c": float(record.get("MAX", 0) or 0) / 10.0,
                    "temp_min_c": float(record.get("MIN", 0) or 0) / 10.0,
                    "precip_mm": float(record.get("PRCP", 0) or 0) / 10.0,
                    "wind_speed_ms": float(record.get("WDSP", 0) or 0) / 10.0,
                })
        except Exception as e:
            print(f"  NOAA GSOD {year}: {type(e).__name__}: {str(e)[:80]}")
            continue
        print(f"  NOAA {station_id} {year}: {sum(1 for r in rows if str(year) in r.get('date',''))} days")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# BLS (Bureau of Labor Statistics) — US employment, wages, CPI
# ═══════════════════════════════════════════════════════════════════════

BLS_SERIES = {
    "unemployment_national": "LNS14000000",
    "unemployment_ny": "LASST360000000000003",
    "unemployment_ca": "LASST060000000000003",
    "labor_force": "LNS11000000",
    "avg_hourly_earnings": "CES0500000003",
    "cpi_all_urban": "CUUR0000SA0",
    "cpi_core": "CUUR0000SA0L1E",
    "ppi_final_demand": "WPSFD4",
}


def fetch_bls(series_id: str = "LNS14000000",
              start_year: int = 2010, end_year: int = 2025,
              api_key: str = None) -> pd.DataFrame:
    """
    Fetch BLS time series data (free, registration required for API v2).

    The BLS Public Data API v2 requires registration (free):
    https://www.bls.gov/developers/

    Common series IDs are in BLS_SERIES. You can also pass raw series IDs.

    Parameters
    ----------
    series_id : str
        BLS series ID.
    start_year, end_year : int
    api_key : str
        BLS API key (free registration).

    Returns
    -------
    pd.DataFrame with [entity_id, year, period, value].
    """
    import os
    import urllib.request

    if api_key is None:
        api_key = os.environ.get("BLS_API_KEY", "")

    if not api_key:
        print("  BLS: no API key. Register for free at "
              "https://www.bls.gov/developers/ and set BLS_API_KEY env var")
        return pd.DataFrame()

    series_code = BLS_SERIES.get(series_id, series_id)
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    payload = json.dumps({
        "seriesid": [series_code],
        "startyear": str(start_year),
        "endyear": str(end_year),
        "registrationkey": api_key,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload,
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  BLS API error: {type(e).__name__}: {str(e)[:120]}")
        return pd.DataFrame()

    if data.get("status") != "REQUEST_SUCCEEDED":
        print(f"  BLS error: {data.get('message', 'unknown')}")
        return pd.DataFrame()

    rows = []
    for series in data.get("Results", {}).get("series", []):
        for d in series.get("data", []):
            rows.append({
                "entity_id": series.get("seriesID", series_id),
                "year": int(d.get("year", 0)),
                "period": d.get("period", ""),
                "period_name": d.get("periodName", ""),
                "value": float(d.get("value", 0)),
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# FEC (Federal Election Commission) — US campaign finance
# ═══════════════════════════════════════════════════════════════════════

def fetch_fec(endpoint: str = "candidates",
               params: dict = None) -> pd.DataFrame:
    """
    Fetch US campaign finance data from the FEC API.

    Free, no registration. Rate limited to 1,000 calls/hour.
    API docs: https://api.open.fec.gov/

    Useful endpoints:
      - /candidates/           — candidate filings
      - /committees/           — PAC/party committees
      - /schedules/schedule_a/ — itemized contributions
      - /schedules/schedule_b/ — itemized expenditures

    Parameters
    ----------
    endpoint : str
        API endpoint path.
    params : dict
        Query parameters (e.g., {"state": "NY", "election_year": 2020}).

    Returns
    -------
    pd.DataFrame
    """
    import urllib.request

    base_url = "https://api.open.fec.gov/v1"
    url = f"{base_url}/{endpoint.lstrip('/')}"
    if params:
        query_parts = []
        for k, v in params.items():
            if isinstance(v, list):
                query_parts.append(f"{k}={','.join(str(x) for x in v)}")
            else:
                query_parts.append(f"{k}={v}")
        url += "?" + "&".join(query_parts)
    # Pagination
    url += ("&" if "?" in url else "?") + "per_page=100&page=1"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "policy-eval/1.0")
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  FEC API error: {type(e).__name__}: {str(e)[:100]}")
        return pd.DataFrame()

    results = data.get("results", [])
    pagination = data.get("pagination", {})
    total = pagination.get("count", len(results))

    df = pd.DataFrame(results)
    if not df.empty:
        df["entity_id"] = "fec"
        print(f"  ✓ FEC {endpoint}: {len(df)} rows (total: {total})")

    return df


# ═══════════════════════════════════════════════════════════════════════
# Zillow Home Value Index — US housing prices
# ═══════════════════════════════════════════════════════════════════════

def fetch_zillow_hvi(geography: str = "metro",
                      geo_name: str = "New York, NY") -> pd.DataFrame:
    """
    Fetch Zillow Home Value Index (ZHVI) or rental data.

    Free, no registration. Data at:
    https://www.zillow.com/research/data/

    Parameters
    ----------
    geography : str
        "metro", "city", "zip", "county", "state", or "national".
    geo_name : str
        Name of geography (e.g., "New York, NY", "San Francisco, CA").

    Returns
    -------
    pd.DataFrame with [entity_id, date, value].
    """
    geo_map = {
        "metro": "Metro",
        "city": "City",
        "zip": "Zip",
        "county": "County",
        "state": "State",
        "national": "Country",
    }
    geo_type = geo_map.get(geography, geography)

    url = (
        f"https://files.zillowstatic.com/research/public_csvs/zhvi/"
        f"{geo_type}_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    )
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"  Zillow fetch error: {type(e).__name__}: {str(e)[:100]}")
        return pd.DataFrame()

    # Filter to the requested geography
    region_col = "RegionName" if "RegionName" in df.columns else "RegionID"
    if "RegionName" in df.columns:
        mask = df["RegionName"].str.contains(
            geo_name.replace(",", "").split()[0], case=False, na=False)
        df = df[mask]

    if df.empty:
        print(f"  Zillow: no data for '{geo_name}' in {geography}")
        return pd.DataFrame()

    # Melt wide date columns to long format
    date_cols = [c for c in df.columns
                 if re.match(r'\d{4}-\d{2}-\d{2}', str(c))]
    id_cols = [c for c in df.columns
               if c not in date_cols and "Region" in c]
    entity_id_col = id_cols[0] if id_cols else df.columns[0]

    rows = []
    for _, row in df.iterrows():
        for dc in date_cols:
            val = row[dc]
            if pd.notna(val):
                rows.append({
                    "entity_id": str(row.get(entity_id_col, geo_name)),
                    "date": dc,
                    "value": float(val),
                })

    result = pd.DataFrame(rows)
    print(f"  ✓ Zillow: {len(result)} rows for {geo_name} "
          f"({result['date'].min()} to {result['date'].max()})")
    return result


# ═══════════════════════════════════════════════════════════════════════
# IPUMS USA — Census/ACS microdata (individual-level)
# ═══════════════════════════════════════════════════════════════════════

def fetch_ipums_usa(api_key: str = None,
                     dataset: str = "acs1",
                     sample: str = "us2023a",
                     variables: list[str] = None) -> pd.DataFrame:
    """
    Fetch IPUMS USA microdata (census/ACS individual-level samples).

    Free registration required at: https://usa.ipums.org/
    After registering, create an API key from your account page.

    IPUMS provides individual-level census/ACS data with hundreds of
    harmonized variables (income, education, occupation, migration, etc.).

    Parameters
    ----------
    api_key : str
        IPUMS API key (free after registration).
    dataset : str
        "acs1", "acs3", "acs5", "decennial".
    sample : str
        IPUMS sample ID (e.g., "us2023a" for ACS 2023 1-year).
    variables : list[str]
        Variable codes (e.g., ["AGE", "SEX", "INCTOT", "EDUC"]).

    Returns
    -------
    pd.DataFrame with individual-level microdata.
    """
    import os
    import urllib.request

    if api_key is None:
        api_key = os.environ.get("IPUMS_API_KEY", "")

    if not api_key:
        print("  IPUMS: register for free at https://usa.ipums.org/ "
              "and set IPUMS_API_KEY env var")
        return pd.DataFrame()

    variables = variables or ["AGE", "SEX", "INCTOT", "EDUC", "RACE",
                              "MARST", "EMPSTAT", "OCC"]

    # Step 1: Submit extract request
    extract_url = "https://api.ipums.org/extracts/"
    payload = json.dumps({
        "collection": dataset,
        "description": "policy-eval auto-extract",
        "dataStructure": {"rectangular": {"on": "P"}},
        "samples": {dataset: {sample: {}}},
        "variables": {dataset: {v: {} for v in variables}},
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            extract_url, data=payload,
            headers={"Authorization": api_key,
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            extract_info = json.loads(resp.read())
        extract_id = extract_info.get("number")
        if not extract_id:
            print(f"  IPUMS extract error: {extract_info}")
            return pd.DataFrame()
        print(f"  IPUMS: extract #{extract_id} submitted "
              f"({len(variables)} variables)")
    except Exception as e:
        print(f"  IPUMS API error: {type(e).__name__}: {str(e)[:120]}")
        print(f"  This is expected — IPUMS extracts are asynchronous and "
              f"may take minutes to hours. Use the IPUMS web interface at "
              f"https://usa.ipums.org/ for large extracts.")
        return pd.DataFrame()

    # Step 2: Poll for completion
    poll_url = f"{extract_url}{extract_id}/?version=2"
    for _ in range(60):  # poll for up to 10 minutes
        time.sleep(10)
        try:
            req = urllib.request.Request(
                poll_url, headers={"Authorization": api_key})
            with urllib.request.urlopen(req, timeout=15) as resp:
                status = json.loads(resp.read())
            if status.get("status") == "completed":
                break
        except Exception:
            continue

    # Step 3: Download data
    download_url = (f"https://api.ipums.org/extracts/"
                    f"{extract_id}/download/")
    try:
        req = urllib.request.Request(
            download_url, headers={"Authorization": api_key})
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = resp.read()
    except Exception as e:
        print(f"  IPUMS download error: {type(e).__name__}. "
              f"Extract #{extract_id} is available at https://usa.ipums.org/")
        return pd.DataFrame()

    # Try to parse as compressed CSV
    import gzip
    import io
    try:
        df = pd.read_csv(gzip.GzipFile(fileobj=io.BytesIO(content)), nrows=50000)
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(content), nrows=50000)
        except Exception:
            print(f"  IPUMS: downloaded {len(content)} bytes but couldn't parse. "
                  f"Download manually from https://usa.ipums.org/")
            return pd.DataFrame()

    df["entity_id"] = f"ipums_{dataset}_{sample}"
    print(f"  ✓ IPUMS: {len(df):,} individual records")
    return df


# ═══════════════════════════════════════════════════════════════════════
# NYC 311 Service Requests — public complaints data
# ═══════════════════════════════════════════════════════════════════════

def fetch_nyc_311(limit: int = 10000,
                   complaint_type: str = None,
                   start_date: str = "2024-01-01",
                   end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    Fetch NYC 311 service requests via the SODA API.

    Free, no registration. Data at:
    https://data.cityofnewyork.us/Social-Services/311-Service-Requests/erm2-nwe9

    Parameters
    ----------
    limit : int
        Max rows to fetch (SODA default: 1000, max: 50000 per call).
    complaint_type : str or None
        Filter by type (e.g., "Noise - Residential", "Illegal Parking").
    start_date, end_date : str
        Date range filter.

    Returns
    -------
    pd.DataFrame with [entity_id, date, complaint_type, location, ...].
    """
    import urllib.request

    base_url = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
    params = {
        "$limit": str(min(limit, 50000)),
        "$order": "created_date DESC",
    }
    if complaint_type:
        params["complaint_type"] = complaint_type
    if start_date:
        params["$where"] = (f"created_date between '{start_date}T00:00:00' "
                           f"and '{end_date}T23:59:59'")

    query_string = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_string}"

    try:
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        req.add_header("X-App-Token", "")  # optional, raises rate limit
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  NYC 311 API error: {type(e).__name__}: {str(e)[:100]}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "created_date" in df.columns:
        df["date"] = df["created_date"].str[:10]
    df["entity_id"] = "nyc_311"
    print(f"  ✓ NYC 311: {len(df)} records")
    return df


# ═══════════════════════════════════════════════════════════════════════
# Chicago Crimes — public crime incident data
# ═══════════════════════════════════════════════════════════════════════

def fetch_chicago_crimes(limit: int = 10000,
                          year: int = 2024,
                          crime_type: str = None) -> pd.DataFrame:
    """
    Fetch Chicago crime incident data via the SODA API.

    Free, no registration. Data at:
    https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2

    Parameters
    ----------
    limit : int
        Max rows.
    year : int
        Data year.
    crime_type : str or None
        Filter by type (e.g., "THEFT", "BATTERY", "HOMICIDE").

    Returns
    -------
    pd.DataFrame with [entity_id, date, crime_type, location, arrest, ...].
    """
    import urllib.request

    base_url = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
    params = {
        "$limit": str(min(limit, 50000)),
        "$order": "date DESC",
        "$where": f"year = {year}",
    }
    if crime_type:
        params["$where"] += f" AND primary_type = '{crime_type}'"

    query_string = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_string}"

    try:
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  Chicago crimes API error: {type(e).__name__}: {str(e)[:100]}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["entity_id"] = "chicago_crimes"
    print(f"  ✓ Chicago crimes: {len(df)} records ({year})")
    return df


# ═══════════════════════════════════════════════════════════════════════
# Batch fetch from variable map
# ═══════════════════════════════════════════════════════════════════════

def _fetch_one_variable(var_name: str, entry: dict, start_year: int,
                        end_year: int, start_date: str, end_date: str,
                        save: bool, force: bool,
                        indicator_override: str = None) -> FetchResult:
    """Fetch a single variable from its variable_map entry. Thread-safe."""
    source = entry.get("source", "")
    source_label, description, tier = _source_meta(source, entry.get("description", ""))
    print(f"  Fetching {var_name} ({description}) via {source}...")

    try:
        if source == "world_bank":
            indicator = entry.get("wb_indicator", var_name)
            countries = entry.get("countries", ["CN"])
            df = fetch_wb_indicator(indicator, countries,
                                    start_year, end_year)

        elif source == "akshare_city":
            indicator = entry.get("akshare_indicator", var_name)
            df = fetch_cn_city_macro(indicator, start_year, end_year,
                                     incremental=True)

        elif source == "akshare_province":
            indicator = entry.get("akshare_indicator", var_name)
            df = fetch_cn_province_macro(indicator, start_year, end_year,
                                         incremental=True)

        elif source == "akshare":
            df = fetch_akshare(entry, start_date, end_date,
                                indicator_override)

        elif source == "yfinance_global_index":
            index_name = entry.get("index_name", var_name)
            df = fetch_global_stock_index(index_name, start_date, end_date)

        elif source == "akshare_custom":
            custom = entry.get("akshare_custom", "")
            if custom == "aqi":
                city = (indicator_override or
                        entry.get("akshare_indicator", "北京"))
                df = fetch_cn_aqi(city, start_date, end_date)
            else:
                return FetchResult(
                    variable=var_name, status="error",
                    errors=[f"Unknown custom source: {custom}"],
                    source_label=source_label, description=description, tier=tier)

        elif source == "census":
            indicator = entry.get("census_indicator", var_name)
            census_year = entry.get("census_year", 2020)
            level = entry.get("level", "city")
            df = fetch_cn_census(indicator, census_year, level)

        elif source == "nbs":
            indicator = entry.get("nbs_indicator", var_name)
            level = entry.get("level", "city")
            df = fetch_nbs_data(indicator, start_year=start_year,
                                end_year=end_year, level=level)

        elif source == "fred":
            series = entry.get("fred_series", var_name)
            df = fetch_fred(series, start_date, end_date)

        elif source == "oecd":
            dataset = entry.get("oecd_dataset", var_name)
            countries = entry.get("countries", ["usa", "deu", "jpn"])
            df = fetch_oecd(dataset, countries, start_year, end_year)

        elif source == "eurostat":
            dataset = entry.get("eurostat_dataset", var_name)
            countries = entry.get("countries", ["DE", "FR", "IT"])
            df = fetch_eurostat(dataset, countries, start_year, end_year)

        elif source == "nyc_taxi":
            ds = entry.get("taxi_dataset", "yellow")
            yr = entry.get("data_year", start_year)
            m = entry.get("data_month", 1)
            sample = entry.get("sample_rows", 50000)
            df = fetch_nyc_taxi(ds, yr, m, sample_rows=sample)

        elif source == "nyc_taxi_agg":
            ds = entry.get("taxi_dataset", "yellow")
            yr = entry.get("data_year", start_year)
            months = entry.get("months", None)
            agg_col = entry.get("agg_col", "total_amount")
            agg_func = entry.get("agg_func", "sum")
            sample = entry.get("sample_rows", 50000)
            df = fetch_nyc_taxi_aggregate(ds, yr, months,
                                          agg_col, agg_func, sample)

        elif source == "citi_bike":
            yr = entry.get("data_year", start_year)
            m = entry.get("data_month", 1)
            sample = entry.get("sample_rows", None)
            df = fetch_citi_bike(yr, m, sample_rows=sample)

        elif source == "epa_aqs":
            param = entry.get("epa_param", "pm25")
            state = entry.get("state_code", "36")
            df = fetch_epa_aqs(param, state, start_year, end_year)

        elif source == "noaa_gsod":
            station = entry.get("station_id", "725030")
            df = fetch_noaa_gsod(station, start_year, end_year)

        elif source == "bls":
            series = entry.get("bls_series", var_name)
            df = fetch_bls(series, start_year, end_year)

        elif source == "fec":
            ep = entry.get("fec_endpoint", "candidates")
            params = entry.get("fec_params", {})
            df = fetch_fec(ep, params)

        elif source == "zillow":
            geo = entry.get("zillow_geography", "metro")
            geo_name = entry.get("zillow_geoname", "New York, NY")
            df = fetch_zillow_hvi(geo, geo_name)

        elif source == "nyc_311":
            limit = entry.get("limit", 10000)
            ctype = entry.get("complaint_type", None)
            df = fetch_nyc_311(limit, ctype,
                               f"{start_year}-01-01",
                               f"{end_year}-12-31")

        elif source == "chicago_crimes":
            yr = entry.get("data_year", start_year)
            limit = entry.get("limit", 10000)
            ctype = entry.get("crime_type", None)
            df = fetch_chicago_crimes(limit, yr, ctype)

        elif source == "ipums":
            dataset = entry.get("ipums_dataset", "acs1")
            sample = entry.get("ipums_sample", f"us{start_year}a")
            variables = entry.get("ipums_variables", None)
            df = fetch_ipums_usa(dataset=dataset, sample=sample,
                                 variables=variables)

        else:
            return FetchResult(
                variable=var_name, status="error",
                errors=[f"Unknown source type: {source}"],
                source_label=source_label, description=description, tier=tier)

        if df.empty:
            return FetchResult(
                variable=var_name, status="empty",
                warnings=["no data returned"],
                errors=[f"Source {source} returned no data for "
                        f"'{description}'"],
                source_label=source_label, description=description, tier=tier)
        elif save:
            meta = {
                "source": source,
                "description": entry.get("description", ""),
                "start_year": start_year,
                "end_year": end_year,
            }
            path = save_to_auto_with_meta(df, var_name, force=force,
                                           meta=meta)
            return FetchResult(
                variable=var_name, status="success",
                df=df, path=path, rows=len(df),
                source_label=source_label, description=description, tier=tier)
        else:
            return FetchResult(
                variable=var_name, status="success",
                df=df, rows=len(df),
                source_label=source_label, description=description, tier=tier)

    except Exception as e:
        return FetchResult(
            variable=var_name, status="error",
            errors=[f"{type(e).__name__}: {str(e)}",
                    traceback.format_exc()[-500:]],
            source_label=source_label, description=description, tier=tier)


def fetch_from_variable_map(variable_names: list[str],
                            country_or_region: str = "cn",
                            start_year: int = 2010, end_year: int = 2025,
                            save: bool = True, force: bool = False,
                            indicator_override: str = None,
                            max_age_hours: float = None,
                            max_workers: int = 6) -> dict[str, FetchResult]:
    """
    Fetch multiple variables using variable_map.json.

    Two-phase: (1) check cache serially, (2) fetch misses in parallel.

    Routes each variable to the appropriate fetch function based on
    the 'source' field in variable_map.json.

    Returns dict: variable_name → FetchResult.
    """
    map_path = SKILL_ROOT / "references" / "variable_map.json"
    if not map_path.exists():
        print(f"  variable_map.json not found at {map_path}")
        return {}
    with open(map_path, encoding="utf-8") as f:
        var_map = json.load(f)

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    results: dict[str, FetchResult] = {}

    # ── Phase 1: validate & check cache (serial, fast) ─────────────────
    to_fetch: list[tuple[str, dict]] = []
    for var_name in variable_names:
        if var_name not in var_map:
            result = FetchResult(
                variable=var_name, status="error",
                errors=[f"'{var_name}' not found in variable_map.json"],
                description=var_name)
            results[var_name] = result
            print(f"  ✗ {var_name}: not in variable map")
            continue

        entry = var_map[var_name]

        if not force and is_cached(var_name, max_age_hours):
            cached_path = str(DATA_AUTO / f"{var_name}.json")
            sl, desc, tier = _source_meta(entry.get("source", ""), entry.get("description", ""))
            result = FetchResult(
                variable=var_name, status="cached", path=cached_path,
                source_label=sl, description=desc, tier=tier)
            results[var_name] = result
            print(f"  ✓ {var_name}: cached (use --force to re-fetch)")
            continue

        to_fetch.append((var_name, entry))

    if not to_fetch:
        return results

    n_cached = len(variable_names) - len(to_fetch)
    n_fetch = len(to_fetch)
    print(f"\n  {'─'*50}")
    print(f"  Phase 1 complete: {n_cached} cached, {n_fetch} to fetch")
    print(f"  Phase 2: fetching {n_fetch} variables "
          f"(max {max_workers} parallel)...")
    print(f"  {'─'*50}\n")

    # ── Phase 2: fetch in parallel ─────────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=min(max_workers, n_fetch)) as executor:
        future_to_var = {
            executor.submit(
                _fetch_one_variable, var_name, entry,
                start_year, end_year, start_date, end_date,
                save, force, indicator_override,
            ): var_name
            for var_name, entry in to_fetch
        }

        for future in as_completed(future_to_var):
            var_name = future_to_var[future]
            try:
                result = future.result()
            except Exception as e:
                result = FetchResult(
                    variable=var_name, status="error",
                    errors=[f"Thread error: {type(e).__name__}: {str(e)}"])
            results[var_name] = result
            # Print result immediately so user sees progress
            if result.is_ok():
                print(f"  ✓ {var_name}: {result.status} ({result.rows} rows)")
            else:
                print(f"  ✗ {var_name}: {result.status} — "
                      f"{'; '.join(result.errors[:1])}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Smart variable matching — Stage 4 → Stage 5 bridge
# ═══════════════════════════════════════════════════════════════════════

# Keywords that indicate geographic level in a variable description
_LEVEL_KEYWORDS = {
    "city": ["city", "城市", "municipal", "prefecture", "地级"],
    "province": ["province", "省", "provincial", "省级"],
    "country": ["country", "national", "国家", "全国", "nationwide"],
    "daily": ["daily", "日频", "日度", "daily frequency"],
    "monthly": ["monthly", "月度", "月频"],
    "yearly": ["yearly", "annual", "年度", "每年"],
}

# Keywords that indicate data domain
_DOMAIN_KEYWORDS = {
    "macro": ["gdp", "population", "人口", "生产总值", "经济", "economy",
              "inflation", "cpi", "unemployment", "失业", "employment", "就业",
              "fiscal", "财政", "revenue", "收入", "expenditure", "支出",
              "trade", "贸易", "export", "import", "fdi", "investment", "投资"],
    "financial": ["stock", "index", "指数", "bond", "债券", "yield", "收益率",
                  "exchange", "汇率", "interest", "利率", "shibor", "lpr",
                  "money supply", "货币", "m0", "m1", "m2", "pmi"],
    "social": ["fertility", "生育", "birth", "出生", "mortality", "死亡",
               "education", "教育", "health", "健康", "medical", "医疗",
               "hospital", "医院", "census", "普查", "elderly", "老龄",
               "urbanization", "城镇化", "migrant", "流动人口"],
    "environment": ["aqi", "空气质量", "pm25", "pm10", "ozone", "so2", "no2",
                    "co2", "emission", "排放", "weather", "天气", "temperature",
                    "precipitation", "降水"],
    "housing": ["house price", "房价", "real estate", "房地产", "zillow",
                "home value", "rent", "租金"],
    "crime": ["crime", "犯罪", "theft", "盗窃", "311", "complaint", "投诉"],
    "transport": ["taxi", "出租车", "bike", "自行车", "transit", "交通",
                  "trip", "出行", "ride"],
    "census_micro": ["ipums", "microdata", "微观", "individual", "个人",
                     "household", "家庭", "cfps", "charls", "chfs"],
}

# Chinese → English keyword translation for fuzzy matching
_CN_EN_KEYWORDS = {
    "收入": "income disposable wage earnings",
    "工资": "wage earnings salary",
    "消费": "consumption retail",
    "房价": "house price real estate housing",
    "投资": "investment fixed asset",
    "出口": "export trade",
    "进口": "import trade",
    "医院": "hospital health medical",
    "医疗": "medical health insurance hospital",
    "教育": "education school student teacher college",
    "学生": "student school education",
    "教师": "teacher education",
    "医生": "doctor physician health",
    "床位": "bed hospital",
    "道路": "road area infrastructure",
    "绿地": "green park area",
    "供水": "water supply",
    "供气": "gas supply",
    "用电": "electricity consumption power",
    "汽车": "car vehicle ownership",
    "手机": "mobile phone telecom",
    "互联网": "internet broadband",
    "专利": "patent innovation",
    "养老": "pension insurance social security",
    "失业": "unemployment insurance",
    "污水处理": "sewage treatment waste",
    "垃圾": "waste treatment",
    "公共交通": "public transport transit bus",
    "出租车": "taxi",
    "建成区": "built up area urban",
    "研发": "rd research development science technology expenditure",
    "生育率": "fertility rate birth births per woman TFR",
    "生育": "fertility birth",
    "死亡率": "mortality death rate IMR",
    "死亡": "mortality death",
    "老龄化": "elderly aging old age",
    "城镇化": "urbanization urban",
    "流动人口": "migrant migration",
    "财政": "fiscal revenue expenditure",
    "外资": "fdi foreign investment",
    "卫生": "health hygiene medical sanitary",
    "社保": "social security insurance pension",
    "居民": "resident household urban rural",
    "城市": "city urban municipal",
    "农村": "rural village countryside",
    "人均": "per capita",
}

# Geography hints for region matching
_GEO_HINTS = {
    "cn": ["china", "中国", "cn", "chinese", "beijing", "shanghai", "城市",
           "province", "省", "city", "akshare", "nbs"],
    "us": ["us", "usa", "united states", "美国", "nyc", "new york", "chicago",
           "epa", "bls", "fec", "fred", "zillow", "citi bike"],
    "eu": ["eu", "europe", "eurostat", "oecd", "european", "germany",
           "france", "uk", "italy", "spain"],
    "global": ["world", "global", "international", "world bank", "wb",
               "cross-country", "跨国"],
}


def _expand_query_cn(query: str) -> str:
    """Expand a Chinese query with English keyword translations."""
    expanded = query
    for cn_term, en_terms in _CN_EN_KEYWORDS.items():
        if cn_term in query:
            expanded += " " + en_terms
    return expanded


def _has_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return any('一' <= c <= '鿿' for c in text)


def _tokenize(text: str) -> set[str]:
    """Tokenize text for matching. Handles Chinese via bigrams + char-level."""
    tokens = set()
    # Split on whitespace/punctuation for English
    english_part = ''.join(
        c if not ('一' <= c <= '鿿') else ' '
        for c in text
    )
    for token in english_part.replace("_", " ").replace("-", " ").split():
        if len(token) > 1:
            tokens.add(token.lower())

    # For Chinese: use bigrams + individual characters
    chinese_chars = [c for c in text if '一' <= c <= '鿿']
    # Add 2-char bigrams (most Chinese words are 2 chars)
    for i in range(len(chinese_chars) - 1):
        tokens.add(chinese_chars[i] + chinese_chars[i+1])
    # Also add individual chars for single-char matches
    for c in chinese_chars:
        tokens.add(c)
    return tokens


def _score_text_match(query: str, target: str) -> float:
    """Score how well query matches target text. 1.0 = perfect match."""
    query_lower = query.lower().strip()
    target_lower = target.lower().strip()

    if not query_lower or not target_lower:
        return 0.0

    # Exact match
    if query_lower == target_lower:
        return 1.0
    # Substring match
    if query_lower in target_lower:
        return 0.85
    if target_lower in query_lower:
        return 0.80

    # Chinese character-level substring match
    if _has_chinese(query_lower) and _has_chinese(target_lower):
        if any(query_lower[i:i+2] in target_lower
               for i in range(len(query_lower) - 1)):
            return 0.70

    # Token overlap
    query_tokens = _tokenize(query_lower)
    target_tokens = _tokenize(target_lower)

    # Also try expanded query (Chinese → English keywords)
    if _has_chinese(query_lower):
        expanded = _expand_query_cn(query_lower)
        query_tokens |= _tokenize(expanded)

    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & target_tokens)
    return 0.5 * min(1.0, overlap / len(query_tokens))


def _infer_level(description: str, level_hint: str = "") -> str:
    """Infer data level from description text and optional hint."""
    if level_hint:
        for level_name, keywords in _LEVEL_KEYWORDS.items():
            if level_hint.lower() in keywords or level_hint.lower() == level_name:
                return level_name
    for level_name, keywords in _LEVEL_KEYWORDS.items():
        for kw in keywords:
            if kw in description.lower():
                return level_name
    return "yearly"


def _infer_geo(description: str, geo_hint: str = "") -> str:
    """Infer geography from description text and optional hint."""
    combined = (description + " " + geo_hint).lower()
    for geo, keywords in _GEO_HINTS.items():
        for kw in keywords:
            if kw in combined:
                return geo
    return "global"


def search_variable_map(requirements: list[dict],
                         top_k: int = 3,
                         min_score: float = 0.3) -> dict[str, list[dict]]:
    """
    Match Stage 4 data requirements to variable_map.json entries.

    Each requirement dict should have:
      - name: str (variable name, e.g., "city_gdp")
      - description: str (what this variable measures)
      - level: str (optional hint: "city", "province", "country", "daily")
      - geo: str (optional hint: "cn", "us", "eu", "global")

    Returns dict: requirement_name → [{match, score, source, level}, ...]
    """
    map_path = SKILL_ROOT / "references" / "variable_map.json"
    if not map_path.exists():
        print(f"  variable_map.json not found at {map_path}")
        return {}
    with open(map_path, encoding="utf-8") as f:
        var_map = json.load(f)

    # Build search index from variable_map
    entries = []
    for var_name, entry in var_map.items():
        if var_name.startswith("_"):
            continue
        # Combine all searchable text
        search_text = entry.get("description", "")
        keywords = entry.get("keywords", "")
        notes = entry.get("notes", "")
        if keywords:
            search_text += " " + keywords
        if notes:
            search_text += " " + notes
        entries.append({
            "variable": var_name,
            "description": entry.get("description", ""),
            "search_text": search_text,
            "source": entry.get("source", ""),
            "level": entry.get("level", ""),
            "notes": entry.get("notes", ""),
        })

    results: dict[str, list[dict]] = {}

    for req in requirements:
        req_name = req.get("name", "")
        req_desc = req.get("description", req_name)
        req_level = req.get("level", "")
        req_geo = req.get("geo", "")

        target_level = _infer_level(req_desc, req_level)
        target_geo = _infer_geo(req_desc, req_geo)

        scored = []
        for entry in entries:
            # Score by combined search text (description + keywords + notes)
            desc_score = _score_text_match(req_desc, entry["search_text"])
            # Score by variable name match
            name_score = _score_text_match(req_name, entry["variable"])
            # Score by level compatibility
            entry_level = entry.get("level", "")
            level_score = 1.0 if (target_level in entry_level
                                  or entry_level in target_level) else 0.5

            # Combined score (weighted)
            total_score = 0.35 * desc_score + 0.35 * name_score + 0.15 * level_score
            # Bonus for exact name match
            if req_name.lower() == entry["variable"].lower():
                total_score += 0.15

            if total_score >= min_score:
                scored.append({
                    "variable": entry["variable"],
                    "score": round(total_score, 3),
                    "description": entry["description"],
                    "source": entry["source"],
                    "level": entry["level"],
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        results[req_name] = scored[:top_k]

    return results


def generate_gap_report(requirements: list[dict],
                         search_results: dict[str, list[dict]] = None,
                         fetch_results: dict[str, FetchResult] = None,
                         geo_hint: str = "cn") -> str:
    """
    Generate a structured data gap report from Stage 4 requirements.

    Parameters
    ----------
    requirements : list[dict]
        Stage 4 data requirements. Each with name, description, level, geo, essential.
    search_results : dict or None
        Pre-computed search results from search_variable_map().
    fetch_results : dict or None
        Results from fetch_from_variable_map() for variables already attempted.

    Returns
    -------
    str : Formatted gap report (Markdown).
    """
    if search_results is None:
        search_results = search_variable_map(requirements)

    fetch_results = fetch_results or {}
    lines = []
    lines.append("═══════════════════════════════════")
    lines.append("Data Gap Report")
    lines.append("═══════════════════════════════════")

    # ── Section 1: Auto-fetched ─────────────────────────────────────
    fetched_vars = []
    for var_name, result in fetch_results.items():
        if result.is_ok():
            fetched_vars.append((var_name, result))

    if fetched_vars:
        lines.append("")
        lines.append("Auto-fetched (✓):")
        for var_name, result in fetched_vars:
            lines.append(f"├── {var_name} → {result.describe()}")
    else:
        lines.append("")
        lines.append("Auto-fetched (✓): (none yet)")

    # ── Section 2: Available in variable map (Tier A) ─────────────────
    tier_a = []
    for req in requirements:
        req_name = req.get("name", "")
        if req_name in fetch_results and fetch_results[req_name].is_ok():
            continue  # already fetched
        matches = search_results.get(req_name, [])
        if matches and matches[0]["score"] >= 0.5:
            tier_a.append((req, matches))

    if tier_a:
        lines.append("")
        lines.append("Available via fetch_data.py (Tier A):")
        for req, matches in tier_a:
            best = matches[0]
            lines.append(f"├── {req['name']} → {best['variable']} "
                         f"(score: {best['score']}, source: {best['source']})")
            lines.append(f"│   Description: {best['description']}")
            lines.append(f"│   Command: python scripts/fetch_data.py "
                         f"--source {best['source']} "
                         f"--indicator {best['variable']}")
            if len(matches) > 1:
                alts = [m['variable'] for m in matches[1:3]]
                lines.append(f"│   Alternatives: {', '.join(alts)}")

    # ── Section 3: Not found — needs manual (Tier B/C) ───────────────
    tier_bc = []
    for req in requirements:
        req_name = req.get("name", "")
        if req_name in fetch_results and fetch_results[req_name].is_ok():
            continue
        matches = search_results.get(req_name, [])
        if not matches or matches[0]["score"] < 0.5:
            tier_bc.append(req)

    if tier_bc:
        lines.append("")
        lines.append("Still needed — manual (Tier B/C):")
        for req in tier_bc:
            req_name = req.get("name", "")
            req_desc = req.get("description", req_name)
            req_level = req.get("level", "")
            req_geo = req.get("geo", geo_hint)

            # Determine tier based on data type
            level = _infer_level(req_desc, req_level)
            geo = _infer_geo(req_desc, req_geo)

            if "micro" in req_desc.lower() or "survey" in req_desc.lower():
                tier = "B"
                suggestion = "Micro/survey data. Requires registration."
            elif level in ("city", "province") and geo == "cn":
                tier = "C"
                suggestion = ("May be available in statistical yearbooks. "
                             "Use extract_yearbook.py or manual collection.")
            elif "policy" in req_desc.lower() or "pilot" in req_desc.lower():
                tier = "C"
                suggestion = "Policy documents / manual collection."
            else:
                tier = "B"
                suggestion = "Not found in public APIs. Check data_sources.md."

            lines.append(f"├── ✗ {req_name} → Tier {tier}")
            lines.append(f"│   Description: {req_desc}")
            lines.append(f"│   Suggestion: {suggestion}")
            if tier == "C":
                lines.append(f"│   Template: data/manual/{req_name}.xlsx")

    # ── Summary ────────────────────────────────────────────────────
    n_auto = len(fetched_vars)
    n_tier_a = len(tier_a)
    n_tier_bc = len(tier_bc)
    lines.append("")
    lines.append("─────────────────────────────────")
    lines.append(f"Summary: {n_auto} fetched | {n_tier_a} Tier A available "
                 f"| {n_tier_bc} Tier B/C needed")
    lines.append("═══════════════════════════════════")

    return "\n".join(lines)


def match_stage4_requirements(essential_vars: list[dict],
                               optional_vars: list[dict] = None,
                               geo_hint: str = "cn",
                               auto_fetch: bool = True,
                               start_year: int = 2010,
                               end_year: int = 2025,
                               max_workers: int = 6) -> dict:
    """
    Full Stage 4 → Stage 5 bridge: match, fetch, and report gaps.

    This is the main entry point for the LLM in Stage 5. Given the
    data requirements from Stage 4, it:
      1. Searches variable_map.json for matches
      2. Auto-fetches Tier A variables with good matches
      3. Generates a gap report for Tier B/C variables

    Parameters
    ----------
    essential_vars : list[dict]
        Required variables from Stage 4. Each with name, description, level, geo.
    optional_vars : list[dict] or None
        Optional control variables.
    geo_hint : str
        Geographic context hint ("cn", "us", "eu", "global").
    auto_fetch : bool
        If True, automatically fetch Tier A matches.
    start_year, end_year : int
    max_workers : int
        Parallel workers for fetching.

    Returns
    -------
    dict with keys:
      - search_results: full search output
      - fetch_results: FetchResult dict for auto-fetched variables
      - gap_report: formatted gap report string
      - tier_a_vars: list of variable_map names ready to fetch
      - tier_bc_vars: list of requirements needing manual collection
    """
    all_vars = list(essential_vars)
    if optional_vars:
        all_vars.extend(optional_vars)

    # Step 1: Search
    print("  Searching variable_map.json for matches...")
    search_results = search_variable_map(all_vars)
    n_matched = sum(1 for v, matches in search_results.items()
                    if matches and matches[0]["score"] >= 0.5)
    print(f"  Found matches for {n_matched}/{len(all_vars)} required variables")

    # Step 2: Determine Tier A variables to fetch
    tier_a_vars = []
    for req in all_vars:
        req_name = req.get("name", "")
        matches = search_results.get(req_name, [])
        if matches and matches[0]["score"] >= 0.5:
            best_var = matches[0]["variable"]
            if best_var not in tier_a_vars:
                tier_a_vars.append(best_var)

    # Step 3: Auto-fetch Tier A
    fetch_results = {}
    if auto_fetch and tier_a_vars:
        print(f"  Auto-fetching {len(tier_a_vars)} Tier A variables...")
        fetch_results = fetch_from_variable_map(
            tier_a_vars, geo_hint, start_year, end_year,
            save=True, force=False, max_workers=max_workers)
    elif tier_a_vars:
        fetch_results = {v: FetchResult(variable=v, status="pending",
                          warnings=["auto_fetch disabled"])
                         for v in tier_a_vars}

    # Step 4: Generate gap report
    gap_report = generate_gap_report(all_vars, search_results, fetch_results,
                                     geo_hint)

    # Step 5: Identify still-needed Tier B/C
    tier_bc_vars = []
    for req in all_vars:
        req_name = req.get("name", "")
        if req_name in fetch_results and fetch_results[req_name].is_ok():
            continue
        matches = search_results.get(req_name, [])
        if not matches or matches[0]["score"] < 0.5:
            tier_bc_vars.append(req)

    return {
        "search_results": search_results,
        "fetch_results": fetch_results,
        "gap_report": gap_report,
        "tier_a_vars": tier_a_vars,
        "tier_bc_vars": tier_bc_vars,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Data fetch module for policy evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search variable map for matching data sources
  python fetch_data.py --match "city level GDP" --geo cn
  python fetch_data.py --match "fertility rate"

  # Generate gap report from Stage 4 requirements
  python fetch_data.py --gap-report data/auto/stage4_requirements.json \\
      --geo cn --start 2015 --end 2020

  # World Bank indicator
  python fetch_data.py --source wb --indicator gdp_per_capita \\
      --countries CN,IN --start 2010 --end 2020

  # China city/province macro
  python fetch_data.py --source akshare_city --indicator gdp --start 2015 --end 2020

  # Any akshare config-driven entry (stock indices, bonds, PMI, exchange rates, etc.)
  python fetch_data.py --source akshare --indicator shanghai_index
  python fetch_data.py --source akshare --indicator bond_yield_curve
  python fetch_data.py --source akshare --indicator a_stock_daily --symbol 600519
  python fetch_data.py --source akshare --indicator cny_usd
  python fetch_data.py --source akshare --indicator pmi
  python fetch_data.py --source akshare --indicator money_supply

  # AQI
  python fetch_data.py --source akshare_custom --indicator aqi --symbol 北京

  # Global index
  python fetch_data.py --source yfinance_global_index --indicator sp500

  # US open data
  python fetch_data.py --source nyc_taxi --indicator yellow --start 2024
  python fetch_data.py --source citi_bike --start 2024
  python fetch_data.py --source epa_aqs --indicator pm25 --symbol 36
  python fetch_data.py --source noaa_gsod --symbol 725030
  python fetch_data.py --source bls --indicator unemployment_national
  python fetch_data.py --source zillow --indicator metro --symbol "New York, NY"
  python fetch_data.py --source nyc_311 --start 2024 --end 2024
  python fetch_data.py --source chicago_crimes --start 2024
  python fetch_data.py --source ipums --indicator acs1 --start 2023

  # Census / NBS / FRED
  python fetch_data.py --source census --indicator total_population
  python fetch_data.py --source nbs --indicator gdp --start 2015 --end 2020
  python fetch_data.py --source fred --indicator gdp
  python fetch_data.py --source oecd --indicator gdp_quarterly
  python fetch_data.py --source eurostat --indicator gdp

  # Batch from variable map
  python fetch_data.py --from-map gdp,shanghai_index,cny_usd,pmi \\
      --start 2010 --end 2020

  # Batch with cache age limit (re-fetch if older than 7 days)
  python fetch_data.py --from-map gdp,population --max-age 168
""",
    )
    parser.add_argument("--source", default=None,
                        choices=["wb", "akshare", "akshare_city", "akshare_province",
                                 "akshare_custom", "yfinance_global_index",
                                 "census", "nbs", "fred", "oecd", "eurostat",
                                 "nyc_taxi", "nyc_taxi_agg", "citi_bike",
                                 "epa_aqs", "noaa_gsod", "bls", "fec",
                                 "zillow", "nyc_311", "chicago_crimes", "ipums"],
                        help="Data source type")
    parser.add_argument("--indicator", default=None, help="Variable name or indicator code")
    parser.add_argument("--symbol", default=None, help="Stock code / city name override")
    parser.add_argument("--countries", default="CN,US,IN", help="Country codes (for WB)")
    parser.add_argument("--start", type=int, default=2010, help="Start year")
    parser.add_argument("--end", type=int, default=2025, help="End year")
    parser.add_argument("--from-map", default=None,
                        help="Comma-separated variable names from variable_map.json")
    parser.add_argument("--region", default="cn")
    parser.add_argument("--max-age", type=float, default=None,
                        help="Max cache age in hours (re-fetch if older)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Max parallel workers for batch fetch (default: 6)")
    parser.add_argument("--match", default=None,
                        help="Search variable_map for a description or keyword")
    parser.add_argument("--gap-report", default=None,
                        help="Generate gap report: path to JSON file with "
                             "Stage 4 requirements array")
    parser.add_argument("--geo", default="cn",
                        help="Geography hint for matching (cn/us/eu/global)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if cached")

    args = parser.parse_args()

    global DATA_AUTO
    if args.output_dir:
        DATA_AUTO = Path(args.output_dir)

    # ── Mode 0a: Search variable map ───────────────────────────────────
    if args.match:
        requirements = [{"name": args.match, "description": args.match,
                         "level": "", "geo": args.geo}]
        results = search_variable_map(requirements, top_k=10)
        matches = results.get(args.match, [])
        if matches:
            print(f"\nTop matches for '{args.match}' (geo={args.geo}):\n")
            for i, m in enumerate(matches, 1):
                print(f"  {i}. {m['variable']} (score: {m['score']})")
                print(f"     {m['description']}")
                print(f"     source: {m['source']}, level: {m['level']}")
                print()
        else:
            print(f"No matches found for '{args.match}'")
        return

    # ── Mode 0b: Generate gap report ───────────────────────────────────
    elif args.gap_report:
        req_path = Path(args.gap_report)
        if not req_path.exists():
            print(f"Requirements file not found: {req_path}")
            return
        with open(req_path, encoding="utf-8") as f:
            data = json.load(f)

        # Accept two formats:
        #   Format A: [{"name": "gdp", ...}, ...]
        #   Format B: {"essential": [...], "optional": [...]}
        if isinstance(data, list):
            essential_vars = data
            optional_vars = []
        elif isinstance(data, dict):
            essential_vars = data.get("essential", data.get("required", []))
            optional_vars = data.get("optional", data.get("controls", []))
        else:
            print("Requirements must be a JSON array or {essential: [...], optional: [...]}")
            return

        result = match_stage4_requirements(
            essential_vars, optional_vars,
            geo_hint=args.geo,
            auto_fetch=False,  # Don't auto-fetch from gap report mode
            start_year=args.start,
            end_year=args.end,
            max_workers=args.workers,
        )
        print(f"\n{result['gap_report']}")
        return

    # ── Mode 1: Single source + indicator ─────────────────────────────
    elif args.source and args.indicator:
        if args.source == "wb":
            countries = [c.strip() for c in args.countries.split(",")]
            df = fetch_wb_indicator(args.indicator, countries, args.start, args.end)

        elif args.source == "akshare_city":
            df = fetch_cn_city_macro(args.indicator, args.start, args.end,
                                     incremental=True)

        elif args.source == "akshare_province":
            df = fetch_cn_province_macro(args.indicator, args.start, args.end,
                                         incremental=True)

        elif args.source == "akshare":
            # Load the entry from variable_map.json
            map_path = SKILL_ROOT / "references" / "variable_map.json"
            with open(map_path, encoding="utf-8") as f:
                var_map = json.load(f)
            entry = var_map.get(args.indicator, {})
            if not entry:
                print(f"'{args.indicator}' not found in variable_map.json")
                return
            df = fetch_akshare(entry, indicator_override=args.symbol)

        elif args.source == "akshare_custom":
            if args.indicator == "aqi":
                city = args.symbol or "北京"
                df = fetch_cn_aqi(city)
            else:
                print(f"Unknown custom indicator: {args.indicator}")
                return

        elif args.source == "yfinance_global_index":
            df = fetch_global_stock_index(args.indicator)

        elif args.source == "census":
            df = fetch_cn_census(args.indicator)

        elif args.source == "nbs":
            df = fetch_nbs_data(args.indicator, start_year=args.start,
                                end_year=args.end)

        elif args.source == "fred":
            df = fetch_fred(args.indicator)

        elif args.source == "oecd":
            countries = [c.strip() for c in args.countries.split(",")]
            df = fetch_oecd(args.indicator, countries, args.start, args.end)

        elif args.source == "eurostat":
            countries = [c.strip() for c in args.countries.split(",")]
            df = fetch_eurostat(args.indicator, countries, args.start, args.end)

        elif args.source == "nyc_taxi":
            df = fetch_nyc_taxi(dataset=args.indicator or "yellow",
                                year=args.start, month=1,
                                sample_rows=args.symbol and int(args.symbol) or 50000)

        elif args.source == "nyc_taxi_agg":
            df = fetch_nyc_taxi_aggregate(dataset=args.indicator or "yellow",
                                          year=args.start,
                                          agg_col=args.symbol or "total_amount")

        elif args.source == "citi_bike":
            df = fetch_citi_bike(year=args.start, month=1,
                                 sample_rows=args.symbol and int(args.symbol) or None)

        elif args.source == "epa_aqs":
            param = args.indicator or "pm25"
            state_code = args.symbol or "36"
            df = fetch_epa_aqs(param=param, state_code=state_code,
                               start_year=args.start, end_year=args.end)

        elif args.source == "noaa_gsod":
            station = args.symbol or args.indicator or "725030"
            df = fetch_noaa_gsod(station_id=station,
                                 start_year=args.start, end_year=args.end)

        elif args.source == "bls":
            df = fetch_bls(series_id=args.indicator or "LNS14000000",
                           start_year=args.start, end_year=args.end)

        elif args.source == "fec":
            df = fetch_fec(endpoint=args.indicator or "candidates",
                           params={"election_year": args.start, "state": args.symbol or "NY"})

        elif args.source == "zillow":
            df = fetch_zillow_hvi(geography=args.indicator or "metro",
                                  geo_name=args.symbol or "New York, NY")

        elif args.source == "nyc_311":
            df = fetch_nyc_311(limit=args.symbol and int(args.symbol) or 10000,
                               complaint_type=args.indicator or None,
                               start_date=f"{args.start}-01-01",
                               end_date=f"{args.end}-12-31")

        elif args.source == "chicago_crimes":
            df = fetch_chicago_crimes(limit=args.symbol and int(args.symbol) or 10000,
                                      year=args.start,
                                      crime_type=args.indicator or None)

        elif args.source == "ipums":
            df = fetch_ipums_usa(dataset=args.indicator or "acs1",
                                 sample=f"us{args.start}a",
                                 variables=args.symbol.split(",") if args.symbol else None)

        else:
            print(f"Unknown source: {args.source}")
            return

        if df.empty:
            print("No data returned.")
        else:
            print(f"\nFetched {len(df)} rows.")
            print(df.head(10))
            save_to_auto_with_meta(df, args.indicator, force=args.force,
                                    meta={"source": args.source})

    # ── Mode 2: Batch from variable map ───────────────────────────────
    elif args.from_map:
        variables = [v.strip() for v in args.from_map.split(",")]
        results = fetch_from_variable_map(
            variables, args.region, args.start, args.end,
            save=True, force=args.force,
            indicator_override=args.symbol,
            max_age_hours=args.max_age,
            max_workers=args.workers,
        )
        print(f"\n─── Summary ({len(variables)} variables) ───")
        ok = sum(1 for r in results.values() if r.is_ok())
        failed = len(results) - ok
        for var, result in results.items():
            status = "✓" if result.is_ok() else "✗"
            print(f"  {status} {result.describe()}")
        print(f"\n{ok} ok, {failed} failed")

        # Save a batch manifest
        manifest = {
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "variables": {v: {"status": r.status, "path": r.path,
                              "rows": r.rows, "errors": r.errors}
                         for v, r in results.items()},
        }
        manifest_path = DATA_AUTO / "batch_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
