"""
Extract tabular data from statistical yearbooks (HTML and PDF).

Used as a fallback in fetch_data.py's fetch_cn_city_macro() when akshare
and NBS API both fail for a given year. The system automatically:

  1. Searches known government statistics sites for yearbook pages
  2. Downloads the HTML page or PDF
  3. Locates the relevant table (by indicator name)
  4. Extracts, cleans, and standardizes → [entity_id, year, value]

Prefer HTML extraction (stats.gov.cn yearbooks are HTML tables).
Fall back to pdfplumber for PDF sources.

Usage:
    from extract_yearbook import extract_yearbook_table
    df = extract_yearbook_table(indicator="gdp", year=2019, level="city")
"""

import json
import re
import time
import urllib.request
import urllib.parse
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# Yearbook URL templates — government statistics sites
# ═══════════════════════════════════════════════════════════════════════

# China Statistical Yearbook (国家统计年鉴) — national/province level
# Format: HTML tables at stats.gov.cn
NATIONAL_YEARBOOK_URLS = [
    "https://www.stats.gov.cn/sj/ndsj/{year}/indexch.htm",
    "https://www.stats.gov.cn/tjsj/ndsj/{year}/indexch.htm",
]

# China City Statistical Yearbook (中国城市统计年鉴) — city level
# Usually available as PDF from various sources
CITY_YEARBOOK_SEARCH_URLS = [
    "https://www.stats.gov.cn/sj/ndsj/",
    "https://data.stats.gov.cn/easyquery.htm",
]

# Provincial yearbook patterns
PROVINCE_YEARBOOK_PATTERNS = [
    "https://tjj.{province}.gov.cn/tjsj/ndsj/{year}/",
    "https://tjj.{province}.gov.cn/sj/ndsj/{year}/",
]

# Province name → pinyin abbreviation for URLs
PROVINCE_PINYIN = {
    "北京": "beijing", "天津": "tianjin", "河北": "hebei",
    "山西": "shanxi", "内蒙古": "nmg", "辽宁": "ln",
    "吉林": "jl", "黑龙江": "hlj", "上海": "shanghai",
    "江苏": "jiangsu", "浙江": "zj", "安徽": "ah",
    "福建": "fujian", "江西": "jiangxi", "山东": "shandong",
    "河南": "henan", "湖北": "hubei", "湖南": "hunan",
    "广东": "gd", "广西": "gx", "海南": "hainan",
    "重庆": "cq", "四川": "sc", "贵州": "gz",
    "云南": "yn", "西藏": "xizang", "陕西": "shaanxi",
    "甘肃": "gansu", "青海": "qh", "宁夏": "nx",
    "新疆": "xinjiang",
}


# ═══════════════════════════════════════════════════════════════════════
# Indicator name → table keyword mapping (Chinese)
# ═══════════════════════════════════════════════════════════════════════

INDICATOR_TABLE_KEYWORDS = {
    "gdp": ["地区生产总值", "生产总值", "GDP", "国内生产总值"],
    "population": ["年末总人口", "常住人口", "总人口", "人口"],
    "fiscal_revenue": ["地方财政一般预算收入", "一般公共预算收入", "财政收入"],
    "fiscal_expenditure": ["地方财政一般预算支出", "一般公共预算支出", "财政支出"],
    "retail_sales": ["社会消费品零售总额", "消费品零售"],
    "industrial_output": ["规模以上工业总产值", "工业总产值"],
    "fixed_investment": ["固定资产投资", "固定资产投资总额"],
    "savings": ["城乡居民储蓄存款余额", "储蓄存款"],
    "foreign_trade": ["进出口总额", "进出口"],
    "education_expenditure": ["教育支出"],
    "urbanization_rate": ["城镇化率", "城镇人口比重"],
    "employment": ["从业人员", "就业人员", "城镇从业人员"],
    "avg_wage": ["职工平均工资", "平均工资", "在岗职工平均工资"],
    "hospital_beds": ["医院卫生院床位数", "医院床位数", "医疗卫生机构床位数"],
    "secondary_students": ["普通中学在校学生", "中学在校学生"],
    "gdp_per_capita": ["人均地区生产总值", "人均GDP"],
}


# ═══════════════════════════════════════════════════════════════════════
# HTML table extraction (preferred — stats.gov.cn)
# ═══════════════════════════════════════════════════════════════════════

def _fetch_html(url: str, timeout: int = 20) -> Optional[str]:
    """Download an HTML page and return its text content."""
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36")
        req.add_header("Accept", "text/html,application/xhtml+xml")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # Detect encoding
            content_type = resp.headers.get("Content-Type", "")
            if "charset=gb" in content_type.lower() or "charset=gb2312" in content_type.lower():
                return resp.read().decode("gb2312", errors="replace")
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    HTML fetch failed: {type(e).__name__}: {str(e)[:100]}")
        return None


def _extract_html_tables(html: str) -> list[pd.DataFrame]:
    """Extract all tables from an HTML page using pandas.read_html."""
    try:
        from io import StringIO
        tables = pd.read_html(StringIO(html))
        # Filter to tables with at least 5 rows and 3 columns
        return [t for t in tables if t.shape[0] >= 5 and t.shape[1] >= 3]
    except Exception:
        # Fallback: manual extraction of <table> elements
        tables = []
        table_pattern = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
        for match in table_pattern.finditer(html):
            try:
                t = pd.read_html(match.group(0))
                if t:
                    tables.extend(t)
            except Exception:
                continue
        return [t for t in tables if t.shape[0] >= 5 and t.shape[1] >= 3]


def _find_indicator_table(tables: list[pd.DataFrame],
                           indicator: str) -> Optional[pd.DataFrame]:
    """Find the table most likely to contain the target indicator."""
    keywords = INDICATOR_TABLE_KEYWORDS.get(indicator, [indicator])

    best_table = None
    best_score = 0

    for t in tables:
        score = 0
        # Check first row (likely header)
        first_row = t.iloc[0].astype(str).str.cat(sep=" ")
        for kw in keywords:
            if kw in first_row:
                score += 10
        # Check column names
        for col in t.columns:
            col_str = str(col)
            for kw in keywords:
                if kw in col_str:
                    score += 15
        # Prefer tables with city-like entries in first column
        first_col = t.iloc[:, 0].astype(str)
        city_indicators = first_col.str.contains("市|省|区|州|盟|地区")
        score += city_indicators.sum()

        if score > best_score:
            best_score = score
            best_table = t

    return best_table if best_score >= 5 else None


def _parse_yearbook_html_table(df: pd.DataFrame, indicator: str,
                                year: int) -> pd.DataFrame:
    """
    Parse a raw HTML table into standardized [entity_id, year, value] format.

    Handles common yearbook table layouts:
      - Merged title rows (same value repeated across all columns) → skip
      - Multi-level headers (地区生产总值 / 绝对值 / 增长%)
      - City/region names in first column
      - Indicator values in columns matching keywords
    """
    df = df.copy()

    # ── Skip merged title rows ──────────────────────────────────────
    # A row where all values are the same string is a merged title
    while len(df) > 0:
        first_vals = df.iloc[0].astype(str)
        unique_vals = first_vals.nunique()
        # All same value = merged title row; skip it
        if unique_vals <= 1:
            df = df.iloc[1:].reset_index(drop=True)
            continue
        # Unit row: contains "单位：" → skip
        if first_vals.str.contains("单位：").any():
            df = df.iloc[1:].reset_index(drop=True)
            continue
        break

    if len(df) < 2:
        return pd.DataFrame()

    # ── Handle multi-level headers ───────────────────────────────────
    # Check if the first 1-2 rows look like headers (contain indicator keywords)
    keywords = INDICATOR_TABLE_KEYWORDS.get(indicator, [indicator])
    header_rows = 0

    for i in range(min(3, len(df))):
        row_text = df.iloc[i].astype(str).str.cat(sep=" ")
        has_keyword = any(kw in row_text for kw in keywords)
        has_unit = "单位" in row_text or "增长" in row_text or "绝对" in row_text
        # Also check: is this row mostly text (not numbers)?
        numeric_count = 0
        for v in df.iloc[i]:
            try:
                float(str(v).replace(",", "").replace(" ", ""))
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        mostly_text = numeric_count < len(df.columns) * 0.3

        if has_keyword or has_unit or (mostly_text and i < 2):
            header_rows = i + 1
        else:
            break

    if header_rows > 0:
        # Build column names from the last header row
        header_cols = []
        for j in range(len(df.columns)):
            col_parts = []
            for i in range(header_rows):
                val = str(df.iloc[i, j]).strip()
                if val and val != "nan":
                    col_parts.append(val)
            header_cols.append("_".join(col_parts) if col_parts else str(df.columns[j]))
        df = df.iloc[header_rows:].reset_index(drop=True)
        df.columns = header_cols
    else:
        # No header found, try column names
        col_names_are_numeric = any(
            str(c).replace(".", "").replace(" ", "").isdigit()
            for c in df.columns[:3]
        )
        if col_names_are_numeric and len(df) > 1:
            new_cols = [str(c).strip().replace("\n", "").replace("\r", "")
                        for c in df.iloc[0]]
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = new_cols

    # Find entity column (first column usually, contains region names)
    entity_col_real = df.columns[0]
    # If a better candidate exists (with "地区" in name), use it
    for col in df.columns:
        col_str = str(col)
        if "地区" in col_str and "生产总值" not in col_str:
            entity_col_real = col
            break

    # Find value column — prefer "绝对值" for indicators, or keyword match
    value_col = None

    # Priority 1: column containing both the indicator keyword AND "绝对值"
    for col in df.columns[1:]:
        col_str = str(col).replace(" ", "")
        for kw in keywords:
            kw_clean = kw.replace(" ", "")
            if kw_clean[:3] in col_str and "绝对值" in col_str:
                value_col = col
                break
        if value_col:
            break

    # Priority 2: column with just the indicator keyword
    if value_col is None:
        for col in df.columns[1:]:
            col_str = str(col).replace(" ", "")
            for kw in keywords:
                kw_clean = kw.replace(" ", "")
                if kw_clean[:3] in col_str or kw_clean in col_str:
                    value_col = col
                    break
            if value_col:
                break

    # Priority 3: first numeric column after entity
    if value_col is None:
        for col in df.columns[1:]:
            try:
                pd.to_numeric(df[col], errors="raise")
                value_col = col
                break
            except (ValueError, TypeError):
                continue

    # Priority 4: just use second column
    if value_col is None and len(df.columns) > 1:
        value_col = df.columns[1]

    # Build standardized output
    rows = []
    for _, row in df.iterrows():
        entity_raw = str(row[entity_col_real]).strip()
        # Skip non-entity rows
        if not entity_raw or len(entity_raw) < 2:
            continue
        if entity_raw in ("nan", "None", ""):
            continue
        # Skip footnote/annotation rows
        if any(skip in entity_raw for skip in
               ["注：", "说明：", "数据来源", "资料来源", "注:", "—"]):
            continue

        val = row[value_col]
        try:
            val = float(str(val).replace(",", "").replace(" ", "")
                       .replace("—", "").replace("…", ""))
        except (ValueError, TypeError):
            continue
        if pd.isna(val) or val <= 0:
            continue

        rows.append({
            "entity_id": entity_raw,
            "year": year,
            "value": val,
        })

    return pd.DataFrame(rows)


def _try_parse_value(val_str: str) -> Optional[float]:
    """Parse a Chinese-formatted number string to float."""
    val_str = str(val_str).strip()
    # Handle Chinese units
    multipliers = {"万": 1e4, "亿": 1e8, "万亿": 1e12,
                   "千": 1e3, "百": 1e2, "%": 0.01}
    for unit, mult in multipliers.items():
        if unit in val_str:
            num_part = val_str.replace(unit, "").replace(",", "").replace(" ", "")
            try:
                return float(num_part) * mult
            except ValueError:
                continue
    # Plain number
    try:
        return float(val_str.replace(",", "").replace(" ", ""))
    except ValueError:
        return None


# ═══════════════════════════════════════════════════════════════════════
# PDF table extraction (fallback)
# ═══════════════════════════════════════════════════════════════════════

def _download_pdf(url: str, timeout: int = 60) -> Optional[Path]:
    """Download a PDF file to a temp location. Returns path or None."""
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
                # Check if it's actually HTML (redirect to login page etc.)
                body = resp.read(min(1024, int(resp.headers.get("Content-Length", 1024))))
                if body.startswith(b"<!DOCTYPE") or body.startswith(b"<html"):
                    return None
            # Re-download full content
            with urllib.request.urlopen(req, timeout=timeout) as resp2:
                content = resp2.read()
            if len(content) < 1000:  # Too small to be a PDF
                return None
            tmp = Path(tempfile.gettempdir()) / f"yearbook_{hash(url)}.pdf"
            tmp.write_bytes(content)
            return tmp
    except Exception as e:
        print(f"    PDF download failed: {type(e).__name__}: {str(e)[:100]}")
        return None


def _extract_pdf_tables(pdf_path: Path) -> list[list[list]]:
    """Extract all tables from a PDF using pdfplumber. Returns list of 2D arrays."""
    try:
        import pdfplumber
    except ImportError:
        print("    pdfplumber not installed. Install: pip install pdfplumber")
        return []

    tables = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    for t in page_tables:
                        if len(t) >= 3:  # at least header + 2 data rows
                            tables.append(t)
    except Exception as e:
        print(f"    PDF extraction error: {type(e).__name__}: {str(e)[:100]}")
        return []

    return tables


def _parse_pdf_table(table_2d: list[list],
                      indicator: str, year: int) -> Optional[pd.DataFrame]:
    """
    Parse a 2D table from PDF into standardized [entity_id, year, value].

    Handles:
      - Merged cells (repeated values via ffill)
      - Header detection
      - City name column identification
      - Numeric value extraction with Chinese unit handling
    """
    if not table_2d or len(table_2d) < 2:
        return None

    # Convert to DataFrame
    max_cols = max(len(row) for row in table_2d if row)
    padded = [list(row) + [None] * (max_cols - len(row))
              for row in table_2d if row]
    df = pd.DataFrame(padded)

    # Forward-fill merged cells
    df = df.fillna(method="ffill")

    # Drop entirely empty columns
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 2:
        return None

    # Try to identify header
    keywords = INDICATOR_TABLE_KEYWORDS.get(indicator, [indicator])
    header_row_idx = 0
    for i in range(min(5, len(df))):
        row_text = df.iloc[i].astype(str).str.cat(sep=" ")
        if any(kw in row_text for kw in keywords):
            header_row_idx = i
            break

    # Set header
    if header_row_idx > 0:
        new_cols = df.iloc[header_row_idx].astype(str).tolist()
        df = df.iloc[header_row_idx + 1:].reset_index(drop=True)
        df.columns = new_cols

    # Find entity column (first column or column with city/region names)
    entity_col = df.columns[0]
    # Find value column
    value_col = None
    for col in df.columns[1:]:
        col_str = str(col)
        for kw in keywords:
            if kw[:3] in col_str:
                value_col = col
                break
        if value_col:
            break
    if value_col is None:
        # Try the column with most numeric values
        best_numeric = 0
        for col in df.columns[1:]:
            numeric_count = df[col].apply(
                lambda x: _try_parse_value(str(x)) is not None).sum()
            if numeric_count > best_numeric:
                best_numeric = numeric_count
                value_col = col
    if value_col is None and len(df.columns) > 1:
        value_col = df.columns[1]

    # Build rows
    rows = []
    for _, row in df.iterrows():
        entity_raw = str(row[entity_col]).strip()
        if not entity_raw or entity_raw in ("None", "nan", ""):
            continue

        val = _try_parse_value(str(row[value_col]))
        if val is None or val <= 0:
            continue

        rows.append({
            "entity_id": entity_raw,
            "year": year,
            "value": val,
        })

    return pd.DataFrame(rows) if rows else None


# ═══════════════════════════════════════════════════════════════════════
# Search for yearbook PDFs online
# ═══════════════════════════════════════════════════════════════════════

def _navigate_yearbook_index(index_url: str, indicator: str) -> list[str]:
    """
    Navigate from a yearbook index page to relevant table sub-pages.

    stats.gov.cn yearbook index pages (e.g., indexch.htm) list links to
    individual HTML table pages. This function extracts those links and
    returns sub-page URLs matching the indicator keywords.
    """
    html = _fetch_html(index_url)
    if html is None:
        return []

    keywords = INDICATOR_TABLE_KEYWORDS.get(indicator, [indicator])

    # Extract <a> tags with href attributes
    links = re.findall(r'<a[^>]*href=["\']([^"\']*\.htm[l]?)["\'][^>]*>'
                       r'([^<]*)</a>', html, re.IGNORECASE)

    base_url = index_url.rsplit("/", 1)[0]
    sub_pages = []

    for href, text in links:
        text_clean = text.strip()
        for kw in keywords:
            if kw[:3] in text_clean or kw[:2] in text_clean:
                # Resolve relative URL
                if href.startswith("http"):
                    full_url = href
                elif href.startswith("/"):
                    # Absolute path from domain root
                    from urllib.parse import urlparse
                    parsed = urlparse(index_url)
                    full_url = f"{parsed.scheme}://{parsed.netloc}{href}"
                else:
                    full_url = f"{base_url}/{href}"
                sub_pages.append(full_url)
                break

    return sub_pages


def _search_yearbook_pdf_url(indicator: str, year: int) -> list[str]:
    """
    Search for statistical yearbook URLs.

    Uses known URL templates for sites confirmed to have accessible
    HTML tables. The yearbook year is typically data_year + 1
    (e.g., 2019 data is in the 2020 yearbook).

    Returns a list of candidate URLs ordered by likelihood.
    """
    yearbook_year = year + 1
    urls = []

    # 1. Known accessible government yearbook servers
    # TRS CMS static file servers (confirmed working — Quanzhou pattern)
    known_trs_servers = [
        # Quanzhou (confirmed working)
        f"http://tjj.quanzhou.gov.cn/tsys/UpLoadFiles/43sjfb/129ndsj/qztjnj{yearbook_year}/cn/html/1601.htm",
        f"http://tjj.quanzhou.gov.cn/tsys/UpLoadFiles/43sjfb/129ndsj/qztjnj{yearbook_year}/cn/html/1704.htm",
        # Try neighboring cities with similar CMS
        f"http://tjj.zhangzhou.gov.cn/tsys/UpLoadFiles/tjnj{yearbook_year}/cn/html/1601.htm",
        f"http://tjj.xm.gov.cn/tsys/UpLoadFiles/tjnj{yearbook_year}/cn/html/1601.htm",
    ]

    # 2. Older static HTML yearbooks (confirmed accessible — Shandong pattern)
    known_static_servers = [
        f"http://tjj.shandong.gov.cn/tjnj/nj{yearbook_year}/NJF1.htm",
        f"http://tjj.shandong.gov.cn/tjnj/nj{yearbook_year}/indexch.htm",
    ]

    # 3. stats.gov.cn yearbook index (may redirect, try anyway)
    national_templates = [
        f"https://www.stats.gov.cn/sj/ndsj/{yearbook_year}/indexch.htm",
        f"https://www.stats.gov.cn/sj/ndsj/{year}/indexch.htm",
    ]

    # 4. Aggregator sites (may need JS, try as fallback)
    aggregator_urls = [
        f"https://www.tjnj.net/navipage-n3020013291000111.html",
        f"https://www.tjnj.net/navipage-n3023071310000417.html",
    ]

    urls.extend(known_trs_servers)
    urls.extend(known_static_servers)
    urls.extend(national_templates)
    urls.extend(aggregator_urls)

    return urls


# ═══════════════════════════════════════════════════════════════════════
# Main extraction function — the one called by fetch_data.py
# ═══════════════════════════════════════════════════════════════════════

def extract_from_html_content(html: str, indicator: str, year: int,
                               level: str = "city") -> pd.DataFrame:
    """
    Extract indicator data from raw HTML content.

    This is the key entry point for LLM-orchestrated extraction:
      1. LLM uses WebSearch to find yearbook pages
      2. LLM uses WebFetch to get the HTML content
      3. LLM calls this function with the fetched HTML

    Parameters
    ----------
    html : str
        Raw HTML content (from WebFetch or any HTTP client).
    indicator : str
        Indicator key (e.g., "gdp", "population").
    year : int
        Data year.
    level : str
        "city" or "province".

    Returns
    -------
    pd.DataFrame with [entity_id, year, value].
    """
    tables = _extract_html_tables(html)
    if not tables:
        print(f"    [extract_from_html] No tables found in {len(html)} chars")
        return pd.DataFrame()

    # Find the sub-pages if this is an index page
    indicator_url = None
    sub_pages = _navigate_yearbook_index_str(html, indicator)
    if sub_pages:
        for sub_url in sub_pages[:3]:
            sub_html = _fetch_html(sub_url)
            if sub_html:
                sub_tables = _extract_html_tables(sub_html)
                if sub_tables:
                    tables.extend(sub_tables)

    target = _find_indicator_table(tables, indicator)
    if target is None:
        # Try fuzzy: accept any table with city-like entries
        for t in tables:
            first_col = t.iloc[:, 0].astype(str)
            if first_col.str.contains("市|省|区|州|盟|县").sum() >= 3:
                target = t
                break

    if target is None:
        return pd.DataFrame()

    df = _parse_yearbook_html_table(target, indicator, year)
    if not df.empty:
        n = df["entity_id"].nunique()
        print(f"    ✓ Extracted from HTML: {len(df)} rows, {n} entities")
    return df


def _navigate_yearbook_index_str(html: str, indicator: str) -> list[str]:
    """Same as _navigate_yearbook_index but takes HTML string instead of URL."""
    keywords = INDICATOR_TABLE_KEYWORDS.get(indicator, [indicator])
    links = re.findall(r'<a[^>]*href=["\']([^"\']*\.htm[l]?)["\'][^>]*>'
                       r'([^<]*)</a>', html, re.IGNORECASE)

    sub_pages = []
    for href, text in links:
        text_clean = text.strip()
        for kw in keywords:
            if kw[:3] in text_clean or kw[:2] in text_clean:
                sub_pages.append(href)
                break
    return sub_pages


def extract_yearbook_table(indicator: str, year: int,
                            level: str = "city") -> pd.DataFrame:
    """
    Auto-extract data from statistical yearbooks as a fallback data source.

    Called by fetch_cn_city_macro() / fetch_cn_province_macro() when akshare
    and NBS API both fail for a given year.

    Strategy (in order):
      1. Try stats.gov.cn HTML yearbook pages (fast and reliable)
      2. Search for and download PDF yearbooks, then extract tables
      3. If all fail, return empty DataFrame

    Parameters
    ----------
    indicator : str
        Indicator key (e.g., "gdp", "population").
    year : int
        The year of data to fetch (NOT the yearbook publication year).
        For yearbooks, this is usually the data year (e.g., 2019 data is
        in the 2020 yearbook).
    level : str
        "city" or "province".

    Returns
    -------
    pd.DataFrame with [entity_id, year, value].
    """
    print(f"    [yearbook] Attempting statistical yearbook for "
          f"{indicator} year={year}...")

    # Yearbooks contain previous year's data
    # e.g., 2020 yearbook has 2019 data
    yearbook_year = year + 1

    candidate_urls = _search_yearbook_pdf_url(indicator, year)
    errors = []

    # ── Phase 1: Try HTML pages (stats.gov.cn) ─────────────────────────
    for url in candidate_urls:
        if url.endswith(".pdf"):
            continue  # Will try in Phase 2

        html = _fetch_html(url)
        if html is None:
            errors.append(f"HTML fetch failed: {url}")
            continue

        # If this is an index page, navigate to sub-pages
        tables = _extract_html_tables(html)
        if not tables:
            sub_pages = _navigate_yearbook_index(url, indicator)
            if sub_pages:
                for sub_url in sub_pages[:3]:  # Try up to 3 sub-pages
                    sub_html = _fetch_html(sub_url)
                    if sub_html is None:
                        continue
                    sub_tables = _extract_html_tables(sub_html)
                    if sub_tables:
                        tables.extend(sub_tables)

        if not tables:
            errors.append(f"No tables found in HTML: {url}")
            continue

        target = _find_indicator_table(tables, indicator)
        if target is None:
            errors.append(f"Indicator '{indicator}' not found in "
                          f"{len(tables)} tables at {url[:80]}")
            continue

        df = _parse_yearbook_html_table(target, indicator, year)
        if not df.empty:
            n_entities = df["entity_id"].nunique()
            print(f"    ✓ Yearbook HTML: {len(df)} rows, "
                  f"{n_entities} entities from {url[:80]}")
            return df

    # ── Phase 2: Try PDF yearbooks ─────────────────────────────────────
    pdf_urls = [u for u in candidate_urls if u.endswith(".pdf")]
    # Also search common PDF repositories
    search_terms = [
        f"中国城市统计年鉴 {yearbook_year} PDF",
        f"城市统计年鉴 {yearbook_year} 下载",
        f"China City Statistical Yearbook {yearbook_year} PDF",
    ]

    for url in pdf_urls:
        pdf_path = _download_pdf(url)
        if pdf_path is None:
            errors.append(f"PDF download failed: {url[:80]}")
            continue

        tables_2d = _extract_pdf_tables(pdf_path)
        # Clean up temp file
        try:
            pdf_path.unlink()
        except Exception:
            pass

        if not tables_2d:
            errors.append(f"No tables extracted from PDF: {url[:80]}")
            continue

        for table_2d in tables_2d:
            df = _parse_pdf_table(table_2d, indicator, year)
            if df is not None and not df.empty:
                n_entities = df["entity_id"].nunique()
                print(f"    ✓ Yearbook PDF: {len(df)} rows, "
                      f"{n_entities} entities from {url[:80]}")
                return df

    # ── All attempts failed ────────────────────────────────────────────
    print(f"    ✗ Yearbook extraction failed for {indicator} year={year}")
    if errors:
        print(f"    Attempts: {len(errors)}")
        for err in errors[-3:]:  # Show last 3 errors
            print(f"      - {err[:120]}")

    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
# CLI (for standalone use)
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract tabular data from statistical yearbooks")
    parser.add_argument("--indicator", required=True,
                        help="Indicator key (e.g., gdp, population)")
    parser.add_argument("--year", type=int, required=True,
                        help="Data year (e.g., 2019)")
    parser.add_argument("--level", default="city", choices=["city", "province"])
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    df = extract_yearbook_table(args.indicator, args.year, args.level)

    if df.empty:
        print("No data extracted.")
    else:
        print(f"\nExtracted {len(df)} rows:")
        print(df.head(15))
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            df.to_json(args.output, orient="records", force_ascii=False)
            print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
