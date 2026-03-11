import os
import io
import re
import json
import time
import requests
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# PURE MOMENTUM RADAR
# ------------------------------------------------------------
# 철학
# - 펀더멘털 완전 배제
# - 순수 가격 모멘텀 돌파
# - 핵심 = 돌파 + RS + 거래량
# - 시장 레짐/브레드스/산업 강도는 해석 및 우선순위에 사용
# - 후보는 시장이 약세여도 항상 출력
#
# 이번 수정 반영
# 1) 유니버스 과도 축소 완화
#    - Healthcare 섹터 전체 제외 삭제
#    - 거래대금 기준 20M -> 10M 완화
# 2) 52주 신고가 근접 필터 완화
#    - 85% -> 80%
# 3) 베이스 깊이 완화
#    - 20% -> 25%
# 4) 계좌 A = 55일 돌파 / 계좌 B = 20일 돌파
# 5) 순수 모멘텀 돌파: 종가 > 50일선만 유지
# ============================================================

# ============================================================
# PATH / ENV
# ============================================================

DATA_DIR = "signals"
SUMMARY_JSON = os.path.join(DATA_DIR, "momentum_summary.json")
ALL_CSV = os.path.join(DATA_DIR, "momentum_all_candidates.csv")
A_CSV = os.path.join(DATA_DIR, "momentum_account_a_candidates.csv")
B_CSV = os.path.join(DATA_DIR, "momentum_account_b_candidates.csv")
META_CACHE_JSON = os.path.join(DATA_DIR, "momentum_meta_cache.json")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

os.makedirs(DATA_DIR, exist_ok=True)

USER_AGENT = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# ============================================================
# CONFIG
# ============================================================

# 유니버스 / 유동성
MIN_PRICE = 10.0
MIN_AVG_DOLLAR_VOLUME = 10_000_000
MIN_MARKET_CAP = 1_000_000_000
LOOKBACK_PERIOD = "1y"
CHUNK_SIZE = 100
CHUNK_SLEEP_SEC = 0.35

# 시장 레짐
SPY_50MA_DAYS = 50
SPY_200MA_DAYS = 200
SPY_3M_LOOKBACK = 63
SPY_6M_LOOKBACK = 126

# 브레드스
BREADTH_STRONG_50MA = 0.60
BREADTH_NEUTRAL_50MA = 0.45

# 모멘텀 점수
MOM_3M_LOOKBACK = 63
MOM_6M_LOOKBACK = 126
MOM_6M_WEIGHT = 0.60
MOM_3M_WEIGHT = 0.40

# RS 퍼센타일
MIN_RS_PERCENTILE_A = 80.0
MIN_RS_PERCENTILE_B = 75.0

# 돌파
BREAKOUT_A_DAYS = 55
BREAKOUT_B_DAYS = 20
NEAR_BREAKOUT_MIN = -0.03   # 돌파가 대비 -3%
NEAR_BREAKOUT_MAX = 0.02    # 돌파가 대비 +2%
ENTRY_BUFFER_PCT = 0.001    # 돌파가 0.1% 위
LIMIT_BUFFER_PCT = 0.03     # 허용 갭 +3%

# 손절
STOP_A_PCT = 0.12
STOP_B_PCT = 0.10

# 구조
TIGHT_10D_MAX = 0.08
STRICT_TIGHT_10D_MAX = 0.05
MAX_EXTENSION_FROM_50MA = 0.25
MAX_BASE_DEPTH = 0.25

# 거래량
MIN_VOLUME_RATIO = 1.0
RELATIVE_VOLUME_STRONG = 1.50
RELATIVE_VOLUME_VERY_STRONG = 2.00

# 52주 신고가 근접
MIN_HIGH_52W_PROXIMITY = 0.80
PREFERRED_HIGH_52W_PROXIMITY = 0.90

# 출력 개수
MAX_OUTPUT_PER_BLOCK = 12

# 산업 ETF
SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Real Estate": "XLRE",
}

# 제외 대상
EXCLUDED_SECTORS = set()

EXCLUDED_INDUSTRY_KEYWORDS = [
    "biotech",
    "biotechnology",
    "drug manufacturers",
    "pharmaceutical",
    "pharmaceuticals",
    "therapeutics",
    "clinical",
    "genomics",
    "shell companies",
    "shell company",
    "specialty pharma",
    "specialty pharmaceutical",
    "spac",
    "blank check",
]

EXCLUDED_NAME_KEYWORDS = [
    "adr",
    "sponsored adr",
    "depositary",
    "biotech",
    "biotechnology",
    "acquisition corp",
    "acquisition corporation",
    "holdings acquisition",
    "capital acquisition",
]

EXCLUDED_SYMBOL_PATTERNS = [
    r".*W$",
    r".*R$",
    r".*U$",
]

# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Candidate:
    symbol: str
    company_name: Optional[str]
    account: str
    state: str
    score: float
    close: float
    breakout_price: float
    trigger_price: float
    limit_price: float
    stop_price: float
    allowed_gap_pct: float
    rs_score: float
    rs_percentile: float
    ret_3m: float
    ret_6m: float
    avg_dollar_volume_20d: float
    market_cap: Optional[float]
    volume_ratio_20d: float
    ma50: float
    ma200: float
    extension_from_50ma_pct: float
    distance_to_breakout_pct: float
    high_52w: Optional[float]
    high_52w_proximity_pct: Optional[float]
    rs_line_new_high: bool
    tight_10d: bool
    tight_10d_width_pct: Optional[float]
    base_depth_pct: Optional[float]
    sector: Optional[str]
    industry: Optional[str]
    sector_etf: Optional[str]
    sector_ret_3m: Optional[float]
    sector_ret_6m: Optional[float]
    industry_bonus: float
    breakout_group_rank: Optional[int] = None


# ============================================================
# COMMON UTILS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_float(v, default=np.nan) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def pct_str(v: Optional[float], multiply_100: bool = True) -> str:
    if v is None or not np.isfinite(v):
        return "-"
    return f"{v * 100:.2f}%" if multiply_100 else f"{v:.2f}%"

def price_str(v: Optional[float]) -> str:
    if v is None or not np.isfinite(v):
        return "-"
    return f"{v:.2f}"

def market_cap_str(v: Optional[float]) -> str:
    if v is None or not np.isfinite(v):
        return "-"
    if v >= 1_000_000_000_000:
        return f"${v/1_000_000_000_000:.2f}T"
    if v >= 1_000_000_000:
        return f"${v/1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    return f"${v:,.0f}"

def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def chunked(seq: List[str], size: int) -> List[List[str]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]

def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    s = s.replace(".", "-")
    s = s.replace("/", "-")
    return s

def display_name(company_name: Optional[str], symbol: str) -> str:
    if company_name and str(company_name).strip():
        return f"{symbol} {company_name}"
    return symbol

def rolling_ma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def compute_return(series: pd.Series, lookback: int) -> float:
    if len(series) <= lookback:
        return np.nan
    start = safe_float(series.iloc[-lookback - 1])
    end = safe_float(series.iloc[-1])
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
        return np.nan
    return (end / start) - 1.0

def avg_dollar_volume(df: pd.DataFrame, n: int = 20) -> float:
    if len(df) < n:
        return np.nan
    return safe_float((df["Close"] * df["Volume"]).tail(n).mean())

def volume_ratio(df: pd.DataFrame, n: int = 20) -> float:
    if len(df) < n + 1:
        return np.nan
    curr = safe_float(df["Volume"].iloc[-1])
    avg = safe_float(df["Volume"].tail(n).mean())
    if not np.isfinite(curr) or not np.isfinite(avg) or avg <= 0:
        return np.nan
    return curr / avg

def tight_range_ratio(df: pd.DataFrame, window: int = 10) -> float:
    if df is None or df.empty or len(df) < window:
        return np.nan
    hi = safe_float(df["High"].tail(window).max())
    lo = safe_float(df["Low"].tail(window).min())
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= 0:
        return np.nan
    return (hi - lo) / hi

def compute_base_depth(df: pd.DataFrame, breakout_days: int) -> float:
    if df is None or df.empty or len(df) < breakout_days:
        return np.nan
    base = df.tail(breakout_days).copy()
    hi = safe_float(base["High"].max())
    lo = safe_float(base["Low"].min())
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= 0:
        return np.nan
    return (hi - lo) / hi

def download_chunk(symbols: List[str], period: str = LOOKBACK_PERIOD) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    return yf.download(
        tickers=symbols,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
        interval="1d",
    )

def extract_symbol_ohlcv(downloaded: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    if downloaded is None or downloaded.empty:
        return None

    if isinstance(downloaded.columns, pd.MultiIndex):
        if symbol not in downloaded.columns.get_level_values(0):
            return None
        sub = downloaded[symbol].copy()
    else:
        sub = downloaded.copy()

    sub.columns = [str(c).strip().title() for c in sub.columns]
    expected = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in sub.columns for c in expected):
        return None

    sub = sub[expected].copy().dropna(how="any")
    if sub.empty:
        return None

    sub = sub[(sub["Close"] > 0) & (sub["Volume"] >= 0)].copy()
    if sub.empty:
        return None

    return sub


# ============================================================
# UNIVERSE
# ============================================================

def fetch_nasdaq_listed() -> pd.DataFrame:
    url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    resp = requests.get(url, headers=USER_AGENT, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), sep="|")
    df = df[df["Symbol"].notna()].copy()
    df = df[df["Test Issue"] == "N"].copy()
    if "ETF" in df.columns:
        df = df[df["ETF"] == "N"].copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    return df

def fetch_other_listed() -> pd.DataFrame:
    url = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
    resp = requests.get(url, headers=USER_AGENT, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), sep="|")
    df = df[df["ACT Symbol"].notna()].copy()
    df = df[df["Test Issue"] == "N"].copy()
    if "ETF" in df.columns:
        df = df[df["ETF"] == "N"].copy()
    df["ACT Symbol"] = df["ACT Symbol"].astype(str).str.strip()
    return df

def load_universe_symbols() -> List[str]:
    nasdaq = fetch_nasdaq_listed()
    other = fetch_other_listed()
    raw = nasdaq["Symbol"].tolist() + other["ACT Symbol"].tolist()

    cleaned = []
    for s in raw:
        x = normalize_symbol(s)
        if not x:
            continue
        if any(ch.isdigit() for ch in x):
            continue
        if len(x) > 6:
            continue
        if not re.fullmatch(r"[A-Z\-]+", x):
            continue
        blocked = False
        for pat in EXCLUDED_SYMBOL_PATTERNS:
            if re.fullmatch(pat, x):
                blocked = True
                break
        if blocked:
            continue
        cleaned.append(x)

    return sorted(list(set(cleaned)))


# ============================================================
# META / SECTOR
# ============================================================

def load_meta_cache() -> Dict[str, dict]:
    return load_json(META_CACHE_JSON, {})

def save_meta_cache(cache: Dict[str, dict]) -> None:
    save_json(META_CACHE_JSON, cache)

def shorten_company_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    short = str(name).strip()
    replacements = [
        "Corporation", "Corp.", "Corp", "Inc.", "Inc", "Class A", "Class B",
        "Holdings", "Holding", "Ltd.", "Ltd", "PLC", "plc", "Company", "Co."
    ]
    for r in replacements:
        short = short.replace(r, "")
    short = re.sub(r"\s+", " ", short).strip(" ,-")
    return short[:40] if short else None

def get_symbol_meta(symbol: str, cache: Dict[str, dict]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[float]]:
    cached = cache.get(symbol)
    if cached and cached.get("updated_at"):
        return (
            cached.get("name"),
            cached.get("sector"),
            cached.get("industry"),
            cached.get("sector_etf"),
            cached.get("market_cap"),
        )

    company_name = None
    sector = None
    industry = None
    sector_etf = None
    market_cap = None

    try:
        tk = yf.Ticker(symbol)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        company_name = shorten_company_name(
            info.get("shortName")
            or info.get("longName")
            or info.get("displayName")
        )
        sector = info.get("sector")
        industry = info.get("industry")
        sector_etf = SECTOR_TO_ETF.get(sector)

        market_cap = safe_float(
            info.get("marketCap") or info.get("enterpriseValue"),
            default=np.nan,
        )
        market_cap = market_cap if np.isfinite(market_cap) else None

    except Exception:
        company_name = None
        sector = None
        industry = None
        sector_etf = None
        market_cap = None

    cache[symbol] = {
        "name": company_name,
        "sector": sector,
        "industry": industry,
        "sector_etf": sector_etf,
        "market_cap": market_cap,
        "updated_at": utc_now_iso(),
    }
    return company_name, sector, industry, sector_etf, market_cap

def get_sector_return_map() -> Dict[str, dict]:
    etfs = ["SPY"] + sorted(set(v for v in SECTOR_TO_ETF.values() if v))
    downloaded = download_chunk(etfs, period=LOOKBACK_PERIOD)
    result: Dict[str, dict] = {}

    for etf in etfs:
        df = extract_symbol_ohlcv(downloaded, etf)
        if df is None or df.empty:
            result[etf] = {"ret_3m": np.nan, "ret_6m": np.nan}
            continue
        result[etf] = {
            "ret_3m": compute_return(df["Close"], MOM_3M_LOOKBACK),
            "ret_6m": compute_return(df["Close"], MOM_6M_LOOKBACK),
        }

    return result

def industry_strength_table(sector_return_map: Dict[str, dict]) -> List[Tuple[str, float, float, float]]:
    rows = []
    for sector_name, etf in SECTOR_TO_ETF.items():
        data = sector_return_map.get(etf, {})
        r3 = safe_float(data.get("ret_3m"))
        r6 = safe_float(data.get("ret_6m"))
        if np.isfinite(r3) and np.isfinite(r6):
            score = 0.60 * r6 + 0.40 * r3
            rows.append((sector_name, r3, r6, score))
    rows.sort(key=lambda x: x[3], reverse=True)
    return rows

def build_industry_bonus_map(top_industries: List[Tuple[str, float, float, float]]) -> Dict[str, float]:
    bonus_map: Dict[str, float] = {}
    for rank, (sector_name, _, _, _) in enumerate(top_industries, start=1):
        if rank <= 3:
            bonus_map[sector_name] = 5.0
        elif rank <= 5:
            bonus_map[sector_name] = 3.0
        else:
            bonus_map[sector_name] = 0.0
    return bonus_map

def is_excluded_meta(
    company_name: Optional[str],
    sector: Optional[str],
    industry: Optional[str],
) -> bool:
    sector_text = (sector or "").strip().lower()
    industry_text = (industry or "").strip().lower()
    name_text = (company_name or "").strip().lower()

    if sector and sector in EXCLUDED_SECTORS:
        return True

    for kw in EXCLUDED_INDUSTRY_KEYWORDS:
        if kw in industry_text:
            return True

    for kw in EXCLUDED_NAME_KEYWORDS:
        if kw in name_text:
            return True

    return False


# ============================================================
# MARKET REGIME / BREADTH
# ============================================================

def compute_breadth(all_frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    above_50 = 0
    above_200 = 0
    total = 0

    for _, df in all_frames.items():
        if df is None or df.empty or len(df) < 200:
            continue

        close = safe_float(df["Close"].iloc[-1])
        ma50 = safe_float(rolling_ma(df["Close"], 50).iloc[-1])
        ma200 = safe_float(rolling_ma(df["Close"], 200).iloc[-1])

        if not np.isfinite(close) or not np.isfinite(ma50) or not np.isfinite(ma200):
            continue

        total += 1
        if close > ma50:
            above_50 += 1
        if close > ma200:
            above_200 += 1

    if total == 0:
        return {
            "breadth_50ma": np.nan,
            "breadth_200ma": np.nan,
        }

    return {
        "breadth_50ma": above_50 / total,
        "breadth_200ma": above_200 / total,
    }

def get_market_regime(spy_df: pd.DataFrame, breadth_50ma: float) -> Tuple[str, str, Dict[str, float]]:
    if spy_df is None or spy_df.empty or len(spy_df) < SPY_200MA_DAYS:
        return "데이터부족", "판단 불가", {
            "spy_close": np.nan,
            "spy_50ma": np.nan,
            "spy_200ma": np.nan,
            "spy_3m_ret": np.nan,
            "spy_6m_ret": np.nan,
        }

    close = spy_df["Close"]
    spy_close = safe_float(close.iloc[-1])
    spy_50ma = safe_float(rolling_ma(close, SPY_50MA_DAYS).iloc[-1])
    spy_200ma = safe_float(rolling_ma(close, SPY_200MA_DAYS).iloc[-1])
    spy_3m_ret = compute_return(close, SPY_3M_LOOKBACK)
    spy_6m_ret = compute_return(close, SPY_6M_LOOKBACK)

    info = {
        "spy_close": spy_close,
        "spy_50ma": spy_50ma,
        "spy_200ma": spy_200ma,
        "spy_3m_ret": spy_3m_ret,
        "spy_6m_ret": spy_6m_ret,
    }

    strong = (
        np.isfinite(spy_close)
        and np.isfinite(spy_50ma)
        and np.isfinite(spy_200ma)
        and np.isfinite(spy_6m_ret)
        and spy_close > spy_200ma
        and spy_50ma > spy_200ma
        and spy_6m_ret > 0
        and np.isfinite(breadth_50ma)
        and breadth_50ma >= BREADTH_STRONG_50MA
    )
    neutral = (
        np.isfinite(spy_close)
        and np.isfinite(spy_200ma)
        and spy_close > spy_200ma
        and np.isfinite(breadth_50ma)
        and breadth_50ma >= BREADTH_NEUTRAL_50MA
    )

    if strong:
        return "강세", "정상 진입 가능", info
    if neutral:
        return "중립", "상위 후보만 선별 진입", info
    return "약세", "후보는 관찰, 진입은 매우 보수적", info

def compute_last_3_trading_dates(spy_df: pd.DataFrame) -> List[str]:
    if spy_df is None or spy_df.empty:
        return []
    dates = spy_df.index.tolist()[-3:]
    out = []
    for d in dates:
        try:
            out.append(pd.Timestamp(d).strftime("%Y-%m-%d"))
        except Exception:
            continue
    return out

def breadth_comment(breadth_50ma: float) -> str:
    if not np.isfinite(breadth_50ma):
        return "브레드스 계산 불가"
    if breadth_50ma >= BREADTH_STRONG_50MA:
        return "브레드스 강함, 후보 수 넓게 유지"
    if breadth_50ma >= BREADTH_NEUTRAL_50MA:
        return "브레드스 중립, 후보 수 축소"
    return "브레드스 약함, 최상위 후보만 유지"

def candidate_limit_by_breadth(breadth_50ma: float) -> int:
    if not np.isfinite(breadth_50ma):
        return 4
    if breadth_50ma >= BREADTH_STRONG_50MA:
        return 10
    if breadth_50ma >= BREADTH_NEUTRAL_50MA:
        return 6
    return 3


# ============================================================
# RS / STRUCTURE / SCORE
# ============================================================

def compute_momentum_score(close: pd.Series) -> Tuple[float, float, float]:
    ret_3m = compute_return(close, MOM_3M_LOOKBACK)
    ret_6m = compute_return(close, MOM_6M_LOOKBACK)
    if not np.isfinite(ret_3m) or not np.isfinite(ret_6m):
        return np.nan, ret_3m, ret_6m
    score = MOM_6M_WEIGHT * ret_6m + MOM_3M_WEIGHT * ret_3m
    return float(score), ret_3m, ret_6m

def compute_rs_line_new_high(stock_df: pd.DataFrame, spy_df: pd.DataFrame, lookback: int = 126) -> bool:
    if stock_df is None or stock_df.empty or spy_df is None or spy_df.empty:
        return False

    merged = stock_df[["Close"]].rename(columns={"Close": "stock_close"}).merge(
        spy_df[["Close"]].rename(columns={"Close": "spy_close"}),
        left_index=True,
        right_index=True,
        how="inner",
    )
    if merged.empty or len(merged) < lookback:
        return False

    merged["rs_line"] = merged["stock_close"] / merged["spy_close"]
    current = safe_float(merged["rs_line"].iloc[-1])
    prev_high = safe_float(merged["rs_line"].tail(lookback).iloc[:-1].max())
    if not np.isfinite(current) or not np.isfinite(prev_high):
        return False
    return current >= prev_high

def score_tightness(width_10d: float) -> float:
    if not np.isfinite(width_10d):
        return 0.0
    if width_10d <= STRICT_TIGHT_10D_MAX:
        return 1.0
    if width_10d <= TIGHT_10D_MAX:
        return 0.65
    if width_10d <= 0.10:
        return 0.30
    return 0.0

def score_relative_volume(vr: float) -> float:
    if not np.isfinite(vr):
        return 0.0
    if vr >= RELATIVE_VOLUME_VERY_STRONG:
        return 1.0
    if vr >= RELATIVE_VOLUME_STRONG:
        return 0.70
    if vr >= 1.20:
        return 0.40
    if vr >= 1.00:
        return 0.20
    return 0.0

def score_breakout_distance(distance_pct: float) -> float:
    if not np.isfinite(distance_pct):
        return 0.0
    abs_dist = abs(distance_pct)
    if abs_dist <= 1.0:
        return 1.0
    if abs_dist <= 2.0:
        return 0.75
    if abs_dist <= 3.0:
        return 0.50
    return 0.20

def score_base_depth(base_depth: float) -> float:
    if not np.isfinite(base_depth):
        return 0.0
    if base_depth <= 0.12:
        return 1.0
    if base_depth <= 0.18:
        return 0.75
    if base_depth <= MAX_BASE_DEPTH:
        return 0.45
    return 0.0

def score_high_52w_proximity(proximity: float) -> float:
    if not np.isfinite(proximity):
        return 0.0
    if proximity >= 0.98:
        return 1.0
    if proximity >= PREFERRED_HIGH_52W_PROXIMITY:
        return 0.70
    if proximity >= MIN_HIGH_52W_PROXIMITY:
        return 0.35
    return 0.0

def build_total_score(
    rs_percentile: float,
    rs_line_new_high: bool,
    width_10d: float,
    volume_ratio_20d: float,
    distance_to_breakout_pct: float,
    base_depth: float,
    high_52w_proximity: float,
    industry_bonus: float,
) -> float:
    rs_component = rs_percentile / 100.0
    rs_line_component = 1.0 if rs_line_new_high else 0.0
    tight_component = score_tightness(width_10d)
    volume_component = score_relative_volume(volume_ratio_20d)
    distance_component = score_breakout_distance(distance_to_breakout_pct)
    base_component = score_base_depth(base_depth)
    high52_component = score_high_52w_proximity(high_52w_proximity)

    raw = (
        0.30 * rs_component
        + 0.18 * rs_line_component
        + 0.12 * tight_component
        + 0.10 * volume_component
        + 0.10 * distance_component
        + 0.10 * base_component
        + 0.10 * high52_component
    ) * 100.0

    return round(raw + industry_bonus, 2)


# ============================================================
# CANDIDATE BUILD
# ============================================================

def build_candidate_for_account(
    symbol: str,
    company_name: Optional[str],
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    sector: Optional[str],
    industry: Optional[str],
    sector_etf: Optional[str],
    sector_ret_3m: Optional[float],
    sector_ret_6m: Optional[float],
    market_cap: Optional[float],
    industry_bonus: float,
    rs_percentile: float,
    rs_score: float,
    breakout_days: int,
    account: str,
    stop_pct: float,
    min_rs_percentile: float,
) -> Optional[Candidate]:
    if df is None or df.empty or len(df) < max(252, breakout_days + 5):
        return None

    close = df["Close"]
    high = df["High"]

    latest_close = safe_float(close.iloc[-1])
    if not np.isfinite(latest_close) or latest_close < MIN_PRICE:
        return None

    if market_cap is None or not np.isfinite(market_cap) or market_cap < MIN_MARKET_CAP:
        return None

    adv20 = avg_dollar_volume(df, 20)
    if not np.isfinite(adv20) or adv20 < MIN_AVG_DOLLAR_VOLUME:
        return None

    vol_ratio_20 = volume_ratio(df, 20)
    if not np.isfinite(vol_ratio_20) or vol_ratio_20 < MIN_VOLUME_RATIO:
        return None

    ma50 = safe_float(rolling_ma(close, 50).iloc[-1])
    ma200 = safe_float(rolling_ma(close, 200).iloc[-1]) if len(df) >= 200 else np.nan

    if not np.isfinite(ma50):
        return None

    if latest_close <= ma50:
        return None

    extension_from_50ma = (latest_close / ma50) - 1.0
    if extension_from_50ma > MAX_EXTENSION_FROM_50MA:
        return None

    ret_3m = compute_return(close, MOM_3M_LOOKBACK)
    ret_6m = compute_return(close, MOM_6M_LOOKBACK)
    if not np.isfinite(ret_3m) or not np.isfinite(ret_6m):
        return None

    if rs_percentile < min_rs_percentile:
        return None

    breakout_price = safe_float(high.iloc[-(breakout_days + 1):-1].max())
    if not np.isfinite(breakout_price) or breakout_price <= 0:
        return None

    distance_to_breakout = (latest_close / breakout_price) - 1.0
    if not (NEAR_BREAKOUT_MIN <= distance_to_breakout <= NEAR_BREAKOUT_MAX):
        return None

    width_10d = tight_range_ratio(df, 10)
    tight_10d = bool(np.isfinite(width_10d) and width_10d <= TIGHT_10D_MAX)

    base_depth = compute_base_depth(df, breakout_days)
    if not np.isfinite(base_depth) or base_depth > MAX_BASE_DEPTH:
        return None

    rs_line_new_high = compute_rs_line_new_high(df, spy_df, lookback=126)

    high_52w = safe_float(high.tail(252).max())
    if not np.isfinite(high_52w) or high_52w <= 0:
        return None
    high_52w_proximity = latest_close / high_52w
    if high_52w_proximity < MIN_HIGH_52W_PROXIMITY:
        return None

    trigger_price = breakout_price * (1.0 + ENTRY_BUFFER_PCT)
    limit_price = breakout_price * (1.0 + LIMIT_BUFFER_PCT)
    stop_price = trigger_price * (1.0 - stop_pct)

    state = f"{account}_돌파직전"
    if latest_close > breakout_price:
        state = f"{account}_돌파확인"

    score = build_total_score(
        rs_percentile=rs_percentile,
        rs_line_new_high=rs_line_new_high,
        width_10d=width_10d,
        volume_ratio_20d=vol_ratio_20,
        distance_to_breakout_pct=distance_to_breakout * 100.0,
        base_depth=base_depth,
        high_52w_proximity=high_52w_proximity,
        industry_bonus=industry_bonus,
    )

    return Candidate(
        symbol=symbol,
        company_name=company_name,
        account=account,
        state=state,
        score=score,
        close=latest_close,
        breakout_price=breakout_price,
        trigger_price=trigger_price,
        limit_price=limit_price,
        stop_price=stop_price,
        allowed_gap_pct=LIMIT_BUFFER_PCT * 100.0,
        rs_score=rs_score,
        rs_percentile=rs_percentile,
        ret_3m=ret_3m,
        ret_6m=ret_6m,
        avg_dollar_volume_20d=adv20,
        market_cap=market_cap,
        volume_ratio_20d=vol_ratio_20,
        ma50=ma50,
        ma200=ma200 if np.isfinite(ma200) else np.nan,
        extension_from_50ma_pct=extension_from_50ma * 100.0,
        distance_to_breakout_pct=distance_to_breakout * 100.0,
        high_52w=high_52w,
        high_52w_proximity_pct=high_52w_proximity * 100.0,
        rs_line_new_high=rs_line_new_high,
        tight_10d=tight_10d,
        tight_10d_width_pct=width_10d * 100.0 if np.isfinite(width_10d) else None,
        base_depth_pct=base_depth * 100.0 if np.isfinite(base_depth) else None,
        sector=sector,
        industry=industry,
        sector_etf=sector_etf,
        sector_ret_3m=sector_ret_3m,
        sector_ret_6m=sector_ret_6m,
        industry_bonus=industry_bonus,
    )

def rank_candidates_by_group(candidates: List[Candidate]) -> List[Candidate]:
    out = sorted(
        candidates,
        key=lambda x: (
            0 if "확인" in x.state else 1,
            -x.score,
            -x.rs_percentile,
            -x.ret_6m,
            -x.ret_3m,
            abs(x.distance_to_breakout_pct),
        )
    )
    for i, c in enumerate(out, start=1):
        c.breakout_group_rank = i
    return out


# ============================================================
# TELEGRAM
# ============================================================

def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

def send_telegram_message(text: str) -> None:
    if not telegram_enabled():
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text[:4000],
        "disable_web_page_preview": True,
    }

    try:
        requests.post(url, json=payload, timeout=20)
    except Exception:
        pass

def send_telegram_messages(texts: List[str], delay_sec: float = 1.0) -> None:
    for text in texts:
        if text and text.strip():
            send_telegram_message(text)
            time.sleep(delay_sec)


# ============================================================
# MESSAGE FORMAT
# ============================================================

def format_market_summary_message(
    regime_level: str,
    regime_action: str,
    regime_info: Dict[str, float],
    breadth_info: Dict[str, float],
    last_3_dates: List[str],
    universe_count: int,
    filtered_universe_count: int,
    downloaded_count: int,
    ranked_count: int,
    all_candidate_count: int,
    a_count: int,
    b_count: int,
    display_limit: int,
    top_industries: List[Tuple[str, float, float, float]],
) -> str:
    lines = []
    lines.append("순수 모멘텀 레이더")
    lines.append(f"스캔 시각(UTC): {utc_now_iso()}")
    lines.append("")

    if last_3_dates:
        lines.append("최근 3거래일")
        for d in last_3_dates:
            lines.append(f"- {d}")
        lines.append("")

    lines.append("시장 레짐")
    lines.append(f"- 단계: {regime_level}")
    lines.append(f"- 해석: {regime_action}")
    lines.append(f"- SPY 종가: {price_str(regime_info.get('spy_close'))}")
    lines.append(f"- SPY 50일선: {price_str(regime_info.get('spy_50ma'))}")
    lines.append(f"- SPY 200일선: {price_str(regime_info.get('spy_200ma'))}")
    lines.append(f"- SPY 3개월 수익률: {pct_str(regime_info.get('spy_3m_ret'))}")
    lines.append(f"- SPY 6개월 수익률: {pct_str(regime_info.get('spy_6m_ret'))}")
    lines.append("")

    lines.append("시장 브레드스")
    lines.append(f"- 50일선 위 종목 비율: {pct_str(breadth_info.get('breadth_50ma'))}")
    lines.append(f"- 200일선 위 종목 비율: {pct_str(breadth_info.get('breadth_200ma'))}")
    lines.append(f"- 해석: {breadth_comment(breadth_info.get('breadth_50ma'))}")
    lines.append(f"- 이번 출력 후보 제한 수: 계좌별 상위 {display_limit}개")
    lines.append("")

    lines.append("현재 강한 산업")
    if top_industries:
        for i, (name, r3, r6, _) in enumerate(top_industries[:5], start=1):
            lines.append(f"- {i}. {name} | 3개월 {pct_str(r3)} | 6개월 {pct_str(r6)}")
    else:
        lines.append("- 계산 불가")
    lines.append("")

    lines.append("스캔 요약")
    lines.append(f"- 원시 유니버스 수: {universe_count}")
    lines.append(f"- 메타 필터 통과 유니버스 수: {filtered_universe_count}")
    lines.append(f"- 다운로드 성공 수: {downloaded_count}")
    lines.append(f"- 모멘텀 점수 계산 수: {ranked_count}")
    lines.append(f"- 전체 후보 수: {all_candidate_count}")
    lines.append(f"- 계좌 A(55일) 후보 수: {a_count}")
    lines.append(f"- 계좌 B(20일) 후보 수: {b_count}")

    return "\n".join(lines).strip()

def format_candidate_block(title: str, candidates: List[Candidate]) -> str:
    lines = []
    lines.append(title)
    lines.append(f"스캔 시각(UTC): {utc_now_iso()}")
    lines.append("")

    if not candidates:
        lines.append("해당 없음")
        return "\n".join(lines).strip()

    for c in candidates[:MAX_OUTPUT_PER_BLOCK]:
        lines.append(f"{c.breakout_group_rank}. {display_name(c.company_name, c.symbol)}")
        lines.append(f"상태: {c.state}")
        lines.append(f"점수: {c.score:.2f}")
        lines.append(f"현재가: {price_str(c.close)}")
        lines.append(f"돌파가: {price_str(c.breakout_price)}")
        lines.append(f"매수 트리거: {price_str(c.trigger_price)}")
        lines.append(f"허용 갭 / 리밋 상단: +{c.allowed_gap_pct:.2f}% ({price_str(c.limit_price)})")
        lines.append(f"손절가: {price_str(c.stop_price)}")
        lines.append(f"돌파까지 거리: {c.distance_to_breakout_pct:+.2f}%")
        lines.append(
            f"52주 신고가 근접도: {c.high_52w_proximity_pct:.2f}%"
            if c.high_52w_proximity_pct is not None
            else "52주 신고가 근접도: -"
        )
        lines.append(f"RS 백분위: {c.rs_percentile:.1f}")
        lines.append(f"3개월 수익률: {pct_str(c.ret_3m)} | 6개월 수익률: {pct_str(c.ret_6m)}")
        lines.append(f"20일 평균 거래대금: ${c.avg_dollar_volume_20d:,.0f}")
        lines.append(f"시가총액: {market_cap_str(c.market_cap)}")
        lines.append(f"거래량 배수(20일): {c.volume_ratio_20d:.2f}x")
        lines.append(f"RS 라인 신고가: {'예' if c.rs_line_new_high else '아니오'}")
        lines.append(f"10일 타이트 구조: {'예' if c.tight_10d else '아니오'}")
        lines.append(
            f"10일 변동폭: {c.tight_10d_width_pct:.2f}%"
            if c.tight_10d_width_pct is not None
            else "10일 변동폭: -"
        )
        lines.append(
            f"베이스 깊이: {c.base_depth_pct:.2f}%"
            if c.base_depth_pct is not None
            else "베이스 깊이: -"
        )
        lines.append(f"50일선 대비 확장: {c.extension_from_50ma_pct:.2f}%")
        lines.append(f"섹터: {c.sector or '-'}")
        lines.append(f"산업: {c.industry or '-'}")
        if c.sector_ret_3m is not None and c.sector_ret_6m is not None:
            lines.append(f"산업 강도(ETF): 3개월 {pct_str(c.sector_ret_3m)} | 6개월 {pct_str(c.sector_ret_6m)}")
        else:
            lines.append("산업 강도(ETF): -")
        lines.append(f"산업 보너스: +{c.industry_bonus:.1f}")
        lines.append("")
    return "\n".join(lines).strip()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=== Pure Momentum Radar Start ===")

    meta_cache = load_meta_cache()

    universe = load_universe_symbols()
    print(f"[1] 원시 유니버스 수: {len(universe)}")

    spy_download = download_chunk(["SPY"], period=LOOKBACK_PERIOD)
    spy_df = extract_symbol_ohlcv(spy_download, "SPY")
    if spy_df is None or spy_df.empty:
        raise RuntimeError("SPY 다운로드 실패")

    last_3_dates = compute_last_3_trading_dates(spy_df)

    sector_return_map = get_sector_return_map()
    top_industries = industry_strength_table(sector_return_map)
    industry_bonus_map = build_industry_bonus_map(top_industries)
    print("[2] 산업 강도 계산 완료")

    filtered_universe = []
    for sym in universe:
        company_name, sector, industry, sector_etf, market_cap = get_symbol_meta(sym, meta_cache)

        if is_excluded_meta(company_name, sector, industry):
            continue

        if market_cap is None or not np.isfinite(market_cap) or market_cap < MIN_MARKET_CAP:
            continue

        filtered_universe.append(sym)

    print(f"[3] 메타 필터 통과 유니버스 수: {len(filtered_universe)}")
    if not filtered_universe:
        raise RuntimeError("메타 필터 통과 종목 없음")

    all_frames: Dict[str, pd.DataFrame] = {}
    chunks = chunked(filtered_universe, CHUNK_SIZE)
    print(f"[4] 다운로드 청크 수: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"  - 청크 {i}/{len(chunks)} | 크기={len(chunk)}")
        try:
            downloaded = download_chunk(chunk, period=LOOKBACK_PERIOD)
        except Exception as e:
            print(f"    다운로드 오류: {e}")
            time.sleep(CHUNK_SLEEP_SEC)
            continue

        if downloaded is None or downloaded.empty:
            time.sleep(CHUNK_SLEEP_SEC)
            continue

        for sym in chunk:
            try:
                sub = extract_symbol_ohlcv(downloaded, sym)
                if sub is None or sub.empty:
                    continue
                all_frames[sym] = sub
            except Exception:
                continue

        time.sleep(CHUNK_SLEEP_SEC)

    print(f"[5] 다운로드 성공 수: {len(all_frames)}")
    if not all_frames:
        raise RuntimeError("다운로드 성공 종목 없음")

    breadth_info = compute_breadth(all_frames)
    regime_level, regime_action, regime_info = get_market_regime(
        spy_df=spy_df,
        breadth_50ma=safe_float(breadth_info.get("breadth_50ma")),
    )
    display_limit = candidate_limit_by_breadth(safe_float(breadth_info.get("breadth_50ma")))
    print(f"[6] 시장 레짐: {regime_level}")
    print(f"[7] 브레드스 제한 수: {display_limit}")

    rs_rows = []
    for sym, df in all_frames.items():
        try:
            score, ret_3m, ret_6m = compute_momentum_score(df["Close"])
            if np.isfinite(score) and np.isfinite(ret_3m) and np.isfinite(ret_6m):
                rs_rows.append((sym, score, ret_3m, ret_6m))
        except Exception:
            continue

    rs_df = pd.DataFrame(rs_rows, columns=["symbol", "rs_score", "ret_3m", "ret_6m"])
    if rs_df.empty:
        raise RuntimeError("RS 점수 계산 실패")

    rs_df["rs_percentile"] = rs_df["rs_score"].rank(pct=True) * 100.0
    rs_map = rs_df.set_index("symbol").to_dict(orient="index")
    print(f"[8] RS 점수 계산 수: {len(rs_df)}")

    all_candidates: List[Candidate] = []
    account_a: List[Candidate] = []
    account_b: List[Candidate] = []

    for sym, df in all_frames.items():
        if sym not in rs_map:
            continue

        company_name, sector, industry, sector_etf, market_cap = get_symbol_meta(sym, meta_cache)

        if is_excluded_meta(company_name, sector, industry):
            continue
        if market_cap is None or not np.isfinite(market_cap) or market_cap < MIN_MARKET_CAP:
            continue

        sector_ret_3m = None
        sector_ret_6m = None
        if sector_etf:
            r3 = safe_float(sector_return_map.get(sector_etf, {}).get("ret_3m"))
            r6 = safe_float(sector_return_map.get(sector_etf, {}).get("ret_6m"))
            sector_ret_3m = r3 if np.isfinite(r3) else None
            sector_ret_6m = r6 if np.isfinite(r6) else None

        industry_bonus = industry_bonus_map.get(sector, 0.0) if sector else 0.0
        info = rs_map[sym]

        candidate_a = build_candidate_for_account(
            symbol=sym,
            company_name=company_name,
            df=df,
            spy_df=spy_df,
            sector=sector,
            industry=industry,
            sector_etf=sector_etf,
            sector_ret_3m=sector_ret_3m,
            sector_ret_6m=sector_ret_6m,
            market_cap=market_cap,
            industry_bonus=industry_bonus,
            rs_percentile=float(info["rs_percentile"]),
            rs_score=float(info["rs_score"]),
            breakout_days=BREAKOUT_A_DAYS,
            account="A",
            stop_pct=STOP_A_PCT,
            min_rs_percentile=MIN_RS_PERCENTILE_A,
        )
        if candidate_a is not None:
            account_a.append(candidate_a)
            all_candidates.append(candidate_a)

        candidate_b = build_candidate_for_account(
            symbol=sym,
            company_name=company_name,
            df=df,
            spy_df=spy_df,
            sector=sector,
            industry=industry,
            sector_etf=sector_etf,
            sector_ret_3m=sector_ret_3m,
            sector_ret_6m=sector_ret_6m,
            market_cap=market_cap,
            industry_bonus=industry_bonus,
            rs_percentile=float(info["rs_percentile"]),
            rs_score=float(info["rs_score"]),
            breakout_days=BREAKOUT_B_DAYS,
            account="B",
            stop_pct=STOP_B_PCT,
            min_rs_percentile=MIN_RS_PERCENTILE_B,
        )
        if candidate_b is not None:
            account_b.append(candidate_b)
            all_candidates.append(candidate_b)

    account_a = rank_candidates_by_group(account_a)
    account_b = rank_candidates_by_group(account_b)

    account_a_display = account_a[:display_limit]
    account_b_display = account_b[:display_limit]

    all_candidates = sorted(
        all_candidates,
        key=lambda x: (
            0 if x.account == "A" else 1,
            0 if "확인" in x.state else 1,
            -x.score,
            -x.rs_percentile,
        )
    )

    print(f"[9] 전체 후보 수: {len(all_candidates)}")
    print(f"[10] 계좌 A 후보 수: {len(account_a)}")
    print(f"[11] 계좌 B 후보 수: {len(account_b)}")

    all_df = pd.DataFrame([asdict(c) for c in all_candidates]) if all_candidates else pd.DataFrame()
    a_df = pd.DataFrame([asdict(c) for c in account_a]) if account_a else pd.DataFrame()
    b_df = pd.DataFrame([asdict(c) for c in account_b]) if account_b else pd.DataFrame()

    cols = [f.name for f in Candidate.__dataclass_fields__.values()]

    if all_df.empty:
        pd.DataFrame(columns=cols).to_csv(ALL_CSV, index=False, encoding="utf-8-sig")
    else:
        all_df.to_csv(ALL_CSV, index=False, encoding="utf-8-sig")

    if a_df.empty:
        pd.DataFrame(columns=cols).to_csv(A_CSV, index=False, encoding="utf-8-sig")
    else:
        a_df.to_csv(A_CSV, index=False, encoding="utf-8-sig")

    if b_df.empty:
        pd.DataFrame(columns=cols).to_csv(B_CSV, index=False, encoding="utf-8-sig")
    else:
        b_df.to_csv(B_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "updated_at": utc_now_iso(),
        "market_regime": regime_level,
        "market_action": regime_action,
        "regime_info": regime_info,
        "breadth_info": breadth_info,
        "display_limit": display_limit,
        "last_3_trading_dates": last_3_dates,
        "raw_universe_count": len(universe),
        "filtered_universe_count": len(filtered_universe),
        "downloaded_count": len(all_frames),
        "ranked_count": len(rs_df),
        "all_candidate_count": len(all_candidates),
        "account_a_count": len(account_a),
        "account_b_count": len(account_b),
        "top_industries": [
            {"sector": x[0], "ret_3m": x[1], "ret_6m": x[2], "score": x[3]}
            for x in top_industries[:10]
        ],
        "top_candidates": [asdict(c) for c in all_candidates[:20]],
    }
    save_json(SUMMARY_JSON, summary)
    save_meta_cache(meta_cache)

    msg_summary = format_market_summary_message(
        regime_level=regime_level,
        regime_action=regime_action,
        regime_info=regime_info,
        breadth_info=breadth_info,
        last_3_dates=last_3_dates,
        universe_count=len(universe),
        filtered_universe_count=len(filtered_universe),
        downloaded_count=len(all_frames),
        ranked_count=len(rs_df),
        all_candidate_count=len(all_candidates),
        a_count=len(account_a),
        b_count=len(account_b),
        display_limit=display_limit,
        top_industries=top_industries,
    )
    msg_a = format_candidate_block("계좌 A | 55일 돌파 후보", account_a_display)
    msg_b = format_candidate_block("계좌 B | 20일 돌파 후보", account_b_display)

    send_telegram_messages([msg_summary, msg_a, msg_b], delay_sec=1.0)

    print("[12] 상위 후보 미리보기")
    for c in all_candidates[:20]:
        print(
            f"{c.symbol:6} | {c.account} | {c.state:10} | "
            f"점수={c.score:6.2f} | RS={c.rs_percentile:6.1f} | "
            f"돌파가={c.breakout_price:8.2f} | 트리거={c.trigger_price:8.2f} | 손절={c.stop_price:8.2f}"
        )

    print("=== Pure Momentum Radar End ===")


if __name__ == "__main__":
    main()
