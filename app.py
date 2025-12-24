import os
import json
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import requests
import ccxt
import xml.etree.ElementTree as ET

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import plotly.graph_objects as go
import plotly.express as px
from email.utils import parsedate_to_datetime


# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="GRT Pulse", page_icon="🟢", layout="wide")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(DATA_DIR, "grt_pulse_history.csv")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")

FALLBACK_EXCHANGES = ["kraken", "coinbase", "bitstamp"]
GRAPH_NETWORK_SUBGRAPH_ID = "GgwLf9BTFBJi6Z5iYHssMAGEE4w5dR3Jox2dMLrBxnCT"

DEFAULT_SETTINGS = {
    "preferred_exchange": "binance",
    "symbol": "GRT/USDT",
    "benchmark": "BTC/USDT",
    "days": 900,
    "timeframe": "1d",
    "news": {
        "enable": True,
        "lookback_days": 14,
        "rss_timeout": 15
    },
    "api_keys": {
        "thegraph_gateway": ""  # opcional
    }
}


# ==========================================================
# NEW UI THEME (glass + teal)
# ==========================================================
NEW_CSS = """
<style>
:root{
  --bg0:#070B10;
  --bg1:#0A1320;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --line: rgba(255,255,255,0.10);
  --text:#EAF2FF;
  --muted: rgba(234,242,255,0.72);
  --teal:#22C3A6;
  --amber:#F8B84A;
  --red:#FF5A6B;
}

html, body, [class*="css"]{
  background:
    radial-gradient(1100px 700px at 10% -15%, rgba(34,195,166,0.35), transparent 60%),
    radial-gradient(900px 600px at 110% 5%, rgba(248,184,74,0.18), transparent 55%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
  color: var(--text) !important;
}

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)) !important;
  border-right: 1px solid var(--line);
}

.block-container{ padding-top: 1rem; }

.glass-hero{
  padding: 16px 18px;
  border-radius: 18px;
  border: 1px solid var(--line);
  background: linear-gradient(135deg, rgba(34,195,166,0.18), rgba(255,255,255,0.05));
  box-shadow: 0 18px 60px rgba(0,0,0,0.35);
}

.hero-title{ font-size: 26px; font-weight: 900; letter-spacing: -0.3px; }
.hero-sub{ color: var(--muted); font-size: 13px; margin-top: 6px; line-height: 1.35rem; }

.badge{
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  margin-left: 10px;
  font-size: 12px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.06);
  color: var(--text);
}

.kpi{
  border-radius: 16px;
  padding: 14px 14px;
  background: linear-gradient(180deg, var(--card), var(--card2));
  border: 1px solid var(--line);
  box-shadow: 0 14px 40px rgba(0,0,0,0.22);
}

.kpi .label{ color: var(--muted); font-size: 12px; margin-bottom: 6px; }
.kpi .value{ font-size: 20px; font-weight: 900; letter-spacing: -0.2px; }
.kpi .hint{ color: var(--muted); font-size: 11px; margin-top: 6px; line-height: 1.25rem; }

hr{ border-color: var(--line) !important; }

.small-muted{ color: var(--muted); font-size: 12px; }
</style>
"""
st.markdown(NEW_CSS, unsafe_allow_html=True)


# ==========================================================
# HELPERS
# ==========================================================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def safe_float(x):
    try:
        if x is None:
            return np.nan
        v = float(x)
        if np.isinf(v):
            return np.nan
        return v
    except Exception:
        return np.nan

def _fmt_pct(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return f"{p*100:.1f}%"

def _fmt_num(x: float, digits=3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"

def _clean_text(s: str) -> str:
    if not s:
        return ""
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s)

def load_settings() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2, ensure_ascii=False)
        return DEFAULT_SETTINGS
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        s = json.load(f)

    # backfill
    def _backfill(dst, src):
        for k, v in src.items():
            if k not in dst:
                dst[k] = v
            elif isinstance(v, dict) and isinstance(dst.get(k), dict):
                _backfill(dst[k], v)

    _backfill(s, DEFAULT_SETTINGS)
    return s

def save_settings(s: dict) -> None:
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)

def _symbol_variants(symbol: str) -> List[str]:
    base, quote = symbol.split("/")
    variants = [symbol]
    if quote.upper() == "USDT":
        variants += [f"{base}/USD", f"{base}/USDC"]
    return variants

def explain_score(score: float) -> str:
    if score is None or not np.isfinite(score):
        return "Sin cálculo aún. Pulsa “Actualizar”."
    if score >= 75:
        return "Confluencia alta. Aun así: gestión de riesgo obligatoria."
    if score >= 60:
        return "Confluencia moderada. Bien, pero no es “piloto automático”."
    if score >= 45:
        return "Zona neutral. Mejor esperar confirmación."
    if score >= 30:
        return "Confluencia baja. Cautela."
    return "Riesgo elevado. Enfócate más en proteger capital que en buscar subida."

def explain_prob(p: float, h: int) -> str:
    if p is None or not np.isfinite(p):
        return f"Sin probabilidad calculada para {h} días."
    if p >= 0.70:
        return f"Alta probabilidad de subida a {h}d según el modelo (no garantía)."
    if p >= 0.58:
        return f"Ventaja ligera a favor de subida a {h}d."
    if p >= 0.50:
        return f"Escenario equilibrado a {h}d (casi 50/50)."
    if p >= 0.42:
        return f"Ventaja ligera a favor de bajada/lateral a {h}d."
    return f"Probabilidad baja de subida a {h}d (más riesgo)."

def explain_auc(auc: float) -> str:
    if auc is None or not np.isfinite(auc):
        return "AUC no disponible (pocos datos o test con una sola clase)."
    if auc >= 0.70:
        return "Muy buena: separación clara (en su test)."
    if auc >= 0.60:
        return "Aceptable: hay algo de ventaja."
    if auc >= 0.55:
        return "Débil: úsalo con prudencia."
    if auc >= 0.50:
        return "Casi azar."
    return "Peor que azar: ojo, puede estar invertido."

def verdict_tag(score: float) -> str:
    if not np.isfinite(score):
        return "—"
    if score >= 70:
        return "🟢 Favorable"
    if score >= 55:
        return "🟡 Moderado"
    if score >= 40:
        return "🟠 Neutral/Riesgo"
    return "🔴 Riesgo"

def save_row_to_csv(row: dict, path: str = RESULTS_PATH) -> None:
    df = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        out = pd.concat([old, df], ignore_index=True)
        out = out.drop_duplicates(subset=["as_of_date","exchange","symbol"], keep="last")
    else:
        out = df
    out.to_csv(path, index=False)

def load_history(path: str = RESULTS_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


# ==========================================================
# DATA FETCH
# ==========================================================
@st.cache_data(ttl=60*60)
def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    exchanges_to_try = [exchange_id] + [ex for ex in FALLBACK_EXCHANGES if ex != exchange_id]
    last_err = None

    for ex_id in exchanges_to_try:
        try:
            ex_class = getattr(ccxt, ex_id)
            ex = ex_class({"enableRateLimit": True})
            ex.load_markets()

            for sym in _symbol_variants(symbol):
                if sym not in ex.markets:
                    continue

                ohlcv = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
                df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.date
                df = df.drop(columns=["ts"]).set_index("date")

                df.attrs["exchange_used"] = ex_id
                df.attrs["symbol_used"] = sym
                return df

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if ("451" in msg) or ("restricted location" in msg) or ("eligibility" in msg):
                continue
            continue

    raise RuntimeError(f"No se pudo descargar OHLCV. Último error: {last_err}")

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["close"] = d["close"].astype(float)
    d["volume"] = d["volume"].astype(float)
    d["log_close"] = np.log(d["close"])
    d["ret"] = d["close"].pct_change()
    return d


# ==========================================================
# FUNDAMENTALS (CoinGecko + Graph Gateway optional)
# ==========================================================
@st.cache_data(ttl=6*60*60)
def fetch_grt_fundamentals_coingecko() -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/the-graph"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    d = r.json()

    m = d.get("market_data", {})
    row = {
        "as_of": str(pd.Timestamp.utcnow().date()),
        "cg_price_usd": safe_float(m.get("current_price", {}).get("usd")),
        "cg_marketcap_usd": safe_float(m.get("market_cap", {}).get("usd")),
        "cg_volume_24h_usd": safe_float(m.get("total_volume", {}).get("usd")),
        "cg_circulating_supply": safe_float(m.get("circulating_supply")),
        "cg_total_supply": safe_float(m.get("total_supply")),
        "cg_price_change_24h_pct": safe_float(m.get("price_change_percentage_24h")),
        "cg_price_change_7d_pct": safe_float(m.get("price_change_percentage_7d")),
        "cg_price_change_30d_pct": safe_float(m.get("price_change_percentage_30d")),
    }
    return pd.DataFrame([row]).set_index("as_of")

def _graphql_post(url: str, query: str, variables=None, timeout=25) -> dict:
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=6*60*60)
def fetch_grt_network_fundamentals_gateway(thegraph_api_key: str) -> pd.DataFrame:
    if not thegraph_api_key:
        raise RuntimeError("Falta THE_GRAPH_API_KEY (Gateway).")

    url = f"https://gateway.thegraph.com/api/{thegraph_api_key}/subgraphs/id/{GRAPH_NETWORK_SUBGRAPH_ID}"
    q = """
    query {
      graphNetwork(id: "1") {
        totalTokensStaked
        totalTokensAllocated
        totalDelegatedTokens
        totalSupply
      }
      indexers(first: 1000, where: {active: true}) { id }
    }
    """
    data = _graphql_post(url, q)
    if "errors" in data:
        raise RuntimeError(f"Graph gateway error: {data['errors']}")

    d = data["data"]
    gn = d.get("graphNetwork") or {}
    idx_count = len(d.get("indexers") or [])

    row = {
        "as_of": str(pd.Timestamp.utcnow().date()),
        "totalTokensStaked": safe_float(gn.get("totalTokensStaked")),
        "totalTokensAllocated": safe_float(gn.get("totalTokensAllocated")),
        "totalDelegatedTokens": safe_float(gn.get("totalDelegatedTokens")),
        "totalSupply": safe_float(gn.get("totalSupply")),
        "activeIndexers": float(idx_count),
    }
    return pd.DataFrame([row]).set_index("as_of")


# ==========================================================
# NEWS (RSS) -> sentiment score 0..100
# ==========================================================
POS_WORDS = {
    "upgrade","partnership","launch","released","adoption","growth","record","surge","bull",
    "win","success","breakthrough","milestone","approval","support","integrates","expands",
    "strong","beats","positive","accumulate","accumulation","listing","listed"
}
NEG_WORDS = {
    "hack","exploit","lawsuit","ban","down","dump","bear","crash","collapse","fraud","scam",
    "risk","warning","investigation","liquidation","delay","weak","negative","sec","rejected",
    "outage","attack","security","breach","concern"
}

def score_sentiment(text: str) -> float:
    t = _clean_text(text)
    if not t.strip():
        return 0.0
    words = [w for w in t.split() if len(w) > 2]
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    raw = (pos - neg) / max(1, (pos + neg))
    return float(raw)

def _parse_pubdate_to_utc_date(s: str) -> Optional[pd.Timestamp]:
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return pd.Timestamp(dt.astimezone(timezone.utc)).normalize()
    except Exception:
        return None

@st.cache_data(ttl=60*30)
def fetch_rss_items(url: str, timeout: int = 15, max_items: int = 25) -> List[dict]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    root = ET.fromstring(r.text)

    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        desc = (item.findtext("description") or "").strip()
        items.append({"title": title, "link": link, "pubDate": pub, "desc": desc})
        if len(items) >= max_items:
            break
    return items

def build_news_panel(enable: bool, lookback_days: int, timeout: int) -> Tuple[float, pd.DataFrame]:
    if not enable:
        return np.nan, pd.DataFrame()

    queries = ["The+Graph+GRT", "The+Graph+protocol", "GRT+token"]
    rss_urls = [f"https://news.google.com/rss/search?q={q}&hl=en&gl=US&ceid=US:en" for q in queries]

    all_items = []
    for u in rss_urls:
        try:
            all_items.extend(fetch_rss_items(u, timeout=timeout, max_items=25))
        except Exception:
            continue

    if not all_items:
        return np.nan, pd.DataFrame()

    cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=int(lookback_days))

    rows = []
    for it in all_items:
        dt = _parse_pubdate_to_utc_date(it.get("pubDate",""))
        if dt is None or dt < cutoff:
            continue
        txt = f"{it.get('title','')} {it.get('desc','')}"
        s = score_sentiment(txt)
        rows.append({
            "date": dt,
            "title": it.get("title",""),
            "link": it.get("link",""),
            "sent": float(s)
        })

    df_news = pd.DataFrame(rows).drop_duplicates(subset=["title"]).sort_values("date", ascending=False).head(60)
    if df_news.empty:
        return np.nan, df_news

    svals = df_news["sent"].astype(float).values
    svals = svals[np.isfinite(svals)]
    if len(svals) == 0:
        return np.nan, df_news

    # recorte robusto 10% extremos
    svals_sorted = np.sort(svals)
    k = max(0, int(0.1 * len(svals_sorted)))
    core = svals_sorted[k:len(svals_sorted)-k] if len(svals_sorted) > 10 else svals_sorted
    m = float(np.mean(core)) if len(core) else float(np.mean(svals_sorted))
    score_0_100 = float(np.clip(50 + 50*m, 0, 100))

    df_news["sent_0_100"] = (50 + 50*df_news["sent"]).clip(0, 100)
    return score_0_100, df_news

def explain_news_score(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "No disponible (RSS falló o está desactivado)."
    if x >= 65:
        return "Titulares claramente positivos (ojo con hype)."
    if x >= 55:
        return "Ligero sesgo positivo."
    if x >= 45:
        return "Neutral."
    if x >= 35:
        return "Sesgo negativo."
    return "Muy negativo: cuidado con shocks."


# ==========================================================
# SIMPLE INDICATORS
# ==========================================================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m_fast = ema(series, fast)
    m_slow = ema(series, slow)
    line = m_fast - m_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def realized_vol(ret: pd.Series, window: int = 30) -> float:
    x = ret.dropna().tail(window)
    if len(x) < 20:
        return np.nan
    return float(x.std() * np.sqrt(365))

def simple_trend(log_close: pd.Series, window: int = 90) -> Dict[str, float]:
    s = log_close.dropna().tail(window)
    if len(s) < 30:
        return {"trend_slope": np.nan, "trend_r2": np.nan}
    y = s.values
    x = np.arange(len(y), dtype=float)
    # y = a + b x
    b, a = np.polyfit(x, y, 1)
    yhat = a + b*x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res/ss_tot
    return {"trend_slope": float(b), "trend_r2": float(r2)}

def momentum(ret: pd.Series, n: int) -> float:
    r = ret.dropna().tail(n)
    if len(r) < n:
        return np.nan
    return float((1 + r).prod() - 1)

def vol_z_14(volume: pd.Series) -> float:
    v = volume.astype(float)
    hist = v.tail(180)
    if len(hist) < 60:
        return np.nan
    recent = v.tail(14).mean()
    return float((recent - hist.mean()) / (hist.std() + 1e-12))

def rolling_corr(a: pd.Series, b: pd.Series, window: int = 60) -> float:
    tmp = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    tmp = tmp.tail(window)
    if len(tmp) < 40:
        return np.nan
    return float(tmp["a"].corr(tmp["b"]))


# ==========================================================
# FEATURES + MODEL (simple, 30d)
# ==========================================================
def build_features(df: pd.DataFrame, dfb: pd.DataFrame, news_score: float, fund_last: Optional[pd.Series]) -> pd.DataFrame:
    d = df.copy()

    close = d["close"].astype(float)
    ret = d["ret"].astype(float)

    d["ret_1"] = ret
    d["ret_7"] = close.pct_change(7)
    d["ret_30"] = close.pct_change(30)
    d["vol_14"] = ret.rolling(14).std()
    d["vol_60"] = ret.rolling(60).std()

    d["rsi_14"] = rsi(close, 14)
    _, _, mh = macd(close)
    d["macd_hist"] = mh

    d["vol_z_14"] = (d["volume"].rolling(14).mean() - d["volume"].rolling(180).mean()) / (d["volume"].rolling(180).std() + 1e-12)
    d["corr_60_btc"] = rolling_corr(d["ret"], dfb["ret"], 60) if (dfb is not None and not dfb.empty) else np.nan

    d["news_score_0_100"] = float(news_score) if np.isfinite(news_score) else np.nan

    # fundamentals (broadcast)
    if fund_last is not None and isinstance(fund_last, pd.Series) and not fund_last.empty:
        for k, v in fund_last.to_dict().items():
            d[f"fund_{k}"] = safe_float(v)

        if "fund_totalTokensStaked" in d.columns and "fund_totalSupply" in d.columns:
            d["fund_stake_ratio"] = d["fund_totalTokensStaked"] / (d["fund_totalSupply"] + 1e-12)

        if "fund_cg_marketcap_usd" in d.columns and "fund_cg_volume_24h_usd" in d.columns:
            d["fund_mcap_to_vol"] = d["fund_cg_marketcap_usd"] / (d["fund_cg_volume_24h_usd"] + 1e-12)

    return d

def fit_prob_30d(feat: pd.DataFrame, min_rows: int = 260) -> Tuple[float, float, int, int]:
    """
    Returns:
      p_today, auc_test, n_train, n_test
    """
    drop_cols = {"open","high","low","close","volume","log_close","ret"}
    candidates = [c for c in feat.columns if c not in drop_cols]

    X_all = feat[candidates].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_all = X_all.ffill().bfill()

    h = 30
    y = (((feat["close"].shift(-h) / feat["close"]) - 1.0) > 0).astype(int)

    data = pd.concat([X_all, y.rename("y")], axis=1).dropna()
    if len(data) < min_rows:
        return np.nan, np.nan, 0, 0

    cut = int(len(data) * 0.8)
    train = data.iloc[:cut]
    test = data.iloc[cut:]

    X_train = train.drop(columns=["y"])
    y_train = train["y"].astype(int)
    X_test = test.drop(columns=["y"])
    y_test = test["y"].astype(int)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        # si el test queda con una sola clase, AUC no se puede
        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=600))])
        model.fit(X_train, y_train)
        p_today = float(model.predict_proba(X_all.iloc[[-1]])[:, 1][0])
        return p_today, np.nan, int(len(train)), int(len(test))

    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=600))])
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, p_test))
    p_today = float(model.predict_proba(X_all.iloc[[-1]])[:, 1][0])

    return p_today, auc, int(len(train)), int(len(test))


# ==========================================================
# SCORE (simple + interpretable)
# ==========================================================
def grt_pulse_score(df: pd.DataFrame, dfb: pd.DataFrame, news_score: float, p30: float, auc: float, fund_last: Optional[pd.Series]) -> Dict[str, float]:
    close = df["close"].astype(float)
    ret = df["ret"].astype(float)

    tr = simple_trend(df["log_close"], 90)
    slope = tr["trend_slope"]
    r2 = tr["trend_r2"]
    mom30 = momentum(ret, 30)
    mom90 = momentum(ret, 90)
    rsi14 = float(rsi(close, 14).iloc[-1]) if len(close) > 40 else np.nan
    rv30 = realized_vol(ret, 30)
    vz = vol_z_14(df["volume"])
    corr60 = rolling_corr(df["ret"], dfb["ret"], 60) if (dfb is not None and not dfb.empty) else np.nan

    stake_ratio = np.nan
    mcap_to_vol = np.nan
    if fund_last is not None and isinstance(fund_last, pd.Series) and not fund_last.empty:
        if ("totalTokensStaked" in fund_last) and ("totalSupply" in fund_last):
            stake_ratio = safe_float(fund_last.get("totalTokensStaked")) / (safe_float(fund_last.get("totalSupply")) + 1e-12)
        if ("cg_marketcap_usd" in fund_last) and ("cg_volume_24h_usd" in fund_last):
            mcap_to_vol = safe_float(fund_last.get("cg_marketcap_usd")) / (safe_float(fund_last.get("cg_volume_24h_usd")) + 1e-12)
        if ("fund_totalTokensStaked" in fund_last) and ("fund_totalSupply" in fund_last):
            # por si viniera ya con prefijo
            stake_ratio = safe_float(fund_last.get("fund_totalTokensStaked")) / (safe_float(fund_last.get("fund_totalSupply")) + 1e-12)

    # Score base
    s = 50.0

    # Tendencia (slope)
    if np.isfinite(slope):
        s += 12 if slope > 0 else -12

    # Momento
    if np.isfinite(mom30):
        s += 10 if mom30 > 0 else -10
    if np.isfinite(mom90):
        s += 6 if mom90 > 0 else -6

    # RSI extremos (ligeros)
    if np.isfinite(rsi14):
        if rsi14 < 30:
            s += 3
        elif rsi14 > 70:
            s -= 3

    # Volatilidad (penaliza si está muy alta)
    if np.isfinite(rv30):
        if rv30 > 1.10:     # ~110% anualizada (muy cripto)
            s -= 8
        elif rv30 > 0.80:
            s -= 4
        else:
            s += 2

    # Volumen anómalo (si es muy bajo, peor)
    if np.isfinite(vz):
        if vz > 0.5:
            s += 3
        elif vz < -0.5:
            s -= 3

    # Dependencia BTC (si es altísima, más “beta” que tesis)
    if np.isfinite(corr60):
        if corr60 > 0.75:
            s -= 2

    # Noticias (ligero)
    if np.isfinite(news_score):
        if news_score >= 65:
            s += 4
        elif news_score <= 35:
            s -= 4

    # Modelo (solo si AUC decente)
    if np.isfinite(p30):
        if np.isfinite(auc) and auc >= 0.58:
            s += 12 if p30 >= 0.60 else (-12 if p30 <= 0.42 else 0)
        else:
            # si AUC es flojo, su peso baja
            s += 6 if p30 >= 0.62 else (-6 if p30 <= 0.38 else 0)

    # Fundamentals (muy suave)
    if np.isfinite(stake_ratio):
        s += 3 if stake_ratio >= 0.20 else (-3 if stake_ratio < 0.10 else 0)

    # Ajuste por coherencia tendencia (R2)
    if np.isfinite(r2):
        s += 2 if r2 >= 0.55 else 0

    s = float(np.clip(s, 0, 100))

    return {
        "score": s,
        "trend_slope": slope,
        "trend_r2": r2,
        "mom30": mom30,
        "mom90": mom90,
        "rsi14": rsi14,
        "rv30": rv30,
        "vol_z14": vz,
        "corr60_btc": corr60,
        "stake_ratio": stake_ratio,
        "mcap_to_vol": mcap_to_vol,
    }


# ==========================================================
# SIDEBAR
# ==========================================================
settings = load_settings()

with st.sidebar:
    st.markdown("## 🟢 GRT Pulse")
    st.caption("Simple · Diario · Enfocado")

    preferred_exchange = st.selectbox(
        "Exchange",
        ["binance", "kraken", "coinbase", "bitstamp"],
        index=["binance","kraken","coinbase","bitstamp"].index(settings.get("preferred_exchange","binance"))
    )
    symbol = st.text_input("Símbolo", value=settings.get("symbol", "GRT/USDT"))
    benchmark = st.text_input("Benchmark", value=settings.get("benchmark", "BTC/USDT"))
    days = st.slider("Histórico (días)", 250, 2000, int(settings.get("days", 900)), step=50)

    st.divider()
    st.markdown("### 📰 Noticias")
    news_cfg = settings.get("news", DEFAULT_SETTINGS["news"])
    news_enable = st.toggle("Activar RSS", value=bool(news_cfg.get("enable", True)))
    news_lookback = st.slider("Ventana titulares (días)", 3, 30, int(news_cfg.get("lookback_days", 14)))

    st.divider()
    st.markdown("### 🔌 Gateway (opcional)")
    api_keys = settings.get("api_keys", {})
    thegraph_key = st.text_input("THE_GRAPH_API_KEY", value=api_keys.get("thegraph_gateway",""), type="password")

    st.divider()
    auto_save = st.toggle("Guardar en histórico al actualizar", value=True)
    run_update = st.button("🔄 Actualizar", type="primary")

    colA, colB = st.columns(2)
    with colA:
        if st.button("💾 Guardar ajustes"):
            settings["preferred_exchange"] = preferred_exchange
            settings["symbol"] = symbol.strip()
            settings["benchmark"] = benchmark.strip()
            settings["days"] = int(days)
            settings["news"] = {
                "enable": bool(news_enable),
                "lookback_days": int(news_lookback),
                "rss_timeout": int(news_cfg.get("rss_timeout", 15))
            }
            settings.setdefault("api_keys", {})
            settings["api_keys"]["thegraph_gateway"] = thegraph_key.strip()
            save_settings(settings)
            st.success("Ajustes guardados.")
    with colB:
        if st.button("🧹 Limpiar caché"):
            st.cache_data.clear()
            st.success("Caché limpiada.")


# ==========================================================
# HERO
# ==========================================================
st.markdown(
    """
    <div class="glass-hero">
      <div class="hero-title">🟢 GRT Pulse <span class="badge">Señales básicas · Modelo simple · News · Fundamentals</span></div>
      <div class="hero-sub">
        Hecha para uso diario: un “pulso” claro. Si quieres profundidad académica, la versión QuantLab es mejor; aquí priorizamos velocidad y claridad.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")


# ==========================================================
# RUN STATE
# ==========================================================
df = pd.DataFrame()
dfb = pd.DataFrame()
fund = pd.DataFrame()
fund_last = None
fund_source = "N/A"

news_score = np.nan
news_df = pd.DataFrame()

p30 = np.nan
auc = np.nan
n_train = 0
n_test = 0

pulse = {}
score = np.nan

progress = st.empty()
status = st.empty()

if run_update:
    try:
        bar = progress.progress(0, text="Preparando…")
        status.info("Descargando mercado…")

        tf = settings.get("timeframe", "1d")
        bar.progress(20, text="OHLCV…")
        df = compute_returns(fetch_ohlcv(preferred_exchange, symbol, tf, limit=int(days)))
        dfb = compute_returns(fetch_ohlcv(preferred_exchange, benchmark, tf, limit=int(days)))

        used_ex = df.attrs.get("exchange_used", preferred_exchange)
        used_sym = df.attrs.get("symbol_used", symbol)
        used_bench = dfb.attrs.get("symbol_used", benchmark)

        bar.progress(40, text="Fundamentals…")
        status.info("Cargando fundamentals…")
        tgk = (settings.get("api_keys", {}).get("thegraph_gateway","") or "").strip()
        if tgk:
            try:
                fund = fetch_grt_network_fundamentals_gateway(tgk)
                # añadimos también CoinGecko (enriquecimiento)
                cg = fetch_grt_fundamentals_coingecko()
                fund = pd.concat([cg, fund], axis=1)
                fund_source = "Graph Gateway + CoinGecko"
            except Exception as fe:
                fund = fetch_grt_fundamentals_coingecko()
                fund_source = f"CoinGecko (fallback: {fe})"
        else:
            fund = fetch_grt_fundamentals_coingecko()
            fund_source = "CoinGecko"

        fund_last = fund.iloc[-1] if (fund is not None and not fund.empty) else None

        bar.progress(60, text="Noticias…")
        status.info("Leyendo titulares…")
        try:
            news_score, news_df = build_news_panel(
                enable=bool(settings.get("news", {}).get("enable", True)),
                lookback_days=int(settings.get("news", {}).get("lookback_days", 14)),
                timeout=int(settings.get("news", {}).get("rss_timeout", 15)),
            )
        except Exception:
            news_score, news_df = np.nan, pd.DataFrame()

        bar.progress(80, text="Modelo…")
        status.info("Entrenando modelo simple (30d)…")
        feat = build_features(df, dfb, news_score, fund_last)
        p30, auc, n_train, n_test = fit_prob_30d(feat, min_rows=260)

        bar.progress(92, text="Score…")
        status.info("Calculando score…")
        pulse = grt_pulse_score(df, dfb, news_score, p30, auc, fund_last)
        score = pulse.get("score", np.nan)

        if auto_save and np.isfinite(score):
            bar.progress(97, text="Guardando…")
            row = {
                "as_of_date": str(df.index.max()),
                "updated_at_utc": now_utc_iso(),
                "exchange": used_ex,
                "symbol": used_sym,
                "benchmark": used_bench,
                "fund_source": fund_source,
                "score_0_100": float(score),
                "verdict": verdict_tag(score),
                "news_score_0_100": safe_float(news_score),
                "p_up_30d": safe_float(p30),
                "auc_test": safe_float(auc),
                "n_train": int(n_train),
                "n_test": int(n_test),
                # extras (debug útil)
                "trend_slope": safe_float(pulse.get("trend_slope", np.nan)),
                "trend_r2": safe_float(pulse.get("trend_r2", np.nan)),
                "mom30": safe_float(pulse.get("mom30", np.nan)),
                "mom90": safe_float(pulse.get("mom90", np.nan)),
                "rsi14": safe_float(pulse.get("rsi14", np.nan)),
                "rv30": safe_float(pulse.get("rv30", np.nan)),
                "vol_z14": safe_float(pulse.get("vol_z14", np.nan)),
                "corr60_btc": safe_float(pulse.get("corr60_btc", np.nan)),
                "stake_ratio": safe_float(pulse.get("stake_ratio", np.nan)),
                "mcap_to_vol": safe_float(pulse.get("mcap_to_vol", np.nan)),
            }
            save_row_to_csv(row)

        bar.progress(100, text="Listo ✅")
        status.success(f"✅ Actualizado ({used_ex} · {used_sym}) — Score: {score:.1f}/100 · {verdict_tag(score)}")
        st.caption(f"Benchmark: {dfb.attrs.get('exchange_used', used_ex)} · {used_bench} · Fundamentals: {fund_source}")

    except Exception as e:
        status.error(f"Error al actualizar: {e}")

# fallback: última fila del histórico si existe
hist = load_history()
if (not run_update) and (hist is not None) and (not hist.empty):
    h = hist.copy()
    h["as_of_date"] = pd.to_datetime(h["as_of_date"], errors="coerce")
    h = h.dropna(subset=["as_of_date"]).sort_values("as_of_date")
    last = h.iloc[-1].to_dict()
    score = safe_float(last.get("score_0_100"))
    news_score = safe_float(last.get("news_score_0_100"))
    p30 = safe_float(last.get("p_up_30d"))
    auc = safe_float(last.get("auc_test"))
    n_train = int(safe_float(last.get("n_train")) or 0)
    n_test = int(safe_float(last.get("n_test")) or 0)


# ==========================================================
# TOP KPIs
# ==========================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>Pulse Score</div>
          <div class='value'>{(score if np.isfinite(score) else np.nan):.1f}/100</div>
          <div class='hint'><b>{verdict_tag(score)}</b> · {explain_score(score)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>P(↑ 30 días)</div>
          <div class='value'>{_fmt_pct(p30)}</div>
          <div class='hint'>{explain_prob(p30, 30)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>AUC (test)</div>
          <div class='value'>{_fmt_num(auc, 3)}</div>
          <div class='hint'>{explain_auc(auc)}<br/>N train {n_train} · N test {n_test}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>News Score</div>
          <div class='value'>{_fmt_num(news_score, 1) if np.isfinite(news_score) else "—"}</div>
          <div class='hint'>{explain_news_score(news_score)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

if np.isfinite(score):
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        number={"suffix": "/100"},
        gauge={"axis": {"range": [0, 100]}},
        title={"text": "Pulse Score"}
    ))
    fig_g.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_g, use_container_width=True)

st.divider()


# ==========================================================
# TABS (simple)
# ==========================================================
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "📈 Market", "📰 News", "🧾 History"])

with tab1:
    st.subheader("Overview (lo esencial)")

    if df.empty:
        st.info("Pulsa **Actualizar** para calcular el pulso de hoy.")
    else:
        # Tabla de señales clave
        sig = pd.DataFrame([{
            "Tendencia 90d (slope log)": safe_float(pulse.get("trend_slope")),
            "Calidad tendencia (R²)": safe_float(pulse.get("trend_r2")),
            "Momentum 30d": safe_float(pulse.get("mom30")),
            "Momentum 90d": safe_float(pulse.get("mom90")),
            "RSI14": safe_float(pulse.get("rsi14")),
            "Volatilidad 30d (anualiz.)": safe_float(pulse.get("rv30")),
            "Volumen z(14d)": safe_float(pulse.get("vol_z14")),
            "Corr 60d vs BTC": safe_float(pulse.get("corr60_btc")),
            "Stake ratio (si disponible)": safe_float(pulse.get("stake_ratio")),
            "Mcap/Vol (si disponible)": safe_float(pulse.get("mcap_to_vol")),
        }])

        st.dataframe(sig.T.rename(columns={0: "valor"}), use_container_width=True)

        st.write("")
        st.markdown("### Lectura rápida (sin humo)")
        bullets = []
        if np.isfinite(pulse.get("trend_slope", np.nan)):
            bullets.append("✅ Tendencia: **alcista**" if pulse["trend_slope"] > 0 else "⚠️ Tendencia: **bajista**")
        if np.isfinite(pulse.get("mom30", np.nan)):
            bullets.append("✅ Momento 30d: **positivo**" if pulse["mom30"] > 0 else "⚠️ Momento 30d: **negativo**")
        if np.isfinite(p30):
            bullets.append(f"Modelo 30d: **P(↑) {_fmt_pct(p30)}** (AUC {_fmt_num(auc,3)})")
        if np.isfinite(news_score):
            bullets.append(f"Noticias: **{_fmt_num(news_score,1)} / 100**")

        st.markdown("\n".join([f"- {b}" for b in bullets]) if bullets else "—")

with tab2:
    st.subheader("Market")

    if df.empty:
        st.info("Pulsa **Actualizar**.")
    else:
        cd = df.reset_index().rename(columns={"date": "Date"})
        fig_c = go.Figure(data=[go.Candlestick(
            x=cd["Date"], open=cd["open"], high=cd["high"], low=cd["low"], close=cd["close"]
        )])
        fig_c.update_layout(height=440, margin=dict(l=20,r=20,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c, use_container_width=True)

        # Indicadores (RSI + MACD hist)
        close = df["close"].astype(float)
        rsi14 = rsi(close, 14)
        _, _, mh = macd(close)

        mini = pd.DataFrame({
            "date": df.index.astype(str),
            "RSI14": rsi14.values,
            "MACD_hist": mh.values,
        }).tail(220)

        colA, colB = st.columns(2)
        with colA:
            fig_r = px.line(mini, x="date", y="RSI14", title="RSI(14) – últimos 220 días")
            fig_r.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_r, use_container_width=True)
        with colB:
            fig_m = px.bar(mini, x="date", y="MACD_hist", title="MACD hist – últimos 220 días")
            fig_m.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_m, use_container_width=True)

with tab3:
    st.subheader("News")

    if news_df is None or news_df.empty:
        st.info("No hay titulares en la ventana (o RSS desactivado / falló).")
    else:
        st.markdown(
            f"""
            <div class='kpi'>
              <div class='label'>News Score (0–100)</div>
              <div class='value'>{_fmt_num(news_score, 1) if np.isfinite(news_score) else "—"}</div>
              <div class='hint'>{explain_news_score(news_score)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")
        q = st.text_input("Buscar en titulares", value="")
        df_show = news_df.copy()
        if q.strip():
            qq = _clean_text(q.strip())
            df_show = df_show[df_show["title"].apply(lambda s: qq in _clean_text(str(s)))]
        st.dataframe(df_show[["date", "title", "sent_0_100", "link"]], use_container_width=True)

with tab4:
    st.subheader("History")

    if hist is None or hist.empty:
        st.info("Aún no hay histórico. Activa guardar y pulsa **Actualizar**.")
    else:
        h = hist.copy()
        h["as_of_date"] = pd.to_datetime(h["as_of_date"], errors="coerce")
        h = h.dropna(subset=["as_of_date"]).sort_values("as_of_date", ascending=False)

        st.markdown("### Últimos registros")
        st.dataframe(h.head(200), use_container_width=True)

        st.write("")
        st.markdown("### Evolución Score + Prob + News")
        hh = h.sort_values("as_of_date")
        plot = pd.DataFrame({
            "date": hh["as_of_date"].dt.date.astype(str),
            "score": pd.to_numeric(hh.get("score_0_100", np.nan), errors="coerce"),
            "p30": pd.to_numeric(hh.get("p_up_30d", np.nan), errors="coerce"),
            "news": pd.to_numeric(hh.get("news_score_0_100", np.nan), errors="coerce"),
        })

        fig1 = px.line(plot, x="date", y=["score", "news"], title="Score y News (histórico)")
        fig1.update_layout(height=320, margin=dict(l=20,r=20,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(plot, x="date", y=["p30"], title="Probabilidad 30d (histórico)")
        fig2.update_layout(height=320, margin=dict(l=20,r=20,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

        st.write("")
        st.markdown("### Exportar CSV")
        csv_bytes = h.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar histórico (CSV)",
            data=csv_bytes,
            file_name="grt_pulse_history.csv",
            mime="text/csv"
        )

st.write("")
st.caption("⚠️ Esto no es asesoramiento financiero. Es analítica educativa: en cripto el riesgo de pérdida es alto.")
