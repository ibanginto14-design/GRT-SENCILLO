# daily_update.py
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import ccxt
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from arch import arch_model
import ruptures as rpt

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(DATA_DIR, "daily_results.csv")

CONFIG_PATH = os.path.join(DATA_DIR, "watchlist.json")

DEFAULT_CONFIG = {
    "exchange": "binance",
    "timeframe": "1d",
    "limit": 600,
    "benchmark": "BTC/USDT",
    "symbols": ["BTC/USDT", "ETH/USDT"]
}

def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(pd.io.json.dumps(DEFAULT_CONFIG, indent=2))
        return DEFAULT_CONFIG
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return pd.read_json(f, typ="series").to_dict() if False else __import__("json").load(f)

def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int):
    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({"enableRateLimit": True})
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
    df = df.drop(columns=["timestamp"]).set_index("date")
    return df

def compute_returns(df):
    df = df.copy()
    df["log_close"] = np.log(df["close"])
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = df["log_close"].diff()
    return df

def trend_regression(df, window=90):
    d = df.dropna().tail(window).copy()
    y = d["log_close"].values
    x = np.arange(len(d))
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return {
        "trend_slope": float(model.params[1]),
        "trend_pvalue": float(model.pvalues[1]),
        "trend_r2": float(model.rsquared),
    }

def stationarity_tests(df):
    d = df["log_ret"].dropna()
    adf = adfuller(d, autolag="AIC")
    kpss_stat, kpss_p, _, _ = kpss(d, regression="c", nlags="auto")
    return {
        "adf_stat": float(adf[0]),
        "adf_pvalue": float(adf[1]),
        "kpss_stat": float(kpss_stat),
        "kpss_pvalue": float(kpss_p),
    }

def momentum_metrics(df):
    r = df["ret"].dropna()
    r30 = float((1 + r.tail(30)).prod() - 1) if len(r) >= 30 else np.nan
    r90 = float((1 + r.tail(90)).prod() - 1) if len(r) >= 90 else np.nan
    recent = r.tail(14)
    z = float((recent.mean() - r.mean()) / (r.std() + 1e-12)) if len(r) > 30 else np.nan
    return {"mom_ret_30d": r30, "mom_ret_90d": r90, "mom_z_14d": z}

def volume_signal(df):
    v = df["volume"].dropna()
    hist = v.tail(180)
    recent = v.tail(14).mean()
    z = float((recent - hist.mean()) / (hist.std() + 1e-12)) if len(hist) > 30 else np.nan
    return {"vol_z_14d": z}

def garch_volatility(df):
    r = df["ret"].dropna() * 100.0
    if len(r) < 200:
        return {"garch_vol_now": np.nan}
    am = arch_model(r, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
    res = am.fit(disp="off")
    return {"garch_vol_now": float(res.conditional_volatility.iloc[-1])}

def structural_breaks(df):
    y = df["log_close"].dropna().values
    if len(y) < 120:
        return {"breakpoints_n": np.nan}
    algo = rpt.Pelt(model="rbf").fit(y)
    bkps = algo.predict(pen=8)
    n = max(0, len(bkps) - 1)
    return {"breakpoints_n": float(n)}

def rolling_correlation(df_asset, df_bench, window=60):
    a = df_asset["ret"].rename("asset")
    b = df_bench["ret"].rename("bench")
    joined = pd.concat([a,b], axis=1).dropna()
    if len(joined) < window + 5:
        return {"corr_60d": np.nan}
    return {"corr_60d": float(joined["asset"].rolling(window).corr(joined["bench"]).iloc[-1])}

def simple_backtest_prob(df):
    d = df.dropna().copy()
    d["ret_fwd_14"] = d["close"].shift(-14) / d["close"] - 1
    d["mom_30"] = (1 + d["ret"]).rolling(30).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    vol_hist = d["volume"].rolling(180)
    d["vol_z_14"] = (d["volume"].rolling(14).mean() - vol_hist.mean()) / (vol_hist.std() + 1e-12)
    sub = d.dropna(subset=["mom_30","vol_z_14","ret_fwd_14"])
    if len(sub) < 200:
        return {"bt_p_up_14d": np.nan, "bt_n": np.nan}
    cond = (sub["vol_z_14"] > 0) & (sub["mom_30"] > 0)
    hits = (sub.loc[cond, "ret_fwd_14"] > 0).mean() if cond.sum() > 20 else np.nan
    return {"bt_p_up_14d": float(hits) if hits == hits else np.nan, "bt_n": float(cond.sum())}

def scoreboard(m):
    score = 50.0
    slope, pval = m.get("trend_slope", np.nan), m.get("trend_pvalue", np.nan)
    if slope == slope and pval == pval:
        score += 15 if (slope > 0 and pval < 0.05) else (7 if slope > 0 else (-15 if (slope < 0 and pval < 0.05) else (-7 if slope < 0 else 0)))
    m30 = m.get("mom_ret_30d", np.nan)
    if m30 == m30: score += 10 if m30 > 0 else -10
    vz = m.get("vol_z_14d", np.nan)
    if vz == vz: score += 8 if vz > 0.5 else (-8 if vz < -0.5 else 0)
    gv = m.get("garch_vol_now", np.nan)
    if gv == gv: score += 3 if gv < 4 else (-6 if gv > 8 else -2)
    corr = m.get("corr_60d", np.nan)
    if corr == corr: score += 4 if corr < 0.5 else 0
    bp = m.get("breakpoints_n", np.nan)
    if bp == bp: score -= min(10, bp * 2)
    p_up = m.get("bt_p_up_14d", np.nan)
    if p_up == p_up: score += 10 if p_up > 0.6 else (-10 if p_up < 0.45 else 0)
    return float(np.clip(score, 0, 100))

def append_results(row):
    df = pd.DataFrame([row])
    if os.path.exists(RESULTS_PATH):
        old = pd.read_csv(RESULTS_PATH)
        out = pd.concat([old, df], ignore_index=True)
        out = out.drop_duplicates(subset=["as_of_date","exchange","symbol"], keep="last")
    else:
        out = df
    out.to_csv(RESULTS_PATH, index=False)

def main():
    cfg = load_config()
    exchange = cfg["exchange"]
    timeframe = cfg.get("timeframe","1d")
    limit = int(cfg.get("limit", 600))
    benchmark = cfg.get("benchmark","BTC/USDT")
    symbols = cfg.get("symbols", [])

    dfb = compute_returns(fetch_ohlcv(exchange, benchmark, timeframe, limit))

    for sym in symbols:
        df = compute_returns(fetch_ohlcv(exchange, sym, timeframe, limit))
        metrics = {}
        metrics.update(trend_regression(df, 90))
        metrics.update(stationarity_tests(df))
        metrics.update(momentum_metrics(df))
        metrics.update(volume_signal(df))
        metrics.update(garch_volatility(df))
        metrics.update(structural_breaks(df))
        metrics.update(rolling_correlation(df, dfb, 60))
        metrics.update(simple_backtest_prob(df))

        row = {
            "as_of_date": str(df.index.max()),
            "exchange": exchange,
            "symbol": sym,
            "benchmark": benchmark,
            "updated_at_utc": now_utc_iso(),
            "score_0_100": scoreboard(metrics),
            **{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k,v in metrics.items()}
        }
        append_results(row)

if __name__ == "__main__":
    main()
