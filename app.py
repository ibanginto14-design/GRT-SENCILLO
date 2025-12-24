# app.py
import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

# Data sources
import ccxt

# Stats
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from arch import arch_model
import ruptures as rpt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="CryptoStatLab", page_icon="📈", layout="wide")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(DATA_DIR, "daily_results.csv")

# -----------------------------
# HELPERS
# -----------------------------
def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

@st.cache_data(ttl=60*60)  # cache 1h
def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int = 365):
    """
    Descarga OHLCV desde un exchange (ccxt).
    timeframe típico: '1d'
    """
    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({"enableRateLimit": True})
    ex.load_markets()

    if symbol not in ex.markets:
        raise ValueError(f"El símbolo {symbol} no existe en {exchange_id}. Prueba otro (ej: BTC/USDT).")

    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
    df = df.drop(columns=["timestamp"]).set_index("date")
    return df

def compute_returns(df: pd.DataFrame):
    df = df.copy()
    df["log_close"] = np.log(df["close"])
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = df["log_close"].diff()
    return df

def trend_regression(df: pd.DataFrame, window: int = 90):
    """
    Regresión lineal de log(precio) vs tiempo en ventana.
    Devuelve slope, pvalue, r2.
    """
    d = df.dropna().tail(window).copy()
    y = d["log_close"].values
    x = np.arange(len(d))
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    slope = model.params[1]
    pval = model.pvalues[1]
    r2 = model.rsquared
    return {"trend_slope": slope, "trend_pvalue": pval, "trend_r2": r2}

def stationarity_tests(df: pd.DataFrame):
    """
    ADF y KPSS sobre retornos log.
    """
    d = df["log_ret"].dropna()
    # ADF: H0 = raíz unitaria (no estacionario)
    adf = adfuller(d, autolag="AIC")
    # KPSS: H0 = estacionario
    kpss_stat, kpss_p, _, _ = kpss(d, regression="c", nlags="auto")
    return {
        "adf_stat": float(adf[0]),
        "adf_pvalue": float(adf[1]),
        "kpss_stat": float(kpss_stat),
        "kpss_pvalue": float(kpss_p),
    }

def momentum_metrics(df: pd.DataFrame):
    """
    Momentum simple: retorno acumulado 30/90d, zscore del retorno reciente.
    """
    d = df.dropna().copy()
    r = d["ret"].dropna()
    if len(r) < 100:
        # fallback si hay pocos datos
        r30 = float((1 + r.tail(30)).prod() - 1) if len(r) >= 30 else np.nan
        r90 = float((1 + r.tail(90)).prod() - 1) if len(r) >= 90 else np.nan
    else:
        r30 = float((1 + r.tail(30)).prod() - 1)
        r90 = float((1 + r.tail(90)).prod() - 1)

    recent = r.tail(14)
    z = float((recent.mean() - r.mean()) / (r.std() + 1e-12))
    return {"mom_ret_30d": r30, "mom_ret_90d": r90, "mom_z_14d": z}

def volume_signal(df: pd.DataFrame):
    """
    Z-score de volumen reciente (14d) vs histórico (últimos 180d).
    """
    d = df.dropna().copy()
    v = d["volume"]
    hist = v.tail(180)
    recent = v.tail(14).mean()
    z = float((recent - hist.mean()) / (hist.std() + 1e-12))
    return {"vol_z_14d": z}

def garch_volatility(df: pd.DataFrame):
    """
    GARCH(1,1) en retornos %.
    Devuelve volatilidad condicional actual (último día).
    """
    r = df["ret"].dropna() * 100.0
    if len(r) < 200:
        return {"garch_vol_now": np.nan}

    am = arch_model(r, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
    res = am.fit(disp="off")
    cond_vol = float(res.conditional_volatility.iloc[-1])
    return {"garch_vol_now": cond_vol}

def structural_breaks(df: pd.DataFrame):
    """
    Rupturas en la serie de log_close usando ruptures (Pelt).
    Devuelve número de cambios detectados en últimos 365d (aprox).
    """
    d = df.dropna().copy()
    y = d["log_close"].values
    if len(y) < 120:
        return {"breakpoints_n": np.nan}

    algo = rpt.Pelt(model="rbf").fit(y)
    # penalty ajustable; aquí un valor razonable para 1d
    bkps = algo.predict(pen=8)
    # ruptures devuelve el último punto como len(y)
    n = max(0, len(bkps) - 1)
    return {"breakpoints_n": float(n)}

def rolling_correlation(df_asset: pd.DataFrame, df_bench: pd.DataFrame, window: int = 60):
    """
    Correlación rolling de retornos con benchmark (BTC/USDT por defecto).
    """
    a = df_asset["ret"].rename("asset")
    b = df_bench["ret"].rename("bench")
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < window + 5:
        return {"corr_60d": np.nan}
    corr = joined["asset"].rolling(window).corr(joined["bench"]).iloc[-1]
    return {"corr_60d": float(corr)}

def simple_backtest_prob(df: pd.DataFrame):
    """
    Probabilidad empírica sencilla:
    Condición: vol_z_14d > 0 y mom_ret_30d > 0 (en cada fecha donde se puede calcular)
    Target: subida a 14 días (retorno 14d > 0)
    """
    d = df.dropna().copy()
    d["ret_fwd_14"] = d["close"].shift(-14) / d["close"] - 1

    # features
    d["mom_30"] = (1 + d["ret"]).rolling(30).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    vol_hist = d["volume"].rolling(180)
    d["vol_z_14"] = (d["volume"].rolling(14).mean() - vol_hist.mean()) / (vol_hist.std() + 1e-12)

    sub = d.dropna(subset=["mom_30", "vol_z_14", "ret_fwd_14"]).copy()
    if len(sub) < 200:
        return {"bt_p_up_14d": np.nan, "bt_n": np.nan}

    cond = (sub["vol_z_14"] > 0) & (sub["mom_30"] > 0)
    hits = (sub.loc[cond, "ret_fwd_14"] > 0).mean() if cond.sum() > 20 else np.nan
    return {"bt_p_up_14d": float(hits) if hits == hits else np.nan, "bt_n": float(cond.sum())}

def scoreboard(metrics: dict):
    """
    Semáforo simple 0..100 a partir de señales.
    No es “predicción”, es un resumen de confluencia.
    """
    score = 50.0

    # Trend slope
    slope = metrics.get("trend_slope", np.nan)
    pval = metrics.get("trend_pvalue", np.nan)
    if slope == slope and pval == pval:
        if slope > 0 and pval < 0.05:
            score += 15
        elif slope > 0:
            score += 7
        elif slope < 0 and pval < 0.05:
            score -= 15
        elif slope < 0:
            score -= 7

    # Momentum
    m30 = metrics.get("mom_ret_30d", np.nan)
    if m30 == m30:
        score += 10 if m30 > 0 else -10

    # Volume z
    vz = metrics.get("vol_z_14d", np.nan)
    if vz == vz:
        score += 8 if vz > 0.5 else (-8 if vz < -0.5 else 0)

    # GARCH vol: si está muy alta, señal de riesgo (no direccional)
    gv = metrics.get("garch_vol_now", np.nan)
    if gv == gv:
        score += 3 if gv < 4 else (-6 if gv > 8 else -2)

    # Correlation: desacople puede ser bueno (si sube con menos dependencia)
    corr = metrics.get("corr_60d", np.nan)
    if corr == corr:
        score += 4 if corr < 0.5 else 0

    # Breakpoints: muchos cambios = régimen inestable (penaliza un poco)
    bp = metrics.get("breakpoints_n", np.nan)
    if bp == bp:
        score -= min(10, bp * 2)

    # Backtest prob
    p_up = metrics.get("bt_p_up_14d", np.nan)
    if p_up == p_up:
        score += 10 if p_up > 0.6 else (-10 if p_up < 0.45 else 0)

    score = float(np.clip(score, 0, 100))
    return score

def append_results(row: dict, path: str = RESULTS_PATH):
    df = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        out = pd.concat([old, df], ignore_index=True)
        # evita duplicar el mismo "as_of_date"
        out = out.drop_duplicates(subset=["as_of_date", "exchange", "symbol"], keep="last")
    else:
        out = df
    out.to_csv(path, index=False)

def load_results(path: str = RESULTS_PATH):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

# -----------------------------
# UI
# -----------------------------
st.title("📈 CryptoStatLab — Panel de pruebas estadísticas (actualizable diario)")
st.caption("Esto NO es una bola de cristal: son **indicadores probabilísticos**. Úsalos como confluencia, no como certeza.")

colA, colB, colC, colD = st.columns([1.2, 1.2, 1, 1])
with colA:
    exchange_id = st.selectbox("Exchange (ccxt)", ["binance", "coinbase", "kraken"], index=0)
with colB:
    symbol = st.text_input("Símbolo (ej: BTC/USDT, ETH/USDT)", value="BTC/USDT")
with colC:
    limit = st.number_input("Días (OHLCV)", min_value=200, max_value=2000, value=600, step=50)
with colD:
    bench_symbol = st.text_input("Benchmark (para correlación)", value="BTC/USDT")

timeframe = "1d"

st.divider()
c1, c2 = st.columns([1, 2])

with c1:
    do_update = st.button("🔄 Actualizar ahora (calcular y guardar)", type="primary")
    st.write("")
    st.info("Para actualización automática diaria, usa `daily_update.py` con el Programador de tareas (Windows) o cron (Linux).")
    st.write(f"🕒 Hora actual: **{now_utc_iso()}**")

with c2:
    st.subheader("Histórico de resultados guardados")
    res_df = load_results()
    if res_df.empty:
        st.write("Aún no hay resultados guardados.")
    else:
        # muestra últimos 30
        st.dataframe(res_df.sort_values("as_of_date").tail(30), use_container_width=True)

st.divider()

# -----------------------------
# MAIN UPDATE LOGIC
# -----------------------------
if do_update:
    try:
        df = fetch_ohlcv(exchange_id, symbol, timeframe, limit=int(limit))
        df = compute_returns(df)

        # benchmark
        dfb = fetch_ohlcv(exchange_id, bench_symbol, timeframe, limit=int(limit))
        dfb = compute_returns(dfb)

        metrics = {}
        metrics.update(trend_regression(df, window=90))
        metrics.update(stationarity_tests(df))
        metrics.update(momentum_metrics(df))
        metrics.update(volume_signal(df))
        metrics.update(garch_volatility(df))
        metrics.update(structural_breaks(df))
        metrics.update(rolling_correlation(df, dfb, window=60))
        metrics.update(simple_backtest_prob(df))

        score = scoreboard(metrics)

        as_of_date = str(df.index.max())
        row = {
            "as_of_date": as_of_date,
            "exchange": exchange_id,
            "symbol": symbol,
            "benchmark": bench_symbol,
            "updated_at_utc": now_utc_iso(),
            "score_0_100": score,
            **{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in metrics.items()},
        }
        append_results(row)

        st.success(f"✅ Actualizado y guardado para {symbol} (fecha datos: {as_of_date}). Score: {score:.1f}/100")

    except Exception as e:
        st.error(f"Error al actualizar: {e}")

# -----------------------------
# LIVE VIEW (cálculo on-demand sin guardar)
# -----------------------------
st.subheader("Vista actual (sin guardar)")

try:
    df_live = fetch_ohlcv(exchange_id, symbol, timeframe, limit=int(limit))
    df_live = compute_returns(df_live)

    # benchmark
    dfb_live = fetch_ohlcv(exchange_id, bench_symbol, timeframe, limit=int(limit))
    dfb_live = compute_returns(dfb_live)

    metrics_live = {}
    metrics_live.update(trend_regression(df_live, window=90))
    metrics_live.update(stationarity_tests(df_live))
    metrics_live.update(momentum_metrics(df_live))
    metrics_live.update(volume_signal(df_live))
    metrics_live.update(garch_volatility(df_live))
    metrics_live.update(structural_breaks(df_live))
    metrics_live.update(rolling_correlation(df_live, dfb_live, window=60))
    metrics_live.update(simple_backtest_prob(df_live))
    score_live = scoreboard(metrics_live)

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Score (0-100)", f"{score_live:.1f}")
    top2.metric("Trend slope (90d)", f"{metrics_live['trend_slope']:.4f}")
    top3.metric("Momentum 30d", f"{metrics_live['mom_ret_30d']*100:.2f}%")
    top4.metric("Vol z (14d)", f"{metrics_live['vol_z_14d']:.2f}")

    st.write("")
    st.dataframe(pd.DataFrame(metrics_live, index=["value"]).T, use_container_width=True)

    st.write("")
    st.subheader("Precio y volumen")
    chart_df = df_live.copy()
    chart_df["close"] = chart_df["close"].astype(float)
    st.line_chart(chart_df["close"])
    st.line_chart(chart_df["volume"])

except Exception as e:
    st.warning(f"No se pudo cargar vista actual: {e}")

st.divider()
st.subheader("Exportar histórico")
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "rb") as f:
        st.download_button("⬇️ Descargar daily_results.csv", f, file_name="daily_results.csv", mime="text/csv")
else:
    st.caption("Cuando actualices al menos una vez, aparecerá el export.")
