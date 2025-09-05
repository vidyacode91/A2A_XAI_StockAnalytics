
import os, json, warnings
warnings.filterwarnings("ignore")

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from langgraph.graph import StateGraph
    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False

# ---------- Defaults ----------
FE_PATH_DEFAULT           = "FE_Agent/features_engineered.csv"
PRED_DIR_DEFAULT          = "Predictive_Model/predictions"
FCAST_DIR_DEFAULT         = "Predictive_Model/advanced_forecasts"
SHAP_DIR_DEFAULT          = "SHAP_XAI"  
OUT_DIR_DEFAULT           = "Risk_Assessment"

ALPHA_DEFAULT             = 0.10         # 90% PIs
CALIB_FRAC_DEFAULT        = 0.70         # residuals split: first 70% for calibration
CALIB_MIN_DEFAULT         = 60           # min calibration points
ROLL_COVER_WIN_DEFAULT    = 60
BOOTSTRAP_PATHS_DEFAULT   = 200
VOL_LOOKBACK_DEFAULT      = 20
ADAPT_WIDTH_WIN           = 252          # derive "wide band" threshold from last 1y widths

rng = np.random.RandomState(42)

# ---------- run_base helper ----------
def _under_run_base(state: Dict[str, Any], rel_or_abs: str) -> str:
    """If `run_base` is present, resolve paths under it; otherwise use as-is."""
    if not rel_or_abs:
        return rel_or_abs
    rb = state.get("run_base")
    if rb and not os.path.isabs(rel_or_abs):
        return os.path.join(rb, rel_or_abs)
    return rel_or_abs

# ---------- Basic helpers ----------
def pct_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()

def rolling_vol(returns: pd.Series, window: int = VOL_LOOKBACK_DEFAULT) -> pd.Series:
    return returns.rolling(window).std()

def drawdown(close: pd.Series) -> pd.Series:
    peak = close.cummax()
    return (close - peak) / peak

def sharpe_ratio(returns: pd.Series, annualization: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0: return float("nan")
    return float(np.sqrt(annualization) * r.mean() / r.std())

def var_percentile(returns: pd.Series, q: float = 0.05) -> float:
    r = returns.dropna()
    if len(r) == 0: return float("nan")
    return float(np.quantile(r, q))

def winkler_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    width = hi - lo
    below, above = y < lo, y > hi
    score = width.copy()
    score[below] += (2.0/alpha) * (lo[below] - y[below])
    score[above] += (2.0/alpha) * (y[above] - hi[above])
    return score

# ---------- I/O helpers ----------
def load_test_predictions(pred_dir: str, ticker: str) -> Optional[pd.DataFrame]:
    path = os.path.join(pred_dir, f"{ticker}_test_predictions.csv")
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    # normalize names
    ren = {}
    if "Actual_Close_Return" in df.columns: ren["Actual_Close_Return"] = "Actual_Return"
    if "Pred_Close_Return"   in df.columns: ren["Pred_Close_Return"]   = "Pred_Return"
    df = df.rename(columns=ren)
    need = {"Actual_Return","Pred_Return"}
    if not need.issubset(df.columns): return None
    if "Date" in df.columns: df["Date"] = pd.to_datetime(df["Date"])
    return df

def load_tomorrow(pred_dir: str, ticker: str) -> Optional[dict]:
    path = os.path.join(pred_dir, f"{ticker}_tomorrow.csv")
    if not os.path.exists(path): return None
    return pd.read_csv(path).iloc[0].to_dict()

def load_forecast(fdir: str, ticker: str) -> Optional[pd.DataFrame]:
    path = os.path.join(fdir, f"{ticker}_forecast_7d.csv")
    if not os.path.exists(path): return None
    df = pd.read_csv(path, parse_dates=["Date"])
    if "Pred_Close_Return" in df.columns: df = df.rename(columns={"Pred_Close_Return":"Pred_Return"})
    if "Predicted_Close" in df.columns: df["Pred_Close"] = df["Predicted_Close"]
    need = {"Date","Pred_Return","Pred_Close"}
    return df[list(need)] if need.issubset(df.columns) else None

def load_top_features(shap_dir: str, ticker: str, head: str = "close", top_n: int = 5) -> Optional[List[str]]:
    path = os.path.join(shap_dir, ticker, f"{ticker}_feature_importance_{head}.csv")
    if not os.path.exists(path): return None
    try:
        imp = pd.read_csv(path)
        col_feat = "feature" if "feature" in imp.columns else imp.columns[0]
        col_score= "mean_abs_shap" if "mean_abs_shap" in imp.columns else imp.columns[1]
        imp = imp.sort_values(col_score, ascending=False)
        return imp[col_feat].head(top_n).tolist()
    except Exception:
        return None

# ---------- Risk helpers ----------
def volatility_marks(returns: pd.Series, lookback: int = VOL_LOOKBACK_DEFAULT):
    rv = returns.rolling(lookback).std().dropna()
    if len(rv) < 30: return None
    return {"recent": float(rv.iloc[-1]),
            "p75": float(np.quantile(rv, 0.75)),
            "p90": float(np.quantile(rv, 0.90))}

def risk_label(interval_pct: float, crosses_zero: bool, marks: Optional[dict], width_thresh: float):
    triggers = []
    if interval_pct >= width_thresh: triggers.append("uncertainty_wide")
    if crosses_zero: triggers.append("direction_uncertain")
    if marks:
        if marks["recent"] >= marks["p90"]: triggers.append("volatility_extreme")
        elif marks["recent"] >= marks["p75"]: triggers.append("volatility_high")
    if "volatility_extreme" in triggers or ("uncertainty_wide" in triggers and "direction_uncertain" in triggers):
        lvl = "HIGH"
    elif triggers:
        lvl = "MEDIUM"
    else:
        lvl = "LOW"
    return lvl, triggers

def return_to_price_interval(latest_close: float, pred_return: float, q_abs_ret: float):
    lo_r, hi_r = pred_return - q_abs_ret, pred_return + q_abs_ret
    lo_p = latest_close * (1.0 + lo_r)
    hi_p = latest_close * (1.0 + hi_r)
    return (lo_r, hi_r), (lo_p, hi_p)

# ---------- Conformal calibration ----------
def calibrate_from_residuals(df_test: pd.DataFrame,
                             alpha: float,
                             calib_frac: float,
                             calib_min: int,
                             roll_cov_win: int) -> Dict[str, Any]:
    y  = df_test["Actual_Return"].values.astype(float)
    yh = df_test["Pred_Return"].values.astype(float)
    resid = y - yh
    abs_resid = np.abs(resid)
    n = len(abs_resid)
    if n < max(calib_min, 20):
        return {"ok": False, "reason": "not_enough_points"}

    n_cal = max(calib_min, int(calib_frac * n))
    n_cal = min(n_cal, n - 5)
    cal = abs_resid[:n_cal].copy()
    lo, hi = np.quantile(cal, [0.01, 0.99])
    cal = np.clip(cal, lo, hi)
    q = float(np.quantile(cal, 1 - alpha))

    # Diagnostics on evaluation tail
    y_ev, yh_ev = y[n_cal:], yh[n_cal:]
    lo_ev, hi_ev = yh_ev - q, yh_ev + q
    cov = float(np.mean((y_ev >= lo_ev) & (y_ev <= hi_ev))) if len(y_ev) else np.nan
    wink = float(np.mean(winkler_score(y_ev, lo_ev, hi_ev, alpha))) if len(y_ev) else np.nan

    # Rolling coverage (on full series)
    lo_all, hi_all = yh - q, yh + q
    ok_all = (y >= lo_all) & (y <= hi_all)
    roll_cov = pd.Series(ok_all).rolling(roll_cov_win).mean().values

    # Adaptive width history 
    width_hist = None
    if {"Pred_Close","Actual_Close"}.issubset(df_test.columns):
        base = df_test["Actual_Close"].shift(1).values.astype(float)
        width_price = base * (2.0 * q)  # approximate price width from return width
        rel_width = width_price / np.maximum(df_test["Pred_Close"].values.astype(float), 1e-8)
        width_hist = pd.Series(rel_width).rolling(ADAPT_WIDTH_WIN).mean().dropna().values

    return {
        "ok": True,
        "alpha": alpha,
        "q_abs_ret": q,
        "coverage_eval": cov,
        "winkler_mean_eval": wink,
        "indices": {"calib_end": int(n_cal), "total": int(n)},
        "roll_cov": roll_cov.tolist(),
        "width_hist": width_hist.tolist() if width_hist is not None else None
    }

# ---------- Fan chart (residual bootstrapping) ----------
def bootstrap_fan_chart(start_price: float,
                        pred_returns: np.ndarray,
                        calib_residuals: np.ndarray,
                        n_paths: int = BOOTSTRAP_PATHS_DEFAULT) -> Dict[str, np.ndarray]:
    T = len(pred_returns)
    paths = np.zeros((n_paths, T))
    for i in range(n_paths):
        p = start_price
        eps = rng.choice(calib_residuals, size=T, replace=True)
        for t in range(T):
            r = float(pred_returns[t]) + float(eps[t])
            p = p * (1.0 + r)
            paths[i, t] = p
    return {"p10": np.percentile(paths, 10, axis=0),
            "p50": np.percentile(paths, 50, axis=0),
            "p90": np.percentile(paths, 90, axis=0)}

# =========================================================
# -------------------- Pipeline nodes ---------------------
# =========================================================
def node_load(state: Dict[str, Any]) -> Dict[str, Any]:
    
    fe_path   = _under_run_base(state, state.get("data_path", FE_PATH_DEFAULT))
    pred_dir  = _under_run_base(state, state.get("predictions_dir", PRED_DIR_DEFAULT))
    fcast_dir = _under_run_base(state, state.get("forecasts_dir", FCAST_DIR_DEFAULT))
    shap_dir  = _under_run_base(state, state.get("shap_dir", SHAP_DIR_DEFAULT))

    tickers   = state.get("tickers")
    days_win  = state.get("days_window", None)

    if not os.path.exists(fe_path):
        raise FileNotFoundError(f"Engineered features not found: {fe_path}")

    fe = pd.read_csv(fe_path, parse_dates=["Date"])
    need = {"Date","Symbol","Close"}
    if not need.issubset(fe.columns):
        raise ValueError(f"{fe_path} missing columns {need}")
    fe = fe[["Date","Symbol","Close"]].dropna().sort_values(["Symbol","Date"]).reset_index(drop=True)
    if not tickers:
        tickers = sorted(fe["Symbol"].unique().tolist())

    per_ticker = {}
    for t in tickers:
        hist = fe[fe["Symbol"] == t].copy()
        if hist.empty: continue
        if days_win and len(hist) > days_win:
            hist = hist.tail(days_win).copy()

        test = load_test_predictions(pred_dir, t)
        tom  = load_tomorrow(pred_dir, t)
        fc   = load_forecast(fcast_dir, t)
        top5 = load_top_features(shap_dir, t, head=state.get("head","close"))

        per_ticker[t] = {
            "hist": hist,              # Date, Close
            "test_preds": test,        # residuals source
            "tomorrow": tom,           # next-day point forecast
            "forecast": fc,            # 7-day point forecasts
            "top_features": top5       # enrichment
        }

    return {**state,
            "per_ticker": per_ticker,
            "tickers": list(per_ticker.keys()),
            "fe_path": fe_path,
            "pred_dir": pred_dir,
            "fcast_dir": fcast_dir,
            "shap_dir": shap_dir,
            "status": "loaded"}

def node_assess(state: Dict[str, Any]) -> Dict[str, Any]:
    alpha      = float(state.get("alpha", ALPHA_DEFAULT))
    calib_frac = float(state.get("calib_frac", CALIB_FRAC_DEFAULT))
    calib_min  = int(state.get("calib_min", CALIB_MIN_DEFAULT))
    roll_win   = int(state.get("roll_cov_win", ROLL_COVER_WIN_DEFAULT))
    ann        = int(state.get("annualization", 252))
    lookback   = int(state.get("vol_lookback", VOL_LOOKBACK_DEFAULT))

    for t, bundle in state["per_ticker"].items():
        hist = bundle["hist"].copy().set_index("Date")
        close = hist["Close"].astype(float)
        ret   = pct_returns(close)
        vol   = rolling_vol(ret, window=lookback)
        dd    = drawdown(close)
        marks = volatility_marks(ret, lookback=lookback)

        # Metrics (historical)
        metrics = {
            "Obs": int(len(close)),
            "Daily_Vol": float(ret.std()) if len(ret) else np.nan,
            "Sharpe": sharpe_ratio(ret, ann),
            "Max_Drawdown": float(dd.min()) if len(dd) else np.nan,
            "VaR_5pct_1d": var_percentile(ret, 0.05),
        }

        # Conformal calibration
        calib = {"ok": False, "reason": "no_test_predictions"}
        if bundle["test_preds"] is not None and len(bundle["test_preds"]) >= max(calib_min, 20):
            calib = calibrate_from_residuals(bundle["test_preds"], alpha, calib_frac, calib_min, roll_win)

        # Adaptive width threshold
        width_thresh = 0.02
        if calib.get("width_hist"):
            w = np.array(calib["width_hist"])
            if len(w) > 20:
                width_thresh = float(np.nanpercentile(w, 80))

        # Tomorrow assessment
        risk = None
        predicted_rows = []
        if bundle["tomorrow"] and calib.get("ok"):
            tom   = bundle["tomorrow"]
            last_close = float(tom.get("Latest_Actual_Close", close.iloc[-1]))
            r_pred = float(tom.get("Pred_Close_Return", tom.get("Pred_Return", 0.0)))
            p_pred = float(tom.get("Pred_Close", last_close * (1 + r_pred)))
            (lo_r, hi_r), (lo_p, hi_p) = return_to_price_interval(last_close, r_pred, calib["q_abs_ret"])
            band_pct = (hi_p - lo_p) / max(p_pred, 1e-8)
            crosses0 = (lo_r <= 0.0 <= hi_r)
            label, triggers = risk_label(band_pct, crosses0, marks, width_thresh)

            snr = abs(r_pred) / max(calib["q_abs_ret"], 1e-12)

            risk = {
                "Pred_Close": p_pred,
                "PI_Return": [lo_r, hi_r],
                "PI_Close": [lo_p, hi_p],
                "Uncertainty_Band_Pct": float(band_pct),
                "Crosses_Zero": bool(crosses0),
                "Risk_Label": label,
                "Risk_Triggers": triggers,
                "SNR": float(snr),
                "Width_Thresh_Used": width_thresh,
                "Top_Features": bundle.get("top_features")
            }

            predicted_rows.append({
                "For_Date": tom.get("For_Date"),
                "Pred_Close": p_pred,
                "PI_Close_Lo": lo_p,
                "PI_Close_Hi": hi_p,
                "BandPct": band_pct,
                "CrossesZero": crosses0,
                "Risk_Label": label,
                "Triggers": ";".join(triggers),
                "Pred_Close_Return": r_pred,
                "SNR": snr,
                "Top_Features": ",".join(bundle.get("top_features") or [])
            })

        # 7-day fan chart (forecast + calibration)
        forecast_bands = None
        if bundle["forecast"] is not None and calib.get("ok"):
            df_test = bundle["test_preds"]
            n_cal = calib["indices"]["calib_end"] if calib.get("indices") else max(60, int(0.7*len(df_test)))
            eps = (df_test["Actual_Return"].values[:n_cal] - df_test["Pred_Return"].values[:n_cal])
            fc = bundle["forecast"].copy()
            fan = bootstrap_fan_chart(float(close.iloc[-1]),
                                      fc["Pred_Return"].values.astype(float),
                                      eps,
                                      n_paths=state.get("bootstrap_paths", BOOTSTRAP_PATHS_DEFAULT))
            fc["PI10"] = fan["p10"]; fc["PI50"] = fan["p50"]; fc["PI90"] = fan["p90"]
            forecast_bands = fc

            
            q = calib["q_abs_ret"]
            base = float(close.iloc[-1])
            for i, r_k in enumerate(fc["Pred_Return"].values.astype(float)):
                (lo_r, hi_r), (lo_p, hi_p) = return_to_price_interval(base, r_k, q)
                band_pct = (hi_p - lo_p) / max(fc["Pred_Close"].iloc[i], 1e-8)
                crosses0 = (lo_r <= 0.0 <= hi_r)
                label, triggers = risk_label(band_pct, crosses0, marks, width_thresh)
                predicted_rows.append({
                    "For_Date": fc["Date"].iloc[i].strftime("%Y-%m-%d"),
                    "Pred_Close": float(fc["Pred_Close"].iloc[i]),
                    "PI_Close_Lo": lo_p,
                    "PI_Close_Hi": hi_p,
                    "BandPct": band_pct,
                    "CrossesZero": crosses0,
                    "Risk_Label": label,
                    "Triggers": ";".join(triggers),
                    "Pred_Close_Return": r_k,
                    "SNR": abs(r_k)/max(q,1e-12),
                    "Top_Features": ",".join(bundle.get("top_features") or [])
                })

        # Persist in bundle
        bundle.update({
            "close": close, "returns": ret, "rolling_vol": vol, "drawdown": dd,
            "vol_marks": marks, "calibration": calib, "risk": risk,
            "forecast_bands": forecast_bands,
            "metrics": metrics,
            "predicted_rows": predicted_rows
        })

    return {**state, "status": "assessed"}

def node_save(state: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = _under_run_base(state, state.get("out_dir", OUT_DIR_DEFAULT))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    summary, rows = {}, []

    for t, b in state["per_ticker"].items():
        hist_idx = b["close"].index
        # ---- Historical CSV
        pd.DataFrame({
            "Date": hist_idx,
            "Close": b["close"].values,
            "Return": b["returns"].reindex(hist_idx).values,
            "RollingVol20": b["rolling_vol"].reindex(hist_idx).values,
            "Drawdown": b["drawdown"].reindex(hist_idx).values
        }).to_csv(os.path.join(out_dir, f"{t}_historical_risk.csv"), index=False)

        # ---- Predicted CSV (tomorrow + 7-day)
        if b.get("predicted_rows"):
            pd.DataFrame(b["predicted_rows"]).to_csv(os.path.join(out_dir, f"{t}_predicted_risk.csv"), index=False)

        # ---- Plots
        idx = b["close"].index
        n = len(idx)
        i_tr = int(0.70 * n)          # end of train
        i_va = int(0.85 * n)          # end of validation
        split_dates = []
        if n >= 10:  # safety
            split_dates = [idx[i_tr], idx[i_va]]

        def _plot_with_splits(series, title, ylabel, fname):
            plt.figure(figsize=(12, 4))  
            plt.plot(series.index, series.values, linewidth=1.3)
            
            for x in split_dates:
                plt.axvline(x=x, linestyle="--", linewidth=1)
           
            if split_dates:
                ylim_top = plt.gca().get_ylim()[1]
                plt.text(split_dates[0], ylim_top, "Train/Val split", va="bottom", ha="left", fontsize=8)
                plt.text(split_dates[1], ylim_top, "Val/Test split",  va="bottom", ha="left", fontsize=8)
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel(ylabel)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "plots", fname), dpi=300)
            plt.close()

        
        _plot_with_splits(b["close"],      f"{t} — Close",                    "Price (USD)",          f"{t}_price.png")
        # Drawdown and rolling volatility plot
        _plot_with_splits(b["drawdown"],   f"{t} — Drawdown",                 "Drawdown (fraction)",  f"{t}_drawdown.png")
        _plot_with_splits(b["rolling_vol"],f"{t} — Rolling Volatility (20d)", "Std. dev. (daily)",    f"{t}_rolling_vol.png")

        # Daily returns histogram
        rets = b["returns"].dropna()
        plt.figure(figsize=(8, 5))
        plt.hist(rets, bins=60, alpha=0.85)
        plt.axvline(0.0, linestyle="--", linewidth=1)
        if len(rets) > 0:
            plt.axvline(rets.mean(), linestyle=":", linewidth=1)
        plt.title(f"{t} — Daily Returns Histogram")
        plt.xlabel("Daily return")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plots", f"{t}_return_hist.png"), dpi=300)
        plt.close()

        # ---- Residual histogram & rolling coverage
        calib = b.get("calibration", {})
        if b.get("test_preds") is not None:
            resid = (b["test_preds"]["Actual_Return"] - b["test_preds"]["Pred_Return"]).values
            plt.figure(figsize=(8, 5))
            plt.hist(resid, bins=40, alpha=0.85)
            if calib.get("ok"):
                q = calib["q_abs_ret"]
                plt.axvline(+q, color="r", linestyle="--", label=f"+q @ {(1-calib['alpha'])*100:.0f}%")
                plt.axvline(-q, color="r", linestyle="--")
            plt.title(f"{t} — Test Residuals (returns)")
            plt.xlabel("Residual (Actual − Pred)")
            plt.ylabel("Count")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "plots", f"{t}_residual_hist.png"), dpi=300)
            plt.close()

        if calib.get("ok") and calib.get("roll_cov"):
            rc = np.array(calib["roll_cov"])
            plt.figure(figsize=(10, 3.5))
            plt.plot(rc, label="Rolling coverage")
            plt.axhline(1 - calib["alpha"], color="k", linestyle="--", label="Target")
            plt.ylim(0, 1)
            plt.title(f"{t} — Rolling Coverage")
            plt.xlabel("Window index")
            plt.ylabel("Coverage")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "plots", f"{t}_rolling_coverage.png"), dpi=300)
            plt.close()

        # ---- Tomorrow band plot
        if b.get("risk"):
            r = b["risk"]
            p_mid = r["Pred_Close"]
            lo_p, hi_p = r["PI_Close"]
            plt.figure(figsize=(7.8, 5))
            plt.errorbar([0], [p_mid], yerr=[[p_mid - lo_p], [hi_p - p_mid]], fmt="o")
            ttl = f"{t} — Tomorrow Close Uncertainty ({r['Risk_Label']})"
            plt.title(ttl)
            plt.xticks([])
            plt.ylabel("Price")
            plt.grid(alpha=0.3)
            txt = (
                f"Pred: {p_mid:.2f}\n"
                f"PI: [{lo_p:.2f}, {hi_p:.2f}] (width {100*r['Uncertainty_Band_Pct']:.2f}%)\n"
                f"Signal-to-noise ratio: {r['SNR']:.2f}"
            )
            plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes, va="top")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "plots", f"{t}_tomorrow_band.png"), dpi=300)
            plt.close()

        # ---- Fan chart
        if isinstance(b.get("forecast_bands"), pd.DataFrame):
            fc = b["forecast_bands"]
            plt.figure(figsize=(10, 4))
            plt.plot(fc["Date"], fc["PI50"], label="Median", linewidth=2)
            plt.fill_between(fc["Date"], fc["PI10"], fc["PI90"], alpha=0.25, label="10–90% band")
            plt.plot(fc["Date"], fc["Pred_Close"], linestyle="--", label="Model path")
            plt.title(f"{t} — 7-Day Forecast Fan Chart")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "plots", f"{t}_fan_chart.png"), dpi=300)
            plt.close()
            fc.to_csv(os.path.join(out_dir, f"{t}_forecast_fanchart.csv"), index=False)

        # Summary
        summary[t] = {
            "metrics": b["metrics"],
            "calibration": {k:v for k,v in calib.items() if k in
                            {"ok","alpha","q_abs_ret","coverage_eval","winkler_mean_eval","indices"}},
            "risk": b.get("risk"),
            "top_features": b.get("top_features")
        }
        rows.append({
            "Ticker": t,
            "Obs": b["metrics"]["Obs"],
            "Daily_Vol": round(b["metrics"]["Daily_Vol"],6) if not np.isnan(b["metrics"]["Daily_Vol"]) else np.nan,
            "Sharpe": round(b["metrics"]["Sharpe"],3) if not np.isnan(b["metrics"]["Sharpe"]) else np.nan,
            "Max_Drawdown": round(b["metrics"]["Max_Drawdown"],4) if not np.isnan(b["metrics"]["Max_Drawdown"]) else np.nan,
            "VaR_5pct_1d": round(b["metrics"]["VaR_5pct_1d"],4) if not np.isnan(b["metrics"]["VaR_5pct_1d"]) else np.nan,
            "Risk_Label_Tomorrow": b.get("risk",{}).get("Risk_Label") if b.get("risk") else None
        })

    # Write summary files
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if rows:
        pd.DataFrame(rows).sort_values("Ticker").to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print(f" Risk assessment saved → {out_dir}")
    return {**state, "status": "done", "report_dir": out_dir}

# ---------- Build / run ----------
def build_risk_workflow():
    g = StateGraph(dict)
    g.add_node("load", node_load)
    g.add_node("assess", node_assess)
    g.add_node("save", node_save)
    g.set_entry_point("load")
    g.add_edge("load", "assess")
    g.add_edge("assess", "save")
    g.set_finish_point("save")
    return g.compile()

if __name__ == "__main__":
    payload = {
        
        # "run_base": "/content/drive/MyDrive/A2A_prediction_system/RUN_YYYYMMDD_HHMMSS",

        "data_path": FE_PATH_DEFAULT,
        "predictions_dir": PRED_DIR_DEFAULT,
        "forecasts_dir": FCAST_DIR_DEFAULT,
        "shap_dir": SHAP_DIR_DEFAULT,      
        "out_dir": OUT_DIR_DEFAULT,

        # Order of agents: Prediction → SHAP/IG → Risk (this agent) → Evaluation
        # tickers: omit to auto-detect
        # "tickers": ["AAPL"],

        # Calibration & diagnostics
        "alpha": ALPHA_DEFAULT,
        "calib_frac": CALIB_FRAC_DEFAULT,
        "calib_min": CALIB_MIN_DEFAULT,
        "roll_cov_win": ROLL_COVER_WIN_DEFAULT,

        # Risk context
        "annualization": 252,
        "vol_lookback": VOL_LOOKBACK_DEFAULT,

        # Fan chart
        "bootstrap_paths": BOOTSTRAP_PATHS_DEFAULT,

        # Data window
        "days_window": 252,

        # SHAP head (for top-features enrichment)
        "head": "close"
    }
    if HAS_LANGGRAPH:
        app = build_risk_workflow()
        _ = app.invoke(payload)
    else:
        s = node_load(payload)
        s = node_assess(s)
        s = node_save(s)
