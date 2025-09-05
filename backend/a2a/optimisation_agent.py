

import os, json, warnings
warnings.filterwarnings("ignore")

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

# ---- LangGraph wiring ----
try:
    from langgraph.graph import StateGraph
    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False

# ---------------- utilities ----------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def _under_run_base(state: Dict[str, Any], path: str) -> str:
    """
    If state has run_base and 'path' is relative, join them.
    Otherwise, return path unchanged.
    """
    rb = state.get("run_base")
    if rb and not os.path.isabs(path):
        return os.path.join(rb, path)
    return path

def kpis(equity: pd.Series, daily_ret: pd.Series) -> Dict[str, float]:
    equity = equity.dropna(); daily_ret = daily_ret.dropna()
    if len(equity) < 2 or len(daily_ret) < 2:
        return {k: float("nan") for k in
                ["CAGR","Sharpe","MaxDrawdown","HitRate","Volatility_Daily","TotalReturn"]}
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max((pd.to_datetime(equity.index[-1]) - pd.to_datetime(equity.index[0])).days / 365.25, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1.0)
    vol = float(daily_ret.std())
    sharpe = float(np.sqrt(252) * daily_ret.mean() / vol) if vol > 0 else float("nan")
    dd = float((equity / equity.cummax() - 1.0).min())
    hit = float((daily_ret > 0).mean() * 100.0)
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDrawdown": dd,
            "HitRate": hit, "Volatility_Daily": vol, "TotalReturn": total_return}

# ------------- schema helpers -------------
def _norm_predictions(df: pd.DataFrame, head: str) -> pd.DataFrame:
    """Return Date, Pred_Return, Actual_Return."""
    df = df.copy()
    if "Date" not in df.columns: raise ValueError("Predictions CSV missing 'Date'")
    pred_two = f"Pred_{head.capitalize()}_Return"
    act_two  = f"Actual_{head.capitalize()}_Return"
    if {"Pred_Return","Actual_Return"}.issubset(df.columns):
        out = df[["Date","Pred_Return","Actual_Return"]].copy()
    elif {pred_two, act_two}.issubset(df.columns):
        out = df[["Date", pred_two, act_two]].rename(
            columns={pred_two:"Pred_Return", act_two:"Actual_Return"})
    else:
        raise ValueError("Predictions CSV must have Pred_Return & Actual_Return (or head-specific columns).")
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date").reset_index(drop=True)

def _norm_forecast(df: pd.DataFrame, head: str) -> pd.DataFrame:
    """Return Date, Pred_Return."""
    df = df.copy()
    if "Date" not in df.columns: raise ValueError("Forecast CSV missing 'Date'")
    pr_two = f"Pred_{head.capitalize()}_Return"; pp_two = f"Pred_{head.capitalize()}"
    if {"Pred_Return","Pred_Close"}.issubset(df.columns):
        out = df[["Date","Pred_Return","Pred_Close"]].copy()
    elif {pr_two, pp_two}.issubset(df.columns):
        out = df[["Date", pr_two, pp_two]].rename(columns={pr_two:"Pred_Return", pp_two:"Pred_Close"})
    else:
        if "Pred_Return" in df.columns:
            out = df[["Date","Pred_Return"]].copy(); out["Pred_Close"] = np.nan
        elif pr_two in df.columns:
            out = df[["Date", pr_two]].rename(columns={pr_two:"Pred_Return"}); out["Pred_Close"] = np.nan
        else:
            raise ValueError("Forecast CSV must contain Pred_Return (and ideally Pred_Close).")
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date").reset_index(drop=True)

# ---------------- risk helpers ---------------
def _load_q_from_summary(risk_dir: str, ticker: str) -> Optional[float]:
    p = os.path.join(risk_dir, "summary.json")
    if not os.path.exists(p): return None
    try:
        obj = json.load(open(p, "r"))
        r = obj.get(ticker, {})
        cal = r.get("calibration", {})
        q = cal.get("q_abs_ret")
        return float(q) if q is not None else None
    except Exception:
        return None

def _load_predicted_risk_rows(risk_dir: str, ticker: str) -> Optional[pd.DataFrame]:
    p = os.path.join(risk_dir, f"{ticker}_predicted_risk.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    if "For_Date" in df.columns: df["Date"] = pd.to_datetime(df["For_Date"])
    elif "Date" in df.columns:   df["Date"] = pd.to_datetime(df["Date"])
    else: return None
    return df

# compute q from predictions if Risk summary not present ---
def _compute_q_from_predictions(pred_df: pd.DataFrame, alpha: float) -> Optional[float]:
    try:
        y  = pred_df["Actual_Return"].astype(float).values
        yh = pred_df["Pred_Return"].astype(float).values
        resid = np.abs(y - yh)
        if len(resid) < 30:  
            return None
        lo, hi = np.quantile(resid, [0.01, 0.99])
        resid = np.clip(resid, lo, hi)
        return float(np.quantile(resid, 1 - alpha))
    except Exception:
        return None

# =========================================================
#                       NODES
# =========================================================
def node_load(state: Dict[str, Any]) -> Dict[str, Any]:

    # ticker handling (single ticker for now)
    tickers = state.get("tickers") or ["AAPL"]
    ticker = tickers[0]

    mode     = str(state.get("mode","historical")).lower()
    head     = str(state.get("head","close")).lower()

    pred_dir = _under_run_base(state, state.get("predictions_dir","Predictive_Model/predictions"))
    fcast_dir= _under_run_base(state, state.get("forecasts_dir","Predictive_Model/advanced_forecasts"))
    fe_path  = _under_run_base(state, state.get("features_path","FE_Agent/features_engineered.csv"))
    risk_dir = _under_run_base(state, state.get("risk_dir","Risk_Assessment"))
    out_dir  = _under_run_base(state, state.get("out_dir","Opt_results"))
    ensure_dir(out_dir)

    # try to get q from Risk assessment 
    q = _load_q_from_summary(risk_dir, ticker)

    risk_rows = _load_predicted_risk_rows(risk_dir, ticker)  

    if mode == "historical":
        ppath = os.path.join(pred_dir, f"{ticker}_test_predictions.csv")
        if not os.path.exists(ppath):
            raise FileNotFoundError(f"Missing predictions: {ppath}")
        pred = _norm_predictions(pd.read_csv(ppath), head=head)
        hist = pred[["Date","Actual_Return"]].copy()
        hist["NextDay_Return"] = hist["Actual_Return"].shift(-1)
        hist["Vol20"] = hist["Actual_Return"].rolling(20).std()

        # fallback q if Risk summary not available
        if q is None:
            alpha = float(state.get("alpha", 0.10))
            q = _compute_q_from_predictions(pred, alpha)

        bundle = {"pred": pred, "hist": hist, "q": q, "risk_rows": None}

    elif mode == "forecast":
        if not os.path.exists(fe_path):
            raise FileNotFoundError(f"Engineered features not found: {fe_path}")
        feats = pd.read_csv(fe_path, parse_dates=["Date"])
        feats = feats[feats["Symbol"] == ticker].sort_values("Date").reset_index(drop=True)
        if len(feats) < 30:
            raise RuntimeError(f"Not enough {ticker} rows in engineered features.")
        fpath = os.path.join(fcast_dir, f"{ticker}_forecast_7d.csv")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing forecast: {fpath}")
        fc = _norm_forecast(pd.read_csv(fpath), head=head)
        feats["Actual_Return"] = feats["Close"].pct_change()
        feats["Vol20"] = feats["Actual_Return"].rolling(20).std()
        bundle = {"forecast": fc, "hist": feats[["Date","Actual_Return","Vol20"]],
                  "q": q, "risk_rows": risk_rows}
    else:
        raise ValueError("mode must be 'historical' or 'forecast'")

    print(f"[Opt] Loaded mode={mode}, ticker={ticker}, head={head}")
    if q is None:
        print("[Opt] Note: q (uncertainty band) unavailable — SNR gating will only use strong-pct / risk rows.")
    else:
        print(f"[Opt] q_abs_ret={q:.6f}")

    return {**state, "ticker": ticker, "bundle": bundle, "head": head, "out_dir": out_dir, "status":"loaded"}

def node_signals(state: Dict[str, Any]) -> Dict[str, Any]:

    mode = state.get("mode","historical").lower()
    allow_shorts = bool(state.get("allow_shorts", False))
    w_max   = float(state.get("w_max", 0.30))
    snr_min = float(state.get("snr_min", 1.0))
    strong_pct = int(state.get("strong_pct", 75))

    b = state["bundle"]
    rows = []

    if mode == "historical":
        df = b["pred"].copy()  # Date, Pred_Return, Actual_Return
        hist = b["hist"].copy()
        q = b.get("q", None)

        # strong-signal threshold from |Pred_Return|
        x = np.abs(df["Pred_Return"].values)
        thr = float(np.percentile(x, strong_pct)) if len(x) >= 10 else 0.0

        for _, r in df.iterrows():
            dt = pd.to_datetime(r["Date"])
            r_pred = float(r["Pred_Return"])
            vol = hist[hist["Date"] <= dt]["Vol20"].dropna()
            vol = float(vol.iloc[-1]) if len(vol) else float(hist["Vol20"].median() or 1e-6)

            passes = True
            if q is None:
                passes = (abs(r_pred) >= thr)
            else:
                snr = abs(r_pred) / max(q, 1e-12)
                crosses_zero = (r_pred - q) <= 0.0 <= (r_pred + q)
                passes = (snr >= snr_min) and (not crosses_zero) and (abs(r_pred) >= thr)

            if not passes or vol == 0 or np.isnan(vol):
                w = 0.0
            else:
                w = r_pred / vol
                if not allow_shorts and w < 0: w = 0.0
                w = float(np.clip(w, -w_max, w_max))
            rows.append({"Date": dt, "Weight": w})

    else:  # forecast
        fc = b["forecast"].copy()   # Date, Pred_Return (+ Pred_Close)
        hist = b["hist"].copy()
        risk_rows = b.get("risk_rows")
        q = b.get("q", None)

        for _, r in fc.iterrows():
            dt = pd.to_datetime(r["Date"])
            r_pred = float(r["Pred_Return"])
            vol = hist[hist["Date"] <= dt]["Vol20"].dropna()
            vol = float(vol.iloc[-1]) if len(vol) else float(hist["Vol20"].median() or 1e-6)

            passes = True
            if isinstance(risk_rows, pd.DataFrame):
                m = risk_rows[risk_rows["Date"] == dt]
                if not m.empty:
                    lbl = str(m["Risk_Label"].iloc[0]) if "Risk_Label" in m.columns else None
                    crosses = bool(m["CrossesZero"].iloc[0]) if "CrossesZero" in m.columns else None
                    snr_val = float(m["SNR"].iloc[0]) if "SNR" in m.columns else None
                    if lbl and lbl.upper() == "HIGH": passes = False
                    if crosses is True:               passes = False
                    if (snr_val is not None) and (snr_val < snr_min): passes = False
                elif q is not None:
                    snr = abs(r_pred) / max(q, 1e-12)
                    crosses_zero = (r_pred - q) <= 0.0 <= (r_pred + q)
                    passes = (snr >= snr_min) and (not crosses_zero)
            elif q is not None:
                snr = abs(r_pred) / max(q, 1e-12)
                crosses_zero = (r_pred - q) <= 0.0 <= (r_pred + q)
                passes = (snr >= snr_min) and (not crosses_zero)

            if not passes or vol == 0 or np.isnan(vol):
                w = 0.0
            else:
                w = r_pred / vol
                if not allow_shorts and w < 0: w = 0.0
                w = float(np.clip(w, -w_max, w_max))
            rows.append({"Date": dt, "Weight": w})

    W = pd.DataFrame(rows).set_index("Date").sort_index()
    print(f"[Opt] Built weights: {len(W)} rows (nonzero {int((W['Weight']!=0).sum())})")
    return {**state, "weights": W, "status":"signals_built"}

def node_simulate(state: Dict[str, Any]) -> Dict[str, Any]:

    cost_rate = float(state.get("cost_bps", 0.001))   # e.g., 10 bps per unit turnover
    alignment = str(state.get("alignment","next_day")).lower()

    W = state["weights"].copy()              # index Date, col Weight
    b = state["bundle"]
    if "NextDay_Return" in b["hist"].columns:
        R = b["hist"][["Date","NextDay_Return"]].copy().set_index("Date").rename(columns={"NextDay_Return":"Ret"})
    else:
        # forecast fallback 
        H = b["hist"].copy()
        H["NextDay_Return"] = H["Actual_Return"].shift(-1)
        R = H[["Date","NextDay_Return"]].set_index("Date").rename(columns={"NextDay_Return":"Ret"})

    idx = W.index.intersection(R.index)
    if len(idx) == 0:
        equity_df = pd.DataFrame(columns=["Equity","DailyReturn"], index=W.index)
        print("[Opt] No overlap between weights and returns — empty equity curve.")
        return {**state, "equity_df": equity_df, "weights_used": W, "status":"backtested"}

    W = W.loc[idx]
    R = R.loc[idx]

    W_used = W.shift(1) if alignment == "next_day" else W
    W_used = W_used.fillna(0.0)

    turnover = W.diff().abs().fillna(0.0)["Weight"]
    gross = (W_used["Weight"] * R["Ret"])
    costs = turnover * cost_rate
    net = gross - costs

    equity = (1.0 + net).cumprod()
    equity_df = pd.DataFrame({"Equity": equity, "DailyReturn": net}, index=idx)
    print(f"[Opt] Backtest built: {len(equity_df)} days, equity {equity_df['Equity'].iloc[-1]:.4f}")
    return {**state, "equity_df": equity_df, "weights_used": W_used, "status":"backtested"}

def node_save_min(state: Dict[str, Any]) -> Dict[str, Any]:

    out_dir = state.get("out_dir","Opt_results"); ensure_dir(out_dir)
    go_min_weight = float(state.get("go_min_weight", 0.10))  # threshold to act
    ticker = state.get("ticker","AAPL")

    # Build GO/NO-GO series from weights_used
    W = state["weights_used"].copy() if "weights_used" in state else pd.DataFrame()
    if not W.empty and "Weight" in W.columns:
        go_series = np.where(np.abs(W["Weight"].values) >= go_min_weight, "GO", "NO-GO")
        go_idx = W.index
    else:
        go_series = np.array([], dtype=str)
        go_idx = pd.DatetimeIndex([])

    # Compose equity_curve.csv
    ec = state["equity_df"].copy() if "equity_df" in state else pd.DataFrame()
    if not ec.empty and len(go_series):
        go_t = pd.Series(go_series, index=go_idx).reindex(ec.index).fillna("NO-GO")
        ec_out = ec.copy()
        
        ec_out[f"GO_{ticker}"] = go_t.values
    elif not ec.empty:
        ec_out = ec.copy()
        ec_out[f"GO_{ticker}"] = "NO-GO"
    else:
        ec_out = pd.DataFrame(columns=["Equity","DailyReturn",f"GO_{ticker}"])

    ec_out.reset_index().rename(columns={"index":"Date"}).to_csv(
        os.path.join(out_dir, "equity_curve.csv"), index=False
    )

    # Latest position/decision/go-no-go
    if not W.empty:
        last_dt = W.index.max()
        last_w = float(W.loc[last_dt, "Weight"])
        last_decision = "LONG" if last_w > 0 else ("SHORT" if last_w < 0 else "FLAT")
        last_go = "GO" if abs(last_w) >= go_min_weight else "NO-GO"
    else:
        last_dt, last_w, last_decision, last_go = None, 0.0, "FLAT", "NO-GO"

    # KPIs
    if "Date" in ec_out.columns and len(ec_out) > 0:
        eq_series = ec_out.set_index("Date")["Equity"]
        dr_series = ec_out.set_index("Date")["DailyReturn"]
    else:
        eq_series = ec_out["Equity"]
        dr_series = ec_out["DailyReturn"]

    k = kpis(eq_series, dr_series)

    summary = {
        "ticker": ticker,
        "kpis": k,
        "params": {
            "mode": state.get("mode","historical"),
            "head": state.get("head","close"),
            "alignment": state.get("alignment","next_day"),
            "allow_shorts": bool(state.get("allow_shorts", False)),
            "w_max": float(state.get("w_max", 0.30)),
            "snr_min": float(state.get("snr_min", 1.0)),
            "strong_pct": int(state.get("strong_pct", 75)),
            "cost_bps": float(state.get("cost_bps", 0.001)),
            "go_min_weight": go_min_weight
        },
        "window": {
            "start": str(ec_out["Date"].min()) if "Date" in ec_out.columns and len(ec_out)>0 else None,
            "end":   str(ec_out["Date"].max()) if "Date" in ec_out.columns and len(ec_out)>0 else None
        },
        "latest": {
            "timestamp": str(last_dt) if last_dt is not None else None,
            "weight": last_w,
            "decision": last_decision,  # LONG / SHORT / FLAT
            "go_nogo": last_go          # GO / NO-GO
        }
    }
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)

    print(f"[Opt] Saved: equity_curve.csv  & summary.json → {out_dir}")
    print(f"[Opt] Last: {summary['latest']}")
    return {**state, "status":"saved"}

# ---------------- workflow glue ----------------
def build_workflow():
    if not HAS_LANGGRAPH:
        raise RuntimeError("LangGraph not available. Install it or call nodes manually.")
    g = StateGraph(dict)
    g.add_node("load",     node_load)
    g.add_node("signals",  node_signals)
    g.add_node("simulate", node_simulate)
    g.add_node("save",     node_save_min)

    g.set_entry_point("load")
    g.add_edge("load", "signals")
    g.add_edge("signals", "simulate")
    g.add_edge("simulate", "save")
    g.set_finish_point("save")
    return g.compile()


def build_optimization_workflow():
    g = StateGraph(dict)
    g.add_node("load", node_load)
    g.add_node("signals", node_signals)
    g.add_node("simulate", node_simulate)
    g.add_node("save", node_save_min)
    g.set_entry_point("load")
    g.add_edge("load","signals")
    g.add_edge("signals","simulate")
    g.add_edge("simulate","save")
    g.set_finish_point("save")
    return g.compile()

# --------------------- run ---------------------
if __name__ == "__main__":
    print(" Running Optimization Agent…")
    app = build_optimization_workflow()
    _ = app.invoke({
        "run_base": "/content/drive/MyDrive/A2A_prediction_system/RUN_XXXXXXXX_XXXXXX",  
        "tickers": ["AAPL"],
        "mode": "historical",                 
        "head": "close",
        "predictions_dir": "Predictive_Model/predictions",
        "forecasts_dir": "Predictive_Model/advanced_forecasts",
        "features_path": "FE_Agent/features_engineered.csv",
        "risk_dir": "Risk_Assessment",

        # risk gating & sizing
        "allow_shorts": False,                # True = long/short
        "w_max": 0.30,
        "snr_min": 1.0,
        "strong_pct": 75,
        "alignment": "next_day",
        "cost_bps": 0.001,                    # 10 bps per unit turnover
        "go_min_weight": 0.10,                # threshold for GO / NO-GO

        "alpha": 0.10,                        # used only if q must be computed from predictions
        "out_dir": "Opt_results"
    })

