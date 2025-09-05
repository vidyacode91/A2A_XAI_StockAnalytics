import os, json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# LangGraph
try:
    from langgraph.graph import StateGraph
    HAS_LG = True
except Exception:
    HAS_LG = False

# ------------------ helpers ------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _load_json(path: str) -> Any:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _pick_run_base(state: Dict[str, Any]) -> str:
    return state.get("run_base") or "."

def _default_paths(run_base: str) -> Dict[str, str]:
    
    return {
        "predictions_dir":       os.path.join(run_base, "Predictive_Model", "predictions"),
        "models_dir":            os.path.join(run_base, "Predictive_Model", "lstm_models"),
        "eval_summary_json":     os.path.join(run_base, "Model_Evaluation", "summaries", "model_evaluation_summary.json"),

        
        "desc_summary_json":     os.path.join(run_base, "Descriptive", "summary.json"),

        # risk / opt
        "risk_summary_json":     os.path.join(run_base, "Risk_Assessment", "summary.json"),
        "opt_summary_json":      os.path.join(run_base, "Opt_results", "summary.json"),

        # explainability  
        "xai_summary_json":      os.path.join(run_base, "Explainability", "xai_summary.json"),
        "ig_dir":                os.path.join(run_base, "Explainability", "IG_XAI"),
        "shap_dir":              os.path.join(run_base, "Explainability", "SHAP_XAI"),

        # plots from agents
        "eval_plots_dir":        os.path.join(run_base, "Model_Evaluation", "plots"),
        "risk_plots_dir":        os.path.join(run_base, "Risk_Assessment", "plots"),

        # final report
        "out_dir":               os.path.join(run_base, "reports"),
        "out_payload_json":      os.path.join(run_base, "reports", "final_payload.json"),
        "out_assets_json":       os.path.join(run_base, "reports", "assets.json"),
    }

def _discover_tickers(predictions_dir: str, tickers: List[str] = None) -> List[str]:
    if tickers:
        return sorted(list({t.upper() for t in tickers}))
    if not os.path.isdir(predictions_dir):
        return []
    toks = []
    for fn in os.listdir(predictions_dir):
        if fn.endswith("_test_predictions.csv"):
            toks.append(fn.split("_")[0].upper())
    return sorted(list(set(toks)))

def _safe_get(d: dict, *path, default=None):
    cur = d
    try:
        for k in path:
            if cur is None:
                return default
            cur = cur[k]
        return cur if cur is not None else default
    except Exception:
        return default

def _latest_pred_row(pred_csv: str) -> dict:
    df = _load_csv(pred_csv)
    if df.empty or "Date" not in df.columns:
        return {}
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df.tail(1).to_dict(orient="records")[0] if len(df) else {}

def _eval_json_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Evaluation agent writes a LIST of rows.
    Convert to { TICKER: row } for easy lookup.
    """
    if isinstance(obj, list):
        out = {}
        for row in obj:
            t = row.get("Ticker") or row.get("ticker")
            if t:
                out[str(t).upper()] = row
        return out
    return obj if isinstance(obj, dict) else {}

def _compute_strong_thr_from_preds(pred_csv: str, pct: int = 75) -> float:
   
    df = _load_csv(pred_csv)
    if df.empty:
        return 0.002
    if "Pred_Return" not in df.columns:
        if "Pred_Close_Return" in df.columns:
            df = df.rename(columns={"Pred_Close_Return": "Pred_Return"})
        elif "Pred_Open_Return" in df.columns:
            df = df.rename(columns={"Pred_Open_Return": "Pred_Return"})
    if "Pred_Return" not in df.columns:
        return 0.002
    try:
        thr = float(np.percentile(np.abs(df["Pred_Return"].astype(float).values), pct))
        return thr
    except Exception:
        return 0.002

def _derive_decision_for_ticker(t: str,
                                opt_json: dict,
                                risk_json: dict,
                                desc_json: dict,
                                pred_csv_path: str) -> Dict[str, Any]:
   
    go = _safe_get(opt_json, "latest", "go_nogo")
    w  = _safe_get(opt_json, "latest", "weight", default=None)
    reason = None

    if go is None:
        risk_label = _safe_get(risk_json, t, "risk", "Risk_Label")
        thr = _compute_strong_thr_from_preds(pred_csv_path, 75)

        last = _latest_pred_row(pred_csv_path)
        pred_ret = None
        for key in ["Pred_Return", "Pred_Close_Return", "Pred_Open_Return"]:
            if key in last:
                try:
                    pred_ret = float(last[key]); break
                except Exception:
                    pass

        if pred_ret is None:
            go, reason = "NO-GO", "No latest Pred_Return available"
        else:
            if str(risk_label).upper() == "HIGH":
                go, reason = "NO-GO", f"Risk {risk_label}"
            elif abs(pred_ret) < thr:
                go, reason = "NO-GO", f"|Pred_Return|<{thr:.4f}"
            else:
                go, reason = "GO", f"|Pred_Return|≥{thr:.4f}"
        w = None if go == "NO-GO" else 1.0

    return {"go_nogo": go, "weight": w, "reason": reason}

def _find_first(paths: List[str]) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""

def _plots_for_ticker(paths: Dict[str, str], tslug: str) -> Dict[str, str]:
    
    out = {}
    tU, tl = tslug.upper(), tslug.lower()

    # Model Evaluation line/scatter (close head)
    cand_line = [
        os.path.join(paths["eval_plots_dir"], f"{tU}_close_price_line.png"),
        os.path.join(paths["eval_plots_dir"], f"{tl}_close_price_line.png"),
        os.path.join(paths["eval_plots_dir"], f"{tU}_price_line.png"),
        os.path.join(paths["eval_plots_dir"], f"{tl}_price_line.png"),
    ]
    cand_scatter = [
        os.path.join(paths["eval_plots_dir"], f"{tU}_close_price_scatter.png"),
        os.path.join(paths["eval_plots_dir"], f"{tl}_close_price_scatter.png"),
        os.path.join(paths["eval_plots_dir"], f"{tU}_price_scatter.png"),
        os.path.join(paths["eval_plots_dir"], f"{tl}_price_scatter.png"),
    ]
    p_line = _find_first(cand_line)
    p_scatter = _find_first(cand_scatter)
    if p_line:    out["actual_vs_pred_line"] = p_line
    if p_scatter: out["actual_vs_pred_scatter"] = p_scatter

    # Risk plots
    rp = paths.get("risk_plots_dir")
    for name, fnameU, fnamel in [
        ("drawdown",      f"{tU}_drawdown.png",      f"{tl}_drawdown.png"),
        ("rolling_vol",   f"{tU}_rolling_vol.png",   f"{tl}_rolling_vol.png"),
        ("tomorrow_band", f"{tU}_tomorrow_band.png", f"{tl}_tomorrow_band.png"),
        ("fanchart",      f"{tU}_fan_chart.png",     f"{tl}_fan_chart.png"),
    ]:
        p = _find_first([os.path.join(rp, fnameU), os.path.join(rp, fnamel)])
        if p:
            out[f"risk_{name}"] = p

    # ---- XAI (IG & SHAP) ----
    # IG (heatmap + importance)
    ig_heat = _find_first([
        os.path.join(paths["ig_dir"],   tU, "ig_heatmap_close.png"),
        os.path.join(paths["ig_dir"],   tl, "ig_heatmap_close.png"),
        os.path.join(paths["ig_dir"],   tU, "AAPL_heatmap_close.png"),
        os.path.join(paths["ig_dir"],   tl, "AAPL_heatmap_close.png"),
    ])
    ig_import = _find_first([
        os.path.join(paths["ig_dir"],   tU, "ig_global_importance_close.png"),
        os.path.join(paths["ig_dir"],   tl, "ig_global_importance_close.png"),
    ])

    # SHAP (global importance / beeswarm / local)
    shap_global = _find_first([
        os.path.join(paths["shap_dir"], tU, f"{tU}_global_importance_close.png"),
        os.path.join(paths["shap_dir"], tl, f"{tl}_global_importance_close.png"),
    ])
    shap_bees = _find_first([
        os.path.join(paths["shap_dir"], tU, f"{tU}_summary_beeswarm_close.png"),
        os.path.join(paths["shap_dir"], tl, f"{tl}_summary_beeswarm_close.png"),
        os.path.join(paths["shap_dir"], tU, "AAPL_summary_beeswarm_close.png"),
        os.path.join(paths["shap_dir"], tl, "AAPL_summary_beeswarm_close.png"),
    ])
    shap_local = _find_first([
        os.path.join(paths["shap_dir"], tU, f"{tU}_local_waterfall_last_close.png"),
        os.path.join(paths["shap_dir"], tl, f"{tl}_local_waterfall_last_close.png"),
        os.path.join(paths["shap_dir"], tU, "AAPL_local_waterfall_last_close.png"),
        os.path.join(paths["shap_dir"], tl, "AAPL_local_waterfall_last_close.png"),
    ])

    # What to expose to the Report Agent
    if ig_heat:    out["ig_heatmap"]   = ig_heat
    if ig_import:  out["ig_importance"] = ig_import
    if shap_global: out["shap_bar"]    = shap_global
    elif shap_bees: out["shap_bar"]    = shap_bees
    if shap_local:  out["shap_local"]  = shap_local

    return out

# ------------------ Sharpe & MaxDD from risk json ------------------
def _extract_sharpe_and_mdd(risk_json_for_ticker: dict) -> Dict[str, Any]:
    """
    Try multiple common key paths/names for Sharpe (annualised) and Max Drawdown.
    Returns dict with keys: {'sharpe_ann': float|None, 'max_drawdown': float|None}
    """
    # Check nested 'kpis' 
    sharpe = _safe_get(risk_json_for_ticker, "kpis", "Sharpe_ann")
    if sharpe is None: sharpe = _safe_get(risk_json_for_ticker, "kpis", "Sharpe")
    if sharpe is None: sharpe = _safe_get(risk_json_for_ticker, "risk", "Sharpe_ann")
    if sharpe is None: sharpe = _safe_get(risk_json_for_ticker, "Sharpe_ann")
    if sharpe is None: sharpe = _safe_get(risk_json_for_ticker, "Sharpe")
    if sharpe is None: sharpe = _safe_get(risk_json_for_ticker, "SharpeRatio")

    mdd = _safe_get(risk_json_for_ticker, "kpis", "Max_Drawdown")
    if mdd is None: mdd = _safe_get(risk_json_for_ticker, "risk", "Max_Drawdown")
    if mdd is None: mdd = _safe_get(risk_json_for_ticker, "Max_Drawdown")
    if mdd is None: mdd = _safe_get(risk_json_for_ticker, "max_drawdown")
    if mdd is None: mdd = _safe_get(risk_json_for_ticker, "MDD")

    # Cast to floats where possible
    try: sharpe = None if sharpe is None else float(sharpe)
    except: sharpe = None
    try: mdd = None if mdd is None else float(mdd)
    except: mdd = None

    return {"sharpe_ann": sharpe, "max_drawdown": mdd}

# ------------------ LangGraph nodes ------------------
def node_collect(state: Dict[str, Any]) -> Dict[str, Any]:
    run_base = _pick_run_base(state)
    paths = _default_paths(run_base)

    # allow overrides
    for k, v in (state.get("paths") or {}).items():
        paths[k] = v

    tickers = _discover_tickers(paths["predictions_dir"], state.get("tickers"))

    # load json artifacts
    eval_js_raw = _load_json(paths["eval_summary_json"]) or []
    eval_js     = _eval_json_to_dict(eval_js_raw)  # normalize to dict by ticker
    desc_js     = _load_json(paths["desc_summary_json"]) or {}
    risk_js     = _load_json(paths["risk_summary_json"]) or {}
    opt_js      = _load_json(paths["opt_summary_json"])  or {}
    xai_js      = _load_json(paths["xai_summary_json"])  or {}

    _ensure_dir(paths["out_dir"])

    return {
        **state,
        "paths": paths,
        "tickers": tickers,
        "eval_json": eval_js,
        "desc_json": desc_js,
        "risk_json": risk_js,
        "opt_json": opt_js,
        "xai_json": xai_js,
        "status": "collected"
    }

def node_summarize(state: Dict[str, Any]) -> Dict[str, Any]:
    tickers = state["tickers"]
    P = state["paths"]

    # labels 
    ui_labels = {
        "prediction_accuracy": "Prediction Accuracy",   # R², RMSE, MAPE
        "trend_prediction":    "Trend Prediction",      # Directional Accuracy %
        "statistical_insights":"Statistical Insights",  # Sharpe (ann.) and Max Drawdown
        "risk_assessment":     "Risk Assessment",       # Label, SNR, Crosses Zero
        "model_explainability":"Model Explainability",  # SHAP / IG status
        "final_recommendation":"Final Recommendation",  # Go / No-Go
    }

    out = {
        "as_of": None,
        "run_base": _pick_run_base(state),
        "tickers": tickers,
        "ui_labels": ui_labels,   
        "per_ticker": {}
    }

    for t in tickers:
        tslug = t.upper()

        # evaluation 
        e = state["eval_json"].get(tslug, {})

        # risk (summary per ticker)
        r_all = state["risk_json"] if isinstance(state["risk_json"], dict) else {}
        r_t   = r_all.get(tslug, {}) if isinstance(r_all, dict) else {}
        # Extract Sharpe & MaxDD 
        kpis  = _extract_sharpe_and_mdd(r_t)

        # opt (single summary)
        o = state["opt_json"] or {}

        # Pred file path 
        pred_csv = os.path.join(P["predictions_dir"], f"{tslug}_test_predictions.csv")
        decision = _derive_decision_for_ticker(tslug, o, state["risk_json"], state["desc_json"], pred_csv)

        # assemble
        per = {
            "ticker": tslug,
            "head": "close",

            # --- Prediction Accuracy ---
            "evaluation": {
                "R2_price": e.get("R2(price)"),
                "RMSE_price": e.get("RMSE_price"),
                "MAE_price": e.get("MAE_price"),
                "MAPE_pct": e.get("MAPE"),
                "DirAcc_pct": e.get("DirAcc(%)") or e.get("DirAcc"),
                "Rows": e.get("Rows"),
                "Test_Start": e.get("Test_Start"),
                "Test_End": e.get("Test_End"),
            },

            # --- Statistical Insights  ---
            "statistical": {
                "Sharpe_ann": kpis.get("sharpe_ann"),      # annualised Sharpe 
                "Max_Drawdown": kpis.get("max_drawdown"),  # max drawdown 
            },

            # --- Risk Assessment ---
            "risk": {
                "label": _safe_get(r_t, "risk", "Risk_Label"),
                "snr_latest": _safe_get(r_t, "risk", "SNR"),
                "crosses_zero_latest": _safe_get(r_t, "risk", "Crosses_Zero"),
            },

            # --- Optimization / Decision ---
            "optimization": {
                "decision": _safe_get(state["opt_json"], "latest", "go_nogo"),
                "weight": _safe_get(state["opt_json"], "latest", "weight"),
                "kpis": state["opt_json"].get("kpis", {})
            },

            # XAI summary 
            "xai": {"ig_top": None, "shap_top": None, "notes": "Populate if you persist a combined XAI summary."},

            # Final decision 
            "decision": decision,

            # Collect plots 
            "plots": _plots_for_ticker(P, tslug)
        }

        out["per_ticker"][tslug] = per
        if out["as_of"] is None:
           
            out["as_of"] = e.get("Test_End")

    return {**state, "payload": out, "status": "summarized"}

def node_save(state: Dict[str, Any]) -> Dict[str, Any]:
    P = state["paths"]
    _ensure_dir(P["out_dir"])
    with open(P["out_payload_json"], "w") as f:
        json.dump(state["payload"], f, indent=2)

    assets = {t: state["payload"]["per_ticker"][t]["plots"] for t in state["payload"]["per_ticker"]}
    with open(P["out_assets_json"], "w") as f:
        json.dump(assets, f, indent=2)

    print(f"Summarizer wrote:\n  - {P['out_payload_json']}\n  - {P['out_assets_json']}")
    return {**state, "status": "saved", "out_payload": P["out_payload_json"], "out_assets": P["out_assets_json"]}

# ------------------ Build workflow ------------------
def build_summarizer_workflow():
    g = StateGraph(dict)
    g.add_node("collect", node_collect)
    g.add_node("summarize", node_summarize)
    g.add_node("save", node_save)
    g.set_entry_point("collect")
    g.add_edge("collect", "summarize")
    g.add_edge("summarize", "save")
    g.set_finish_point("save")
    return g.compile()

# ------------------ Standalone run ------------------
if __name__ == "__main__":
    print(" Running Agent Summarizer…")
    app = build_summarizer_workflow()
    _ = app.invoke({
        "run_base": "/content/drive/MyDrive/A2A_prediction_system/RUN_XXXXXXXX_XXXXXX",
        "tickers": ["AAPL"],
    })

