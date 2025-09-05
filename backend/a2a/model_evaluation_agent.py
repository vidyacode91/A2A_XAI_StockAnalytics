import os, json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil


# LangGraph
try:
    from langgraph.graph import StateGraph
    HAS_LG = True
except Exception:
    HAS_LG = False

# ----------------- IO utils -----------------
def _atomic_write_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def _infer_run_base(state: Dict[str, Any]) -> str:
    if state.get("run_base"):
        return str(state["run_base"])
    pred_dir = state.get("predictions_dir")
    if pred_dir:
        # expected path .../<run_base>/Predictive_Model/predictions
        return os.path.dirname(os.path.dirname(pred_dir))
    return "."

# ----------------- Normalizer -----------------
def normalize_pred_file(df: pd.DataFrame, head: str = "close") -> pd.DataFrame:
    """
    Standardize to columns: Date, Actual_Return, Pred_Return, Actual_Close, Pred_Close.
    """
    required_single = {"Actual_Return", "Pred_Return", "Actual_Close", "Pred_Close"}
    if required_single.issubset(df.columns):
        return df

    head = (head or "close").lower().strip()
    if head not in {"close", "open"}:
        head = "close"

    if head == "close":
        req = {"Actual_Return", "Pred_Return", "Actual_Close", "Pred_Close"}
        # Predictive file already uses these names for close
        mapped = {"Actual_Close_Return": "Actual_Return", "Pred_Close_Return": "Pred_Return"}
        if req.issubset(df.columns):
            return df
        if set(mapped.keys()).issubset(df.columns) and {"Actual_Close","Pred_Close"}.issubset(df.columns):
            df2 = df.copy()
            df2.rename(columns=mapped, inplace=True)
            return df2

    if head == "open":
        req = {"Actual_Open", "Pred_Open"}
        ret = {"Actual_Open_Return": "Actual_Return", "Pred_Open_Return": "Pred_Return"}
        if req.issubset(df.columns) and set(ret.keys()).issubset(df.columns):
            df2 = df.copy()
            df2.rename(columns=ret, inplace=True)
            # For generic naming, map open->close slots so metrics/plots reuse code
            df2["Actual_Close"] = df2["Actual_Open"]
            df2["Pred_Close"]   = df2["Pred_Open"]
            return df2

    raise ValueError(
        "Prediction file missing required columns. "
        "Expected either {Actual_Return, Pred_Return, Actual_Close, Pred_Close} "
        "or open-head equivalents."
    )

# ----------------- Metrics  -----------------
def _price_metrics(a_p: np.ndarray, p_p: np.ndarray) -> Dict[str, float]:
    if len(a_p) <= 1 or len(p_p) <= 1:
        return {"RMSE_price": np.nan, "MAE_price": np.nan, "MAPE": np.nan, "R2(price)": np.nan}
    mse  = float(np.mean((a_p - p_p) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(a_p - p_p)))
    r2   = float(r2_score(a_p, p_p))
    mape = float(pd.Series(a_p).pipe(lambda s: np.mean(np.abs((a_p - p_p) / np.where(s != 0, s, np.nan)))) * 100.0)
    return {"RMSE_price": rmse, "MAE_price": mae, "MAPE": mape, "R2(price)": r2}

from sklearn.metrics import r2_score, mean_squared_error

def eval_metrics(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {k: (np.nan) for k in
                ["RMSE_price","MAE_price","MAPE","R2(price)","RMSE_ret","R2(returns)","DirAcc(%)","Sharpe_ann",
                 "Rows","Test_Start","Test_End"]}

    if "Date" in df.columns:
        d = pd.to_datetime(df["Date"], errors="coerce")
        test_start, test_end = d.min(), d.max()
    else:
        test_start = test_end = None

    a_r = df["Actual_Return"].to_numpy()
    p_r = df["Pred_Return"].to_numpy()
    a_p = df["Actual_Close"].to_numpy()
    p_p = df["Pred_Close"].to_numpy()

    # price metrics
    price = _price_metrics(a_p, p_p)

    # returns metrics
    if len(a_r) > 1:
        rmse_ret = float(np.sqrt(mean_squared_error(a_r, p_r)))
        r2_ret   = float(r2_score(a_r, p_r))
        diracc   = float(np.mean(np.sign(a_r) == np.sign(p_r)) * 100.0)
        ex_ret   = (p_r - a_r)
        sharpe_d = float(np.mean(ex_ret) / np.std(ex_ret)) if np.std(ex_ret) > 0 else np.nan
        sharpe_a = float(sharpe_d * np.sqrt(252)) if not np.isnan(sharpe_d) else np.nan
    else:
        rmse_ret = r2_ret = diracc = sharpe_a = np.nan

    out = {
        **{k: (round(v, 6) if isinstance(v, float) and not np.isnan(v) else (v if not isinstance(v, float) else np.nan))
           for k, v in price.items()},
        "RMSE_ret": round(rmse_ret, 6) if not np.isnan(rmse_ret) else np.nan,
        "R2(returns)": round(r2_ret, 4) if not np.isnan(r2_ret) else np.nan,
        "DirAcc(%)": round(diracc, 1) if not np.isnan(diracc) else np.nan,
        "Sharpe_ann": round(sharpe_a, 3) if not np.isnan(sharpe_a) else np.nan,
        "Rows": int(len(df)),
        "Test_Start": "" if test_start is None else str(getattr(test_start, "date", lambda: "")()),
        "Test_End": "" if test_end is None else str(getattr(test_end, "date", lambda: "")()),
    }
    return out

# ----------------- Baselines -----------------
def eval_baselines(df: pd.DataFrame) -> Dict[str, float]:
    """
    Baseline A (returns): predict zero return.
    Baseline B (price): carry-forward yesterday's actual close.
    """
    out = {"BL_DirAcc(%)": np.nan, "BL_RMSE_ret": np.nan, "BL_RMSE_price": np.nan, "BL_R2(price)": np.nan}
    if df.empty:
        return out

    # A: returns baseline (0)
    a_r = df["Actual_Return"].to_numpy()
    p0_r = np.zeros_like(a_r)
    if len(a_r) > 1:
        out["BL_RMSE_ret"] = float(np.sqrt(mean_squared_error(a_r, p0_r)))
        out["BL_DirAcc(%)"] = float(np.mean(np.sign(a_r) == np.sign(p0_r)) * 100.0)

    # B: price baseline (carry-forward)
    a_p = df["Actual_Close"].to_numpy()
    if len(a_p) > 2:
        p0_p = np.r_[a_p[0], a_p[:-1]]  # yesterday's price
        pm = _price_metrics(a_p, p0_p)
        out["BL_RMSE_price"] = pm["RMSE_price"]
        out["BL_R2(price)"]  = pm["R2(price)"]

    return out

# ========================= LangGraph NODES =========================
def node_init(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs:
      run_base: str
      predictions_dir: str   (defaults to <run_base>/Predictive_Model/predictions)
      eval_root: str         (defaults to <run_base>/Model_Evaluation)
      eval_head: "close"|"open" (default "close")
      strong_pct: int        (default 75)  # percentile on |Pred_Return|
    """
    run_base = _infer_run_base(state)
    predictions_dir = state.get("predictions_dir") or os.path.join(run_base, "Predictive_Model", "predictions")
    eval_root = state.get("eval_root") or os.path.join(run_base, "Model_Evaluation")
    eval_head = (state.get("eval_head", "close") or "close").lower()
    strong_pct = int(state.get("strong_pct", 75))

    if not os.path.isdir(predictions_dir):
        raise FileNotFoundError(f"Predictions folder not found: {predictions_dir}")

    sum_dir = os.path.join(eval_root, "summaries")
    plot_dir = os.path.join(eval_root, "plots")
    os.makedirs(sum_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    paths = {
        "CSV_FULL":    os.path.join(sum_dir,  "model_evaluation_summary.csv"),
        "JSON_FULL":   os.path.join(sum_dir,  "model_evaluation_summary.json"),
        "CSV_STRONG":  os.path.join(sum_dir,  "model_evaluation_strong_signals.csv"),
        "JSON_STRONG": os.path.join(sum_dir,  "model_evaluation_strong_signals.json"),
        "DIRACC_PNG":  os.path.join(plot_dir, "diracc_full_vs_strong.png"),
        "SCATTER_PNG": lambda ticker: os.path.join(plot_dir, f"{ticker}_strong_signal_pred_vs_actual.png"),
    }

    return {
        **state,
        "run_base": run_base,
        "predictions_dir": predictions_dir,
        "eval_root": eval_root,
        "sum_dir": sum_dir,
        "plot_dir": plot_dir,
        "paths": paths,
        "eval_head": "open" if eval_head == "open" else "close",
        "strong_pct": strong_pct,
        "status": "init_ok",
    }

def node_discover_files(state: Dict[str, Any]) -> Dict[str, Any]:
    files = [fn for fn in sorted(os.listdir(state["predictions_dir"])) if fn.endswith("_test_predictions.csv")]
    tickers = [fn.split("_")[0] for fn in files]
    return {**state, "files": files, "tickers": tickers, "status": "files_ok"}

def node_evaluate(state: Dict[str, Any]) -> Dict[str, Any]:
    head = state["eval_head"]
    pred_dir = state["predictions_dir"]

    full_rows: List[dict] = []
    strong_rows: List[dict] = []

    print(f" Model evaluation (head='{head}') — price & return metrics …\n")
    for fname in state["files"]:
        ticker = fname.split("_")[0]
        path = os.path.join(pred_dir, fname)
        try:
            raw = pd.read_csv(path)
            df = normalize_pred_file(raw, head=head)

            need = {"Actual_Return", "Pred_Return", "Actual_Close", "Pred_Close"}
            if df.empty or not need.issubset(df.columns):
                print(f"  {ticker}: normalized file missing columns — skip.")
                continue

            # Full-sample metrics
            fm = eval_metrics(df)
            bl = eval_baselines(df)
            fm.update(bl)
            fm["Ticker"] = ticker
            fm["Head"] = head
            full_rows.append(fm)

            print(f" {ticker}: R2(price)={fm['R2(price)']} | RMSEp={fm['RMSE_price']} | DirAcc={fm['DirAcc(%)']}%  "
                  f"| BL DirAcc={fm.get('BL_DirAcc(%)')}%")

            # Strong-signal subset (by |Pred_Return|)
            try:
                thr = df["Pred_Return"].abs().quantile(state["strong_pct"]/100.0)
                strong = df.loc[df["Pred_Return"].abs() >= thr].copy()
                if not strong.empty:
                    sm = eval_metrics(strong)
                    sm["Ticker"] = ticker
                    sm["Head"] = head
                    sm["Subset"] = f"|Pred_Return|>={state['strong_pct']}pct"
                    strong_rows.append(sm)
            except Exception:
                pass

        except Exception as e:
            print(f"  {ticker}: error {e}")

    return {**state, "full_rows": full_rows, "strong_rows": strong_rows, "status": "evaluated"}

def node_save_tables(state: Dict[str, Any]) -> Dict[str, Any]:
    paths = state["paths"]
    full_rows = state.get("full_rows", [])
    strong_rows = state.get("strong_rows", [])

    df_full = pd.DataFrame(full_rows).sort_values(["Ticker", "Head"]) if full_rows else pd.DataFrame()
    df_str  = pd.DataFrame(strong_rows).sort_values(["Ticker", "Head"]) if strong_rows else pd.DataFrame()

    if not df_full.empty:
        _atomic_write_df(df_full, paths["CSV_FULL"])
        with open(paths["JSON_FULL"], "w") as f:
            json.dump(full_rows, f, indent=2)

    if not df_str.empty:
        _atomic_write_df(df_str, paths["CSV_STRONG"])
        with open(paths["JSON_STRONG"], "w") as f:
            json.dump(strong_rows, f, indent=2)

    return {**state, "df_full": df_full, "df_strong": df_str, "status": "tables_saved"}

def node_make_plots(state: Dict[str, Any]) -> Dict[str, Any]:
    df_full = state.get("df_full", pd.DataFrame())
    paths = state["paths"]
    head = state["eval_head"]
    pred_dir = state["predictions_dir"]

    # Per-ticker plots from prediction files
    files = [fn for fn in sorted(os.listdir(pred_dir)) if fn.endswith("_test_predictions.csv")]
    for fname in files:
        ticker = fname.split("_")[0]
        path = os.path.join(pred_dir, fname)
        try:
            raw = pd.read_csv(path)
            df = normalize_pred_file(raw, head=head)

            # Sort by date 
            if "Date" in df.columns:
                try:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date")
                    x = df["Date"]
                    x_label = "Date"
                except Exception:
                    x = np.arange(len(df)); x_label = "Index"
            else:
                x = np.arange(len(df)); x_label = "Index"

            # Line plot: Actual vs Predicted Price
            plt.figure(figsize=(12, 5))
            plt.plot(x, df["Actual_Close"], label="Actual Price", linewidth=1.5)
            plt.plot(x, df["Pred_Close"],   label="Predicted Price", linewidth=1.5)
            plt.title(f"{ticker} — {head.capitalize()} Price: Actual vs Predicted")
            plt.xlabel(x_label); plt.ylabel("Price"); plt.legend(); plt.grid(alpha=0.3)
            out_line = os.path.join(state["plot_dir"], f"{ticker}_{head}_price_line.png")
            plt.tight_layout(); plt.savefig(out_line, dpi=300); plt.close()

            # Scatter: Predicted vs Actual Price
            plt.figure(figsize=(6.5, 6))
            plt.scatter(df["Actual_Close"], df["Pred_Close"], alpha=0.6)
            lo = float(min(df["Actual_Close"].min(), df["Pred_Close"].min()))
            hi = float(max(df["Actual_Close"].max(), df["Pred_Close"].max()))
            plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
            plt.title(f"{ticker} — {head.capitalize()} Price: Predicted vs Actual")
            plt.xlabel("Actual Price"); plt.ylabel("Predicted Price"); plt.grid(alpha=0.3)
            out_scatter = os.path.join(state["plot_dir"], f"{ticker}_{head}_price_scatter.png")
            plt.tight_layout(); plt.savefig(out_scatter, dpi=300); plt.close()

            # Copy to the Predictive_Model/evaluation_plots folder 
            legacy_dir = os.path.join(state["run_base"], "Predictive_Model", "evaluation_plots")
            os.makedirs(legacy_dir, exist_ok=True)
            shutil.copyfile(out_line,    os.path.join(legacy_dir, f"{ticker}_{head}_price_line.png"))
            shutil.copyfile(out_scatter, os.path.join(legacy_dir, f"{ticker}_{head}_price_scatter.png"))

        except Exception as e:
            print(f"  {ticker}: plotting error: {e}")

    # Aggregate DirAcc bar 
    df_str = state.get("df_strong", pd.DataFrame())
    if not df_full.empty and not df_str.empty:
        tickers = sorted(set(df_full["Ticker"]).intersection(set(df_str["Ticker"])))
        if tickers:
            full_da = [float(df_full[df_full["Ticker"]==t]["DirAcc(%)"].values[0]) for t in tickers]
            str_da  = [float(df_str[df_str["Ticker"]==t]["DirAcc(%)"].values[0]) for t in tickers]
            x = np.arange(len(tickers))
            plt.figure(figsize=(10,5))
            plt.bar(x - 0.2, full_da, width=0.4, label="Full")
            plt.bar(x + 0.2, str_da,  width=0.4, label="Strong")
            plt.xticks(x, tickers)
            plt.ylabel("Direction Accuracy (%)")
            plt.title("DirAcc: Full vs Strong-signal subset")
            plt.legend(); plt.grid(axis="y", alpha=0.3)
            plt.tight_layout(); plt.savefig(paths["DIRACC_PNG"], dpi=300); plt.close()

    print("\nSaved plots in:", state["plot_dir"])
    return {**state, "status": "plots_saved"}

# ========================= Build + run LangGraph =========================
def build_evaluation_workflow():
    g = StateGraph(dict)
    g.add_node("init", node_init)
    g.add_node("discover", node_discover_files)
    g.add_node("evaluate", node_evaluate)
    g.add_node("tables", node_save_tables)
    g.add_node("plots", node_make_plots)

    g.set_entry_point("init")
    g.add_edge("init", "discover")
    g.add_edge("discover", "evaluate")
    g.add_edge("evaluate", "tables")
    g.add_edge("tables", "plots")
    g.set_finish_point("plots")
    return g.compile()

# -------- Simple function for orchestrator --------
def run_evaluate(state: Dict[str, Any]) -> Dict[str, Any]:
    s = node_init(state)
    s = node_discover_files(s)
    s = node_evaluate(s)
    s = node_save_tables(s)
    s = node_make_plots(s)
    return {
        **state,
        "run_base": s["run_base"],
        "eval_root": s["eval_root"],
        "eval_summary_csv": s["paths"]["CSV_FULL"],
        "eval_summary_json": s["paths"]["JSON_FULL"],
        "eval_strong_csv": s["paths"]["CSV_STRONG"],
        "eval_diracc_png": s["paths"]["DIRACC_PNG"],
        "status": "evaluation_complete",
    }

if __name__ == "__main__":
    print(" Running Evaluation Agent …")
    app = build_evaluation_workflow()
    _ = app.invoke({
        # Provide run_base or predictions_dir; defaults to <run_base>/Predictive_Model/predictions
        
        "eval_head": "close",   # or "open"
        "strong_pct": 75
    })
