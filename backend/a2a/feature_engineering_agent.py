import os, json
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Optional, List, Dict, Any

import numpy as np
import pandas as pd
import logging


try:
    from langgraph.graph import StateGraph
    HAS_LG = True
except Exception:
    HAS_LG = False

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------- State type --------------
class FeatureState(TypedDict, total=False):
    run_base: Optional[str]
    data_path: Optional[str]
    fe_path: Optional[str]
    fe_model_path: Optional[str]
    model_features: Optional[List[str]]
    status: Optional[str]
    error: Optional[str]
   
    data: Optional[pd.DataFrame]

# --------------- Helpers -----------------
def _infer_run_base(state: FeatureState) -> Path:
    
    if state.get("run_base"):
        return Path(str(state["run_base"]))
    dp = state.get("data_path")
    if dp:
        p = Path(dp)
        # expect .../<run_base>/data/filename.csv
        return p.parent.parent
    return Path(".")

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def winsorize_series(s: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    s = s.astype(float)
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)

# ============== Load data =============
def load_stock_data(state: FeatureState) -> FeatureState:
    logger.info("Loading stock data...")
    try:
        data_path = state.get("data_path")
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(f"Raw data file not found: {data_path}")

        df = pd.read_csv(data_path, parse_dates=["Date"])
        logger.info(f"Loaded: {data_path} shape={df.shape}")

        # Normalize column names 
        column_mapping = {
            'symbol': 'Symbol', 'ticker': 'Symbol', 'SYMBOL': 'Symbol',
            'date': 'Date', 'DATE': 'Date',
            'open': 'Open', 'OPEN': 'Open',
            'close': 'Close', 'CLOSE': 'Close',
            'high': 'High', 'HIGH': 'High',
            'low': 'Low', 'LOW': 'Low',
            'volume': 'Volume', 'VOLUME': 'Volume',
        }
        df.rename(columns=column_mapping, inplace=True)

        expected = ["Date", "Symbol", "Open", "Close", "High", "Low", "Volume"]
        miss = [c for c in expected if c not in df.columns]
        if miss:
            raise ValueError(f"Missing required columns: {miss}")

        df = df[expected].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for c in ["Open", "Close", "High", "Low", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Basic validation & cleaning
        n0 = len(df)
        df = df.dropna(subset=["Date", "Symbol", "Open", "Close", "High", "Low", "Volume"])
        df = df[df["Volume"] > 0]
        df = df[df["High"] >= df["Low"]]
        df = df[df["Close"] > 0]
        df = df.drop_duplicates(["Symbol", "Date"]).sort_values(["Symbol", "Date"]).reset_index(drop=True)
        logger.info(f"Validation: {n0} -> {len(df)} rows")

        return {**state, "data": df, "status": "Data loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return {**state, "status": "Failed", "error": f"Error loading data: {e}"}

# ============== Create features =======
def create_enhanced_features(state: FeatureState) -> FeatureState:
    logger.info("Creating features...")
    try:
        df = state["data"]
        syms = df["Symbol"].unique()
        logger.info(f"Symbols: {list(syms)}")

        processed: List[pd.DataFrame] = []

        for sym in syms:
            s = df[df["Symbol"] == sym].copy().sort_values("Date")
            if len(s) < 50:
                logger.warning(f"{sym}: too few rows ({len(s)}), skip")
                continue

            # Returns
            s["Return_1d"] = s["Close"].pct_change(1)

            # MAs
            s["MA_5"]  = s["Close"].rolling(5).mean()
            s["MA_20"] = s["Close"].rolling(20).mean()
            s["Close_MA5_Ratio"]  = s["Close"] / s["MA_5"].replace(0, np.nan)
            s["Close_MA20_Ratio"] = s["Close"] / s["MA_20"].replace(0, np.nan)
            s["MA5_MA20_Ratio"]   = s["MA_5"] / s["MA_20"].replace(0, np.nan)
            s["MA_5_Slope"]  = s["MA_5"].diff(5) / s["MA_5"].shift(5).replace(0, np.nan)
            s["MA_20_Slope"] = s["MA_20"].diff(5) / s["MA_20"].shift(5).replace(0, np.nan)

            # RSI
            delta = s["Close"].diff()
            gain = delta.clip(lower=0).rolling(14, min_periods=7).mean()
            loss = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean().replace(0, np.nan)
            rs = gain / loss
            s["RSI"] = 100 - (100 / (1 + rs))

            # MACD (histogram)
            exp1 = s["Close"].ewm(span=12, adjust=False).mean()
            exp2 = s["Close"].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            macd_sig = macd.ewm(span=9, adjust=False).mean()
            s["MACD_Histogram"] = macd - macd_sig

            # Bollinger position
            bb_ma = s["Close"].rolling(20).mean()
            bb_sd = s["Close"].rolling(20).std()
            bb_up = bb_ma + 2 * bb_sd
            bb_lo = bb_ma - 2 * bb_sd
            span = (bb_up - bb_lo).replace(0, np.nan)
            s["BB_Position"] = (s["Close"] - bb_lo) / span

            # Ratios / ranges
            denom_hl = (s["High"] - s["Low"]).replace(0, np.nan)
            s["High_Low_Ratio"] = (s["Close"] - s["Low"]) / denom_hl
            s["Daily_Range"] = (s["High"] - s["Low"]) / s["Close"].replace(0, np.nan)
            s["Open_Close_Ratio"] = s["Open"] / s["Close"].replace(0, np.nan)

            # Volatility & volume signal
            s["Volatility_20d"] = s["Return_1d"].rolling(20, min_periods=10).std()
            vol_ma10 = s["Volume"].rolling(10, min_periods=5).mean()
            s["Volume_Ratio"] = s["Volume"] / vol_ma10.replace(0, np.nan)

            # Winsorize (cap outliers)
            for col in ["Return_1d", "Volume_Ratio", "Daily_Range"]:
                if col in s.columns:
                    s[col] = winsorize_series(s[col])

            # Targets (analysis only)
            s["Next_Day_Return"] = s["Return_1d"].shift(-1)
            s["Next_Day_Direction"] = (s["Next_Day_Return"] > 0).astype(int)

            s = s.dropna(subset=["Open", "Close", "High", "Low", "Volume"])
            processed.append(s)

        if not processed:
            raise ValueError("No symbols processed successfully")

        final_df = pd.concat(processed, ignore_index=True)
        final_df = final_df.sort_values(["Symbol", "Date"]).replace([np.inf, -np.inf], np.nan)

        # Core features used for modeling (no targets)
        CORE_MODEL_FEATURES = [
            "Open","Close","High","Low","Volume",
            "Return_1d","Volatility_20d",
            "Close_MA5_Ratio","Close_MA20_Ratio","MA5_MA20_Ratio",
            "MA_5_Slope","MA_20_Slope",
            "RSI","MACD_Histogram","BB_Position",
            "High_Low_Ratio","Daily_Range","Open_Close_Ratio",
            "Volume_Ratio"
        ]
        available_core = [c for c in CORE_MODEL_FEATURES if c in final_df.columns]
        logger.info(f"Model features: {len(available_core)}/{len(CORE_MODEL_FEATURES)} available")

        return {**state, "data": final_df, "model_features": available_core,
                "status": "Features created successfully"}
    except Exception as e:
        logger.error(f"Error in create_enhanced_features: {e}")
        return {**state, "status": "Failed", "error": f"Error in create_features: {e}"}

# ============== Handle NaNs ===========
def handle_missing_after_fe(state: FeatureState) -> FeatureState:
    logger.info("Handling NaNs after FE...")
    try:
        df = state["data"].copy().sort_values(["Symbol","Date"])
        target_cols = {"Next_Day_Return", "Next_Day_Direction"}

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feat_cols = [c for c in num_cols if c not in target_cols]

        # ffill/bfill per symbol 
        df[feat_cols] = df.groupby("Symbol", group_keys=False)[feat_cols].transform(lambda g: g.ffill().bfill())

        # median per symbol then global
        per_sym_med = df.groupby("Symbol", group_keys=False)[feat_cols].transform("median")
        df[feat_cols] = df[feat_cols].fillna(per_sym_med)
        df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median(numeric_only=True))

        # drop rows with critical NA and tail NA 
        before = len(df)
        df = df.dropna(subset=["Close", "Return_1d"])
        dropped = before - len(df)
        if "Next_Day_Return" in df.columns:
            before2 = len(df)
            df = df.dropna(subset=["Next_Day_Return"])
            dropped += before2 - len(df)
        logger.info(f"Dropped rows due to NA: {dropped}")

        # report remaining NaNs
        rem = df.isna().sum()
        rem = rem[rem > 0]
        if len(rem):
            logger.warning("Remaining NaNs:\n" + "\n".join([f"  {k}: {v}" for k,v in rem.items()]))
        else:
            logger.info("All NaNs handled.")

        return {**state, "data": df, "status": "NaNs handled after FE"}
    except Exception as e:
        logger.error(f"Error in handle_missing_after_fe: {e}")
        return {**state, "status": "Failed", "error": f"Error in NaN handling: {e}"}

# ============== Validate/Summarize =====
def validate_and_summarize(state: FeatureState) -> FeatureState:
    logger.info("Validating & summarizing...")
    try:
        df = state["data"]
        logger.info(f"Rows: {len(df):,}")
        logger.info(f"Symbols: {df['Symbol'].nunique()}")
        logger.info(f"Date range: {df['Date'].min()} → {df['Date'].max()}")

        inf_cols = [c for c in df.select_dtypes(include=[np.number]).columns if np.isinf(df[c]).any()]
        if inf_cols:
            logger.warning(f"Infinite values in: {inf_cols}")

        const_cols = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].nunique(dropna=True) <= 1]
        if const_cols:
            logger.warning(f"Constant columns (consider drop): {const_cols}")

        return {**state, "status": "Features validated and summarized"}
    except Exception as e:
        logger.error(f"Error in validate_and_summarize: {e}")
        return {**state, "status": "Failed", "error": f"Error in validation: {e}"}

# ============== Save outputs ===========
def save_enhanced_features(state: FeatureState) -> FeatureState:
    logger.info("Saving features...")
    try:
        df = state["data"].copy()
        model_feats = state.get("model_features", []) or []

        run_base = _infer_run_base(state)
        out_dir  = run_base / "FE_Agent"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Full file
        full_path = out_dir / "features_engineered.csv"
        _atomic_write_csv(df, full_path)
        logger.info(f"Full features: {full_path}")

        # model file
        keep_cols = ["Date", "Symbol"] + model_feats
        slim = df[keep_cols].copy()
        slim_path = out_dir / "features_for_model.csv"
        _atomic_write_csv(slim, slim_path)
        logger.info(f"Model features: {slim_path}")

        # Artifacts near full_path
        base = full_path.with_suffix("")  
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        all_feats = [c for c in df.columns if c not in ["Date","Symbol"]]
        (base.with_name(base.name + "_feature_list.json")
         ).write_text(json.dumps({"generated_at": ts, "count": len(all_feats), "features": all_feats}, indent=2))

        (base.with_name(base.name + "_model_feature_list.json")
         ).write_text(json.dumps({"generated_at": ts, "count": len(model_feats), "features": model_feats}, indent=2))

        schema = {c: str(t) for c, t in df.dtypes.items()}
        (base.with_name(base.name + "_schema.json")
         ).write_text(json.dumps({"generated_at": ts, "schema": schema}, indent=2))

        df.describe().to_csv(base.with_name(base.name + "_summary.csv"), index=True)

        (base.with_name(base.name + "_VERSION.txt")
         ).write_text(
            f"features_version: 2.0\n"
            f"generated_at_utc: {ts}\n"
            f"rows: {len(df)}\n"
            f"columns_full: {len(df.columns)}\n"
            f"columns_for_model: {len(keep_cols)}\n"
        )

        # keep state small for next agent/orchestrator
        out_state = {
            **state,
            "fe_path": str(full_path),
            "fe_model_path": str(slim_path),
            "model_features": model_feats,
            "status": "Features saved successfully",
        }
        out_state.pop("data", None)
        return out_state

    except Exception as e:
        logger.error(f"Error saving features: {e}")
        return {**state, "status": "Failed", "error": f"Error saving features: {e}"}

# ======= Simple function for orchestrator =======
def run_fe(state: Dict[str, Any]) -> Dict[str, Any]:
    
    s = FeatureState(**state)  

    for step in (load_stock_data, create_enhanced_features, handle_missing_after_fe,
                 validate_and_summarize, save_enhanced_features):
        s = step(s)  
        if str(s.get("status","")).startswith("Failed"):
            return s
    return s

# ============ LangGraph ============
def build_enhanced_feature_workflow():
    if not HAS_LG:
        raise RuntimeError("LangGraph not available. Install with: pip install langgraph")
    graph = StateGraph(FeatureState)
    graph.add_node("load", load_stock_data)
    graph.add_node("create", create_enhanced_features)
    graph.add_node("nans", handle_missing_after_fe)
    graph.add_node("validate", validate_and_summarize)
    graph.add_node("save", save_enhanced_features)

    graph.set_entry_point("load")
    graph.add_edge("load", "create")
    graph.add_edge("create", "nans")
    graph.add_edge("nans", "validate")
    graph.add_edge("validate", "save")
    graph.set_finish_point("save")
    return graph.compile()

# =============== Demo run =================
if __name__ == "__main__":
    print("Starting Feature Engineering Agent...\n")
   
    example_state: FeatureState = {
        "data_path": "data/Yfinance_real_time_data.csv",  
        "run_base": ".",  
    }
    out = run_fe(example_state)
    print("\nStatus:", out.get("status"))
    if out.get("error"):
        print("Error:", out["error"])
    else:
        print("FE path:", out.get("fe_path"))
        print("Model features path:", out.get("fe_model_path"))
