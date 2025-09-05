import os
from datetime import datetime, timedelta
from typing import TypedDict, Optional, List

import pandas as pd
import yfinance as yf


try:
    from langgraph.graph import StateGraph
    HAS_LG = True
except Exception:
    HAS_LG = False


class DataCollectorState(TypedDict, total=False):
    run_base: Optional[str]
    tickers: Optional[List[str]]
    date_start: Optional[str]
    date_end: Optional[str]
    data_path: Optional[str]
    status: Optional[str]
    error: Optional[str]


def _drive_base() -> str:
    
    drive_root = "/content/drive/MyDrive/A2A_prediction_system"
    local_root = "/content/A2A_prediction_system"
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(drive_root, exist_ok=True)
        return drive_root
    os.makedirs(local_root, exist_ok=True)
    return local_root


def _compute_run_base(state: DataCollectorState) -> str:

    if state.get("run_base"):
        return state["run_base"]  

    env_rb = os.environ.get("RUN_BASE")
    if env_rb:
        os.makedirs(env_rb, exist_ok=True)
        return env_rb

    run_id = pd.Timestamp.now().strftime("RUN_%Y%m%d_%H%M%S")
    return os.path.join(_drive_base(), run_id)



def _as_date(s: Optional[str], fallback: Optional[datetime]) -> datetime:
    if s:
        try:
            return pd.to_datetime(s).to_pydatetime()
        except Exception:
            pass
    return fallback or datetime.today()


def _normalize_downloaded_df(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame with columns: Date, Open, Close, High, Low, Volume
    Works for:
      - standard columns (Open, High, Low, Close, Volume)
      - MultiIndex flattened to either Open_AAPL or AAPL_Open
    """
    if df is None or df.empty:
        return None

    df = df.reset_index()
    base_cols = {"Open", "High", "Low", "Close", "Volume"}

    
    if not isinstance(df.columns, pd.MultiIndex):
        if base_cols.issubset(set(df.columns)):
            return df[["Date", "Open", "Close", "High", "Low", "Volume"]].copy()
     
        return None

    # MultiIndex case: flatten to strings
    flat_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            flat_cols.append("_".join([c for c in col if c]).strip())
        else:
            flat_cols.append(str(col))
    df.columns = flat_cols

    fields = ["Open", "Close", "High", "Low", "Volume"]
    # Accept BOTH orders
    needed_field_first = ["Date"] + [f"{f}_{ticker}" for f in fields]   # Open_AAPL
    needed_ticker_first = ["Date"] + [f"{ticker}_{f}" for f in fields]  # AAPL_Open

    if all(c in df.columns for c in needed_field_first):
        out = df[needed_field_first].copy()
        out.columns = ["Date", "Open", "Close", "High", "Low", "Volume"]
        return out
    if all(c in df.columns for c in needed_ticker_first):
        tmp = df[needed_ticker_first].copy()
        rename_map = {f"{ticker}_Open": "Open",
                      f"{ticker}_Close": "Close",
                      f"{ticker}_High": "High",
                      f"{ticker}_Low": "Low",
                      f"{ticker}_Volume": "Volume"}
        tmp = tmp.rename(columns=rename_map)
        return tmp[["Date", "Open", "Close", "High", "Low", "Volume"]].copy()

   
    return None


def fetch_5yr_stock_data(state: DataCollectorState) -> DataCollectorState:
    print("[DataCollector] incoming run_base:", state.get("run_base"), "| env RUN_BASE:", os.environ.get("RUN_BASE"))

    try:
        # Resolve config (with safe fallbacks)
        tickers = state.get("tickers") or ["AAPL"]
        end_date = _as_date(state.get("date_end"), datetime.today())
        start_date = _as_date(state.get("date_start"), end_date - timedelta(days=5 * 365))
        run_base = _compute_run_base(state)

        output_dir = os.path.join(run_base, "data")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "yfinance_raw_data.csv")

        print(f"\n[DataCollector] Tickers: {tickers}")
        print(f"[DataCollector] Window : {start_date.date()} → {end_date.date()}")
        print(f"[DataCollector] Output : {output_path}")

        # Download per ticker
        all_data = []
        success_count = 0

        for ticker in tickers:
            print(f"\nDownloading: {ticker}")
            try:
                # Use field-first layout 
                df = yf.download(
                    tickers=ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    group_by="column",   
                    auto_adjust=False,
                    actions=False,
                    rounding=False,
                    threads=True,
                )

                df_clean = _normalize_downloaded_df(df, ticker)
                if df_clean is None or df_clean.empty:
                    
                    df2 = yf.download(
                        tickers=ticker,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=False,
                        actions=False,
                        rounding=False,
                        threads=True,
                    )
                    df_clean = _normalize_downloaded_df(df2, ticker)

                if df_clean is None or df_clean.empty:
                    print(f"  {ticker}: Could not normalize columns.")
                    continue

                # Clean types
                df_clean["Symbol"] = ticker
                df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
                for col in ["Open", "Close", "High", "Low", "Volume"]:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
                df_clean.dropna(subset=["Date", "Open", "Close", "High", "Low", "Volume"], inplace=True)

                if df_clean.empty:
                    print(f"  {ticker}: No valid rows after cleaning.")
                    continue

                print(f"  {ticker}: {len(df_clean)} rows ready.")
                all_data.append(df_clean)
                success_count += 1

            except Exception as e:
                print(f"  Error fetching {ticker}: {e}")
                continue

        if not all_data:
            return {**state, "run_base": run_base, "status": "Failed", "error": "No valid stock data collected."}

        # Concatenate & save
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.sort_values(["Symbol", "Date"], inplace=True)
        final_df.to_csv(output_path, index=False)

        print(f"\n[DataCollector] Saved: {output_path}")
        print(f"[DataCollector] Records: {len(final_df)} from {success_count} ticker(s)")
        print(f"[DataCollector] Columns: {list(final_df.columns)}")

        return {
            **state,
            "run_base": run_base,
            "data_path": output_path,
            "status": f"Success: {success_count} ticker(s) collected",
        }

    except Exception as e:
        return {**state, "status": "Error", "error": str(e)}


def build_data_collector_workflow():
    if not HAS_LG:
        raise RuntimeError("LangGraph not available. Install with: pip install langgraph")
    graph = StateGraph(dict)
    graph.add_node("FetchData", fetch_5yr_stock_data)
    graph.set_entry_point("FetchData")
    graph.set_finish_point("FetchData")
    return graph.compile()


if __name__ == "__main__":
    print("Starting 5-Year Stock Data Collection Agent...\n")
    demo_state: DataCollectorState = {
        "tickers": ["AAPL"],
        "date_start": (datetime.today() - timedelta(days=5*365)).date().isoformat(),
        "date_end": datetime.today().date().isoformat(),
        
    }
    try:
        if HAS_LG:
            app = build_data_collector_workflow()
            result = app.invoke(demo_state)
        else:
            result = fetch_5yr_stock_data(demo_state)

        print("\nPipeline Complete")
        print(f" Status: {result.get('status')}")
        if result.get("error"):
            print(f" Error: {result['error']}")
        else:
            print(f" Data saved to: {result['data_path']}")
    except Exception as e:
        print("Fatal error:", e)



