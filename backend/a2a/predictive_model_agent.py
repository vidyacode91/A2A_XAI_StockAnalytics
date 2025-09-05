import os, json, warnings, shutil
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# LangGraph 
try:
    from langgraph.graph import StateGraph
    HAS_LG = True
except Exception:
    HAS_LG = False

np.random.seed(42)
tf.random.set_seed(42)


# ============================== Utilities ==============================
def _infer_run_base(state: Dict[str, Any]) -> str:
    if state.get("run_base"):
        return str(state["run_base"])
    fe_path = state.get("fe_path")
    if fe_path:
        # path../<run_base>/FE_Agent/features_engineered.csv
        return os.path.dirname(os.path.dirname(fe_path))
    
    return "."

def _atomic_write_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ============================== Trainer ==============================
class TwoHeadTrainer:
    def __init__(self, seq_len: int, run_base: str):
        self.sequence_length = int(seq_len)
        self.RUN_BASE = run_base
        self.ROOT = os.path.join(run_base, "Predictive_Model")
        self.DIR_MODELS = os.path.join(self.ROOT, "lstm_models")
        self.DIR_PRED   = os.path.join(self.ROOT, "predictions")
        self.DIR_FCAST  = os.path.join(self.ROOT, "advanced_forecasts")
        self.DIR_PLOTS  = os.path.join(self.ROOT, "evaluation_plots")
        self.RESULTS_CSV= os.path.join(self.ROOT, "lstm_results.csv")
        for d in [self.ROOT, self.DIR_MODELS, self.DIR_PRED, self.DIR_FCAST, self.DIR_PLOTS]:
            os.makedirs(d, exist_ok=True)

        self.models: Dict[str, tf.keras.Model] = {}
        self.scalers: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Any] = {}

    # -------------------------- helpers --------------------------
    def _next_business_day(self, d):
        d = pd.Timestamp(d)
        nxt = d + pd.Timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += pd.Timedelta(days=1)
        return nxt

    # ----------------------- feature builder ----------------------
    def build_features(self, df: pd.DataFrame):
        data = df.copy()

        roll50 = data["Close"].rolling(50)
        data["Price_Norm"] = (data["Close"] - roll50.mean()) / roll50.std()
        data["High_Low_Ratio"]  = data["High"] / data["Low"].replace(0, np.nan)
        data["Open_Close_Ratio"] = data["Open"] / data["Close"].replace(0, np.nan)

        for w in [5, 10, 20]:
            ma = data["Close"].rolling(w).mean()
            data[f"MA_{w}"] = ma
            data[f"Close_MA{w}_Ratio"] = data["Close"] / ma.replace(0, np.nan)
            data[f"MA{w}_Slope"] = ma.diff(5) / ma.shift(5).replace(0, np.nan)

        for p in [1, 3, 5]:
            prev = data["Close"].shift(p).replace(0, np.nan)
            data[f"Log_Return_{p}d"] = np.log(data["Close"] / prev)
            data[f"Return_Volatility_{p}d"] = data[f"Log_Return_{p}d"].rolling(10).std()

        delta = data["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))
        data["RSI_Norm"] = (data["RSI"] - 50) / 50

        exp1 = data["Close"].ewm(span=12, adjust=False).mean()
        exp2 = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = exp1 - exp2
        data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
        data["MACD_Histogram"] = data["MACD"] - data["MACD_Signal"]
        data["MACD_Norm"] = data["MACD"] / data["Close"].replace(0, np.nan)

        vol_ma = data["Volume"].rolling(20).mean().replace(0, np.nan)
        data["Volume_MA"] = vol_ma
        data["Volume_Ratio"] = data["Volume"] / vol_ma
        data["Volume_Price_Correlation"] = data["Volume"].rolling(20).corr(data["Close"])

        data["Price_Volatility"] = data["Log_Return_1d"].rolling(20).std()
        data["High_Low_Volatility"] = np.log((data["High"] / data["Low"]).replace(0, np.nan)).rolling(10).mean()

        roll_min = data["Close"].rolling(20).min()
        roll_max = data["Close"].rolling(20).max()
        den = (roll_max - roll_min).replace(0, np.nan)
        data["Price_Position"] = (data["Close"] - roll_min) / den
        data["Momentum_5"] = data["Close"] / data["Close"].shift(5).replace(0, np.nan)
        data["Momentum_10"] = data["Close"] / data["Close"].shift(10).replace(0, np.nan)

        feature_cols = [
            "Price_Norm","High_Low_Ratio","Open_Close_Ratio",
            "Close_MA5_Ratio","Close_MA10_Ratio","Close_MA20_Ratio",
            "MA5_Slope","MA10_Slope","MA20_Slope",
            "Log_Return_1d","Log_Return_3d","Log_Return_5d",
            "Return_Volatility_1d","Return_Volatility_3d",
            "RSI_Norm","MACD_Norm","MACD_Histogram",
            "Volume_Ratio","Volume_Price_Correlation",
            "Price_Volatility","High_Low_Volatility",
            "Price_Position","Momentum_5","Momentum_10",
        ]
        avail = [c for c in feature_cols if c in data.columns]
        data[avail] = (data[avail].replace([np.inf, -np.inf], np.nan)
                                 .fillna(method="ffill")
                                 .fillna(method="bfill"))
        return data, avail

    def _prepare_sequences(self, X: np.ndarray, Y: np.ndarray, seq: int):
        xs, ys = [], []
        for i in range(seq, len(X)):
            xs.append(X[i-seq:i])
            ys.append(Y[i])
        return np.array(xs), np.array(ys)

    def _build_model(self, input_shape, *,
                     arch="lstm_gru", layers=2,
                     units1=64, units2=32, units3=16,
                     dropout=0.2, batch_norm=True,
                     dense_units=16, lr=1e-3, optimizer="adam"):
        m = Sequential()
        if arch == "lstm_gru":
            m.add(LSTM(units1, return_sequences=True, input_shape=input_shape))
            if batch_norm: m.add(BatchNormalization())
            m.add(Dropout(dropout))
            m.add(GRU(units2, return_sequences=(layers > 2)))
            if batch_norm: m.add(BatchNormalization())
            m.add(Dropout(dropout))
            if layers >= 3:
                m.add(GRU(units3))
                if batch_norm: m.add(BatchNormalization())
                m.add(Dropout(dropout))
        else:
            m.add(GRU(units1, return_sequences=(layers > 1), input_shape=input_shape))
            if batch_norm: m.add(BatchNormalization())
            m.add(Dropout(dropout))
            if layers >= 2:
                m.add(LSTM(units2, return_sequences=(layers > 2)))
                if batch_norm: m.add(BatchNormalization())
                m.add(Dropout(dropout))
            if layers >= 3:
                m.add(LSTM(units3))
                if batch_norm: m.add(BatchNormalization())
                m.add(Dropout(dropout))
        if dense_units > 0:
            m.add(Dense(dense_units, activation="relu"))
            m.add(Dropout(dropout))
        m.add(Dense(2, activation="linear"))
        opt = Adam(learning_rate=lr) if optimizer == "adam" else RMSprop(learning_rate=lr)
        m.compile(optimizer=opt, loss="mse", metrics=["mae"])
        return m

    def _tiny_search(self, Xtr, Ytr, Xva, Yva, input_shape):
        grid = [
            dict(arch="lstm_gru", layers=2, units1=64, units2=32, dropout=0.2, batch_norm=True,  dense_units=16, lr=1e-3, optimizer="adam"),
            dict(arch="lstm_gru", layers=2, units1=64, units2=32, dropout=0.3, batch_norm=True,  dense_units=16, lr=1e-3, optimizer="adam"),
            dict(arch="lstm_gru", layers=3, units1=64, units2=32, dropout=0.2, batch_norm=True,  dense_units=16, lr=5e-4, optimizer="adam"),
        ]
        best = {"val_loss": np.inf, "model": None, "config": None, "history": None}
        for i, cfg in enumerate(grid, 1):
            m = self._build_model(input_shape, **cfg)
            cbs = [
                EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
            ]
            h = m.fit(Xtr, Ytr, validation_data=(Xva, Yva),
                      epochs=60, batch_size=min(64, max(8, len(Xtr)//4)),
                      verbose=0, callbacks=cbs)
            v = float(np.min(h.history["val_loss"]))
            print(f"   tried {i}/{len(grid)} -> val_loss={v:.6f}")
            if v < best["val_loss"]:
                best = {"val_loss": v, "model": m, "config": cfg, "history": h}
        print(f"   best config: {best['config']} (val_loss={best['val_loss']:.6f})")
        return best

    # -------------------------- train one ticker --------------------------
    def train_model(self, ticker: str, fe_path: str):
        print(f"\n Training model for {ticker}...")
        try:
            allfe = pd.read_csv(fe_path)
            tdf = allfe[allfe["Symbol"] == ticker].copy().sort_values("Date").reset_index(drop=True)
            if len(tdf) < 80:
                print(f"   Not enough rows for {ticker}")
                return None

            fe, feats = self.build_features(tdf)
            prev_close = fe["Close"].shift(1)
            fe["Open_Ret"]  = fe["Open"]/prev_close - 1.0
            fe["Close_Ret"] = fe["Close"]/prev_close - 1.0

            cols_needed = feats + ["Open_Ret", "Close_Ret", "Date", "Open", "Close"]
            fe = fe.dropna(subset=cols_needed).reset_index(drop=True)

            dates = pd.to_datetime(fe["Date"]).values
            X = fe[feats].values
            Y = fe[["Open_Ret","Close_Ret"]].values

            N = len(X); trn = int(0.70*N); val = int(0.15*N)
            Xtr, Xva, Xte = X[:trn], X[trn:trn+val], X[trn+val:]
            Ytr, Yva, Yte = Y[:trn], Y[trn:trn+val], Y[trn+val:]
            dates_te = dates[trn+val:]

            fs = StandardScaler()
            Xtr_s = fs.fit_transform(pd.DataFrame(Xtr, columns=feats))
            Xva_s = fs.transform(pd.DataFrame(Xva, columns=feats))
            Xte_s = fs.transform(pd.DataFrame(Xte, columns=feats))

            ys = StandardScaler()
            Ytr_s = ys.fit_transform(Ytr)
            Yva_s = ys.transform(Yva)
            Yte_s = ys.transform(Yte)

            seq = self.sequence_length
            Xtr_seq, Ytr_seq = self._prepare_sequences(Xtr_s, Ytr_s, seq)
            Xva_seq, Yva_seq = self._prepare_sequences(Xva_s, Yva_s, seq)
            Xte_seq, Yte_seq = self._prepare_sequences(Xte_s, Yte_s, seq)
            print(f"   Sequences - Train: {len(Xtr_seq)}, Val: {len(Xva_seq)}, Test: {len(Xte_seq)}")

            input_shape = (seq, len(feats))
            best = self._tiny_search(Xtr_seq, Ytr_seq, Xva_seq, Yva_seq, input_shape)
            model = best["model"]; hist = best["history"]

            tr_pred = ys.inverse_transform(model.predict(Xtr_seq, verbose=0))
            va_pred = ys.inverse_transform(model.predict(Xva_seq, verbose=0))
            te_pred = ys.inverse_transform(model.predict(Xte_seq, verbose=0))

            tr_true = Ytr[seq:]
            va_true = Yva[seq:]
            te_true = Yte[seq:]

            def head_metrics(y_true, y_pred, head=0):
                if len(y_true) < 2:
                    return 0.0, float("nan"), float("nan")
                r2  = r2_score(y_true[:, head], y_pred[:, head])
                rmse= float(np.sqrt(mean_squared_error(y_true[:, head], y_pred[:, head])))
                dir = float(np.mean((y_true[:, head] > 0) == (y_pred[:, head] > 0))*100.0)
                return r2, rmse, dir

            m_tr_open = head_metrics(tr_true, tr_pred, 0)
            m_tr_close= head_metrics(tr_true, tr_pred, 1)
            m_va_open = head_metrics(va_true, va_pred, 0)
            m_va_close= head_metrics(va_true, va_pred, 1)
            m_te_open = head_metrics(te_true, te_pred, 0)
            m_te_close= head_metrics(te_true, te_pred, 1)

            print("   Results (Open/Close heads):")
            print(f"      Train Open : R²={m_tr_open[0]:.4f} RMSE={m_tr_open[1]:.4f} DirAcc={m_tr_open[2]:.1f}%")
            print(f"      Train Close: R²={m_tr_close[0]:.4f} RMSE={m_tr_close[1]:.4f} DirAcc={m_tr_close[2]:.1f}%")
            print(f"      Val   Open : R²={m_va_open[0]:.4f} RMSE={m_va_open[1]:.4f} DirAcc={m_va_open[2]:.1f}%")
            print(f"      Val   Close: R²={m_va_close[0]:.4f} RMSE={m_va_close[1]:.4f} DirAcc={m_va_close[2]:.1f}%")
            print(f"      Test  Open : R²={m_te_open[0]:.4f} RMSE={m_te_open[1]:.4f} DirAcc={m_te_open[2]:.1f}%")
            print(f"      Test  Close: R²={m_te_close[0]:.4f} RMSE={m_te_close[1]:.4f} DirAcc={m_te_close[2]:.1f}%")

            # Align for price reconstruction
            L = len(te_pred)
            base_close = fe["Close"].iloc[trn+val+seq-1 : trn+val+seq-1 + (L+1)].values
            L = min(L, len(base_close)-1)
            te_pred = te_pred[:L]; te_true = te_true[:L]
            dates_te_adj = pd.to_datetime(dates_te[seq: seq+L])

            pred_close_price = base_close[:-1] * (1.0 + te_pred[:, 1])
            actual_close_price = base_close[1:]
            pred_open_price = base_close[:-1] * (1.0 + te_pred[:, 0])
            actual_open_price = fe["Open"].iloc[trn+val+seq : trn+val+seq+L].values

            out_pred = pd.DataFrame({
                "Date": dates_te_adj.astype("datetime64[ns]").astype(str),
                "Actual_Open":  actual_open_price,
                "Pred_Open":    pred_open_price,
                "Actual_Close": actual_close_price,
                "Pred_Close":   pred_close_price,
                "Actual_Return":  te_true[:,1],
                "Pred_Return":    te_pred[:,1]
            })
            _atomic_write_df(out_pred, os.path.join(self.DIR_PRED, f"{ticker}_test_predictions.csv"))

            self.models[ticker] = model
            self.scalers[ticker] = {"feature_scaler": fs, "target_scaler": ys, "features": feats}
            self.results[ticker] = {
                "train_open": m_tr_open, "train_close": m_tr_close,
                "val_open": m_va_open, "val_close": m_va_close,
                "test_open": m_te_open, "test_close": m_te_close,
                "best_config": best["config"], "history": hist.history,
            }

            # Save generated outputs
            model.save(os.path.join(self.DIR_MODELS, f"{ticker}_twohead.keras"))
            joblib.dump(fs, os.path.join(self.DIR_MODELS, f"{ticker}_feature_scaler.pkl"))
            joblib.dump(ys, os.path.join(self.DIR_MODELS, f"{ticker}_target_scaler.pkl"))
            with open(os.path.join(self.DIR_MODELS, f"{ticker}_feature_names.json"), "w") as f:
                json.dump(feats, f)
            with open(os.path.join(self.DIR_MODELS, f"{ticker}_seq_len.txt"), "w") as f:
                f.write(str(self.sequence_length))
            with open(os.path.join(self.DIR_MODELS, f"{ticker}_best_config.json"), "w") as f:
                json.dump(best["config"], f, indent=2)

            # Plot
            plt.figure(figsize=(12,5))
            plt.plot(dates_te_adj, actual_close_price, label="Actual Close")
            plt.plot(dates_te_adj, pred_close_price, label="Pred Close")
            plt.title(f"{ticker} — Test Close: Actual vs Predicted")
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(self.DIR_PLOTS, f"{ticker}_test_close.png"), dpi=220)
            plt.close()

            # Tomorrow-only prediction
            fe["Date"] = pd.to_datetime(fe["Date"])
            latest_date = fe["Date"].iloc[-1]
            latest_close = float(fe["Close"].iloc[-1])
            latest_open  = float(fe["Open"].iloc[-1])

            last_win = fe[feats].tail(self.sequence_length).values
            last_win_s = fs.transform(last_win).reshape(1, self.sequence_length, len(feats))
            tom_pred = ys.inverse_transform(model.predict(last_win_s, verbose=0))[0]
            pred_open_ret, pred_close_ret = float(tom_pred[0]), float(tom_pred[1])

            tom_pred_open  = latest_close * (1.0 + pred_open_ret)
            tom_pred_close = latest_close * (1.0 + pred_close_ret)

            for_date = self._next_business_day(latest_date)
            tomorrow_df = pd.DataFrame([{
                "For_Date": for_date.strftime("%Y-%m-%d"),
                "Latest_Actual_Open":  latest_open,
                "Latest_Actual_Close": latest_close,
                "Pred_Open_Return":  pred_open_ret,
                "Pred_Close_Return": pred_close_ret,
                "Pred_Open":  tom_pred_open,
                "Pred_Close": tom_pred_close
            }])
            _atomic_write_df(tomorrow_df, os.path.join(self.DIR_PRED, f"{ticker}_tomorrow.csv"))
            print(f"   Tomorrow-only prediction saved → {os.path.join(self.DIR_PRED, f'{ticker}_tomorrow.csv')}")

            # XAI compatibility alias 
            src = os.path.join(self.DIR_MODELS, f"{ticker}_twohead.keras")
            dst = os.path.join(self.DIR_MODELS, f"{ticker}_lstm.keras")
            try:
                shutil.copyfile(src, dst)
            except Exception:
                pass

            return self.results[ticker]

        except Exception as e:
            print(f"    Error training {ticker}: {e}")
            import traceback; traceback.print_exc()
            return None

    def forecast_recursive(self, ticker: str, fe_path: str, days: int = 7):
        if ticker not in self.models:
            print("    No trained model found. Train first.")
            return None

        allfe = pd.read_csv(fe_path)
        df = allfe[allfe["Symbol"] == ticker].copy().sort_values("Date").reset_index(drop=True)
        fe, feats = self.build_features(df)
        fe = fe.dropna(subset=feats + ["Date", "Close"]).reset_index(drop=True)
        fe["Date"] = pd.to_datetime(fe["Date"])
        model = self.models[ticker]
        fs = self.scalers[ticker]["feature_scaler"]
        ys = self.scalers[ticker]["target_scaler"]

        seq = fs.transform(fe[feats].tail(self.sequence_length).values)

        try:
            lr1_idx = feats.index("Log_Return_1d")
        except ValueError:
            lr1_idx = None

        current_close = float(fe["Close"].iloc[-1])
        start_date = fe["Date"].iloc[-1]
        dates, pred_open, pred_close, ret_open, ret_close = [], [], [], [], []

        d = self._next_business_day(start_date)
        for _ in range(days):
            x = seq.reshape(1, self.sequence_length, len(feats))
            pred = ys.inverse_transform(model.predict(x, verbose=0))[0]
            oret, cret = float(pred[0]), float(pred[1])

            next_open  = current_close * (1.0 + oret)
            next_close = current_close * (1.0 + cret)

            dates.append(d); pred_open.append(next_open); pred_close.append(next_close)
            ret_open.append(oret); ret_close.append(cret)

            new_row = seq[-1].copy()
            if lr1_idx is not None:
                mu = getattr(fs, "mean_", None); sc = getattr(fs, "scale_", None)
                raw_lr1 = np.log(1.0 + cret)
                if mu is not None and sc is not None and sc[lr1_idx] != 0:
                    new_row[lr1_idx] = (raw_lr1 - mu[lr1_idx]) / sc[lr1_idx]
            seq = np.vstack([seq[1:], new_row])
            current_close = next_close
            d = self._next_business_day(d)

        out = pd.DataFrame({
            "Date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "Pred_Open_Return":  ret_open,
            "Pred_Close_Return": ret_close,
            "Pred_Open": pred_open,
            "Pred_Close": pred_close
        })
        out_path = os.path.join(self.DIR_FCAST, f"{ticker}_forecast_{days}d.csv")
        _atomic_write_df(out, out_path)
        print(f"   {days}-day forecast saved → {out_path}")

        # Plot
        tailN = 60
        tail_dates = fe["Date"].tail(tailN).tolist()
        tail_prices= fe["Close"].tail(tailN).tolist()
        plt.figure(figsize=(12,5))
        plt.plot(tail_dates, tail_prices, label="Actual Close (last 60d)")
        plt.plot(pd.to_datetime(dates), pred_close, label=f"Forecast Close (+{days}d)")
        plt.title(f"{ticker} — Last 60d Close + {days}-Day Forecast")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(self.DIR_PLOTS, f"{ticker}_forecast_close_{days}d.png"), dpi=220)
        plt.close()
        return out

    def save_results_summary(self):
        if not self.results:
            print(" No results to report")
            return None
        rows = []
        for t, r in self.results.items():
            te_o, te_c = r["test_open"], r["test_close"]
            rows.append({
                "Ticker": t,
                "Test_R2_Open":  round(te_o[0], 4),
                "Test_RMSE_Open":round(te_o[1], 4),
                "Test_DirAcc_Open(%)": round(te_o[2], 1),
                "Test_R2_Close":  round(te_c[0], 4),
                "Test_RMSE_Close":round(te_c[1], 4),
                "Test_DirAcc_Close(%)": round(te_c[2], 1),
                "Chosen": "lstm_gru",
            })
        df = pd.DataFrame(rows)
        _atomic_write_df(df, self.RESULTS_CSV)
        print(f" Results summary saved → {self.RESULTS_CSV}")
        return df


# ============================== LangGraph nodes ==============================
def node_init(state: Dict[str, Any]) -> Dict[str, Any]:
    run_base = _infer_run_base(state)
    seq_len  = int(state.get("seq_len", 20))
    tickers  = state.get("tickers") or ["AAPL"]
    days     = int(state.get("forecast_days", 7))
    fe_path  = state.get("fe_path") or os.path.join(run_base, "FE_Agent", "features_engineered.csv")
    if not os.path.exists(fe_path):
        raise FileNotFoundError(f"FE file not found: {fe_path}")
    trainer = TwoHeadTrainer(seq_len=seq_len, run_base=run_base)
    return {**state, "run_base": run_base, "fe_path": fe_path, "trainer": trainer,
            "tickers": tickers, "forecast_days": days, "status": "init_ok"}

def node_train_all(state: Dict[str, Any]) -> Dict[str, Any]:
    tr, fe_path = state["trainer"], state["fe_path"]
    trained = []
    for t in state["tickers"]:
        res = tr.train_model(t, fe_path)
        if res is not None:
            trained.append(t)
    return {**state, "trained": trained, "status": "trained"}

def node_forecast_all(state: Dict[str, Any]) -> Dict[str, Any]:
    tr, fe_path = state["trainer"], state["fe_path"]
    days, tickers = state["forecast_days"], state.get("trained", [])
    for t in tickers:
        tr.forecast_recursive(t, fe_path, days=days)
    return {**state, "status": "forecast_done"}

def node_save_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    tr = state["trainer"]
    df = tr.save_results_summary()
    return {**state, "results_csv": getattr(tr, "RESULTS_CSV", None), "status": "complete"}

def build_predictive_workflow():
    g = StateGraph(dict)
    g.add_node("init", node_init)
    g.add_node("train", node_train_all)
    g.add_node("forecast", node_forecast_all)
    g.add_node("save", node_save_summary)
    g.set_entry_point("init")
    g.add_edge("init", "train")
    g.add_edge("train", "forecast")
    g.add_edge("forecast", "save")
    g.set_finish_point("save")
    return g.compile()


def run_predict(state: Dict[str, Any]) -> Dict[str, Any]:
    s = node_init(state)
    s = node_train_all(s)
    s = node_forecast_all(s)
    s = node_save_summary(s)
    
    tr: TwoHeadTrainer = s["trainer"]
    return {
        **state,
        "run_base": s["run_base"],
        "predict_root": tr.ROOT,
        "models_dir": tr.DIR_MODELS,
        "pred_dir": tr.DIR_PRED,
        "forecast_dir": tr.DIR_FCAST,
        "results_csv": tr.RESULTS_CSV,
        "trained": s.get("trained", []),
        "status": "predict_complete",
    }

if __name__ == "__main__":
    print(" Starting Predictive Agent…\n")
    app = build_predictive_workflow()
    _ = app.invoke({
        
        # "fe_path": "<run_base>/FE_Agent/features_engineered.csv",
        "seq_len": 20,
        "forecast_days": 7,
    })
    print("\n Done. Artifacts saved under <run_base>/Predictive_Model/")
