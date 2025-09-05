# Explainability Agent — runs IG first, then SHAP

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib

import tensorflow as tf
from tensorflow.keras.models import load_model

# LangGraph 
try:
    from langgraph.graph import StateGraph
    HAS_LG = True
except Exception:
    HAS_LG = False

# SHAP 
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

np.random.seed(42)
tf.random.set_seed(42)

# ----------------------------- path helpers -----------------------------
def _infer_paths(state):
    """
    Returns (models_dir, fe_path, explain_root, ig_root, shap_root)
    """
    run_base = state.get("run_base")
    if run_base:
        models_dir   = state.get("models_dir", os.path.join(run_base, "Predictive_Model", "lstm_models"))
        fe_path      = state.get("fe_path",   os.path.join(run_base, "FE_Agent", "features_engineered.csv"))
        explain_root = state.get("explain_root", os.path.join(run_base, "Explainability"))
    else:
        models_dir   = state.get("models_dir", "Predictive_Model/lstm_models")
        fe_path      = state.get("fe_path",   "FE_Agent/features_engineered.csv")
        explain_root = state.get("explain_root", "Explainability")

    ig_root   = os.path.join(explain_root, "IG_XAI")
    shap_root = os.path.join(explain_root, "SHAP_XAI")
    return models_dir, fe_path, explain_root, ig_root, shap_root

def _atomic_write_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

# ======================================================================
# =============================== IG ===============================
# ======================================================================

def build_improved_features(ticker_df: pd.DataFrame):
    df = ticker_df.copy()
    roll50 = df["Close"].rolling(50)
    df["Price_Norm"] = (df["Close"] - roll50.mean()) / roll50.std()
    df["High_Low_Ratio"]  = df["High"] / df["Low"].replace(0, np.nan)
    df["Open_Close_Ratio"] = df["Open"] / df["Close"].replace(0, np.nan)
    for w in [5, 10, 20]:
        ma = df["Close"].rolling(w).mean()
        df[f"MA_{w}"] = ma
        df[f"Close_MA{w}_Ratio"] = df["Close"] / ma.replace(0, np.nan)
        df[f"MA{w}_Slope"] = ma.diff(5) / ma.shift(5).replace(0, np.nan)
    for p in [1, 3, 5]:
        prev = df["Close"].shift(p).replace(0, np.nan)
        df[f"Log_Return_{p}d"] = np.log(df["Close"] / prev)
        df[f"Return_Volatility_{p}d"] = df[f"Log_Return_{p}d"].rolling(10).std()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_Norm"] = (df["RSI"] - 50) / 50
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
    df["MACD_Norm"] = df["MACD"] / df["Close"].replace(0, np.nan)
    vol_ma = df["Volume"].rolling(20).mean().replace(0, np.nan)
    df["Volume_MA"] = vol_ma
    df["Volume_Ratio"] = df["Volume"] / vol_ma
    df["Volume_Price_Correlation"] = df["Volume"].rolling(20).corr(df["Close"])
    df["Price_Volatility"] = df["Log_Return_1d"].rolling(20).std()
    df["High_Low_Volatility"] = np.log((df["High"] / df["Low"]).replace(0, np.nan)).rolling(10).mean()
    roll_min = df["Close"].rolling(20).min()
    roll_max = df["Close"].rolling(20).max()
    den = (roll_max - roll_min).replace(0, np.nan)
    df["Price_Position"] = (df["Close"] - roll_min) / den
    df["Momentum_5"] = df["Close"] / df["Close"].shift(5).replace(0, np.nan)
    df["Momentum_10"] = df["Close"] / df["Close"].shift(10).replace(0, np.nan)

    feature_columns = [
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
    avail = [c for c in feature_columns if c in df.columns]
    df[avail] = df[avail].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df, avail

def prepare_sequences_ig(features: np.ndarray, seq_len: int) -> np.ndarray:
    if len(features) <= seq_len:
        return np.empty((0, seq_len, features.shape[1]))
    return np.array([features[i - seq_len:i] for i in range(seq_len, len(features))])

def integrated_gradients_for_head(model, inputs, head_idx: int, steps: int = 50):
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    baseline = tf.zeros_like(inputs)
    output_dim = int(model.output_shape[-1]) if hasattr(model, "output_shape") else 1

    @tf.function
    def f_head(x):
        y = model(x, training=False)
        if output_dim == 1:
            return y
        return y[:, head_idx:head_idx+1]

    alphas = tf.reshape(tf.linspace(0.0, 1.0, steps + 1), [-1, 1, 1, 1])
    xb = tf.expand_dims(inputs, 0)
    x0 = tf.expand_dims(baseline, 0)
    path_inputs = x0 + alphas * (xb - x0)

    with tf.GradientTape() as tape:
        tape.watch(path_inputs)
        flat = tf.reshape(path_inputs, [-1, inputs.shape[1], inputs.shape[2]])
        preds = f_head(flat)
        s = tf.reduce_sum(preds)
    grads = tape.gradient(s, path_inputs)
    avg_grads = tf.reduce_mean(grads, axis=0)
    ig = (inputs - baseline) * avg_grads
    return ig.numpy()

def ig_completeness_residual(model, x_batch, ig_vals, head_idx=1):
    x = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    x0 = tf.zeros_like(x)
    def f_head(z):
        y = model(z, training=False)
        return y[:, head_idx]
    delta = f_head(x) - f_head(x0)
    ig_sum = tf.reduce_sum(ig_vals, axis=[1,2])
    resid = delta - ig_sum
    return float(tf.reduce_mean(tf.abs(resid)).numpy()), resid.numpy()

def ig_discover(state):
    models_dir = state.get("models_dir", "Predictive_Model/lstm_models")
    data_path  = state.get("data_path",  "FE_Agent/features_engineered.csv")
    out_dir    = state.get("out_dir",    "IG_XAI")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Engineered features not found: {data_path}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump({
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "matplotlib_version": matplotlib.__version__,
        }, f, indent=2)

    tickers, artifacts = [], {}
    for fname in os.listdir(models_dir):
        if fname.endswith("_twohead.keras") or fname.endswith("_lstm.keras"):
            t = fname.replace("_twohead.keras", "").replace("_lstm.keras", "")
            tickers.append(t)
    tickers = sorted(set(tickers))
    for t in tickers:
        two = os.path.join(models_dir, f"{t}_twohead.keras")
        lstm= os.path.join(models_dir, f"{t}_lstm.keras")
        artifacts[t] = {
            "model_path": two if os.path.exists(two) else lstm,
            "feature_scaler_path": os.path.join(models_dir, f"{t}_feature_scaler.pkl"),
            "feature_names_path":  os.path.join(models_dir, f"{t}_feature_names.json"),
            "seq_len_path":        os.path.join(models_dir, f"{t}_seq_len.txt"),
        }
    return {**state, "tickers": tickers, "artifacts": artifacts}

def ig_prepare(state):
    df = pd.read_csv(state["data_path"])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    per_ticker = {}
    for t in state["tickers"]:
        tdf = df[df["Symbol"] == t].copy().sort_values("Date").reset_index(drop=True)
        if tdf.empty:
            continue
        fe_df, rebuilt_feats = build_improved_features(tdf)
        feats_path = state["artifacts"][t]["feature_names_path"]
        if os.path.exists(feats_path):
            with open(feats_path, "r") as f:
                trained_feats = json.load(f)
            feats = [c for c in trained_feats if c in fe_df.columns]
        else:
            feats = rebuilt_feats
        fe_df = fe_df.dropna(subset=feats).copy()
        per_ticker[t] = {"full_df": fe_df, "feature_columns": feats}
    return {**state, "per_ticker": per_ticker}

def ig_compute(state):
    out_dir  = state.get("out_dir", "IG_XAI")
    steps    = int(state.get("ig_steps", 50))
    samples  = int(state.get("ig_samples", 80))
    head     = state.get("ig_head", "close")
    head_idx = 0 if str(head).lower() == "open" else 1

    results = {}
    for t, bundle in state.get("per_ticker", {}).items():
        try:
            art = state["artifacts"][t]
            model = load_model(art["model_path"])
            scaler = joblib.load(art["feature_scaler_path"])

            seq_len = 20
            if os.path.exists(art["seq_len_path"]):
                with open(art["seq_len_path"], "r") as f:
                    seq_len = int(f.read().strip())

            features = bundle["feature_columns"]
            fe_df = bundle["full_df"].copy()
            X_all = fe_df[features].values
            N = len(X_all)
            if N < seq_len + 10:
                print(f"[IG] {t}: not enough rows, skipping")
                continue
            trn = int(0.70 * N); val = int(0.15 * N)
            X_test = X_all[trn + val:]
            X_test_s = scaler.transform(pd.DataFrame(X_test, columns=features))
            X_seq = prepare_sequences_ig(X_test_s, seq_len)
            if len(X_seq) == 0:
                print(f"[IG] {t}: no test sequences, skipping")
                continue
            K = min(samples, len(X_seq))
            X_exp = X_seq[-K:]
            ig_vals = integrated_gradients_for_head(model, X_exp, head_idx=head_idx, steps=steps)
            res_mean, resid = ig_completeness_residual(model, X_exp, ig_vals, head_idx=head_idx)

            mean_abs_ig = np.mean(np.abs(ig_vals), axis=(0,1))
            imp = pd.DataFrame({"feature": features, "mean_abs_ig": mean_abs_ig}).sort_values("mean_abs_ig", ascending=False)

            tdir = os.path.join(out_dir, t); os.makedirs(tdir, exist_ok=True)
            np.savetxt(os.path.join(tdir, f"ig_completeness_residual_{head}.txt"), resid, fmt="%.6e")
            imp.to_csv(os.path.join(tdir, f"ig_feature_importance_{head}.csv"), index=False)

            avg_time_feat = np.mean(np.abs(ig_vals), axis=0)
            vmax = np.percentile(avg_time_feat, 99) if np.isfinite(avg_time_feat).all() else None
            plt.figure(figsize=(10, 4.8))
            plt.imshow(avg_time_feat.T, aspect='auto', interpolation='nearest', vmin=0.0, vmax=(vmax if (vmax and vmax>0) else None))
            plt.colorbar(label="mean |IG|")
            plt.yticks(ticks=np.arange(len(features)), labels=features)
            plt.xlabel("Time step (old → new)"); plt.ylabel("Feature")
            plt.title(f"{t} — IG Heatmap ({head} head)")
            plt.tight_layout(); plt.savefig(os.path.join(tdir, f"ig_heatmap_{head}.png"), dpi=300); plt.close()

            topN = min(20, len(features))
            plt.figure(figsize=(8, max(4, int(topN*0.35))))
            plt.barh(imp["feature"].head(topN)[::-1], imp["mean_abs_ig"].head(topN)[::-1])
            plt.title(f"{t} — Global Feature Importance (mean |IG|, {head} head)")
            plt.tight_layout(); plt.savefig(os.path.join(tdir, f"ig_global_importance_{head}.png"), dpi=300); plt.close()

            results[t] = {
                "head": head, "seq_len": int(seq_len), "rows_used": int(K),
                "ig_steps": int(steps), "ig_completeness_mean_abs_residual": float(res_mean),
                "top_features": imp.head(10).to_dict(orient="records"),
                "outputs_dir": tdir
            }
            print(f"[IG] {t}: saved → {tdir}")
        except Exception as e:
            print(f"[IG] {t}: failed — {e}")

    
    summary_path = os.path.join(out_dir, "ig_xai_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"historical": results, "forecast": {}}, f, indent=2)
    print(f" Saved IG summary → {summary_path}")
    return {**state, "ig_results": results}

# ======================================================================
# ============================== SHAP ==============================
# ======================================================================

def build_features_like_trainer(ticker_df: pd.DataFrame):
    df = ticker_df.copy()
    roll50 = df["Close"].rolling(50)
    df["Price_Norm"] = (df["Close"] - roll50.mean()) / roll50.std()
    df["High_Low_Ratio"] = df["High"] / df["Low"].replace(0, np.nan)
    df["Open_Close_Ratio"] = df["Open"] / df["Close"].replace(0, np.nan)
    for w in [5, 10, 20]:
        ma = df["Close"].rolling(w).mean()
        df[f"MA_{w}"] = ma
        df[f"Close_MA{w}_Ratio"] = df["Close"] / ma.replace(0, np.nan)
        df[f"MA{w}_Slope"] = ma.diff(5) / ma.shift(5).replace(0, np.nan)
    for p in [1, 3, 5]:
        prev = df["Close"].shift(p).replace(0, np.nan)
        df[f"Log_Return_{p}d"] = np.log(df["Close"] / prev)
        df[f"Return_Volatility_{p}d"] = df[f"Log_Return_{p}d"].rolling(10).std()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_Norm"] = (df["RSI"] - 50) / 50
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
    df["MACD_Norm"] = df["MACD"] / df["Close"].replace(0, np.nan)
    vol_ma = df["Volume"].rolling(20).mean().replace(0, np.nan)
    df["Volume_MA"] = vol_ma
    df["Volume_Ratio"] = df["Volume"] / vol_ma
    df["Volume_Price_Correlation"] = df["Volume"].rolling(20).corr(df["Close"])
    df["Price_Volatility"] = df["Log_Return_1d"].rolling(20).std()
    df["High_Low_Volatility"] = np.log((df["High"] / df["Low"]).replace(0, np.nan)).rolling(10).mean()
    roll_min = df["Close"].rolling(20).min()
    roll_max = df["Close"].rolling(20).max()
    den = (roll_max - roll_min).replace(0, np.nan)
    df["Price_Position"] = (df["Close"] - roll_min) / den
    df["Momentum_5"] = df["Close"] / df["Close"].shift(5).replace(0, np.nan)
    df["Momentum_10"] = df["Close"] / df["Close"].shift(10).replace(0, np.nan)

    feature_columns = [
        "Price_Norm", "High_Low_Ratio", "Open_Close_Ratio",
        "Close_MA5_Ratio", "Close_MA10_Ratio", "Close_MA20_Ratio",
        "MA5_Slope", "MA10_Slope", "MA20_Slope",
        "Log_Return_1d", "Log_Return_3d", "Log_Return_5d",
        "Return_Volatility_1d", "Return_Volatility_3d",
        "RSI_Norm", "MACD_Norm", "MACD_Histogram",
        "Volume_Ratio", "Volume_Price_Correlation",
        "Price_Volatility", "High_Low_Volatility",
        "Price_Position", "Momentum_5", "Momentum_10",
    ]
    avail = [c for c in feature_columns if c in df.columns]
    df[avail] = df[avail].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df, avail

def prepare_sequences_shap(features: np.ndarray, seq_len: int) -> np.ndarray:
    return np.array([features[i - seq_len:i] for i in range(seq_len, len(features))])

def shap_discover(state):
    models_dir = state.get("models_dir", "Predictive_Model/lstm_models")
    if not os.path.exists(models_dir):
        print(f" Models dir not found: {models_dir}")
        return {**state, "tickers": [], "artifacts": {}}

    tickers, artifacts = [], {}
    for fname in os.listdir(models_dir):
        if fname.endswith("_twohead.keras"):
            tickers.append(fname.split("_twohead.keras")[0])
        elif fname.endswith("_lstm.keras"):
            tickers.append(fname.split("_lstm.keras")[0])
    tickers = sorted(set(tickers))
    for t in tickers:
        model_path = os.path.join(models_dir, f"{t}_twohead.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(models_dir, f"{t}_lstm.keras")
        artifacts[t] = {
            "model_path": model_path,
            "feature_scaler_path": os.path.join(models_dir, f"{t}_feature_scaler.pkl"),
            "feature_names_path":   os.path.join(models_dir, f"{t}_feature_names.json"),
            "seq_len_path":         os.path.join(models_dir, f"{t}_seq_len.txt"),
        }
    return {**state, "tickers": tickers, "artifacts": artifacts}

def shap_prepare(state):
    data_path = state.get("data_path", "FE_Agent/features_engineered.csv")
    if not os.path.exists(data_path):
        print(f" Engineered data not found: {data_path}")
        return {**state, "per_ticker": {}}

    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    per_ticker = {}
    for t in state.get("tickers", []):
        tdf = df[df["Symbol"] == t].copy().sort_values("Date").reset_index(drop=True)
        if len(tdf) < 200:
            print(f"   {t}: too few rows; skipping")
            continue
        fe_df, built_features = build_features_like_trainer(tdf)
        art = state["artifacts"][t]
        features = built_features
        fpath = art["feature_names_path"]
        if os.path.exists(fpath):
            try:
                with open(fpath, "r") as f:
                    saved_feats = json.load(f)
                features = [c for c in saved_feats if c in fe_df.columns]
            except Exception:
                pass
        fe_df = fe_df.dropna(subset=features + ["Date"]).copy()
        per_ticker[t] = {"full_df": fe_df, "feature_columns": features}
    return {**state, "per_ticker": per_ticker}

def shap_compute(state):
    if not HAS_SHAP:
        print("[SHAP] shap not installed; skipping SHAP")
        return {**state, "results": {}}

    out_dir = state.get("out_dir", "SHAP_XAI")
    os.makedirs(out_dir, exist_ok=True)

    head_name = str(state.get("head", "close")).lower()
    head_idx = 0 if head_name == "open" else 1
    k_last = int(state.get("k_last", 120))
    bg_cap = int(state.get("bg_cap", 100))

    results = {}
    for t, bundle in state.get("per_ticker", {}).items():
        try:
            art = state["artifacts"][t]
            model = load_model(art["model_path"])
            feat_scaler = joblib.load(art["feature_scaler_path"])

            seq_len = 20
            if os.path.exists(art["seq_len_path"]):
                try:
                    with open(art["seq_len_path"], "r") as f:
                        seq_len = int(f.read().strip())
                except Exception:
                    pass

            features = bundle["feature_columns"]
            fe_df = bundle["full_df"].copy()
            X_all = fe_df[features].values
            dates_all = fe_df["Date"].values

            total = len(X_all)
            trn = int(total * 0.70)
            val = int(total * 0.15)
            X_train = X_all[:trn]
            X_val   = X_all[trn: trn + val]
            X_test  = X_all[trn + val:]
            dates_test = dates_all[trn + val:]

            X_train_s = feat_scaler.transform(pd.DataFrame(X_train, columns=features))
            X_val_s   = feat_scaler.transform(pd.DataFrame(X_val,   columns=features))
            X_test_s  = feat_scaler.transform(pd.DataFrame(X_test,  columns=features))

            def _seq(x): return prepare_sequences_shap(x, seq_len)
            X_train_seq = _seq(X_train_s)
            X_val_seq   = _seq(X_val_s)
            X_test_seq  = _seq(X_test_s)
            if len(X_test_seq) < 5 or len(X_train_seq) < 20:
                print(f"    {t}: not enough sequences; skipping")
                continue

            B = min(bg_cap, len(X_train_seq))
            bg_idx = np.random.RandomState(42).choice(len(X_train_seq), size=B, replace=False)
            background = X_train_seq[bg_idx]

            k = min(k_last, len(X_test_seq))
            X_shap = X_test_seq[-k:]

            shap_vals = None
            try:
                explainer = shap.DeepExplainer(model, background)
                sv = explainer.shap_values(X_shap)
                shap_vals = sv[head_idx] if isinstance(sv, list) else sv
            except Exception as e:
                print(f"    {t}: DeepExplainer failed ({e}). Trying GradientExplainer…")
                try:
                    explainer = shap.GradientExplainer(model, background)
                    sv = explainer.shap_values(X_shap)
                    shap_vals = sv[head_idx] if isinstance(sv, list) else sv
                except Exception as e2:
                    print(f"    {t}: GradientExplainer failed ({e2}). Using KernelExplainer (slow)…")
                    X_bg_flat   = background.reshape((background.shape[0], -1))
                    X_shap_flat = X_shap.reshape((X_shap.shape[0], -1))
                    def f(z):
                        z3 = z.reshape((-1, seq_len, len(features)))
                        preds = model.predict(z3, verbose=0)
                        if preds.shape[-1] == 2:
                            return preds[:, head_idx]
                        return preds.reshape((-1,))
                    explainer = shap.KernelExplainer(f, X_bg_flat[:min(50, len(X_bg_flat))])
                    sv = explainer.shap_values(X_shap_flat, nsamples=200)
                    shap_vals = np.array(sv).reshape((X_shap.shape[0], seq_len, len(features)))

            try:
                exp_val_raw = explainer.expected_value
                base_value = float(exp_val_raw[head_idx]) if isinstance(exp_val_raw, (list, tuple, np.ndarray)) else float(exp_val_raw)
            except Exception:
                preds_bg = model.predict(background, verbose=0)
                base_value = float(np.mean(preds_bg[:, head_idx])) if preds_bg.ndim == 2 else float(np.mean(preds_bg))

            mean_abs_shap = np.mean(np.abs(shap_vals), axis=(0, 1))
            feat_importance = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)

            ticker_dir = os.path.join(out_dir, t); os.makedirs(ticker_dir, exist_ok=True)
            suffix = head_name.lower()

            # SHAP files:
            feat_importance.to_csv(os.path.join(ticker_dir, f"{t}_feature_importance_{suffix}.csv"), index=False)

            topN = min(20, len(features))
            plt.figure(figsize=(8, max(4, int(topN * 0.35))))
            plt.barh(feat_importance["feature"].head(topN)[::-1],
                     feat_importance["mean_abs_shap"].head(topN)[::-1])
            plt.title(f"{t} — Global Feature Importance (mean |SHAP|, head='{suffix}')")
            plt.tight_layout()
            plt.savefig(os.path.join(ticker_dir, f"{t}_global_importance_{suffix}.png"), dpi=300)
            plt.close()

            heat = np.mean(np.abs(shap_vals), axis=0)
            vmax = np.percentile(heat, 99) if np.isfinite(heat).all() else None
            plt.figure(figsize=(12, 5))
            plt.imshow(heat.T, aspect='auto', interpolation='nearest',
                       vmin=0.0, vmax=vmax if (vmax is not None and vmax > 0) else None)
            plt.colorbar(label="mean |SHAP|")
            plt.yticks(ticks=np.arange(len(features)), labels=features)
            plt.xlabel("Time step (old → new)"); plt.ylabel("Feature")
            plt.title(f"{t} — SHAP Heatmap (head='{suffix}')")
            plt.tight_layout()
            plt.savefig(os.path.join(ticker_dir, f"{t}_heatmap_{suffix}.png"), dpi=300)
            plt.close()

            # plots 
            X_last_step = X_shap[:, -1, :]
            shap_last_step = shap_vals[:, -1, :]

            try:
                shap.summary_plot(shap_last_step, X_last_step,
                                  feature_names=features, show=False,
                                  plot_type="dot", max_display=20)
                plt.title(f"{t} — SHAP Summary (last step, head='{suffix}')")
                plt.tight_layout()
                plt.savefig(os.path.join(ticker_dir, f"{t}_summary_beeswarm_{suffix}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"    {t}: summary_plot failed ({e}) — continuing.")

            try:
                idx = -1
                fv = X_last_step[idx]
                sv = shap_last_step[idx]
                order = np.argsort(-np.abs(sv))
                vals = shap.Explanation(
                    values=sv[order],
                    base_values=np.array([base_value]),
                    data=fv[order],
                    feature_names=[features[i] for i in order]
                )
                shap.plots.waterfall(vals, max_display=20, show=False)
                plt.title(f"{t} — Local Waterfall (last sample, head='{suffix}')")
                plt.tight_layout()
                plt.savefig(os.path.join(ticker_dir, f"{t}_local_waterfall_last_{suffix}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"    {t}: local waterfall failed ({e}) — continuing.")

            try:
                force = shap.force_plot(base_value,
                                        shap_last_step[-1],
                                        X_last_step[-1],
                                        feature_names=features,
                                        matplotlib=False)
                shap.save_html(os.path.join(ticker_dir, f"{t}_force_last_{suffix}.html"), force)
            except Exception as e:
                print(f"    {t}: force_plot (HTML) failed ({e})")
            try:
                plt.figure(figsize=(10, 2.8))
                shap.force_plot(base_value,
                                shap_last_step[-1],
                                X_last_step[-1],
                                feature_names=features,
                                matplotlib=True,
                                show=False)
                plt.title(f"{t} — SHAP Force (last sample, head='{suffix}')")
                plt.tight_layout()
                plt.savefig(os.path.join(ticker_dir, f"{t}_force_last_{suffix}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"    {t}: force_plot (PNG) failed ({e})")

            try:
                n_overlay = min(50, shap_last_step.shape[0])
                plt.figure(figsize=(10, 5))
                shap.decision_plot(base_value,
                                   shap_last_step[-n_overlay:],
                                   feature_names=features,
                                   show=False)
                plt.title(f"{t} — SHAP Decision Plot (last {n_overlay} samples, head='{suffix}')")
                plt.tight_layout()
                plt.savefig(os.path.join(ticker_dir, f"{t}_decision_overlay_{suffix}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"    {t}: decision_plot (overlay) failed ({e})")
            try:
                plt.figure(figsize=(10, 4))
                shap.decision_plot(base_value,
                                   shap_last_step[-1],
                                   feature_names=features,
                                   show=False)
                plt.title(f"{t} — SHAP Decision Plot (last sample, head='{suffix}')")
                plt.tight_layout()
                plt.savefig(os.path.join(ticker_dir, f"{t}_decision_last_{suffix}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"    {t}: decision_plot (single) failed ({e})")

            results[t] = {
                "head": head_name, "rows_used": int(len(X_shap)),
                "seq_len": int(seq_len), "features": features,
                "top_features": feat_importance.head(10).to_dict(orient="records"),
                "outputs_dir": ticker_dir
            }
            print(f"    {t}: SHAP completed (head='{suffix}')")

        except Exception as e:
            print(f"   {t}: SHAP failed — {e}")

    # summary file
    summary_path = os.path.join(out_dir, "shap_xai_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f" Saved summary → {summary_path}")
    return {**state, "results": results}

# ======================================================================
# =========================== COMBINED RUNNER ===========================
# ======================================================================

def run_explainability(state: dict) -> dict:
    """
    Runs IG first (saving under Explainability/IG_XAI), then SHAP
    (Explainability/SHAP_XAI). Artifacts & filenames saved under this folder.
    """
    models_dir, fe_path, explain_root, ig_root, shap_root = _infer_paths(state)
    os.makedirs(explain_root, exist_ok=True)

    # -------- IG phase --------
    ig_state = {
        "models_dir": models_dir,
        "data_path":  fe_path,
        "out_dir":    ig_root,
        "ig_steps":   int(state.get("ig_steps", 50)),
        "ig_samples": int(state.get("ig_samples", 80)),
        "ig_head":    state.get("ig_head", "close"),
    }
    ig_state = ig_discover(ig_state)
    ig_state = ig_prepare(ig_state)
    ig_state = ig_compute(ig_state)

    # -------- SHAP phase --------
    shap_state = {
        "models_dir": models_dir,
        "data_path":  fe_path,
        "out_dir":    shap_root,
        "head":       state.get("shap_head", state.get("ig_head", "close")),
        "k_last":     int(state.get("k_last", 120)),
        "bg_cap":     int(state.get("bg_cap", 100)),
    }
    shap_state = shap_discover(shap_state)
    shap_state = shap_prepare(shap_state)
    shap_state = shap_compute(shap_state)

    # Combined summary
    summary_path = os.path.join(explain_root, "explainability_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "IG": ig_state.get("ig_results", {}),
            "SHAP": shap_state.get("results", {}),
            "roots": {"IG_XAI": ig_root, "SHAP_XAI": shap_root}
        }, f, indent=2)
    print(f"[Explainability] Combined summary → {summary_path}")

    return {
        "explain_root": explain_root,
        "ig_root": ig_root,
        "shap_root": shap_root,
        "ig_summary": os.path.join(ig_root, "ig_xai_summary.json"),
        "shap_summary": os.path.join(shap_root, "shap_xai_summary.json"),
        "combined_summary": summary_path,
    }

# LangGraph wrapper (runs IG → SHAP)
def build_explainability_workflow():
    if not HAS_LG:
        raise RuntimeError("LangGraph not available. Install with: pip install langgraph")
    
    def _runner(state):
        return {**state, **run_explainability(state)}
    g = StateGraph(dict)
    g.add_node("run", _runner)
    g.set_entry_point("run")
    g.set_finish_point("run")
    return g.compile()

if __name__ == "__main__":
    print(" Running Explainability Agent (IG → SHAP)…")
    _ = run_explainability({
        # "run_base": "/content/drive/MyDrive/A2A_prediction_system/RUN_YYYYMMDD_HHMMSS",
        "models_dir": "Predictive_Model/lstm_models",
        "fe_path":    "FE_Agent/features_engineered.csv",
        "ig_head": "close", "ig_steps": 50, "ig_samples": 80,
        "shap_head": "close", "k_last": 120, "bg_cap": 100,
    })
