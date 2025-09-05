
# backend/a2a/report_agent.py
import os, json, glob, datetime, base64
from typing import Dict, Any, List, Optional

# ---------------------------- utils ----------------------------
def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def _safe_get(d: dict, *path, default=None):
    cur = d
    for k in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, list) and isinstance(k, int):
            cur = cur[k] if 0 <= k < len(cur) else None
        else:
            return default
    return default if cur is None else cur

def _dash(msg: str | None = None, html: bool = True) -> str:
    if html:
        tip = (msg or "not computed").replace("'", "\\'")
        return f"<span title='{tip}'>—</span>"
    return "—"

def _fmt_str(x: Optional[str], html: bool = True) -> str:
    s = (x or "").strip()
    return s if s else _dash(html=html)

def _fmt_bool(x, html: bool = True) -> str:
    if x is True:  return "True"
    if x is False: return "False"
    return _dash("not available", html=html)

def _fmt_num(x, digits=3, money=False):
    try:
        v = float(x)
    except Exception:
        return "—"
    return (f"${v:.{digits}f}" if money else f"{v:.{digits}f}")

def _decision_badge(decision: Optional[str]) -> str:
    d = (decision or "").strip().upper()
    if d == "GO":    return "<b>GO</b>"
    if d == "NO-GO": return "<b>NO-GO</b>"
    return "—"

def _autodetect_run_base(run_base: str) -> str:
    if os.path.exists(os.path.join(run_base, "reports", "final_payload.json")):
        return run_base
    candidates = []
    for p in glob.glob(os.path.join("runs", "*", "reports", "final_payload.json")):
        try:
            mtime = os.path.getmtime(p)
            base = os.path.dirname(os.path.dirname(p))
            candidates.append((mtime, base))
        except Exception:
            pass
    if candidates:
        candidates.sort()
        return candidates[-1][1]
    return run_base

def _img_data_uri(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None

# ---------------------------- main render ----------------------------
def render_report(run_base: str = ".", out_name: str = "llm_report.md",
                  order: Optional[List[str]] = None) -> str:

    run_base = _autodetect_run_base(run_base)

    reports_dir = os.path.join(run_base, "reports")
    payload_path = os.path.join(reports_dir, "final_payload.json")
    assets_path  = os.path.join(reports_dir, "assets.json")

    if not os.path.exists(payload_path):
        raise FileNotFoundError(
            f"Missing {payload_path}. Run the Summarizer agent for the same run_base first."
        )

    payload = json.load(open(payload_path, "r"))
    assets  = json.load(open(assets_path, "r")) if os.path.exists(assets_path) else {}

    as_of = payload.get("as_of") or datetime.date.today().isoformat()
    tickers = order or payload.get("tickers") or list((_safe_get(payload, "per_ticker") or {}).keys())
    tickers = [t.upper() for t in tickers]

    # ---- UI labels 
    ui = payload.get("ui_labels") or {
        "prediction_accuracy": "Prediction Accuracy",
        "trend_prediction":    "Trend Prediction",
        "statistical_insights":"Statistical Insights",
        "risk_assessment":     "Risk Assessment",
        "model_explainability":"Model Explainability",
        "final_recommendation":"Final Recommendation",
    }

    # Helper to choose a plot path from candidates, with fuzzy/glob + path resolution
    def _get_plot(node_plots: Dict[str, str], asset_plots: Dict[str, str], key_candidates: List[str]) -> Optional[str]:
        def _resolve(p: Optional[str]) -> Optional[str]:
            if not p:
                return None
            for cand in (
                p,
                os.path.join(reports_dir, p),
                os.path.join(run_base, p),
            ):
                if os.path.exists(cand):
                    return cand
            return None

        # exact keys
        for k in key_candidates:
            p = (node_plots or {}).get(k) or (asset_plots or {}).get(k)
            rp = _resolve(p)
            if rp:
                return rp

        # fuzzy on dict keys
        def _match_any(d: Dict[str, str], must_tokens: List[str]) -> Optional[str]:
            for k, v in (d or {}).items():
                lab = (k or "").lower()
                if all(tok in lab for tok in must_tokens):
                    rp = _resolve(v)
                    if rp:
                        return rp
            return None

        toks_sets = []
        # infer token sets from candidates provided
        cand_l = " ".join(key_candidates).lower()
        if "ig" in cand_l:
            toks_sets += [["ig", "heatmap"], ["ig", "importance"]]
        if "shap" in cand_l:
            toks_sets += [["shap", "beeswarm"], ["shap", "importance"], ["shap", "global"]]

        for dsrc in (node_plots or {}), (asset_plots or {}):
            for toks in toks_sets:
                hit = _match_any(dsrc, toks)
                if hit:
                    return hit

        # glob search on disk for common patterns
        import glob as _glob
        def _glob_first(patterns: List[str]) -> Optional[str]:
            roots = [reports_dir, os.path.join(run_base, "reports"), run_base]
            for root in roots:
                for pat in patterns:
                    hits = _glob.glob(os.path.join(root, "**", pat), recursive=True)
                    if hits:
                        return hits[0]
            return None

        patterns = []
        if "ig" in cand_l:
            patterns += ["*ig*heatmap*.png", "*ig*importance*.png"]
        if "shap" in cand_l:
            patterns += ["*shap*beeswarm*.png", "*shap*importance*.png", "*global*importance*shap*.png"]
        
        patterns += [
            "ig_heatmap_close.png",
            "ig_global_importance_close.png",
            "*_summary_beeswarm_close.png",
            "*_local_waterfall_last_close.png",
            "*_global_importance_close.png",
        ]
        ghit = _glob_first(patterns)
        if ghit:
            return ghit

        return None

    make_html = out_name.lower().endswith(".html")
    blocks: List[str] = []

    # ---------- Header ----------
    if make_html:
        blocks.append("""<!doctype html>
<html><head><meta charset="utf-8">
<title>Stock Insights Report</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; line-height:1.5; padding:24px; max-width:960px; margin:auto;}
 h1,h2{margin:0.6em 0;}
 .muted{color:#666}
 .section{margin:24px 0;}
 img{max-width:100%; height:auto; border:1px solid #eee; border-radius:8px; padding:4px; background:#fff;}
 details{margin:8px 0 16px 0;}
 table{border-collapse:collapse; width:100%; font-size:14px;}
 th,td{border:1px solid #ddd; padding:6px 8px;}
 th{background:#fafafa; text-align:left;}
 .hr{border-top:1px solid #eee; margin:24px 0;}
</style>
</head><body>
""")
        blocks.append(f"<h1>Stock Insights Report</h1>")
        blocks.append(f"<p class='muted'>Report Date <b>{as_of}</b></p>")
        blocks.append("<p><i>Note: This report summarizes model predictions and risk metrics. It is informational and <b>not investment advice</b>.</i></p>")
    else:
        blocks.append("# Stock Insights Report\n")
        blocks.append(f"_As of **{as_of}**_\n")
        blocks.append("> **Note:** This report summarizes model predictions and risk metrics. It is informational and **not investment advice**.\n")

    # ---------- Per-ticker sections ----------
    for t in tickers:
        node = _safe_get(payload, "per_ticker", t) or {}
        eval_ = node.get("evaluation", {}) or {}
        stat_ = node.get("statistical", {}) or {}  
        risk_ = node.get("risk", {}) or {}
        opt_  = node.get("optimization", {}) or {}
        final = node.get("decision", {}) or {}
        node_plots = node.get("plots", {}) or {}
        asset_plots = assets.get(t, {}) if isinstance(assets, dict) else {}

        r2     = _fmt_num(_safe_get(eval_, "R2_price"), 3)
        rmse   = _fmt_num(_safe_get(eval_, "RMSE_price"), 3, money=True)
        mape   = _fmt_num(_safe_get(eval_, "MAPE_pct"), 2)
        diracc = _fmt_num(_safe_get(eval_, "DirAcc_pct"), 1)
        rows   = _safe_get(eval_, "Rows", default="—")

        # Risk
        html_mode = make_html
        raw_label = risk_.get("label")
        risk_label = (raw_label.upper() if raw_label else "—")
        snr        = _fmt_num(risk_.get("snr_latest"), 2)
        cz         = _fmt_bool(risk_.get("crosses_zero_latest"), html=html_mode)

        # Decision
        raw_decision = final.get("go_nogo") or opt_.get("decision")
        decision     = (raw_decision or "NO-GO")
        weight       = final.get("weight") if final.get("weight") is not None else opt_.get("weight")
        reason       = _fmt_str(final.get("reason") or "", html=html_mode)

        # SHAP / IG plots existence
        p_shap = _get_plot(node_plots, asset_plots, ["shap_bar", "xai_shap_bar", "shap", "beeswarm", "importance"])
        p_ig   = _get_plot(node_plots, asset_plots, ["ig_heatmap", "xai_ig_heatmap", "ig", "importance", "heatmap"])
        shap_status = "Shown" if p_shap else _dash("SHAP skipped or not available", html=html_mode)
        ig_status   = "Shown" if p_ig   else _dash("IG not available", html=html_mode)

        # Section title + executive summary
        if make_html:
            blocks.append(f"<div class='section'><h2>{t}</h2>")
            blocks.append("<h3>Executive Summary</h3>")
            blocks.append(
                f"<ul>"
                f"<li><b>{ui['prediction_accuracy']}</b>: R² <b>{r2}</b>, RMSE <b>{rmse}</b>, MAPE <b>{mape}%</b> (N={rows}).</li>"
                f"<li><b>{ui['risk_assessment']}</b>: label <b>{risk_label}</b>, SNR <b>{snr}</b>, crosses-zero=<code>{cz}</code>.</li>"
                f"<li><b>{ui['final_recommendation']}</b>: " + _decision_badge(decision) +
                (f" (weight={_fmt_num(weight, 2)})" if weight is not None else "") +
                (f" — {reason}" if reason.strip('— ').strip() else "") + ".</li></ul>"
            )

            # Metrics (HTML)
            blocks.append("<details><summary><b>Metrics</b> (click to expand)</summary>")
            blocks.append("<table><thead><tr><th>Category</th><th>Metric</th><th>Value</th></tr></thead><tbody>")
            rows_tbl = [
                (ui["prediction_accuracy"], "R² (price)", r2),
                (ui["prediction_accuracy"], "RMSE (price)", rmse),
                (ui["prediction_accuracy"], "MAPE", f"{mape}%"),
                (ui["trend_prediction"], "Directional Accuracy", f"{diracc}%"),
                (ui["statistical_insights"], "Sharpe (annualised)", _fmt_num(stat_.get("Sharpe_ann"), 2)),
                (ui["statistical_insights"], "Max Drawdown", _fmt_num(stat_.get("Max_Drawdown"), 2)),
                (ui["risk_assessment"], "Label", risk_label),
                (ui["risk_assessment"], "SNR (latest)", snr),
                (ui["risk_assessment"], "Crosses zero", cz),
                (ui["model_explainability"], "SHAP status", shap_status),
                (ui["model_explainability"], "IG status", ig_status),
                (ui["final_recommendation"], "Final", decision),
            ]
            for cat, m, v in rows_tbl:
                blocks.append(f"<tr><td>{cat}</td><td>{m}</td><td>{v}</td></tr>")
            blocks.append("</tbody></table></details>")
        else:
            blocks.append(f"## {t}\n")
            blocks.append("**Executive Summary**\n")
            blocks.append(f"- **{ui['prediction_accuracy']}**: R² **{r2}**, RMSE **{rmse}**, MAPE **{mape}%** _(N={rows})_.")
            blocks.append(f"- **{ui['risk_assessment']}**: label **{risk_label}**, SNR **{snr}**, crosses-zero={cz}.")
            badges = _decision_badge(decision)
            wtxt = (f" (weight={_fmt_num(weight, 2)})" if weight is not None else "")
            rtxt = (f" — {reason}" if reason.strip('— ').strip() else "")
            blocks.append(f"- **{ui['final_recommendation']}**: {badges}{wtxt}{rtxt}.\n")

            # Metrics (Markdown + HTML <details> for collapsible)
            blocks.append("<details><summary><b>Metrics</b> (click to expand)</summary>\n")
            blocks.append("| Category | Metric | Value |")
            blocks.append("|---|---|---|")
            md_rows = [
                (ui["prediction_accuracy"], "R² (price)", r2),
                (ui["prediction_accuracy"], "RMSE (price)", rmse),
                (ui["prediction_accuracy"], "MAPE", f"{mape}%"),
                (ui["trend_prediction"], "Directional Accuracy", f"{diracc}%"),
                (ui["statistical_insights"], "Sharpe (annualised)", _fmt_num(stat_.get("Sharpe_ann"), 2)),
                (ui["statistical_insights"], "Max Drawdown", _fmt_num(stat_.get("Max_Drawdown"), 2)),
                (ui["risk_assessment"], "Label", risk_label),
                (ui["risk_assessment"], "SNR (latest)", snr),
                (ui["risk_assessment"], "Crosses zero", cz),
                (ui["model_explainability"], "SHAP status", ("Shown" if p_shap else "— (SHAP skipped or not available)")),
                (ui["model_explainability"], "IG status", ("Shown" if p_ig else "— (IG not available)")),
                (ui["final_recommendation"], "Final", decision),
            ]
            for cat, m, v in md_rows:
                blocks.append(f"| {cat} | {m} | {v} |")
            blocks.append("\n</details>\n")

        # Plots (auto-hide missing)
        p_line    = _get_plot(node_plots, asset_plots, ["actual_vs_pred_line"])
        p_scatter = _get_plot(node_plots, asset_plots, ["actual_vs_pred_scatter"])
        p_dd      = _get_plot(node_plots, asset_plots, ["risk_drawdown"])
        p_vol     = _get_plot(node_plots, asset_plots, ["risk_rolling_vol"])
        p_tb      = _get_plot(node_plots, asset_plots, ["risk_tomorrow_band"])
        p_fc      = _get_plot(node_plots, asset_plots, ["risk_fanchart"])
        # XAI (broaden keys; fuzzy+glob matching inside _get_plot handles variations)
        p_shap    = _get_plot(node_plots, asset_plots, ["shap_bar", "xai_shap_bar", "shap", "beeswarm", "importance"])
        p_ig      = _get_plot(node_plots, asset_plots, ["ig_heatmap", "xai_ig_heatmap", "ig", "importance", "heatmap"])

        ordered = [
            ("Actual vs Predicted Close (test)", p_line),
            ("Actual vs Predicted Close (scatter)", p_scatter),
            ("Drawdown", p_dd),
            ("Rolling Volatility (20d)", p_vol),
            ("Tomorrow Close Uncertainty", p_tb),
            ("7-Day Forecast Fan Chart", p_fc),
            ("Explainability — SHAP", p_shap),
            ("Explainability — Integrated Gradients", p_ig),
        ]
        for title, src in ordered:
            if not src:
                continue
            if make_html:
                data_uri = _img_data_uri(src)
                if data_uri:
                    blocks.append(f"<p><b>{title}</b><br><img src='{data_uri}' alt='{title}'></p>")
                elif os.path.exists(src):
                    # Fallback to relative path if base64 fails
                    rel = os.path.relpath(src, reports_dir) if os.path.exists(reports_dir) else src
                    blocks.append(f"<p><b>{title}</b><br><img src='{rel}' alt='{title}'></p>")
            else:
                blocks.append(f"**{title}**\n")
                blocks.append(f"![{title}]({src})\n")

        if make_html:
            blocks.append("<div class='hr'></div></div>")
        else:
            blocks.append("---\n")

    # ---------- Footer ----------
    if make_html:
        blocks.append("<p class='muted'><i>This report is automatically generated from model outputs and stock market data. "
                      "It is <b>not</b> investment advice.</i></p></body></html>")
        content = "\n".join(blocks)
    else:
        blocks.append("> _This report is automatically generated from model outputs and stock market data. It is **not** investment advice._\n")
        content = "\n".join(blocks)

    _ensure_dir(reports_dir)
    out_path = os.path.join(reports_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Report written → {out_path}")
    return out_path

# ---------------------------- LangGraph wrapper ----------------------------
try:
    from langgraph.graph import StateGraph
    HAS_LG = True
except Exception:
    HAS_LG = False

def node_render(state: Dict[str, Any]) -> Dict[str, Any]:
    run_base = state.get("run_base") or "."
    order    = state.get("tickers")
    out_name = state.get("out_name", "llm_report.html")  # default to HTML
    path = render_report(run_base=run_base, out_name=out_name, order=order)
    return {**state, "status": "ok", "report_path": path}

def build_llm_report_workflow():
    g = StateGraph(dict)
    g.add_node("render", node_render)
    g.set_entry_point("render")
    g.set_finish_point("render")
    return g.compile()

if __name__ == "__main__":
    run_base = os.environ.get("RUN_BASE", ".")
    tickers  = os.environ.get("TICKERS")
    order = [t.strip().upper() for t in (tickers.split(",") if tickers else [])] if tickers else None
    try:
        render_report(run_base=run_base, out_name="llm_report.html", order=order)
    except Exception as e:
        print("Report generation failed:", e)

