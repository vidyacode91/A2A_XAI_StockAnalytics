import os, datetime
from typing import Dict, Any, Tuple

from langgraph.graph import StateGraph

# import agent modules 
from . import reception                 as rec
from . import data_collector_agent      as dc
from . import feature_engineering_agent as fe
from . import predictive_model_agent    as pm
from . import model_evaluation_agent    as me
from . import explainability_agent      as xai
from . import risk_assessment_agent     as risk
from . import optimisation_agent        as opt
from . import summarizer_agent          as summ
from .report_agent import render_report


# =========================
# Helpers
# =========================
def _ts() -> str:
    return datetime.datetime.now().strftime("RUN_%Y%m%d_%H%M%S")

def _ensure_run_base(state: Dict[str, Any]) -> str:
    rb = state.get("run_base") or os.environ.get("RUN_BASE")
    if not rb:
        root = os.environ.get("A2A_ROOT", "/content/drive/MyDrive/A2A_prediction_system")
        rb = os.path.join(root, _ts())
    os.makedirs(rb, exist_ok=True)
    os.environ["RUN_BASE"] = rb
    return rb

def _ok(res: Dict[str, Any]) -> bool:
    """Consider a step ok unless it explicitly reports an error/failure."""
    if not isinstance(res, dict): return False
    if res.get("error"): return False
    s = str(res.get("status", "")).lower()
    if any(b in s for b in ("fail","error","exception","not found")):
        return False
    return True

def _invoke(build_fn, args: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    app = build_fn()
    out = app.invoke(args)
    return _ok(out), out

def _flag(state: Dict[str, Any], *keys: str, default: bool=True) -> bool:
    """Return first present flag among synonyms, else default."""
    for k in keys:
        if k in state: return bool(state[k])
    return default


# =========================
# Nodes
# =========================
def node_init(state: Dict[str, Any]) -> Dict[str, Any]:
    rb = _ensure_run_base(state)

    # Map common top-level keys 
    st = {**state, "run_base": rb}

    # Collector args: support both start/end and date_start/date_end
    ca = {**st.get("collector_args", {})}
    if "start_date" in st: ca["start"] = st["start_date"]
    if "end_date"   in st: ca["end"]   = st["end_date"]
    if "date_start" in st: ca["start"] = st["date_start"]
    if "date_end"   in st: ca["end"]   = st["date_end"]
    if "tickers"    in st: ca["tickers"] = st["tickers"]
    st["collector_args"] = ca

    # Predict args
    pa = {**st.get("predict_args", {})}
    if "seq_len" in st: pa["seq_len"] = st["seq_len"]
    st["predict_args"] = pa

    return {**st, "init_status": {"ok": True, "run_base": rb}}

def node_reception(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse natural-language request:
      - tickers
      - date_start/date_end 
      - seq_len
    """
    ok, out = _invoke(rec.build_llm_reception_workflow, {
        "user_request": state.get("user_request", ""),
        "default_seq_len": state.get("default_seq_len", 40),
        "run_base": state["run_base"],
        "tickers": state.get("tickers"),
    })

    new_state = {**state, "reception_status": {"ok": ok}, "reception_out": out}
    print("[RECEPTION] ack:", out.get("ack_text"))

    # tickers
    tickers = out.get("tickers") or new_state.get("tickers")
    if tickers:
        new_state["tickers"] = tickers

    # date window 
    ca = {**new_state.get("collector_args", {})}
    if out.get("date_start"): ca["start"] = out["date_start"]
    if out.get("date_end"):   ca["end"]   = out["date_end"]
  
    if out.get("start"): ca["start"] = out["start"]
    if out.get("end"):   ca["end"]   = out["end"]
    new_state["collector_args"] = ca

    # seq_len -> predict args
    pa = {**new_state.get("predict_args", {})}
    if out.get("seq_len"): pa["seq_len"] = out["seq_len"]
    new_state["predict_args"] = pa

    # run_base 
    if out.get("run_base"):
        new_state["run_base"] = out["run_base"]
        os.environ["RUN_BASE"] = out["run_base"]

    return new_state

def node_collect(state: Dict[str, Any]) -> Dict[str, Any]:
    if not _flag(state, "run_collector", default=True):
        return state

    ok0, out = _invoke(dc.build_data_collector_workflow, {
        **state.get("collector_args", {}),
        "run_base": state["run_base"],
        "tickers": state.get("tickers"),
    })

    data_path = out.get("data_path")
   
    if not data_path:
        cand = os.path.join(state["run_base"], "data", "yfinance_raw_data.csv")
        if os.path.exists(cand): data_path = cand

    exists = bool(data_path and os.path.exists(data_path))
    ok = ok0 or exists or ("success" in str(out.get("status","")).lower())

    if ok and exists:
        fe_args = {**state.get("fe_args", {})}
        fe_args["data_path"]     = data_path
        fe_args["raw_data_path"] = data_path
        fe_args["input_csv"]     = data_path
        fe_args["run_base"]      = state["run_base"]
        return {**state,
                "collector_status": {"ok": True},
                "collector_out": out,
                "fe_args": fe_args}
    else:
        return {**state,
                "collector_status": {"ok": False, "error": out.get("error")},
                "collector_out": out,
                # disable downstream steps if collection failed
                "run_fe": False, "run_predict": False, "run_eval": False,
                "run_xai": False, "run_risk": False, "run_opt": False,
                "run_summary": False, "run_report": False}

def node_fe(state: Dict[str, Any]) -> Dict[str, Any]:
    if not _flag(state, "run_fe", default=True):
        return state
    ok, out = _invoke(fe.build_enhanced_feature_workflow, {
        **state.get("fe_args", {}),
        "run_base": state["run_base"],
    })
    return {**state, "fe_status": {"ok": ok}, "fe_out": out}

def node_predict(state: Dict[str, Any]) -> Dict[str, Any]:
    if not _flag(state, "run_predict", default=True):
        return state
    ok, out = _invoke(pm.build_predictive_workflow, {
        **state.get("predict_args", {}),
        "run_base": state["run_base"],
    })
    return {**state, "predict_status": {"ok": ok}, "predict_out": out}

def node_eval(state: Dict[str, Any]) -> Dict[str, Any]:
    if not _flag(state, "run_eval", default=True):
        return state
    ok, out = _invoke(me.build_evaluation_workflow, {
        **state.get("eval_args", {}),
        "run_base": state["run_base"],
    })
    return {**state, "eval_status": {"ok": ok}, "eval_out": out}

def node_xai(state: Dict[str, Any]) -> Dict[str, Any]:
    # accept multiple flag spellings
    if not _flag(state, "run_xai", "run_xai_shap", "run_xai_ig", default=True):
        return state
    ok, out = _invoke(xai.build_explainability_workflow, {
        **state.get("xai_args", {}),
        "run_base": state["run_base"],
    })
    return {**state, "xai_status": {"ok": ok}, "xai_out": out}

def node_risk(state: Dict[str, Any]) -> Dict[str, Any]:
    if not _flag(state, "run_risk", default=True):
        return state
    ok, out = _invoke(risk.build_risk_workflow, {
        **state.get("risk_args", {}),
        "run_base": state["run_base"],
    })
    return {**state, "risk_status": {"ok": ok}, "risk_out": out}

def node_opt(state: Dict[str, Any]) -> Dict[str, Any]:
    if not _flag(state, "run_opt", default=True):
        return state
    ok, out = _invoke(opt.build_optimization_workflow, {
        **state.get("opt_args", {}),
        "run_base": state["run_base"],
    })
    return {**state, "opt_status": {"ok": ok}, "opt_out": out}

def node_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    if not _flag(state, "run_summary", "run_summarizer", default=True):
        return state
    ok, out = _invoke(summ.build_summarizer_workflow, {
        **state.get("summary_args", {}),
        "run_base": state["run_base"],
    })
    return {**state, "summary_status": {"ok": ok}, "summary_out": out}

def node_report(state: Dict[str, Any]) -> Dict[str, Any]:
    if not _flag(state, "run_report", "run_llm_report", default=True):
        return state
    path = render_report(
        run_base=state["run_base"],
        out_name=state.get("report_out_name","llm_report.html"),
        order=state.get("tickers"),
    )
    return {**state, "report_status": {"ok": bool(path)}, "report_path": path}


# =========================
# Workflow
# =========================
def build_orchestrator_workflow():
    g = StateGraph(dict)
    g.add_node("init",      node_init)
    g.add_node("reception", node_reception)
    g.add_node("collect",   node_collect)
    g.add_node("features",  node_fe)
    g.add_node("predict",   node_predict)
    g.add_node("eval",      node_eval)
    g.add_node("xai",       node_xai)
    g.add_node("risk",      node_risk)
    g.add_node("opt",       node_opt)
    g.add_node("summary",   node_summary)
    g.add_node("report",    node_report)

    g.set_entry_point("init")
    g.add_edge("init", "reception")
    g.add_edge("reception","collect")
    g.add_edge("collect","features")
    g.add_edge("features","predict")
    g.add_edge("predict","eval")
    g.add_edge("eval","xai")
    g.add_edge("xai","risk")
    g.add_edge("risk","opt")
    g.add_edge("opt","summary")
    g.add_edge("summary","report")
    g.set_finish_point("report")
    return g.compile()

def run_orchestrator(config: Dict[str, Any]) -> Dict[str, Any]:
    print("[ENTRY]", "NL mode" if config.get("user_request") else "Structured mode")
    app = build_orchestrator_workflow()
    out = app.invoke(config or {})
    import os
    run_base = out.get("run_base") or ""
    run_id   = os.path.basename(run_base) if run_base else None
    return {
        "status": "done",
        "run_id": run_id,                 
        "run_base": run_base,
        "report_path": out.get("report_path"),
        "reception_status": out.get("reception_status"),
        "reception_out": out.get("reception_out"),
        "collector_status": out.get("collector_status"),
        "fe_status": out.get("fe_status"),
        "predict_status": out.get("predict_status"),
        "eval_status": out.get("eval_status"),
        "xai_status": out.get("xai_status"),
        "risk_status": out.get("risk_status"),
        "opt_status": out.get("opt_status"),
        "summary_status": out.get("summary_status"),
        "report_status": out.get("report_status"),
    }
