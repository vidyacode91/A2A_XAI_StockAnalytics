import re
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

try:
    from langgraph.graph import StateGraph
    HAS_LG = True
except Exception:
    HAS_LG = False


# ---------------- helpers ----------------

def _now_date() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()

def _parse_rel_window(text: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Parse phrases like 'last 5 years', 'last 6 months', 'last 30 days'."""
    t = text.lower()
    end = _now_date()
    if re.search(r'\bytd\b|\byear[-\s]*to[-\s]*date\b', t):
        start = pd.Timestamp(end.year, 1, 1)
        return start, end
    m = re.search(r'last\s+(\d+)\s*(year|years|yr|yrs|month|months|mo|mos|week|weeks|day|days)', t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith(('year', 'yr')):
            start = end - pd.DateOffset(years=n)
        elif unit.startswith('month') or unit.startswith('mo'):
            start = end - pd.DateOffset(months=n)
        elif unit.startswith('week'):
            start = end - pd.Timedelta(days=7 * n)
        else:
            start = end - pd.Timedelta(days=n)
        return start.normalize(), end
    return None, None

def _parse_explicit_dates(text: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Parse explicit 'from YYYY-MM-DD' / 'to YYYY-MM-DD' spans."""
    start = end = None
    m = re.search(r'(from|since|start[,:\s]*)\s*([0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2})', text, flags=re.I)
    if m:
        try: start = pd.to_datetime(m.group(2))
        except Exception: pass
    m = re.search(r'(to|until|end[,:\s]*)\s*([0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2})', text, flags=re.I)
    if m:
        try: end = pd.to_datetime(m.group(2))
        except Exception: pass
    return start, end

def _default_dates(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """If dates missing, default to last 3 years."""
    e = (end or _now_date()).normalize()
    s = (start or (e - pd.DateOffset(years=3))).normalize()
    if s > e:
        s, e = e - pd.DateOffset(years=1), e
    return s, e

def _parse_seq_len(text: str, default: int = 30) -> int:
    m = re.search(r'(seq[_\s-]*len|sequence\s*length)\s*[:=]?\s*(\d+)', text, flags=re.I)
    if m:
        try:
            return max(2, int(m.group(2)))
        except Exception:
            return default
    return default

def _parse_tickers(text: str) -> List[str]:
    """General ticker parser."""
    t = text.upper()
    STOP = {
        "I","ME","MY","WE","OUR","YOU","YOUR","PLEASE",
        "WANT","KNOW","SHOW","TELL","TREND","TRENDS","STOCK","MARKET",
        "FOR","OF","THE","A","AN","AND","LAST","YEARS","YEAR",
        "MONTHS","MONTH","WEEKS","WEEK","DAYS","DAY",
        "IS","IT","OKAY","TO","TRADE","NOW","ABOUT","ON","IN","OVER",
        "RUN","FE","SEQ","LEN","YTD","FROM","SINCE","TO","UNTIL","END",
        "PREDICT","EVAL","RISK","OPT","IG","SHAP","SUMMARY","REPORT",
        "NASDAQ","NYSE","LSE","EURONEXT","AMEX","TSX","ASX","BSE","NSE","SHOULD",
    }
    toks: List[str] = []
    for m in re.finditer(r'\b(?:NYSE|NASDAQ|LSE|EURONEXT|AMEX|TSX|ASX|BSE|NSE)[:\s]+([A-Z.-]{1,6})\b', t):
        tok = m.group(1).replace('-', '.'); toks.append(tok)
    for m in re.finditer(r'\b[A-Z][A-Z.-]{0,5}\b', t):
        tok = m.group(0)
        if tok in STOP or len(tok) <= 1: continue
        toks.append(tok.replace('-', '.'))
    m = re.search(r'(?:\brun\b|\btickers?)\s*[:=]\s*([A-Z0-9,.\s-]+)', t)
    if m:
        more = [s.strip().replace('-', '.') for s in re.split(r'[,\s]+', m.group(1)) if s.strip()]
        toks.extend(more)
    seen, out = set(), []
    for tok in toks:
        if re.fullmatch(r'[A-Z.]{1,6}', tok) and tok not in seen and tok not in STOP:
            seen.add(tok); out.append(tok)
    return out or ["AAPL"]

def _user_intends_decision(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in [
        "is it okay to trade", "should i trade", "go or no-go", "buy", "enter", "trade now", "okay to trade"
    ])

def _parse_steps(text: str) -> Dict[str, bool]:
    """Turn words in the prompt into pipeline flags."""
    t = text.lower()
    mentioned = any(k in t for k in ["collect","fe","feature","predict","eval","risk","opt","ig","shap","summary","report"])
    defaults = dict(
        run_collector=True, run_fe=True, run_predict=True, run_eval=True,
        run_xai_ig=True, run_xai_shap=True, run_risk=True, run_opt=True,
        run_portfolio=False, run_summarizer=True, run_llm_report=True,
    )
    if not mentioned: return defaults
    flags = {k: False for k in defaults}
    def has(*keys): return any(k in t for k in keys)
    flags["run_collector"]  = has("collect","collector","download")
    flags["run_fe"]         = has("fe","feature")
    flags["run_predict"]    = has("predict")
    flags["run_eval"]       = has("eval","evaluation")
    flags["run_xai_ig"]     = has(" ig ","integrated gradients")
    flags["run_xai_shap"]   = has("shap")
    flags["run_risk"]       = has("risk")
    flags["run_opt"]        = has("opt","optimization")
    flags["run_portfolio"]  = has("portfolio")
    flags["run_summarizer"] = has("summary","summarizer")
    flags["run_llm_report"] = has("report")
    for k, v in defaults.items():
        if k not in flags: flags[k] = v
    return flags

def _make_run_base(base: Optional[str] = None) -> str:
    if base: return base
    return pd.Timestamp.now().strftime("RUN_%Y%m%d_%H%M%S")


# ---------------- main NL node ----------------

def node_parse(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reception: parse tickers, dates (explicit/relative), seq_len and flags.
    Falls back to sensible defaults when the user doesn't specify.
    """
    txt = (state.get("user_request", "") or "").strip()
    if not txt:
        txt = "Analyze AAPL last 3 years; seq_len 30"

    # tickers
    tickers = state.get("tickers") or _parse_tickers(txt) or ["AAPL"]

    # dates
    s1, e1 = _parse_explicit_dates(txt)
    s2, e2 = _parse_rel_window(txt)
    start, end = _default_dates(s1 or s2, e1 or e2)

    # seq_len
    seq_len = _parse_seq_len(txt, default=int(state.get("default_seq_len", 30)))

    # pipeline flags
    flags = _parse_steps(txt)

    # intent
    wants_decision = _user_intends_decision(txt)

    # run folder
    run_base = _make_run_base(state.get("run_base"))

    # UX text
    pretty_tics = ", ".join(tickers)
    daterange = f"{start.date()} \u2192 {end.date()}"
    ack = (f"Thanks! I’ll analyze {pretty_tics} over {daterange}, then coordinate "
           f"data → features → prediction → evaluation → IG & SHAP → risk → optimization, and return a report.")
    plan = (f"Plan: tickers={tickers}, window={start.date()}..{end.date()}, "
            f"seq_len={seq_len}, steps={[k for k, v in flags.items() if v]}, run_base='{run_base}', "
            f"decision_requested={wants_decision}.")

    # Per-stage args 
    base_args = {"run_base": run_base, "tickers": tickers}
    collector_args = {**base_args, "start": str(start.date()), "end": str(end.date())}
    fe_args = {**base_args}
    predict_args = {**base_args, "seq_len": seq_len}
    eval_args = {**base_args}
    risk_args = {**base_args}
    ig_args = {**base_args, "ig_head": "close"}
    shap_args = {**base_args, "head": "close"}
    opt_args = {**base_args, "mode": "historical", "head": "close"}

    return {
        **state,
        "status": "parsed",
        "ack_text": ack,
        "plan_text": plan,
        "run_base": run_base,
        "tickers": tickers,
        "seq_len": seq_len,
        "date_start": str(start.date()),
        "date_end": str(end.date()),
        "wants_decision": wants_decision,
        **flags,
        "collector_args": collector_args,
        "fe_args": fe_args,
        "predict_args": predict_args,
        "eval_args": eval_args,
        "risk_args": risk_args,
        "ig_args": ig_args,
        "shap_args": shap_args,
        "opt_args": opt_args,
    }


# ---------------- optional workflow ----------------

def build_llm_reception_workflow():
    if not HAS_LG:
        raise RuntimeError("LangGraph not available. Install with: pip install langgraph")
    g = StateGraph(dict)
    g.add_node("parse", node_parse)
    g.set_entry_point("parse")
    g.set_finish_point("parse")
    return g.compile()
