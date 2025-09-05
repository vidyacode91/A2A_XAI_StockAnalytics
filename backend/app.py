# backend/app.py
import os
import datetime
from fastapi import FastAPI, Body, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

from backend.a2a.orchestrator import run_orchestrator

app = FastAPI(title="A2A Orchestrator API")

# --- CORS  ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths ---
A2A_ROOT = os.environ.get("A2A_ROOT", "/content/drive/MyDrive/A2A_prediction_system")
FRONTEND_DIR = os.path.join(A2A_ROOT, "frontend")

# Serve Drive so images in the report load
if os.path.isdir(A2A_ROOT):
    app.mount("/files", StaticFiles(directory=A2A_ROOT), name="files")

# Serve the static UI: <A2A_ROOT>/frontend/index.html -> /ui/index.html
if os.path.isdir(FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _new_run_base() -> tuple[str, str]:
    """Create a fresh run folder and return (run_id, run_base)."""
    run_id = datetime.datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
    run_base = os.path.join(A2A_ROOT, run_id)
    os.makedirs(os.path.join(run_base, "reports"), exist_ok=True)
    return run_id, run_base


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
def root_redirect():
    if os.path.isdir(FRONTEND_DIR):
        return HTMLResponse('<meta http-equiv="refresh" content="0; url=/ui/index.html">')
    return HTMLResponse('<meta http-equiv="refresh" content="0; url=/docs">')


@app.get("/health")
def health():
    return {"ok": True}


# Synchronous run with structured JSON
@app.post("/run")
def run_pipeline(payload: dict = Body(...)):
    payload = dict(payload or {})
    payload.setdefault("report_out_name", "llm_report.html")
    # Ensure a stable run_id in the response
    if "run_base" not in payload:
        run_id, run_base = _new_run_base()
        payload["run_base"] = run_base
    else:
        run_id = os.path.basename(payload["run_base"].rstrip("/"))
    res = run_orchestrator(payload)
    res["run_id"] = run_id
    return JSONResponse(res)


# Synchronous natural-language run 
@app.post("/run-nl")
def run_pipeline_nl(
    user_request: str = Body(..., embed=True),
    report_out_name: str | None = Body(None),
    tickers: list[str] | None = Body(None),
):
    run_id, run_base = _new_run_base()
    cfg = {
        "user_request": user_request,
        "report_out_name": report_out_name or "llm_report.html",
        "run_base": run_base,
    }
    if tickers:
        cfg["tickers"] = tickers
    res = run_orchestrator(cfg)
    res["run_id"] = run_id
    return JSONResponse(res)


# Asynchronous natural-language run 
@app.post("/run-nl-async")
def run_pipeline_nl_async(
    background_tasks: BackgroundTasks,
    user_request: str = Body(..., embed=True),
    report_out_name: str | None = Body(None),
    tickers: list[str] | None = Body(None),
):
    run_id, run_base = _new_run_base()
    cfg = {
        "user_request": user_request,
        "report_out_name": report_out_name or "llm_report.html",
        "run_base": run_base,
    }
    if tickers:
        cfg["tickers"] = tickers

    # Launch heavy pipeline in the background and return immediately
    background_tasks.add_task(run_orchestrator, cfg)

    return {
        "accepted": True,
        "run_id": run_id,
        "run_base": run_base,
        "status_url": f"/status/{run_id}",
        "report_url": f"/report-html/{run_id}",
        "message": "Run started in background.",
    }


# Is the HTML report there yet?
@app.get("/status/{run_id}")
def run_status(run_id: str):
    rpt = os.path.join(A2A_ROOT, run_id, "reports", "llm_report.html")
    ready = os.path.exists(rpt)
    return {
        "run_id": run_id,
        "ready": ready,
        "report": f"/report-html/{run_id}" if ready else None,
    }


# Download any report file (HTML or other)
@app.get("/report/{run_id}/{filename}")
def get_report(run_id: str, filename: str):
    path = os.path.join(A2A_ROOT, run_id, "reports", filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"Report not found: {path}")
    return FileResponse(path)


# HTML preview: rewrite absolute Drive paths to files
@app.get("/report-html/{run_id}")
def preview_html(run_id: str):
    path = os.path.join(A2A_ROOT, run_id, "reports", "llm_report.html")
    if not os.path.exists(path):
        raise HTTPException(404, f"Report not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(A2A_ROOT, "/files")
    return HTMLResponse(content)


