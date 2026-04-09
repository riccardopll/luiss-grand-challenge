from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.engine import CRMDecisionEngine

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

app = FastAPI(title="Fater CRM Decision Engine Demo")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

engine = CRMDecisionEngine(
    project_root=PROJECT_ROOT,
    artifacts_dir=PROJECT_ROOT / "artifacts",
    rules_path=PROJECT_ROOT / "rules" / "decision_rules.txt",
)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, user_id: str | None = Query(default=None)) -> HTMLResponse:
    payload = None
    not_found = False
    if user_id:
        payload = engine.lookup(user_id.strip())
        not_found = payload is None

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "initial_payload": asdict(payload) if payload is not None else None,
            "user_id": user_id or "",
            "not_found": not_found,
            "sample_user_ids": engine.sample_user_ids,
            "latest_reference_date": engine.latest_reference_date,
        },
    )


@app.get("/api/user/{user_id}")
def user_lookup(user_id: str) -> dict[str, object]:
    payload = engine.lookup(user_id.strip())
    if payload is None:
        raise HTTPException(
            status_code=404, detail="User not found in the latest snapshot.")
    return asdict(payload)
