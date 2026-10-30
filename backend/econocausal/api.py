"""REST API for EconoCausal (FastAPI).  Run:  uvicorn econocausal.api:app --reload"""
from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__, data as datagen, service
from .config import DEFAULT_BUDGET
from .dag import graph_json

app = FastAPI(title="EconoCausal API", version=__version__,
              description="Causal ML (Double Machine Learning) for personalized discount allocation.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSION = service.Session()


class AllocateRequest(BaseModel):
    budget: float = Field(DEFAULT_BUDGET, gt=0, le=10_000_000)
    max_rows: int = Field(300, ge=1, le=5000, description="Rows of the prescription table to return")


def _guard(fn, *a, **k):
    try:
        return fn(*a, **k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "data_loaded": SESSION.history is not None,
            "model_trained": SESSION.engine is not None}


@app.get("/dag")
def dag():
    return graph_json()


@app.post("/data/demo")
def data_demo(n: int = 6000):
    if not 1000 <= n <= 50000:
        raise HTTPException(400, "n must be between 1,000 and 50,000")
    return service.load_demo(SESSION, n)


@app.post("/data/upload")
async def data_upload(file: UploadFile = File(...)):
    try:
        df = service.read_csv(await file.read())
    except Exception:
        raise HTTPException(400, "Could not read the file as CSV")
    return _guard(service.load_history, SESSION, df)


@app.get("/data/template")
def data_template():
    csv = datagen.generate_observational(20).drop(columns=["true_p0", "true_tau"]).to_csv(index=False)
    return StreamingResponse(io.BytesIO(csv.encode()), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=econocausal_template.csv"})


@app.post("/train")
def train(trees: int = 300):
    return _guard(service.train, SESSION, trees)


@app.get("/uplift")
def uplift():
    if not SESSION.train_report:
        raise HTTPException(400, "Train the model first")
    return SESSION.train_report


@app.get("/audit")
def audit():
    return _guard(service.audit, SESSION)


@app.post("/allocate")
def allocate(req: AllocateRequest):
    result = _guard(service.allocate, SESSION, req.budget)
    result["table_total_rows"] = len(result["table"])
    result["table"] = result["table"][: req.max_rows]
    return result


@app.get("/allocate/download")
def allocate_download():
    if SESSION.last_allocation is None:
        raise HTTPException(400, "Run an allocation first")
    csv = SESSION.last_allocation.round(4).to_csv(index=False)
    return StreamingResponse(io.BytesIO(csv.encode()), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=prescription.csv"})


@app.get("/users/{user_id}/ite")
def user_ite(user_id: int):
    try:
        return service.user_ite(SESSION, user_id)
    except KeyError:
        raise HTTPException(404, "Unknown user_id")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/drift/demo")
def drift_demo(shift: float = 0.0):
    if not 0 <= shift <= 2:
        raise HTTPException(400, "shift must be between 0 and 2")
    return _guard(service.drift_check, SESSION, datagen.campaign_audience(1500, shift=shift))


@app.post("/drift/upload")
async def drift_upload(file: UploadFile = File(...)):
    try:
        df = service.read_csv(await file.read())
    except Exception:
        raise HTTPException(400, "Could not read the file as CSV")
    return _guard(service.drift_check, SESSION, df)