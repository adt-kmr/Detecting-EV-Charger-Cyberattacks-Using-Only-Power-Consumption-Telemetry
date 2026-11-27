"""
FastAPI server for the EVSE intrusion-detection demo.

Run from the repository root:
    python -m uvicorn app.server:app --port 8000
then open http://localhost:8000
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .pipeline import DATA_PATH, STRATEGIES, Engine

STATIC = Path(__file__).parent / "static"
engine = Engine()


@asynccontextmanager
async def lifespan(_app):
    if DATA_PATH.exists():
        engine.train_async("block")  # warm start with the leakage-aware strategy
    else:
        engine._set("error", f"Dataset not found at {DATA_PATH}", 0.0)
    yield


app = FastAPI(title="EVSE Power-Telemetry Intrusion Detection", lifespan=lifespan)


class TrainReq(BaseModel):
    strategy: str = "block"


def _need_model():
    if engine.trained is None:
        raise HTTPException(409, "Models are not trained yet")
    return engine.trained


@app.get("/api/status")
def status():
    s = dict(engine.status)
    s["strategies"] = STRATEGIES
    s["trained_strategy"] = engine.trained.strategy if engine.trained else None
    return s


@app.get("/api/dataset")
def dataset():
    info = engine.dataset_info()
    if info is None:
        raise HTTPException(409, "Dataset still loading")
    return info


@app.post("/api/train")
def train(req: TrainReq):
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"strategy must be one of {list(STRATEGIES)}")
    if engine.status.get("state") in ("training", "loading"):
        raise HTTPException(409, "Training already in progress")
    engine.train_async(req.strategy)
    return {"ok": True}


@app.get("/api/metrics")
def metrics():
    return _need_model().metrics


@app.get("/api/segments")
def segments():
    return _need_model().segments


@app.get("/api/stream")
def stream(cursor: int = 0, n: int = 1):
    _need_model()
    return engine.stream(cursor, n)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    _need_model()
    try:
        return engine.predict_csv(await file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
