from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.datastructures import State

from app.model import load
from app.schemas import Session

INDEX = Path(__file__).parent / "static" / "index.html"
THRESHOLD = 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.info = load()
    yield


app = FastAPI(title="ShopPredict", lifespan=lifespan)


def loaded(request: Request):
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return request.app.state


Loaded = Annotated[State, Depends(loaded)]


@app.get("/")
def index():
    return FileResponse(INDEX)


@app.get("/ready", dependencies=[Depends(loaded)])
def ready():
    return {"status": "ready"}


@app.get("/model-info")
def model_info(state: Loaded):
    return state.info


@app.post("/predict")
def predict(session: Session, state: Loaded):
    row = session.model_dump()
    row["Weekend"] = int(row["Weekend"])
    probability = round(float(state.model.predict_proba(pd.DataFrame([row]))[0, 1]), 4)
    return {
        "probability": probability,
        "purchase": probability >= THRESHOLD,
        "model_version": state.info["version"],
    }
