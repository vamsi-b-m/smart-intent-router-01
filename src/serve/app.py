import os
import uuid

import mlflow
import mlflow.sklearn

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


APP_NAME = os.getenv(
    "APP_NAME",
    "smart-intent-router",
)

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0",
)

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "local",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "smart-intent-router",
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "champion",
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001",
)


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
)


model = None
model_uri = None


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Text to classify",
    )


class PredictionResponse(BaseModel):
    intent: str
    confidence: float
    model_name: str
    model_alias: str
    model_uri: str
    request_id: str


@app.on_event("startup")
def load_model():
    global model
    global model_uri

    print("Starting model initialization...")
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Model name: {MODEL_NAME}")
    print(f"Model alias: {MODEL_ALIAS}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    print(f"Loading model from: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)

        print("Model loaded successfully.")

    except Exception as exc:
        print(f"Failed to load model: {exc}")
        raise


@app.middleware("http")
async def add_request_id(
    request: Request,
    call_next,
):
    request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id

    return response


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.get("/ready")
def readiness():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "ready",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
    }


@app.get("/version")
def version():
    return {
        "application": APP_NAME,
        "application_version": APP_VERSION,
        "environment": ENVIRONMENT,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_uri": model_uri,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
    http_request: Request,
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    request_id = http_request.state.request_id

    try:
        prediction = model.predict([request.text])[0]

        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                [request.text]
            )[0]

            confidence = float(max(probabilities))

        if confidence is None:
            confidence = 0.0

        return PredictionResponse(
            intent=str(prediction),
            confidence=confidence,
            model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS,
            model_uri=model_uri,
            request_id=request_id,
        )

    except Exception as exc:
        print(
            f"Prediction failed. "
            f"request_id={request_id}, "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        )
