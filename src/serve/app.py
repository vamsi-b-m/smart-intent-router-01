from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import joblib
from fastapi import FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.config import (
    APP_NAME,
    APP_VERSION,
    ENVIRONMENT,
    MODEL_NAME,
    MODEL_PATH,
    MODEL_VERSION,
)


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to classify",
    )


class PredictionResponse(BaseModel):
    request_id: UUID
    intent: str
    confidence: float
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool


class VersionResponse(BaseModel):
    app_name: str
    app_version: str
    model_name: str
    model_version: str
    environment: str


def load_model():
    """Load the trained model artifact."""
    return joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup/shutdown lifecycle.

    The model is loaded once during startup rather than
    loading it for every prediction request.
    """

    print(f"Loading model from: {MODEL_PATH}")

    try:
        app.state.model = load_model()
        app.state.model_loaded = True

        print("Model loaded successfully.")

    except Exception as exc:
        app.state.model = None
        app.state.model_loaded = False

        print(f"Model loading failed: {exc}")

    yield

    # Cleanup during shutdown
    app.state.model = None
    app.state.model_loaded = False


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Production-oriented ML inference service",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    """
    Liveness endpoint.

    This tells Kubernetes/load balancers that the
    application process itself is running.
    """

    return HealthResponse(
        status="healthy",
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
)
def ready(request: Request):
    """
    Readiness endpoint.

    The service is ready only when the ML model
    has been successfully loaded.
    """

    if not request.app.state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    return ReadinessResponse(
        status="ready",
        model_loaded=True,
    )


@app.get(
    "/version",
    response_model=VersionResponse,
)
def version():
    """Return application and model version information."""

    return VersionResponse(
        app_name=APP_NAME,
        app_version=APP_VERSION,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        environment=ENVIRONMENT,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    request: Request,
    x_request_id: str | None = Header(default=None),
):
    """
    Run inference against the loaded model.
    """

    if not request.app.state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready",
        )

    request_id = (
        x_request_id
        if x_request_id
        else str(uuid4())
    )

    model = request.app.state.model

    try:
        prediction = model.predict([payload.text])[0]

        probabilities = model.predict_proba(
            [payload.text]
        )[0]

        confidence = float(probabilities.max())

    except Exception as exc:
        print(
            f"Inference failed. "
            f"request_id={request_id}, "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed",
        ) from exc

    return PredictionResponse(
        request_id=request_id,
        intent=str(prediction),
        confidence=confidence,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )
