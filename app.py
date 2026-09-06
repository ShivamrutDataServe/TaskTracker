import json
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

app = FastAPI(title="TaskTracker API")

MODEL_PATH = Path("model.pkl")
METRICS_PATH = Path("metrics.json")
VERSION_PATH = Path("VERSION")

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "support_calls",
    "is_senior_citizen",
    "has_tech_support",
]

templates = Jinja2Templates(directory="templates")


def _load_model():
    """Load the trained model, failing loudly and clearly if it's missing.

    model.pkl is produced by train_model.py in CI and copied into the
    image at build time - it is never trained here.
    """
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"{MODEL_PATH} not found. Run `python train_model.py` first, "
            "or make sure the CI-trained artifact was copied into the image."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _to_features(
    tenure_months, monthly_charges, total_charges,
    contract_type, support_calls, is_senior_citizen, has_tech_support,
) -> pd.DataFrame:
    """Builds a DataFrame with the same column names/order used at
    training time, so scikit-learn matches features by name, not
    position."""
    return pd.DataFrame(
        [[tenure_months, monthly_charges, total_charges, contract_type,
          support_calls, is_senior_citizen, has_tech_support]],
        columns=FEATURE_COLUMNS,
    )


model = _load_model()
metrics = _load_json(METRICS_PATH)
app_version = VERSION_PATH.read_text().strip() if VERSION_PATH.exists() else "unknown"


class CustomerData(BaseModel):
    tenure_months: int = Field(ge=0, le=100)
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)
    contract_type: int = Field(ge=0, le=2, description="0=month-to-month, 1=one_year, 2=two_year")
    support_calls: int = Field(ge=0, le=20)
    is_senior_citizen: int = Field(ge=0, le=1)
    has_tech_support: int = Field(ge=0, le=1)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"prediction": None, "probability": None})


@app.get("/health")
def health():
    """Used by the Docker HEALTHCHECK and by load balancers/orchestrators."""
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/version")
def version():
    """Reports exactly which code version and which trained model this
    running container is serving."""
    return {
        "app_version": app_version,
        "model_metrics": metrics,
    }


@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    tenure_months: int = Form(...),
    monthly_charges: float = Form(...),
    total_charges: float = Form(...),
    contract_type: int = Form(...),
    support_calls: int = Form(...),
    is_senior_citizen: int = Form(...),
    has_tech_support: int = Form(...),
):
    features = _to_features(
        tenure_months, monthly_charges, total_charges,
        contract_type, support_calls, is_senior_citizen, has_tech_support,
    )
    probability = float(model.predict_proba(features)[0][1])
    prediction = "Likely to churn" if probability >= 0.5 else "Likely to stay"
    return templates.TemplateResponse(
        request, "index.html",
        {"prediction": prediction, "probability": round(probability * 100, 1)},
    )


@app.post("/predict/")
def predict(data: CustomerData):
    features = _to_features(
        data.tenure_months, data.monthly_charges, data.total_charges,
        data.contract_type, data.support_calls, data.is_senior_citizen,
        data.has_tech_support,
    )
    probability = float(model.predict_proba(features)[0][1])
    return {
        "churn_probability": round(probability, 4),
        "churn_prediction": int(probability >= 0.5),
    }
