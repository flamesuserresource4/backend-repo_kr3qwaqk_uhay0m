import os
import time
from typing import List, Optional, Literal, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Fram Figures API", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Models
# -----------------------------
class KPI(BaseModel):
    label: str
    value: float
    unit: Optional[str] = None


class SummaryResponse(BaseModel):
    period: Literal["yesterday", "week", "month", "year"]
    generated_at: float = Field(default_factory=lambda: time.time())
    kpis: List[KPI]
    mock: bool = False


class BreakdownItem(BaseModel):
    name: str
    boardings: int
    departures: int
    vehicles: int


class BreakdownResponse(BaseModel):
    period: Literal["yesterday", "week", "month", "year"]
    dimension: Literal["line", "station", "train_type"]
    items: List[BreakdownItem]
    mock: bool = False


# -----------------------------
# Power BI helpers (proxy layer)
# -----------------------------
TENANT_ID = os.getenv("POWERBI_TENANT_ID")
CLIENT_ID = os.getenv("POWERBI_CLIENT_ID")
CLIENT_SECRET = os.getenv("POWERBI_CLIENT_SECRET")
SCOPE = os.getenv("POWERBI_SCOPE", "https://analysis.windows.net/powerbi/api/.default")
GROUP_ID = os.getenv("POWERBI_GROUP_ID")
DATASET_ID = os.getenv("POWERBI_DATASET_ID")


def powerbi_available() -> bool:
    return all([TENANT_ID, CLIENT_ID, CLIENT_SECRET, GROUP_ID, DATASET_ID])


def get_access_token() -> Optional[str]:
    if not powerbi_available():
        return None
    token_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": SCOPE,
    }
    try:
        resp = requests.post(token_url, data=data, timeout=15)
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception:
        return None


def execute_dax_query(dax: str) -> Dict[str, Any]:
    """Execute a DAX query against a dataset. Returns raw Power BI response.
    Falls back to empty result on error."""
    token = get_access_token()
    if not token:
        return {}
    url = f"https://api.powerbi.com/v1.0/myorg/groups/{GROUP_ID}/datasets/{DATASET_ID}/executeQueries"
    payload = {"queries": [{"query": dax}]}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# -----------------------------
# Mock generators (for development without secrets)
# -----------------------------

def mock_summary(period: Literal["yesterday", "week", "month", "year"]) -> SummaryResponse:
    base = {
        "yesterday": 120_540,
        "week": 815_320,
        "month": 3_520_900,
        "year": 41_230_000,
    }[period]
    return SummaryResponse(
        period=period,
        kpis=[
            KPI(label="Boardings", value=base, unit="passengers"),
            KPI(label="Departures", value=base * 0.98, unit="trips"),
            KPI(label="Vehicles", value=base * 0.015, unit="units"),
            KPI(label="Utilization rate", value=63.4, unit="%"),
            KPI(label="Bus-for-train share", value=1.8, unit="%"),
            KPI(label="Double-set share", value=27.2, unit="%"),
        ],
        mock=True,
    )


def mock_breakdown(period: Literal["yesterday", "week", "month", "year"], dimension: str) -> BreakdownResponse:
    sample = {
        "line": ["L1", "L2", "L3", "R10", "R11"],
        "station": ["Oslo S", "Lillestrøm", "Asker", "Drammen", "Trondheim"],
        "train_type": ["Local", "Regional", "Long-distance"],
    }[dimension]
    items: List[BreakdownItem] = []
    for i, name in enumerate(sample):
        items.append(
            BreakdownItem(
                name=name,
                boardings=10_000 + i * 2_500,
                departures=9_800 + i * 2_400,
                vehicles=150 + i * 15,
            )
        )
    return BreakdownResponse(period=period, dimension=dimension, items=items, mock=True)


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def root():
    return {"service": "Fram Figures API", "status": "ok"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "powerbi_configured": powerbi_available(),
    }


@app.get("/api/powerbi/status")
def powerbi_status():
    has_secrets = powerbi_available()
    token = get_access_token() if has_secrets else None
    return {
        "configured": has_secrets,
        "token": bool(token),
        "group_id": GROUP_ID is not None,
        "dataset_id": DATASET_ID is not None,
    }


@app.get("/api/metrics/summary", response_model=SummaryResponse)
def metrics_summary(period: Literal["yesterday", "week", "month", "year"] = Query("yesterday")):
    # Placeholder for real DAX. Example only.
    dax = "EVALUATE ROW(\"Boardings\", 0)"  # Replace with real query once fields are agreed
    result = execute_dax_query(dax)
    if not result:
        return mock_summary(period)

    # Parse Power BI result (simplified and defensive)
    try:
        # TODO: map DAX result to KPIs when real fields exist
        return mock_summary(period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Power BI response: {e}")


@app.get("/api/metrics/by_line", response_model=BreakdownResponse)
def metrics_by_line(period: Literal["yesterday", "week", "month", "year"] = Query("yesterday")):
    return mock_breakdown(period, "line")


@app.get("/api/metrics/by_station", response_model=BreakdownResponse)
def metrics_by_station(period: Literal["yesterday", "week", "month", "year"] = Query("yesterday")):
    return mock_breakdown(period, "station")


@app.get("/api/metrics/by_train_type", response_model=BreakdownResponse)
def metrics_by_train_type(period: Literal["yesterday", "week", "month", "year"] = Query("yesterday")):
    return mock_breakdown(period, "train_type")


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    try:
        # Try to import database module
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, "name") else "✅ Connected"
            response["connection_status"] = "Connected"

            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
