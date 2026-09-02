from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import alerts, dashboard, risk_score, works

app = FastAPI(
    title="MPLADS Anomaly Detection API",
    description="SIH26102 — AI-powered anomaly/fraud detection for MPLADS scheme implementation.",
    version="0.1.0",
)

# Wide open for hackathon dev; tighten before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(works.router)
app.include_router(risk_score.router)
app.include_router(alerts.router)
app.include_router(dashboard.router)


@app.get("/")
def root():
    return {"status": "ok", "service": "mplads-anomaly-api"}


@app.get("/health")

def health():
    return {"status": "healthy"}