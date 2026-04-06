from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import engine, Base
from routers import shipments, risk

app = FastAPI(
    title="ChainGuard API",
    version="0.1.0",
    description=(
        "Cargo Risk Intelligence Platform — backend API for shipment tracking "
        "and ML-powered risk scoring."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)


app.include_router(shipments.router, prefix="/api/v1")
app.include_router(risk.router, prefix="/api/v1")


@app.get("/", tags=["health"])
def health_check():
    return {"status": "ok", "service": "ChainGuard API"}