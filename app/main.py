from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

app = FastAPI(title="QA Auditor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# Try to include OAuth routes if available; otherwise keep running.
try:
    from app.routes.oauth import router as oauth_router
    app.include_router(oauth_router, prefix="/oauth")
except Exception as e:
    logging.warning("OAuth router not loaded (%s). API will run without /oauth.* routes.", e)
