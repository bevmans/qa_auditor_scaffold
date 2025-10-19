# app/api_server.py
from __future__ import annotations

import os
import json
import hmac
import hashlib
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional  # <-- added Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy import text

# Local modules
from db import engine, SessionLocal
from helpscout_client import oauth_authorize_url, exchange_code_for_token  # shims provided in helpscout_client.py

# Optional: Redis/RQ (don’t crash app if missing during setup)
try:
    from redis import Redis
    from rq import Queue
except Exception:
    Redis = None
    Queue = None

load_dotenv()

app = FastAPI()

# ---- Config ----
HELPSCOUT_AUTH_URL = "https://secure.helpscout.net/authentication/authorizeClientApplication"
CLIENT_ID = os.getenv("HELPSCOUT_CLIENT_ID")
WEBHOOK_SECRET = os.getenv("HELPSCOUT_WEBHOOK_SECRET", "")
POST_LOGIN_REDIRECT = os.getenv("HELPSCOUT_POST_LOGIN_REDIRECT", "http://localhost:8502/?oauth=helpscout_success")

# Queue (optional)
q = None
if Redis and Queue:
    try:
        redis = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
        q = Queue("qa_jobs", connection=redis)
    except Exception:
        q = None

# Import job helper (late import so missing rq/redis won’t crash)
try:
    from worker import queue_import_job  # noqa
except Exception:
    def queue_import_job(_q):  # fallback no-op
        return None


@app.get("/health")
def health():
    return {"ok": True}


# ----------------------------
# OAuth: START (hotfix & safe)
# ----------------------------
@app.get("/oauth/start")
def oauth_start():
    """
    Minimal, uncrashable OAuth start:
    - Uses current Help Scout endpoint
    - Only includes client_id (no legacy params)
    - Never touches DB
    """
    if not CLIENT_ID:
        raise HTTPException(status_code=500, detail="HELPSCOUT_CLIENT_ID missing")

    # Build the new-style URL directly (avoid helper signature differences)
    url = f"{HELPSCOUT_AUTH_URL}?client_id={CLIENT_ID}"
    return RedirectResponse(url, status_code=302)

# ------------------------------
# OAuth: CALLBACK (token save)
# ------------------------------
@app.get("/oauth/callback")
def oauth_callback(code: str, state: str | None = None):
    # (Optional) if you later add a signed state, you can verify it here.

    """
    Exchange code → tokens, upsert account, then kick off import.
    """
    # Handle both (status, json) and plain dict returns
    token_payload = exchange_code_for_token(code)
    if isinstance(token_payload, tuple) and len(token_payload) == 2:
        status, token = token_payload
        if status != 200:
            return JSONResponse({"error": "token_exchange_failed", "detail": token}, status_code=400)
    else:
        token = token_payload

    access = token.get("access_token")
    if not access:
        return JSONResponse({"error": "missing_access_token", "detail": token}, status_code=400)

    refresh = token.get("refresh_token")
    expires_in = int(token.get("expires_in", 3600))
    expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

    # Ensure table exists (lightweight)
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS hs_accounts (
                    id SERIAL PRIMARY KEY,
                    hs_account_id TEXT UNIQUE,
                    hs_email TEXT,
                    access_token TEXT,
                    refresh_token TEXT,
                    token_expires_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """))
    except Exception:
        # Non-fatal: still redirect so the user sees success; diagnostics in logs
        pass

    # Upsert account
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO hs_accounts (hs_account_id, hs_email, access_token, refresh_token, token_expires_at, updated_at)
            VALUES (:aid, :email, :at, :rt, :exp, NOW())
            ON CONFLICT (hs_account_id) DO UPDATE SET
              access_token      = EXCLUDED.access_token,
              refresh_token     = EXCLUDED.refresh_token,
              token_expires_at  = EXCLUDED.token_expires_at,
              updated_at        = NOW()
        """), {
            "aid": token.get("user_id") or "default",
            "email": token.get("email"),
            "at": access,
            "rt": refresh,
            "exp": expires_at,
        })

    # Kick off initial import (if queue available)
    try:
        if q is not None:
            queue_import_job(q)
    except Exception:
        pass

    return RedirectResponse(url=POST_LOGIN_REDIRECT, status_code=302)


# -------------------
# Webhook (optional)
# -------------------
def verify_signature(req: Request, body: bytes) -> bool:
    if not WEBHOOK_SECRET:
        return True
    sig = req.headers.get("X-Helpscout-Signature", "")
    mac = hmac.new(WEBHOOK_SECRET.encode(), msg=body, digestmod=hashlib.sha1).hexdigest()
    return hmac.compare_digest(sig, mac)

@app.post("/webhooks/helpscout")
async def helpscout_webhook(request: Request):
    body = await request.body()
    if not verify_signature(request, body):
        raise HTTPException(status_code=401, detail="Invalid signature")
    # Minimal handler: queue an import on any relevant event
    try:
        if q is not None:
            queue_import_job(q)
    except Exception:
        pass
    return JSONResponse({"queued": True})


# ---------------------------------------------------
# Per-Agent rollups filtered by HelpScout date (with optional agent filter)
# ---------------------------------------------------
def _as_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

@app.get("/rollups/agents")
def per_agent_rollups(
    start: datetime = Query(..., description="UTC start (inclusive)"),
    end:   datetime = Query(..., description="UTC end (exclusive)"),
    agent: Optional[str] = Query(None, description="Optional agent label (COALESCE(agent_name, author, agent_email))"),
):
    """
    Returns per-agent rollups using messages.hs_created_at (the original HelpScout message date).
    If ?agent= is provided, results are scoped to that one agent.
    Response shape: [{ agent, day, scored_cnt, avg_score }]
    """
    start_utc = _as_utc(start)
    end_utc   = _as_utc(end)

    where = ["m.hs_created_at >= :start", "m.hs_created_at < :end"]
    params: Dict[str, Any] = {"start": start_utc, "end": end_utc}

    if agent:
        # Match the UI label used in the selectbox (COALESCE(agent_name, author))
        where.append("COALESCE(m.agent_name, m.author, m.agent_email) = :agent")
        params["agent"] = agent

    sql = f"""
    SELECT
      COALESCE(m.agent_name, m.author, m.agent_email) AS agent,
      date_trunc('day', m.hs_created_at)              AS day,
      COUNT(*)                                        AS scored_cnt,
      AVG(q.total)::float                             AS avg_score
    FROM qa_scores q
    JOIN messages m ON m.hs_message_id = q.hs_message_id
    WHERE {' AND '.join(where)}
    GROUP BY 1,2
    ORDER BY 2 DESC, 1;
    """

    rows: List[Dict[str, Any]] = []
    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        for r in result.fetchall():
            rows.append({
                "agent": r[0],
                "day": r[1].date().isoformat() if r[1] else None,
                "scored_cnt": int(r[2]),
                "avg_score": float(r[3]) if r[3] is not None else None,
            })

    return rows
