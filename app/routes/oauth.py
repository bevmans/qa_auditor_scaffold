# app/routes/oauth.py
from __future__ import annotations

import base64
import hmac
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy import create_engine, text

# --- Load .env early and reliably ---
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

router = APIRouter()

# --- Help Scout OAuth (current endpoints) ---
# Docs: https://developer.helpscout.com/mailbox-api/overview/authentication
HELPSCOUT_AUTH_URL = "https://secure.helpscout.net/authentication/authorizeClientApplication"
HELPSCOUT_TOKEN_URL = "https://api.helpscout.net/v2/oauth2/token"

# --- Env helpers ---
def env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    return val if (val is not None and str(val).strip() != "") else default

CLIENT_ID = env("HELPSCOUT_CLIENT_ID")
CLIENT_SECRET = env("HELPSCOUT_CLIENT_SECRET")
REDIRECT_URI = env("HELPSCOUT_REDIRECT_URI", "http://localhost:8001/oauth/callback")

# Where to send the user after we finish the callback successfully.
# Defaults to Streamlit in Docker (8502). Override if you’re running locally.
POST_LOGIN_REDIRECT = env("HELPSCOUT_POST_LOGIN_REDIRECT", "http://localhost:8502/?oauth=helpscout_success")

# HMAC secret for signing the 'state' (stateless CSRF protection)
SESSION_SECRET = env("SESSION_SECRET", "dev-secret-change-me")

# --- DB (lazy) ---
def get_engine():
    url = env("DATABASE_URL")
    if not url:
        return None
    try:
        return create_engine(url, future=True, pool_pre_ping=True)
    except Exception:
        return None

def ensure_tables():
    engine = get_engine()
    if not engine:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS oauth_tokens (
                id SERIAL PRIMARY KEY,
                provider TEXT NOT NULL,
                access_token TEXT NOT NULL,
                refresh_token TEXT,
                token_type TEXT,
                scope TEXT,
                expires_in INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """))
        # Minimal table so Streamlit sidebar query works (it reads hs_email, token_expires_at)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS hs_accounts (
                id SERIAL PRIMARY KEY,
                hs_email TEXT,
                token_expires_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """))

# --- Stateless 'state' helpers (no Redis required) ---
def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

def _b64u_dec(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode())

def sign_state(payload: dict, ttl_seconds: int = 600) -> str:
    payload = dict(payload or {})
    payload["ts"] = int(time.time())
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    sig = hmac.new(SESSION_SECRET.encode(), body, hashlib.sha256).digest()
    return _b64u(body) + "." + _b64u(sig)

def verify_state(state: str, max_age: int = 900) -> dict:
    try:
        body_b64, sig_b64 = state.split(".", 1)
        body = _b64u_dec(body_b64)
        exp_sig = _b64u_dec(sig_b64)
        got_sig = hmac.new(SESSION_SECRET.encode(), body, hashlib.sha256).digest()
        if not hmac.compare_digest(exp_sig, got_sig):
            raise ValueError("bad signature")
        payload = json.loads(body.decode())
        ts = int(payload.get("ts", 0))
        if int(time.time()) - ts > max_age:
            raise ValueError("state expired")
        return payload
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid state: {e}")

# --- Routes ---
@router.get("/start", summary="Oauth Start")
def oauth_start():
    if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
        raise HTTPException(500, "OAuth env vars missing (HELPSCOUT_CLIENT_ID/SECRET/REDIRECT_URI)")

    # Ensure tables lazily (won't crash import if DB is down)
    ensure_tables()

    # build signed 'state'
    state = sign_state({"nonce": _b64u(os.urandom(8))})

    # Per Help Scout docs, use authorizeClientApplication and only pass client_id (+ optional state)
    params = {
        "client_id": CLIENT_ID,
        "state": state,
    }
    url = f"{HELPSCOUT_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url)

@router.get("/start2", summary="Oauth Start (v2)")
def oauth_start_v2():
    if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
        raise HTTPException(500, "OAuth env vars missing (HELPSCOUT_CLIENT_ID/SECRET/REDIRECT_URI)")

    ensure_tables()
    state = sign_state({"nonce": _b64u(os.urandom(8))})
    params = {"client_id": CLIENT_ID, "state": state}
    url = f"{HELPSCOUT_AUTH_URL}?{urlencode(params)}"

    # helpful log
    import logging
    logging.info("HS authorize URL (v2): %s", url)

    return RedirectResponse(url)

@router.get("/callback", summary="Oauth Callback")
async def oauth_callback(request: Request):
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    if not code or not state:
        raise HTTPException(422, "Missing code/state")

    # Verify state (stateless)
    _ = verify_state(state)

    # Exchange the authorization code for tokens
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,   # must match the one set in My Apps
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(HELPSCOUT_TOKEN_URL, data=data)
    if resp.status_code != 200:
        # Bubble the exact error back for debugging during setup
        return JSONResponse(
            status_code=400,
            content={"error": "token_exchange_failed", "detail": resp.text},
        )

    token = resp.json()
    access_token = token.get("access_token")
    refresh_token = token.get("refresh_token")
    token_type = token.get("token_type")
    scope = token.get("scope")
    expires_in = token.get("expires_in")

    # Persist tokens if DB is available; otherwise, just continue the flow.
    engine = get_engine()
    if engine:
        ensure_tables()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO oauth_tokens (provider, access_token, refresh_token, token_type, scope, expires_in)
                VALUES (:provider, :access_token, :refresh_token, :token_type, :scope, :expires_in)
            """), {
                "provider": "helpscout",
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": token_type,
                "scope": scope,
                "expires_in": expires_in,
            })
            # Touch hs_accounts so Streamlit can display "connected"
            # We don't fetch the email here (keeps this path robust); Streamlit shows 'unknown' if null.
            conn.execute(text("""
                INSERT INTO hs_accounts (hs_email, token_expires_at, updated_at)
                VALUES (NULL, NOW() + (INTERVAL '1 second' * :ttl), NOW())
            """), {"ttl": int(expires_in or 0)})

    # Finally bounce to the UI with success flag
    return RedirectResponse(POST_LOGIN_REDIRECT)