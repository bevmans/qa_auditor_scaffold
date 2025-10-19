# app/helpscout_client.py
import os
import time
import random
import urllib.parse
from typing import Dict, Iterable, List, Optional, Tuple, Any

import requests

# ---------- Config ----------
HELPSCOUT_API_BASE = "https://api.helpscout.net/v2"

# Paging & politeness
MAX_PAGES = int(os.getenv("HELPSCOUT_PAGE_DEPTH", "2"))
SLEEP_BETWEEN_PAGES_SEC = float(os.getenv("HELPSCOUT_PAGE_SLEEP", "0.2"))

# HTTP timeouts and retries
HTTP_TIMEOUT = float(os.getenv("HELPSCOUT_HTTP_TIMEOUT", "20"))
MAX_RETRIES = int(os.getenv("HELPSCOUT_MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.getenv("HELPSCOUT_BACKOFF_BASE", "0.5"))   # seconds
BACKOFF_CAP = float(os.getenv("HELPSCOUT_BACKOFF_CAP", "8.0"))     # max delay between retries
BACKOFF_JITTER = float(os.getenv("HELPSCOUT_BACKOFF_JITTER", "0.2"))  # +/- jitter seconds

# OAuth env (used by shims below)
HS_CLIENT_ID = os.getenv("HELPSCOUT_CLIENT_ID") or ""
HS_CLIENT_SECRET = os.getenv("HELPSCOUT_CLIENT_SECRET") or ""
HS_REDIRECT_URI = os.getenv("HELPSCOUT_REDIRECT_URI") or ""
HS_SCOPE = os.getenv("HELPSCOUT_SCOPES", "conversations.read mailboxes.read customers.read")

USER_AGENT = "qa-auditor/1.0 (+http://localhost)"


# ---------- Helpers ----------
def _auth_headers(access_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }


def _allowlisted_mailboxes() -> List[Optional[int]]:
    raw = os.getenv("HELPSCOUT_MAILBOX_ALLOWLIST", "") or ""
    out: List[Optional[int]] = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            pass
    # If empty, sentinel None means "no mailbox filter"
    return out or [None]


def _sleep_backoff(attempt: int, retry_after: Optional[str] = None):
    # Respect Retry-After header if present and valid
    if retry_after:
        try:
            delay = float(retry_after)
            time.sleep(min(delay, BACKOFF_CAP))
            return
        except Exception:
            pass
    # Exponential backoff with jitter
    base = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** max(0, attempt - 1)))
    jitter = random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER)
    time.sleep(max(0.0, base + jitter))


def _request_with_retries(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    timeout: float = HTTP_TIMEOUT,
) -> requests.Response:
    """
    Retry on 429, 5xx, and network errors. Raises on final failure.
    """
    attempt = 0
    last_exc: Optional[Exception] = None

    while attempt <= MAX_RETRIES:
        try:
            resp = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                data=data,
                timeout=timeout,
            )
            # Happy path
            if resp.status_code < 400:
                return resp

            # Retryable?
            if resp.status_code in (429, 500, 502, 503, 504):
                attempt += 1
                if attempt > MAX_RETRIES:
                    break
                _sleep_backoff(attempt, resp.headers.get("Retry-After"))
                continue

            # Non-retryable HTTP errors
            resp.raise_for_status()
            return resp  # pragma: no cover (raise_for_status above)
        except requests.RequestException as e:
            # Network hiccup -> retry
            attempt += 1
            last_exc = e
            if attempt > MAX_RETRIES:
                break
            _sleep_backoff(attempt)

    # Exhausted
    if last_exc:
        raise last_exc
    raise RuntimeError(f"HTTP error after retries: {url}")


def _get_json(
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    timeout: float = HTTP_TIMEOUT,
) -> dict:
    resp = _request_with_retries("GET", url, headers=headers, params=params, timeout=timeout)
    try:
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from {url}: {e}; body={resp.text[:400]}")


# ---------- Public data helpers ----------
def iter_conversations(access_token: str) -> Iterable[dict]:
    """
    Yields recent conversations across allowlisted mailboxes,
    with configurable pagination depth & politeness sleep.
    """
    headers = _auth_headers(access_token)
    mboxes = _allowlisted_mailboxes()

    for mb in mboxes:
        page = 1
        max_pages = max(1, MAX_PAGES)
        while page <= max_pages:
            params: Dict[str, Any] = {
                "page": str(page),
                "sortField": "createdAt",
                "sortOrder": "desc",
                "status": "all",
            }
            if mb is not None:
                params["mailbox"] = str(mb)

            url = f"{HELPSCOUT_API_BASE}/conversations"
            try:
                data = _get_json(url, headers, params)
            except Exception as e:
                print(f"iter_conversations error (mailbox={mb}, page={page}): {e}")
                break

            convs = (data.get("_embedded") or {}).get("conversations") or []
            if not convs:
                break

            for c in convs:
                yield c

            page += 1
            time.sleep(SLEEP_BETWEEN_PAGES_SEC)


def get_threads(access_token: str, conversation_id: int) -> List[dict]:
    """
    Returns a list of threads for the conversation.
    """
    headers = _auth_headers(access_token)
    url = f"{HELPSCOUT_API_BASE}/conversations/{conversation_id}/threads"
    try:
        data = _get_json(url, headers)
    except Exception as e:
        print(f"get_threads error (conv={conversation_id}): {e}")
        return []

    threads = (data.get("_embedded") or {}).get("threads") or []
    return [t for t in threads if isinstance(t, dict)]


# ---------- OAuth helpers ----------
def oauth_authorize_url(client_id: str, redirect_uri: str, scope: str = HS_SCOPE) -> str:
    """
    Build the Help Scout OAuth authorize URL (classic client app flow).
    """
    base = "https://secure.helpscout.net/authentication/authorize"
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
    }
    return f"{base}?{urllib.parse.urlencode(params)}"


def _exchange_code_for_token(client_id: str, client_secret: str, code: str, redirect_uri: str) -> dict:
    token_url = "https://api.helpscout.net/v2/oauth2/token"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    resp = requests.post(token_url, data=data, timeout=30, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.json()


def exchange_code_for_token(code: str):
    """
    Env-driven shim so api_server.py can call exchange_code_for_token(code)
    without passing client credentials explicitly.
    """
    if not (HS_CLIENT_ID and HS_CLIENT_SECRET and HS_REDIRECT_URI):
        return 400, {"error": "missing_client_env",
                    "detail": "HELPSCOUT_CLIENT_ID/SECRET/REDIRECT_URI must be set"}
    try:
        payload = _exchange_code_for_token(HS_CLIENT_ID, HS_CLIENT_SECRET, code, HS_REDIRECT_URI)
        return payload
    except requests.HTTPError as e:
        return e.response.status_code, {"error": "token_exchange_failed", "body": e.response.text}
    except Exception as e:
        return 500, {"error": "token_exchange_exception", "detail": str(e)}


def refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> dict:
    """
    Standard OAuth refresh. Returns the token payload (access_token, refresh_token?, expires_in, etc.)
    Caller is responsible for persisting new tokens.
    """
    token_url = "https://api.helpscout.net/v2/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    resp = requests.post(token_url, data=data, timeout=30, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.json()
