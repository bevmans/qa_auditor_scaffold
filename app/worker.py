import os, json
from datetime import datetime, timedelta

from dotenv import load_dotenv
from sqlalchemy import text
from db import engine
from helpscout_client import iter_conversations, get_threads, refresh_access_token
from scoring import score_message

# parse HelpScout timestamps from API
from dateutil import parser as dateparser  # requires python-dateutil in requirements

load_dotenv()

DEFAULT_WEIGHTS = {"empathy": 0.25, "accuracy": 0.35, "tone": 0.15, "resolution": 0.25}
SKIP_THREAD_TYPES = {"lineitem", "system", "notification"}  # never store/score these

# Token refresh skew (minutes). If token expires before now + skew, refresh first.
TOKEN_REFRESH_SKEW_MIN = int(os.getenv("HELPSCOUT_TOKEN_REFRESH_SKEW_MIN", "10"))

def get_weights():
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT v FROM kv_store WHERE k='rubric_weights'")).first()
            if row and row[0]:
                w = json.loads(row[0])
                return {**DEFAULT_WEIGHTS, **w}
    except Exception as e:
        print("Weights fetch error:", e)
    return DEFAULT_WEIGHTS

def _allowlisted_mailboxes():
    raw = os.getenv("HELPSCOUT_MAILBOX_ALLOWLIST", "")
    ids = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            ids.append(int(p))
        except Exception:
            pass
    return set(ids)

def _parse_hs_ts(iso_str: str | None):
    if not iso_str:
        return None
    try:
        return dateparser.isoparse(iso_str)  # aware dt (UTC)
    except Exception:
        return None

def _insert_message_dynamic(hs_id, msg_id, author_type, body_plain, created_when, agent_name, agent_email, hs_created_at):
    """
    Insert/upsert into messages table trying both possible schemas:
      A) created_at column present
      B) created column present
    Conflict target: hs_message_id (unique)
    """
    # First attempt with created_at
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO messages
                    (hs_conversation_id, hs_message_id, author, body_plain, created_at, agent_name, agent_email, hs_created_at)
                VALUES
                    (:cid, :mid, :author, :plain, :created_at, :aname, :aemail, :hs_created_at)
                ON CONFLICT (hs_message_id) DO UPDATE SET
                    author       = COALESCE(EXCLUDED.author, messages.author),
                    body_plain   = COALESCE(EXCLUDED.body_plain, messages.body_plain),
                    agent_name   = COALESCE(EXCLUDED.agent_name, messages.agent_name),
                    agent_email  = COALESCE(EXCLUDED.agent_email, messages.agent_email),
                    hs_created_at= COALESCE(EXCLUDED.hs_created_at, messages.hs_created_at)
            """), {
                "cid": hs_id,
                "mid": msg_id,
                "author": author_type,
                "plain": body_plain,
                "created_at": created_when,
                "aname": agent_name,
                "aemail": agent_email,
                "hs_created_at": hs_created_at,
            })
        return True
    except Exception as e1:
        # Second attempt: created (older schema)
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO messages
                        (hs_conversation_id, hs_message_id, author, body_plain, created, agent_name, agent_email, hs_created_at)
                    VALUES
                        (:cid, :mid, :author, :plain, :created, :aname, :aemail, :hs_created_at)
                    ON CONFLICT (hs_message_id) DO UPDATE SET
                        author       = COALESCE(EXCLUDED.author, messages.author),
                        body_plain   = COALESCE(EXCLUDED.body_plain, messages.body_plain),
                        agent_name   = COALESCE(EXCLUDED.agent_name, messages.agent_name),
                        agent_email  = COALESCE(EXCLUDED.agent_email, messages.agent_email),
                        hs_created_at= COALESCE(EXCLUDED.hs_created_at, messages.hs_created_at)
                """), {
                    "cid": hs_id,
                    "mid": msg_id,
                    "author": author_type,
                    "plain": body_plain,
                    "created": created_when,
                    "aname": agent_name,
                    "aemail": agent_email,
                    "hs_created_at": hs_created_at,
                })
            return True
        except Exception as e2:
            print(f"Message upsert failed (mid={msg_id}). created_at err: {e1}; created err: {e2}")
            return False

def _process_conversation(hs_id: int, access_token: str, weights: dict, counters: dict):
    """Fetch threads for a single conversation and insert/score messages."""
    try:
        threads = get_threads(access_token, hs_id) or []
    except Exception as e:
        print(f"Thread fetch error (conv={hs_id}):", e)
        return

    print(f"conv {hs_id}: fetched {len(threads)} threads -> types: {[ (t.get('type') or '').lower() for t in threads[:10] ]}")

    for th in threads:
        th_type = (th.get("type") or "").lower()
        if th_type in SKIP_THREAD_TYPES:
            continue

        try:
            msg_id = int(th.get("id"))
        except Exception:
            continue

        created_by = th.get("createdBy") or {}
        author_type = (created_by.get("type") or "unknown").lower()  # "user" or "customer" | unknown

        body_html = th.get("body", "") or ""
        body_plain = th.get("text", "") or body_html
        created_msg_at = th.get("createdAt")
        hs_ts = _parse_hs_ts(created_msg_at)

        person = created_by.get("person") or created_by
        display = person.get("displayName") or " ".join(
            [person.get("first", "") or "", person.get("last", "") or ""]
        ).strip() or author_type
        email = person.get("email")

        ok = _insert_message_dynamic(
            hs_id=hs_id,
            msg_id=msg_id,
            author_type=author_type,
            body_plain=body_plain,
            created_when=created_msg_at,             # preserve original str if your schema keeps text, else parse
            agent_name=display if author_type == "user" else None,
            agent_email=email if author_type == "user" else None,
            hs_created_at=hs_ts,
        )
        if ok:
            counters["msg_new"] += 1

        # Score only staff/agent messages
        if author_type != "user":
            continue

        try:
            s = score_message(body_plain or body_html)
            e = float(s.get("empathy", 0))
            a = float(s.get("accuracy", 0))
            t = float(s.get("tone", 0))
            r = float(s.get("resolution", 0))
            total = (
                weights["empathy"] * e
                + weights["accuracy"] * a
                + weights["tone"] * t
                + weights["resolution"] * r
            )

            with engine.begin() as conn:
                # idempotent upsert via delete+insert
                conn.execute(text("DELETE FROM qa_scores WHERE hs_message_id = :mid"), {"mid": msg_id})
                conn.execute(text("""
                    INSERT INTO qa_scores
                        (hs_message_id, hs_conversation_id, empathy, accuracy, tone, resolution, total, reasoning)
                    VALUES
                        (:mid, :cid, :e, :a, :t, :r, :tot, :why)
                """), {
                    "mid": msg_id, "cid": hs_id,
                    "e": e, "a": a, "t": t, "r": r,
                    "tot": total, "why": s.get("reasoning", "")
                })
            counters["scored"] += 1
        except Exception as e:
            print(f"Scoring failed (mid={msg_id}):", e)

def _maybe_refresh_token(access_token: str, refresh_token: str | None, token_expires_at) -> str:
    """
    If the token expires within TOKEN_REFRESH_SKEW_MIN minutes, refresh and persist.
    Returns a valid access token (refreshed or original).
    """
    if not refresh_token or not token_expires_at:
        return access_token

    try:
        exp = token_expires_at
        if isinstance(exp, str):
            # If stored as string, best-effort parse
            exp = dateparser.isoparse(exp)
    except Exception:
        exp = None

    skew_dt = datetime.utcnow() + timedelta(minutes=TOKEN_REFRESH_SKEW_MIN)
    if exp and exp > skew_dt:
        return access_token  # still fresh enough

    client_id = os.getenv("HELPSCOUT_CLIENT_ID")
    client_secret = os.getenv("HELPSCOUT_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("Token nearing expiry but missing HELPSCOUT_CLIENT_ID/SECRET; skipping refresh.")
        return access_token

    try:
        new = refresh_access_token(client_id, client_secret, refresh_token)
        new_access = new.get("access_token") or access_token
        new_refresh = new.get("refresh_token") or refresh_token
        expires_in = int(new.get("expires_in", 3600))
        expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE hs_accounts
                SET access_token=:at, refresh_token=:rt, token_expires_at=:exp, updated_at=NOW()
                WHERE id = (
                  SELECT id FROM hs_accounts ORDER BY updated_at DESC LIMIT 1
                )
            """), {"at": new_access, "rt": new_refresh, "exp": expires_at})

        print("HelpScout access token refreshed.")
        return new_access
    except Exception as e:
        print("Token refresh failed; proceeding with existing token:", e)
        return access_token

def import_and_score():
    # latest token
    with engine.begin() as conn:
        row = conn.execute(text("SELECT id, access_token, refresh_token, token_expires_at FROM hs_accounts ORDER BY updated_at DESC LIMIT 1")).first()
        if not row:
            print("No HS account connected.")
            return
        account_id, access_token, refresh_token, expires_at = row

    # refresh if needed
    access_token = _maybe_refresh_token(access_token, refresh_token, expires_at)

    weights = get_weights()
    allowlist = _allowlisted_mailboxes()
    counters = {"convos": 0, "msg_new": 0, "scored": 0}

    # 1) API iterator (paged)
    yielded_any = False
    for convo in iter_conversations(access_token):
        yielded_any = True
        try:
            hs_id = int(convo["id"])
        except Exception:
            continue

        mailbox_id = convo.get("mailboxId")
        if allowlist and mailbox_id not in allowlist:
            continue

        subject = convo.get("subject")
        email = (convo.get("primaryCustomer") or {}).get("email")
        status = convo.get("status")
        created_at = convo.get("createdAt")

        # upsert conversation; try created_at, then created (schema differences)
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO conversations
                        (hs_conversation_id, mailbox_id, subject, customer_email, status, created_at, updated_at)
                    VALUES
                        (:id, :mb, :subj, :email, :status, :created_at, NOW())
                    ON CONFLICT (hs_conversation_id) DO UPDATE SET
                        subject = EXCLUDED.subject,
                        customer_email = COALESCE(EXCLUDED.customer_email, conversations.customer_email),
                        status = EXCLUDED.status,
                        updated_at = NOW()
                """), {
                    "id": hs_id,
                    "mb": mailbox_id,
                    "subj": subject,
                    "email": email,
                    "status": status,
                    "created_at": created_at,
                })
            counters["convos"] += 1
        except Exception as e1:
            try:
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO conversations
                            (hs_conversation_id, mailbox_id, subject, customer_email, status, created, last_updated)
                        VALUES
                            (:id, :mb, :subj, :email, :status, :created, NOW())
                        ON CONFLICT (hs_conversation_id) DO UPDATE SET
                            subject = EXCLUDED.subject,
                            customer_email = COALESCE(EXCLUDED.customer_email, conversations.customer_email),
                            status = EXCLUDED.status,
                            last_updated = NOW()
                    """), {
                        "id": hs_id,
                        "mb": mailbox_id,
                        "subj": subject,
                        "email": email,
                        "status": status,
                        "created": created_at,
                    })
                counters["convos"] += 1
            except Exception as e2:
                print(f"Conversation upsert failed (id={hs_id}). created_at err: {e1}; created err: {e2}")

        _process_conversation(hs_id, access_token, weights, counters)

    # 2) Fallback: if nothing yielded, process recent DB convos
    if not yielded_any:
        print("iter_conversations returned 0; falling back to recent conversations from DB…")
        mailbox_filter_sql = ""
        params = {}
        if allowlist:
            mailbox_filter_sql = "WHERE mailbox_id = ANY(:allowlist)"
            params["allowlist"] = list(allowlist)

        rows = []
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(f"""
                    SELECT hs_conversation_id
                    FROM conversations
                    {mailbox_filter_sql}
                    ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
                    LIMIT 50
                """), params).all()
        except Exception as e:
            print("Fallback DB query failed:", e)
            rows = []

        for (hs_id,) in rows:
            _process_conversation(int(hs_id), access_token, weights, counters)

    print(
        "Import + scoring finished. "
        f"Conversations upserted: {counters['convos']}, "
        f"messages added/updated: {counters['msg_new']}, "
        f"scored: {counters['scored']}"
    )

def queue_import_job(q):
    try:
        from redis import Redis
        from rq import Queue
        redis = Redis.from_url(os.getenv("REDIS_URL"))
        Queue("qa_jobs", connection=redis).enqueue(import_and_score, job_timeout=60*15)
    except Exception as e:
        print("Failed to enqueue job via RQ:", e)

if __name__ == "__main__":
    import_and_score()
