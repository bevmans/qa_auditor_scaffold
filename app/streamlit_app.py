# app/streamlit_app.py
import os
import json
import hmac
import hashlib
import datetime as dt
from pathlib import Path
from decimal import Decimal
from typing import List, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Optional: only needed if you press "Test OpenAI"
try:
    from openai import OpenAI  # noqa: F401
except Exception:
    OpenAI = None

# Optional: used by Job Status + toast polling
try:
    from redis import Redis
    from rq import Queue
    from rq.registry import StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry
except Exception:
    Redis = Queue = StartedJobRegistry = FinishedJobRegistry = FailedJobRegistry = None

# --- Load .env from project root ---
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# --- Environment ---
BACKEND_PORT = os.getenv("BACKEND_PORT", "8001")
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Version / build info (shown under Diagnostics)
APP_VERSION = os.getenv("APP_VERSION", "").strip()
GIT_SHA = os.getenv("GIT_SHA", "").strip()
BUILD_TIME = os.getenv("BUILD_TIME", "").strip()

def _version_display() -> str:
    parts = []
    if APP_VERSION:
        parts.append(f"v{APP_VERSION}")
    if GIT_SHA:
        parts.append(GIT_SHA[:7])
    if BUILD_TIME:
        parts.append(BUILD_TIME)
    if not parts:
        parts = [dt.datetime.utcnow().strftime("build-%Y%m%d-%H%MUTC")]
    return " • ".join(parts)

# API base:
# - Inside Docker: web->api via http://api:8000 (server-side calls use this)
# - From browser/host: localhost:8001 (links/buttons use this)
API_BASE = os.getenv("API_BASE", "http://api:8000")
PUBLIC_API_BASE = os.getenv("PUBLIC_API_BASE", f"http://localhost:{BACKEND_PORT}")

HS_WEBHOOK_SECRET = os.getenv("HELPSCOUT_WEBHOOK_SECRET", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# --- Page config early ---
st.set_page_config(page_title="QA Auditor", layout="wide")

# --- DB engine (guard nicely) ---
if not DATABASE_URL:
    st.error("DATABASE_URL is not set. Please add it to your .env and restart the app.")
    st.stop()
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

# Ensure tiny KV table exists (used by rubric weights)
try:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS kv_store (
                k TEXT PRIMARY KEY,
                v TEXT
            );
        """))
except Exception as e:
    st.error(f"Database not reachable / migration failed: {e}")
    st.stop()

# ---- Small utils ----
def d2f(x):
    if x is None:
        return None
    return float(x) if isinstance(x, Decimal) else x

DEFAULT_WEIGHTS = {"empathy": 0.25, "accuracy": 0.35, "tone": 0.15, "resolution": 0.25}

def get_weights():
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT v FROM kv_store WHERE k='rubric_weights'")).first()
            if not row or not row[0]:
                return DEFAULT_WEIGHTS.copy()
            return {**DEFAULT_WEIGHTS, **json.loads(row[0])}
    except Exception:
        return DEFAULT_WEIGHTS.copy()

def set_weights(w: dict):
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO kv_store (k, v) VALUES ('rubric_weights', :v) "
                 "ON CONFLICT (k) DO UPDATE SET v=:v"),
            {"v": json.dumps(w)}
        )

def post_signed_webhook(url: str, payload: dict, secret: str):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if secret:
        sig = hmac.new(secret.encode("utf-8"), msg=body, digestmod=hashlib.sha1).hexdigest()
        headers["X-Helpscout-Signature"] = sig
    return requests.post(url, data=body, headers=headers, timeout=15)

def _humanize_remaining(exp_dt: dt.datetime | None) -> str | None:
    """Return a friendly 'time left' string until exp_dt in UTC, or None."""
    if not exp_dt:
        return None
    if isinstance(exp_dt, str):
        try:
            exp_dt = pd.to_datetime(exp_dt).to_pydatetime()
        except Exception:
            return None
    now = dt.datetime.utcnow()
    delta = exp_dt - now
    secs = int(delta.total_seconds())
    if secs <= 0:
        return f"expired on {exp_dt:%Y-%m-%d %H:%M} UTC"
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, _ = divmod(rem, 60)
    parts = []
    if days: parts.append(f"{days}d")
    if hours: parts.append(f"{hours}h")
    if mins or not parts: parts.append(f"{mins}m")
    return f"{' '.join(parts)} left (until {exp_dt:%Y-%m-%d %H:%M} UTC)"

def _as_date(x):
    if isinstance(x, dt.datetime):
        return x.date()
    try:
        return x.to_pydatetime().date()
    except Exception:
        pass
    if isinstance(x, dt.date):
        return x
    return dt.datetime.utcnow().date()

# -------- Health chip (cached) --------
@st.cache_data(ttl=10)
def _api_health() -> Tuple[bool, str]:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        ok = (r.status_code == 200 and (r.json() or {}).get("ok", False))
        return ok, "OK" if ok else f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)

# -------- Cached lookups / queries --------
@st.cache_data(ttl=60)
def _get_agents() -> List[str]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT DISTINCT COALESCE(agent_name, author) AS agent "
                "FROM messages "
                "WHERE agent_name IS NOT NULL OR author IS NOT NULL "
                "ORDER BY agent"
            )).all()
        return [r[0] for r in rows]
    except Exception:
        return []

@st.cache_data(ttl=60)
def _get_mailboxes() -> List[int]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT DISTINCT mailbox_id FROM conversations WHERE mailbox_id IS NOT NULL ORDER BY 1"
            )).all()
        return [int(r[0]) for r in rows]
    except Exception:
        return []

@st.cache_data(ttl=60)
def _get_kpis(agent: str, start_dt: dt.datetime, end_dt_excl: dt.datetime, mb_ids: Tuple[int, ...]):
    where = ["m.hs_created_at >= :start", "m.hs_created_at < :end"]
    params = {"start": start_dt, "end": end_dt_excl}
    if agent != "All":
        where.append("(COALESCE(m.agent_name, m.author) = :agent)")
        params["agent"] = agent
    if mb_ids:
        where.append("c.mailbox_id = ANY(:mbox)")
        params["mbox"] = list(mb_ids)

    sql = f"""
        SELECT
          COUNT(DISTINCT m.hs_message_id) AS messages_scored,
          ROUND(AVG(s.total), 2)          AS avg_total,
          ROUND(AVG(s.empathy), 2)        AS empathy,
          ROUND(AVG(s.accuracy), 2)       AS accuracy,
          ROUND(AVG(s.tone), 2)           AS tone,
          ROUND(AVG(s.resolution), 2)     AS resolution
        FROM qa_scores s
        JOIN messages m      ON m.hs_message_id = s.hs_message_id
        JOIN conversations c ON c.hs_conversation_id = s.hs_conversation_id
        WHERE {' AND '.join(where)}
    """
    with engine.connect() as conn:
        return conn.execute(text(sql), params).mappings().first()

@st.cache_data(ttl=60)
def _get_rollups(agent: str, start_dt: dt.datetime, end_dt_excl: dt.datetime, mb_ids: Tuple[int, ...]) -> pd.DataFrame:
    """
    Local rollups (DB), so mailbox filter is supported.
    """
    where = ["m.hs_created_at >= :start", "m.hs_created_at < :end", "m.agent_email IS NOT NULL"]
    params = {"start": start_dt, "end": end_dt_excl}
    if agent != "All":
        where.append("(COALESCE(m.agent_name, m.author) = :agent)")
        params["agent"] = agent
    if mb_ids:
        where.append("c.mailbox_id = ANY(:mbox)")
        params["mbox"] = list(mb_ids)

    sql = f"""
        SELECT
          COALESCE(m.agent_name, m.author)         AS agent,
          date_trunc('day', m.hs_created_at)::date AS day,
          COUNT(*)                                  AS scored_cnt,
          AVG(s.total)                              AS avg_score
        FROM qa_scores s
        JOIN messages m      ON m.hs_message_id = s.hs_message_id
        JOIN conversations c ON c.hs_conversation_id = s.hs_conversation_id
        WHERE {' AND '.join(where)}
        GROUP BY 1,2
        ORDER BY day DESC, agent ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    return df

@st.cache_data(ttl=60)
def _get_recent(agent: str, start_dt: dt.datetime, end_dt_excl: dt.datetime, mb_ids: Tuple[int, ...], staff_only: bool) -> pd.DataFrame:
    where = ["m.hs_created_at >= :start AND m.hs_created_at < :end"]
    params = {"start": start_dt, "end": end_dt_excl}
    if agent != "All":
        where.append("(COALESCE(m.agent_name, m.author) = :agent)")
        params["agent"] = agent
    if mb_ids:
        where.append("c.mailbox_id = ANY(:mbox)")
        params["mbox"] = list(mb_ids)
    if staff_only:
        where.append("m.author = 'user'")  # staff/agent messages

    sql = f"""
        SELECT
            m.hs_created_at,
            COALESCE(m.agent_name, m.author) AS agent,
            s.total, s.empathy, s.accuracy, s.tone, s.resolution,
            c.mailbox_id, c.subject, c.customer_email,
            m.author, m.body_plain
        FROM qa_scores s
        JOIN messages m      ON m.hs_message_id = s.hs_message_id
        JOIN conversations c ON c.hs_conversation_id = s.hs_conversation_id
        WHERE {' AND '.join(where)}
        ORDER BY m.hs_created_at DESC NULLS LAST
        LIMIT 500
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    return df

# ---- Page header + health chip ----
st.title("QA Auditor Dashboard")
ok, msg = _api_health()
chip = "🟢 API healthy" if ok else "🔴 API down"
st.caption(f"{chip} — {_version_display()}")

# --- OAuth success feedback ---
if st.query_params.get("oauth") == "helpscout_success":
    st.success("✅ HelpScout connected successfully! Tokens saved to Postgres.")

# --- HelpScout connect button (single source of truth) ---
if st.button("Connect HelpScout"):
    st.markdown(f"[Start OAuth]({PUBLIC_API_BASE}/oauth/start)")

# ======== Sidebar ========
with st.sidebar:
    st.header("Connections")
    try:
        with engine.connect() as conn:
            acc = conn.execute(
                text("SELECT hs_email, token_expires_at FROM hs_accounts ORDER BY updated_at DESC LIMIT 1")
            ).first()
    except Exception:
        acc = None

    if acc:
        hs_email, token_exp = acc[0], acc[1]
        st.success(f"Help Scout: connected as {hs_email or 'unknown'}")

        status = "Connected"
        exp_display = None
        try:
            exp_display = _humanize_remaining(token_exp)
            if token_exp:
                if isinstance(token_exp, str):
                    try:
                        token_exp = pd.to_datetime(token_exp).to_pydatetime()
                    except Exception:
                        token_exp = None
                if token_exp:
                    remaining = (token_exp - dt.datetime.utcnow()).total_seconds()
                    if remaining < 0:
                        status = "Expired"
                    elif remaining < 3600:
                        status = "Expiring soon"
        except Exception:
            pass

        st.caption(f"OAuth status: **{status}**")
        if exp_display:
            st.caption(f"Token validity: _{exp_display}_")
        st.link_button("Reconnect Help Scout", f"{PUBLIC_API_BASE}/oauth/start")
    else:
        st.link_button("Connect Help Scout", f"{PUBLIC_API_BASE}/oauth/start")

    st.header("Jobs")
    if st.button("Queue Import + Score"):
        try:
            finished_before = None
            rtmp = None
            if Redis and FinishedJobRegistry:
                try:
                    rtmp = Redis.from_url(REDIS_URL)
                    finished_reg = FinishedJobRegistry("qa_jobs", connection=rtmp)
                    finished_before = len(finished_reg)
                except Exception:
                    finished_before = None

            # IMPORTANT: server-side call must use API_BASE (http://api:8000), not PUBLIC_API_BASE
            r = post_signed_webhook(f"{API_BASE}/webhooks/helpscout", {"manual": True}, HS_WEBHOOK_SECRET)
            r.raise_for_status()
            st.success("Queued import job.")

            if finished_before is not None:
                import time
                for _ in range(20):
                    time.sleep(1)
                    try:
                        finished_reg = FinishedJobRegistry("qa_jobs", connection=rtmp)
                        if len(finished_reg) > finished_before:
                            st.toast("✅ Import finished", icon="✅")
                            break
                    except Exception:
                        break
        except Exception as e:
            st.error(f"Failed to queue job: {e}")

    # ====== Filters ======
    st.header("Filters")

    # Parse URL query params (initial load)
    qp = st.query_params
    qp_agent = qp.get("agent")
    qp_start = qp.get("start")
    qp_end   = qp.get("end")
    qp_mbox  = qp.get("mbox")

    # Establish defaults BEFORE creating widgets
    today = dt.datetime.utcnow().date()
    default_start = today - dt.timedelta(days=30)

    # Agents & mailboxes (cached)
    agents = _get_agents()
    mailboxes = _get_mailboxes()

    # Session defaults
    if "agent_select_sidebar" not in st.session_state:
        st.session_state["agent_select_sidebar"] = qp_agent if qp_agent else "All"

    def _parse_mbox_query(val: str | None) -> List[int]:
        if not val:
            return []
        out = []
        for p in val.split(","):
            p = p.strip()
            if not p:
                continue
            try:
                out.append(int(p))
            except Exception:
                pass
        return out

    if "mailbox_multiselect" not in st.session_state:
        st.session_state["mailbox_multiselect"] = _parse_mbox_query(qp_mbox)

    if "hs_date_range" not in st.session_state:
        if qp_start and qp_end:
            try:
                st.session_state["hs_date_range"] = (
                    pd.to_datetime(qp_start).date(),
                    pd.to_datetime(qp_end).date(),
                )
            except Exception:
                st.session_state["hs_date_range"] = (default_start, today)
        else:
            st.session_state["hs_date_range"] = (default_start, today)

    # Clear filters (must run BEFORE widget instantiation)
    if st.button("Clear filters", type="secondary", help="Reset to agent=All, last 30 days, all mailboxes"):
        st.session_state["agent_select_sidebar"] = "All"
        st.session_state["hs_date_range"] = (default_start, today)
        st.session_state["mailbox_multiselect"] = []
        st.rerun()

    st.selectbox("Agent", options=["All"] + agents, key="agent_select_sidebar")
    st.multiselect(
        "Mailbox ID(s)",
        options=mailboxes,
        key="mailbox_multiselect",
        help="Filter results to selected mailbox IDs (from conversations.mailbox_id). Leave empty for all."
    )
    st.date_input("HelpScout message date range", key="hs_date_range")
    staff_only = st.toggle("Show only agent replies", value=False, help="Filters the Recent table to author='user'")

    # ====== Rubric ======
    st.header("Rubric Editor")
    weights = get_weights()
    with st.form("rubric_form"):
        emp = st.number_input("Empathy weight", min_value=0.0, max_value=1.0,
                              value=float(weights["empathy"]), step=0.05)
        acc = st.number_input("Accuracy weight", min_value=0.0, max_value=1.0,
                              value=float(weights["accuracy"]), step=0.05)
        ton = st.number_input("Tone weight", min_value=0.0, max_value=1.0,
                              value=float(weights["tone"]), step=0.05)
        res = st.number_input("Resolution weight", min_value=0.0, max_value=1.0,
                              value=float(weights["resolution"]), step=0.05)
        submitted = st.form_submit_button("Save Weights")
        if submitted:
            total = emp + acc + ton + res
            if abs(total - 1.0) > 1e-6:
                st.error("Weights must sum to 1.0")
            else:
                set_weights({"empathy": emp, "accuracy": acc, "tone": ton, "resolution": res})
                st.success("Weights saved. New scores will use these weights.")

    # ====== Danger Zone (guarded) ======
    st.header("Danger Zone")
    with st.form("purge_form"):
        sure = st.checkbox("I understand this will delete ALL conversations, messages, and scores.")
        do_purge = st.form_submit_button("Purge Data")
        if do_purge:
            if not sure:
                st.error("Please confirm the checkbox before purging.")
            else:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM qa_scores"))
                    conn.execute(text("DELETE FROM messages"))
                    conn.execute(text("DELETE FROM conversations"))
                st.warning("All conversation data purged.")

    # Diagnostics + version
    st.header("Diagnostics")
    if st.button("Test OpenAI"):
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY not set in environment.")
        else:
            try:
                use_responses = os.getenv("OPENAI_USE_RESPONSES", "0") == "1"
                base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                url = f"{base}/responses" if use_responses else f"{base}/chat/completions"
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
                if os.getenv("OPENAI_ORG"):
                    headers["OpenAI-Organization"] = os.getenv("OPENAI_ORG")
                if os.getenv("OPENAI_PROJECT"):
                    headers["OpenAI-Project"] = os.getenv("OPENAI_PROJECT")
                payload = (
                    {"model": OPENAI_MODEL, "input": "Say OK.", "max_output_tokens": 10, "temperature": 0}
                    if use_responses else
                    {"model": OPENAI_MODEL, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 10, "temperature": 0}
                )
                r = requests.post(url, headers=headers, json=payload, timeout=20)
                r.raise_for_status()
                data = r.json()
                text_out = data.get("output_text")
                if not text_out and not use_responses:
                    choices = data.get("choices") or []
                    if choices and "message" in choices[0]:
                        text_out = choices[0]["message"].get("content")
                st.success("OpenAI connectivity OK")
                st.code((text_out or json.dumps(data, indent=2))[:500])
            except Exception as e:
                st.error(f"OpenAI test failed: {e}")

    st.caption(f"App version: **{_version_display()}**")

# -------- Normalize filters (after sidebar) --------
agent = st.session_state.get("agent_select_sidebar", "All")
sel = st.session_state.get("hs_date_range")
mbox_sel = tuple(st.session_state.get("mailbox_multiselect", []))
staff_only = st.session_state.get("Show only agent replies", False)

if isinstance(sel, (list, tuple)):
    if len(sel) == 2:
        start_date, end_date = sel[0], sel[1]
    elif len(sel) == 1:
        start_date = end_date = sel[0]
    else:
        today = dt.datetime.utcnow().date()
        start_date = today - dt.timedelta(days=30)
        end_date = today
else:
    today = dt.datetime.utcnow().date()
    start_date = today - dt.timedelta(days=30)
    end_date = today

start_date = _as_date(start_date)
end_date   = _as_date(end_date)

start_dt    = dt.datetime.combine(start_date, dt.time.min)
end_dt_excl = dt.datetime.combine(end_date,   dt.time.min) + dt.timedelta(days=1)

# ---- Persist filters in URL ----
try:
    qp_out = {
        "agent": agent if agent else "All",
        "start": start_date.isoformat(),
        "end":   end_date.isoformat(),
    }
    if mbox_sel:
        qp_out["mbox"] = ",".join(str(m) for m in mbox_sel)
    st.query_params.update(qp_out)
except Exception:
    pass

# ---- Compact tables toggle (global CSS) ----
compact = st.toggle("Compact tables", value=False, help="Reduce row height / font size in tables")
if compact:
    st.markdown("""
        <style>
        div[data-testid="stDataFrame"] table {
            font-size: 0.85rem;
        }
        div[data-testid="stDataFrame"] td, 
        div[data-testid="stDataFrame"] th {
            padding-top: 2px !important;
            padding-bottom: 2px !important;
            line-height: 1.1 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ---- Job Status Panel ----
st.subheader("Job Status")
try:
    r = Redis.from_url(REDIS_URL)
    q = Queue("qa_jobs", connection=r)

    started_reg = StartedJobRegistry("qa_jobs", connection=r)
    finished_reg = FinishedJobRegistry("qa_jobs", connection=r)
    failed_reg = FailedJobRegistry("qa_jobs", connection=r)

    cols = st.columns(4)
    with cols[0]: st.metric("Queued", len(q.job_ids))
    with cols[1]: st.metric("Started", len(started_reg))
    with cols[2]: st.metric("Finished (recent)", len(finished_reg))
    with cols[3]: st.metric("Failed (recent)", len(failed_reg))
except Exception as e:
    st.info(f"Queue status unavailable: {e}")

st.divider()

# ---- KPIs ----
cols = st.columns([1, 1, 2, 2])
try:
    kpi = _get_kpis(agent, start_dt, end_dt_excl, mbox_sel)
except Exception as e:
    st.error(f"Failed to compute KPIs: {e}")
    kpi = None

# Quick overall counts (uncached; cheap)
try:
    with engine.connect() as conn:
        total_msgs_raw   = conn.execute(text("SELECT COUNT(*) FROM messages")).scalar()
        total_scored_raw = conn.execute(text("SELECT COUNT(*) FROM qa_scores")).scalar()
        total_msgs   = int(total_msgs_raw or 0)
        total_scored = int(total_scored_raw or 0)
except Exception:
    total_msgs, total_scored = 0, 0

with cols[0]: st.metric("Messages", total_msgs)
with cols[1]: st.metric("Scored", total_scored)

avg_total = d2f(kpi["avg_total"]) if kpi and kpi["messages_scored"] else 0.0
with cols[2]: st.metric("Avg QA Score", avg_total)
with cols[3]: st.info("Use the sidebar to connect, queue jobs, edit weights, test OpenAI, and tune filters.")

st.divider()

# ---- Summary for Selected Date Range ----
st.subheader("Summary for Selected Date Range")
if not kpi or not kpi["messages_scored"]:
    scope = f" for **{agent}**" if agent != "All" else ""
    if mbox_sel:
        scope += f" (mailboxes: {', '.join(map(str, mbox_sel))})"
    st.info(f"No scored messages in this range{scope}.")
else:
    label = f"(Agent: {agent})" if agent != "All" else "(All agents)"
    if mbox_sel:
        label += f" — Mailboxes: {', '.join(map(str, mbox_sel))}"
    st.caption(label)
    k_cols = st.columns(6)
    k_cols[0].metric("Messages Scored", int(kpi["messages_scored"]))
    k_cols[1].metric("Avg QA Score", float(kpi["avg_total"]))
    k_cols[2].metric("Empathy",      float(kpi["empathy"]))
    k_cols[3].metric("Accuracy",     float(kpi["accuracy"]))
    k_cols[4].metric("Tone",         float(kpi["tone"]))
    k_cols[5].metric("Resolution",   float(kpi["resolution"]))

st.divider()

# ---- Per-Agent Rollups (DB; supports mailbox + agent filter) ----
st.subheader("Per-Agent Rollups (HelpScout message date)")
try:
    roll_api = _get_rollups(agent, start_dt, end_dt_excl, mbox_sel)
except Exception as e:
    st.error(f"Failed to load rollups: {e}")
    roll_api = pd.DataFrame()

if roll_api.empty:
    scope = f" for {agent}" if agent != "All" else ""
    if mbox_sel:
        scope += f" — mailboxes: {', '.join(map(str, mbox_sel))}"
    st.warning(f"No scores in the selected range{scope}.")
else:
    st.dataframe(roll_api.sort_values(["day", "agent"], ascending=[False, True]), use_container_width=True)

    totals = (
        roll_api.groupby("agent", as_index=False)
                .agg(scored_cnt=("scored_cnt", "sum"),
                     avg_score=("avg_score", "mean"))
                .sort_values(["scored_cnt", "avg_score"], ascending=[False, False])
    )
    st.caption("Agent totals (selected range)")
    st.dataframe(totals, use_container_width=True)
    st.download_button(
        "Download rollups CSV",
        totals.to_csv(index=False).encode("utf-8"),
        file_name="qa_rollups.csv",
        mime="text/csv",
    )

# ---- Trend: Avg QA Score by Day ----
st.subheader("Trend: Avg QA Score by Day")
if not roll_api.empty:
    trend = (
        roll_api.groupby(["day", "agent"], as_index=False)
                .agg(avg_score=("avg_score", "mean"),
                     scored_cnt=("scored_cnt", "sum"))
    )
    trend["day"] = pd.to_datetime(trend["day"])

    if agent != "All":
        t = trend[trend["agent"] == agent].sort_values("day").set_index("day")
        if t.empty:
            st.info(f"No daily data for {agent} in this range.")
        else:
            st.line_chart(t["avg_score"], height=240)
            st.caption(f"Daily average for {agent}")
            st.download_button(
                "Download agent trend CSV",
                t.reset_index()[["day", "avg_score", "scored_cnt"]].to_csv(index=False).encode("utf-8"),
                file_name=f"qa_trend_{agent}.csv",
                mime="text/csv",
            )
    else:
        pivot = (
            trend.pivot_table(index="day", columns="agent", values="avg_score", aggfunc="mean")
                 .sort_index()
        )
        if pivot.empty:
            st.info("No daily data in this range.")
        else:
            st.line_chart(pivot, height=280)
            overall = (
                trend.groupby("day")
                     .apply(lambda g: (g["avg_score"] * g["scored_cnt"]).sum() / g["scored_cnt"].sum())
                     .rename("Overall")
                     .sort_index()
            )
            st.caption("Overall (weighted by message count)")
            st.line_chart(overall, height=180)
            st.download_button(
                "Download trend CSV (per-agent)",
                pivot.reset_index().to_csv(index=False).encode("utf-8"),
                file_name="qa_trend_by_agent.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download overall trend CSV",
                overall.reset_index().rename(columns={"index": "day"}).to_csv(index=False).encode("utf-8"),
                file_name="qa_trend_overall.csv",
                mime="text/csv",
            )
else:
    st.info("No data available for the selected filters to plot a trend.")

st.divider()

# ---- Recent Scored Messages (filtered by hs_created_at + Agent + mailbox + staff_only) ----
st.subheader("Recent Scored Messages")
try:
    df = _get_recent(agent, start_dt, end_dt_excl, mbox_sel, staff_only)
except Exception as e:
    st.error(f"Failed to load scored messages: {e}")
    df = pd.DataFrame()

if df.empty:
    st.info("No messages match the current filters.")
else:
    st.dataframe(df, use_container_width=True, height=450, hide_index=True)
    st.download_button(
        "Download messages CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="qa_messages.csv",
        mime="text/csv",
    )
