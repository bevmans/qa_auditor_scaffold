🧠 QA Auditor — Help Scout + OpenAI

AI-powered QA scoring for customer support teams — built with FastAPI, Streamlit, Redis, and Postgres.
Connect your Help Scout account, import conversations, and let OpenAI automatically evaluate ticket quality and tone.

⚙️ Architecture
Service	Description
api	FastAPI backend (OAuth callback, Help Scout webhooks, job queue interface)
web	Streamlit dashboard (main UI for QA reviewers)
worker	RQ worker that imports conversations and runs AI scoring
redis	Queue broker and caching layer
postgres	Primary data store for conversations and scores
🚀 Quick Start
1️⃣ Copy your environment file
cp .env.example .env


Then open .env and fill in:

HELPSCOUT_CLIENT_ID=your_client_id
HELPSCOUT_CLIENT_SECRET=your_client_secret
OPENAI_API_KEY=your_openai_key


All other settings can stay as-is for local use.

2️⃣ Start the full stack
docker compose up -d --build


This launches:

Service	URL
Frontend (Streamlit)	http://localhost:8502

Backend (FastAPI)	http://localhost:8001

Redis / Postgres	Internal containers
3️⃣ Connect Help Scout (OAuth)

Go to Help Scout Developer Apps
.

Create a new Custom OAuth App.

Set the Redirect URI to:

http://localhost:8001/oauth/callback


Copy your Client ID and Client Secret into your .env.

Then visit http://localhost:8502
 and click “Connect Help Scout”.
You’ll authorize the app, return to your dashboard, and see “OAuth status: Connected.”

4️⃣ Queue and Score Conversations

Once connected:

Click Queue Import + Score in Streamlit.

The worker service imports tickets and runs AI scoring.

Results appear in your dashboard once jobs complete.

If you see all zeros, check:

The worker logs (docker compose logs -f worker)

That your OPENAI_API_KEY is valid

And that Redis is running

🧩 Useful Commands
# View logs for any service
docker compose logs -f api
docker compose logs -f worker

# Stop everything
docker compose down

# Rebuild from scratch
docker compose up -d --build

🧰 Development Notes

Dummy Data — use Purge Demo Data in the Streamlit sidebar.

Re-import conversations — click Queue Import in the UI.

Reconnect Help Scout — refresh tokens after expiry (see status in dashboard).

Schema — defined in app/migrations.sql. For production, consider Alembic.

🔐 OAuth + API Details
Endpoint	Purpose
/health	Basic health check
/oauth/start	Begins Help Scout OAuth flow
/oauth/callback	Handles OAuth redirect
/webhooks/helpscout	Processes Help Scout webhook data
🧠 OpenAI Integration

The QA Auditor uses your OPENAI_API_KEY and OPENAI_MODEL to analyze ticket responses.
Supported models:

gpt-4o-mini — fast and efficient for bulk QA

gpt-4o — for higher-accuracy text evaluation

🧾 License

MIT — free to use and modify.

🙌 Credits

Built by Bev Barenbrug (Mans)
Powered by:
Help Scout API
OpenAI GPT-4o
FastAPI + Streamlit
Redis + Postgres

✅ Submission-ready Summary

QA Auditor — AI-driven quality scoring for Help Scout support conversations.
Built in Docker Compose with FastAPI, Streamlit, and OpenAI.
Fully functional OAuth flow, background worker jobs, and a real scoring pipeline.
