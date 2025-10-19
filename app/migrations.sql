-- app/migrations.sql
-- Idempotent indexes for QA Auditor
-- Run with:
-- docker compose exec postgres psql -U postgres -d qa_auditor -f /code/app/migrations.sql

-- Safety: make sure core tables exist (no-ops if they already do)
CREATE TABLE IF NOT EXISTS conversations (
  hs_conversation_id BIGINT PRIMARY KEY,
  mailbox_id BIGINT,
  subject TEXT,
  customer_email TEXT,
  status TEXT,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMP DEFAULT now(),
  updated_at TIMESTAMP DEFAULT now(),
  hs_conversation_id BIGINT REFERENCES conversations(hs_conversation_id) ON DELETE CASCADE,
  hs_message_id BIGINT UNIQUE,
  author TEXT,
  body TEXT,
  body_plain TEXT,
  created TIMESTAMP,
  agent_name TEXT,
  agent_email TEXT,
  hs_created_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS qa_scores (
  id SERIAL PRIMARY KEY,
  hs_message_id BIGINT UNIQUE,
  hs_conversation_id BIGINT,
  empathy NUMERIC,
  accuracy NUMERIC,
  tone NUMERIC,
  resolution NUMERIC,
  total NUMERIC,
  reasoning TEXT,
  created_at TIMESTAMP DEFAULT now()
);

-- === Indexes you’ll feel immediately ===

-- Filter/joins: messages by HelpScout message date (already created earlier in your DB; keep for completeness)
CREATE INDEX IF NOT EXISTS idx_messages_hs_created_at ON messages(hs_created_at);

-- JOIN speed: qa_scores -> messages via hs_message_id
CREATE INDEX IF NOT EXISTS idx_qascores_hsmsg ON qa_scores(hs_message_id);

-- Agent filtering: COALESCE(agent_name, author)
CREATE INDEX IF NOT EXISTS idx_msgs_agent ON messages((COALESCE(agent_name, author)));

-- If you ever filter/graph by score creation time (used in some pages & exports)
CREATE INDEX IF NOT EXISTS idx_scores_created ON qa_scores(created_at);

-- (Optional but useful) If you frequently filter by both date and agent together:
-- CREATE INDEX IF NOT EXISTS idx_messages_hsdate_agent
--   ON messages(hs_created_at, (COALESCE(agent_name, author)));

-- ===== Optional (not applied automatically): monthly partitioning plan =====
-- If volume grows large, consider migrating messages to monthly partitions on hs_created_at.
-- This requires a separate migration path (create a partitioned table, copy data, swap).
-- Keeping a commented starter here for later:
--
-- -- Example (DO NOT RUN blindly in prod; plan a controlled migration):
-- -- 1) Create a new partitioned table structure
-- -- CREATE TABLE messages_p (
-- --   LIKE messages INCLUDING ALL
-- -- ) PARTITION BY RANGE (hs_created_at);
-- --
-- -- 2) Create monthly partitions (example Oct 2025)
-- -- CREATE TABLE IF NOT EXISTS messages_p_2025_10 PARTITION OF messages_p
-- --   FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
-- --
-- -- 3) Copy data in a maintenance window, then rename swap
-- -- INSERT INTO messages_p SELECT * FROM messages WHERE hs_created_at IS NOT NULL;
-- -- ALTER TABLE messages RENAME TO messages_old;
-- -- ALTER TABLE messages_p RENAME TO messages;
-- -- (Recreate FKs, indexes if needed, then drop messages_old when satisfied.)
