# ARK AI Bot — Production RAG Backend v2

Low-latency RAG-powered WhatsApp AI assistant for **ARK Learning Arena**. Answers questions from an institute Word document in **2–3 seconds** via Supabase pgvector + HuggingFace LLM.

---

## Architecture

```
WhatsApp Message
      |
  Interakt Webhook
      |
  FastAPI  /whatsapp
      |
  In qualification flow?  ─── YES → collect next field
      |  NO
  Hot lead?  ─── YES → 🔥 notify admin (HIGH priority)
      |  NO
  Escalation?  ─── YES → 🚨 classify lead + notify admin
      |  NO
  Course interest?  ─── YES → RAG answer + start qualification
      |  NO
  Cache check  ─── HIT? → instant reply (< 1s)
      |  MISS
  RAG pipeline → reply (cold lead, no admin notify)
```

---

## Project Structure

```
ARK_AI_BOT/
├── api/
│   └── main.py               # FastAPI (POST /ask, POST /whatsapp, GET /health)
├── config/
│   └── settings.py            # Centralised Pydantic Settings
├── rag/
│   ├── __init__.py
│   ├── cache.py               # Question cache (in-memory + Redis)
│   ├── chunking.py            # .docx extraction + chunking + hashing
│   ├── embeddings.py          # Sentence-transformer embeddings
│   ├── escalation.py          # Human handoff detection + admin notification
│   ├── lead_manager.py        # Lead classification, qualification, hot leads
│   └── retriever.py           # Vector search + LLM + cache + memory + perf logging
├── scripts/
│   └── ingest_document.py     # Incremental ingestion CLI
├── documents/
│   └── ark_details.docx       # Your institute document
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install

```bash
cd ARK_AI_BOT
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Configure

```bash
copy .env.example .env
```

Fill in `.env`:

| Variable                | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `SUPABASE_URL`          | Supabase project URL                           |
| `SUPABASE_KEY`          | Supabase service role key                      |
| `HF_API_TOKEN`          | HuggingFace API token                          |
| `LLM_MODEL`             | `mistralai/Mistral-7B-Instruct-v0.2` (default) |
| `AISENSY_API_KEY`       | AiSensy Project API key for WhatsApp           |
| `AISENSY_PROJECT_ID`    | AiSensy project ID (for session replies)       |
| `AISENSY_CAMPAIGN_NAME` | Campaign name for template messages            |
| `ADMIN_WHATSAPP_NUMBER` | Admin phone for escalation alerts              |
| `REDIS_URL`             | Optional. Leave empty for in-memory cache      |

### 3. Database Setup

Run in **Supabase SQL Editor**:

```sql
-- Enable pgvector
create extension if not exists vector;

-- Document chunks table
create table ark_docs (
  id bigserial primary key,
  content text not null,
  content_hash text,
  embedding vector(384)
);

-- Index for incremental ingestion hash lookups
create index idx_ark_docs_hash on ark_docs(content_hash);

-- Similarity search function
create or replace function match_ark_docs (
  query_embedding vector(384),
  match_count int default 5
)
returns table (id bigint, content text, similarity float)
language plpgsql
as $$
begin
  return query
  select
    ark_docs.id,
    ark_docs.content,
    1 - (ark_docs.embedding <=> query_embedding) as similarity
  from ark_docs
  order by ark_docs.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

**Leads table** (for storing qualified leads):

```sql
create table ark_leads (
  id bigserial primary key,
  phone text,
  student_name text,
  class text,
  school text,
  parent_phone text,
  course text,
  lead_type text,
  priority text default 'NORMAL',
  message text,
  created_at timestamptz default now()
);
```

### 4. Ingest Document

```bash
# First time (full ingestion)
python scripts/ingest_document.py --full

# Subsequent updates (incremental — only changed chunks)
python scripts/ingest_document.py
```

### 5. Start Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "version": "2.0.0" }
```

### `POST /ask`

**Request:**
```json
{
  "question": "What courses does ARK offer?",
  "user_id": "user_123"
}
```

**Response:**
```json
{
  "answer": "ARK Learning Arena offers NEET coaching, school tuition for classes 6–12, and foundation courses."
}
```

### `POST /whatsapp`

Receives AiSensy webhook payloads, runs RAG, and replies via AiSensy API.

**Test payload:**
```json
{
  "from": "919876543210",
  "message": "What are the NEET fees?"
}
```

---

## Performance Logging

Every request logs structured timing:

```
12:34:56 | ark.retriever    | INFO  | PERF | embedding_ms=8.2 | search_ms=42.1 | llm_ms=1823.4 | total_ms=1873.7
```

Cache hits return instantly:

```
12:34:57 | ark.retriever    | INFO  | PERF | cache_hit=True | total_ms=0.3
```

---

## Switching LLM Models

Change `LLM_MODEL` in `.env` — no code changes needed:

```env
# Fast & light
LLM_MODEL=microsoft/Phi-3-mini-4k-instruct

# More capable
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

---

## Deployment

### Render

1. Push to GitHub.
2. **New Web Service** on [render.com](https://render.com).
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Add env vars in Render dashboard.
6. Deploy. API URL: `https://your-app.onrender.com`

### Railway

1. Push to GitHub.
2. New project on [railway.app](https://railway.app).
3. Create `Procfile`:
   ```
   web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
   ```
4. Add env vars in Railway dashboard.
5. Deploy. API URL: `https://your-app.up.railway.app`

---

## AiSensy WhatsApp Integration

### Setup Flow

```
WhatsApp message  →  AiSensy  →  Webhook POST  →  FastAPI /whatsapp  →  RAG  →  AiSensy reply API  →  WhatsApp
```

### Configuration Steps

1. Sign up at [aisensy.com](https://www.aisensy.com/).
2. Set up a WhatsApp Business API number.
3. In AiSensy dashboard, set **Webhook URL**:
   ```
   https://your-app.onrender.com/whatsapp
   ```
4. Copy your **AiSensy API Key** and **Project ID** to `.env`:
   ```
   AISENSY_API_KEY=your-key-here
   AISENSY_PROJECT_ID=your-project-id
   AISENSY_CAMPAIGN_NAME=your-campaign-name
   ```
5. Test by sending a WhatsApp message to your business number.

---

## Conversation Memory

The bot remembers the last 3 turns per user (configurable via `MEMORY_MAX_TURNS`):

```
User: What courses do you offer?
Bot:  NEET coaching, school tuition, and foundation courses.

User: What about fees?
Bot:  (understands "fees" refers to the courses mentioned above)
```

Memory is per phone number for WhatsApp and per `user_id` for the `/ask` endpoint.

---

## Tech Stack

| Component        | Technology                                |
| ---------------- | ----------------------------------------- |
| API Framework    | FastAPI (async)                           |
| Embeddings       | sentence-transformers (all-MiniLM-L6-v2)  |
| Vector Database  | Supabase + pgvector                       |
| LLM              | HuggingFace Inference API (configurable)  |
| Cache            | In-memory / Redis                         |
| HTTP Client      | httpx (async)                             |
| Document Parsing | python-docx                               |
| Escalation       | Keyword + regex intent detection          |
| Lead Management  | State machine qualification + Supabase    |
| Language         | Python 3.10+                              |

---

## Lead Management Pipeline

The bot is a **full admission funnel system** that classifies, qualifies, and routes leads.

### Lead Types

| Type | Triggers |
|------|----------|
| Fee Negotiation | "fees", "discount", "scholarship" |
| Callback Request | "call me", "contact me", "speak with" |
| Demo Class | "demo", "trial class" |
| Admission Enquiry | "admission", "join", "enroll" |
| General Enquiry | Everything else |

### Hot Lead Detection

High-intent messages trigger immediate admin notification with **HIGH** priority:

```
User:  "Can you reduce the NEET fees?"
Admin: 🔥 HOT LEAD | Phone: 919xxxx | Priority: HIGH | Contact immediately.
```

### Qualification Flow

When a user shows course interest, the bot collects student details:

```
User:  "I want NEET coaching"
Bot:   "ARK offers NEET coaching for... Please share the student's name."
User:  "Rahul"
Bot:   "Thank you! Which class is the student currently studying in?"
User:  "Class 11"
Bot:   "Got it! Which school is the student studying in?"
User:  "Velammal"
Bot:   "Almost done! Please share the parent's phone number."
User:  "9876543210"
Bot:   "Thank you! Our counsellor will contact you shortly."
Admin: 🚨 Qualified Lead | Student: Rahul | Class: 11 | School: Velammal | Course: NEET
```

### Cold Lead Handling

General questions ("What courses do you offer?") are answered via RAG **without** notifying the admin.

---

## Human Escalation / Admin Notification

When a user asks for human help, the bot automatically:
1. Detects the escalation intent
2. Sends a lead notification to the admin's WhatsApp
3. Replies to the user with a counsellor-callback message

### Trigger Phrases

`fees negotiation`, `call admin`, `talk to someone`, `speak with counsellor`, `contact admin`, `call me`, `I want to speak with a person`, `admission enquiry`, `schedule a call`, and more.

### Example Flow

```
User:  "I want to negotiate fees for NEET batch"
Bot:   "Thank you for your interest! Our academic counsellor will contact you shortly."
Admin: 🚨 New Lead Request — Name: Arrav, Phone: 919xxxx, Message: "I want to negotiate fees..."
```

### Cooldown

If the same user sends multiple escalation messages within **10 minutes**, only the first notification is sent to avoid spamming the admin.

### Log Format

```
13:01:33 | ark.escalation   | INFO  | ESCALATION_TRIGGERED | user=919xxxx | message="fees negotiation" | admin_notified=True
```
