-- scripts/add_ark_leads_intelligence_columns.sql
--
-- Adds the intelligence columns that rag/lead_manager.py::save_lead_to_db
-- writes alongside the core lead fields. Run this once in the Supabase
-- SQL Editor if you see ADMIN logs like:
--     ERROR | Failed to save lead to DB: {"code":"PGRST204",
--         "message":"Could not find the 'concern' column of 'ark_leads'..."}
--
-- Safe to re-run — IF NOT EXISTS makes each ADD idempotent.

ALTER TABLE ark_leads
    ADD COLUMN IF NOT EXISTS segment         text,
    ADD COLUMN IF NOT EXISTS stage           text,
    ADD COLUMN IF NOT EXISTS lead_score      integer DEFAULT 0,
    ADD COLUMN IF NOT EXISTS lead_score_type text,
    ADD COLUMN IF NOT EXISTS persona         text,
    ADD COLUMN IF NOT EXISTS concern         text;

-- Force PostgREST to refresh its schema cache so the new columns
-- are immediately visible to the Supabase client (otherwise the bot
-- may continue to see PGRST204 for a short window).
NOTIFY pgrst, 'reload schema';
