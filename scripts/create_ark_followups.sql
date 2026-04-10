-- =====================================================================
-- Supabase Table: ark_followups
-- =====================================================================
-- Tracks user activity and automated follow-up state for the
-- ARK WhatsApp bot follow-up automation system.
--
-- Run this SQL in the Supabase SQL Editor to create the table.
-- =====================================================================

CREATE TABLE IF NOT EXISTS ark_followups (
    id                BIGSERIAL    PRIMARY KEY,
    phone             TEXT         NOT NULL UNIQUE,
    last_message_time TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    followup_stage    INTEGER      NOT NULL DEFAULT 0,
    status            TEXT         NOT NULL DEFAULT 'active',
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Indexes for efficient queries by the background scheduler
CREATE INDEX IF NOT EXISTS idx_followups_status ON ark_followups(status);
CREATE INDEX IF NOT EXISTS idx_followups_stage  ON ark_followups(followup_stage);

-- =====================================================================
-- Column Reference
-- =====================================================================
-- phone             : User's WhatsApp phone number (unique key)
-- last_message_time : Timestamp of the user's most recent message
-- followup_stage    : 0 = no follow-up sent
--                     1 = first follow-up sent  (after 10 min)
--                     2 = second follow-up sent (after 24 hours)
-- status            : 'active'    = eligible for follow-ups
--                     'completed' = lead qualification finished / max follow-ups sent
--                     'escalated' = hot lead or human escalation occurred
-- created_at        : When the record was first created
