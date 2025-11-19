-- Migration: Add missing columns to chat_messages table
-- Date: 2025-11-19
-- Purpose: Align chat_messages schema with backend expectations

-- Add missing columns to chat_messages
ALTER TABLE IF EXISTS public.chat_messages
ADD COLUMN IF NOT EXISTS response TEXT,
ADD COLUMN IF NOT EXISTS question_type VARCHAR(100),
ADD COLUMN IF NOT EXISTS confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_chat_messages_question_type ON public.chat_messages(question_type);
CREATE INDEX IF NOT EXISTS idx_chat_messages_confidence ON public.chat_messages(confidence);

-- Add comments for documentation
COMMENT ON COLUMN public.chat_messages.response IS 'AI-generated response to the user message';
COMMENT ON COLUMN public.chat_messages.question_type IS 'Classification of question type (e.g., analysis, prediction, insight)';
COMMENT ON COLUMN public.chat_messages.confidence IS 'Confidence score of the response (0.0-1.0)';
COMMENT ON COLUMN public.chat_messages.metadata IS 'Additional metadata for the message (e.g., sources, reasoning)';
