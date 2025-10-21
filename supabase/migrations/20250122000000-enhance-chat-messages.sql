-- Enhance chat_messages table for intelligent chat orchestrator
-- Date: 2025-01-22
-- Purpose: Add fields needed for storing intelligent responses

-- Add new columns for intelligent chat features
ALTER TABLE public.chat_messages
ADD COLUMN IF NOT EXISTS response TEXT,
ADD COLUMN IF NOT EXISTS question_type TEXT,
ADD COLUMN IF NOT EXISTS confidence FLOAT,
ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';

-- Create index on question_type for analytics
CREATE INDEX IF NOT EXISTS idx_chat_messages_question_type 
ON public.chat_messages(question_type) 
WHERE question_type IS NOT NULL;

-- Create GIN index on metadata for JSONB queries
CREATE INDEX IF NOT EXISTS idx_chat_messages_metadata 
ON public.chat_messages USING GIN (metadata);

-- Add comment for documentation
COMMENT ON COLUMN public.chat_messages.response IS 'AI assistant response text';
COMMENT ON COLUMN public.chat_messages.question_type IS 'Type of question: causal, temporal, relationship, what_if, explain, data_query, general';
COMMENT ON COLUMN public.chat_messages.confidence IS 'Confidence score (0.0-1.0) of the AI response';
COMMENT ON COLUMN public.chat_messages.metadata IS 'Full response data including actions, visualizations, follow-up questions';
