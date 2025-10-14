-- Add accuracy enhancement fields to raw_events table
-- This migration implements the 5 critical accuracy fixes for production-grade financial data semantics

-- FIX #1: Amount direction and transaction type fields
ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS transaction_type TEXT CHECK (transaction_type IN ('income', 'expense', 'transfer', 'refund', 'unknown')),
ADD COLUMN IF NOT EXISTS amount_direction TEXT CHECK (amount_direction IN ('credit', 'debit', 'neutral', 'unknown')),
ADD COLUMN IF NOT EXISTS amount_signed_usd DECIMAL(15,2),
ADD COLUMN IF NOT EXISTS affects_cash BOOLEAN DEFAULT true;

-- FIX #2: Standardized timestamp semantics
ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS source_ts TIMESTAMP WITH TIME ZONE,  -- When transaction occurred (from source)
ADD COLUMN IF NOT EXISTS ingested_ts TIMESTAMP WITH TIME ZONE,  -- When we ingested it
ADD COLUMN IF NOT EXISTS processed_ts TIMESTAMP WITH TIME ZONE,  -- When we finished processing
ADD COLUMN IF NOT EXISTS transaction_date DATE,  -- Date of transaction for currency conversion
ADD COLUMN IF NOT EXISTS exchange_rate_date DATE;  -- Date of exchange rate used

-- FIX #3: Data validation flags
ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS validation_flags JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS is_valid BOOLEAN DEFAULT true;

-- FIX #4: Canonical entity IDs
ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS vendor_canonical_id TEXT,
ADD COLUMN IF NOT EXISTS vendor_verified BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS vendor_alternatives JSONB DEFAULT '[]';

-- FIX #5: Confidence-based flagging
ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS overall_confidence DECIMAL(3,2),
ADD COLUMN IF NOT EXISTS requires_review BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS review_reason TEXT,
ADD COLUMN IF NOT EXISTS review_priority TEXT CHECK (review_priority IN ('high', 'medium', 'low', NULL));

-- Accuracy metadata
ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS accuracy_enhanced BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS accuracy_version TEXT;

-- Add indexes for new fields to optimize queries

-- FIX #1 indexes: Amount direction and transaction type
CREATE INDEX IF NOT EXISTS idx_raw_events_transaction_type ON public.raw_events(transaction_type);
CREATE INDEX IF NOT EXISTS idx_raw_events_amount_direction ON public.raw_events(amount_direction);
CREATE INDEX IF NOT EXISTS idx_raw_events_amount_signed_usd ON public.raw_events(amount_signed_usd);
CREATE INDEX IF NOT EXISTS idx_raw_events_affects_cash ON public.raw_events(affects_cash) WHERE affects_cash = true;

-- FIX #2 indexes: Standardized timestamps
CREATE INDEX IF NOT EXISTS idx_raw_events_source_ts ON public.raw_events(source_ts);
CREATE INDEX IF NOT EXISTS idx_raw_events_ingested_ts ON public.raw_events(ingested_ts);
CREATE INDEX IF NOT EXISTS idx_raw_events_processed_ts ON public.raw_events(processed_ts);
CREATE INDEX IF NOT EXISTS idx_raw_events_transaction_date ON public.raw_events(transaction_date);

-- FIX #3 indexes: Data validation
CREATE INDEX IF NOT EXISTS idx_raw_events_is_valid ON public.raw_events(is_valid);
CREATE INDEX IF NOT EXISTS idx_raw_events_validation_flags_gin ON public.raw_events USING GIN (validation_flags);

-- FIX #4 indexes: Canonical entity IDs
CREATE INDEX IF NOT EXISTS idx_raw_events_vendor_canonical_id ON public.raw_events(vendor_canonical_id);
CREATE INDEX IF NOT EXISTS idx_raw_events_vendor_verified ON public.raw_events(vendor_verified) WHERE vendor_verified = true;

-- FIX #5 indexes: Confidence-based flagging
CREATE INDEX IF NOT EXISTS idx_raw_events_overall_confidence ON public.raw_events(overall_confidence);
CREATE INDEX IF NOT EXISTS idx_raw_events_requires_review ON public.raw_events(requires_review) WHERE requires_review = true;
CREATE INDEX IF NOT EXISTS idx_raw_events_review_priority ON public.raw_events(review_priority) WHERE review_priority IS NOT NULL;

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_raw_events_user_review ON public.raw_events(user_id, requires_review, review_priority) WHERE requires_review = true;
CREATE INDEX IF NOT EXISTS idx_raw_events_user_transaction_type ON public.raw_events(user_id, transaction_type, transaction_date);
CREATE INDEX IF NOT EXISTS idx_raw_events_user_amount_signed ON public.raw_events(user_id, amount_signed_usd, transaction_date) WHERE affects_cash = true;

-- Create function to get accuracy-enhanced statistics
CREATE OR REPLACE FUNCTION get_accuracy_stats(user_uuid UUID)
RETURNS TABLE(
    total_events BIGINT,
    accuracy_enhanced_events BIGINT,
    events_requiring_review BIGINT,
    high_priority_reviews BIGINT,
    income_events BIGINT,
    expense_events BIGINT,
    transfer_events BIGINT,
    total_income_usd DECIMAL(15,2),
    total_expenses_usd DECIMAL(15,2),
    net_cash_flow_usd DECIMAL(15,2),
    avg_overall_confidence DECIMAL(3,2),
    validation_error_rate DECIMAL(5,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_events,
        COUNT(*) FILTER (WHERE accuracy_enhanced = true) as accuracy_enhanced_events,
        COUNT(*) FILTER (WHERE requires_review = true) as events_requiring_review,
        COUNT(*) FILTER (WHERE review_priority = 'high') as high_priority_reviews,
        COUNT(*) FILTER (WHERE transaction_type = 'income') as income_events,
        COUNT(*) FILTER (WHERE transaction_type = 'expense') as expense_events,
        COUNT(*) FILTER (WHERE transaction_type = 'transfer') as transfer_events,
        COALESCE(SUM(amount_signed_usd) FILTER (WHERE transaction_type = 'income'), 0) as total_income_usd,
        COALESCE(ABS(SUM(amount_signed_usd)) FILTER (WHERE transaction_type = 'expense'), 0) as total_expenses_usd,
        COALESCE(SUM(amount_signed_usd) FILTER (WHERE affects_cash = true), 0) as net_cash_flow_usd,
        AVG(overall_confidence) FILTER (WHERE overall_confidence IS NOT NULL) as avg_overall_confidence,
        (COUNT(*) FILTER (WHERE is_valid = false)::DECIMAL / NULLIF(COUNT(*), 0)) as validation_error_rate
    FROM public.raw_events
    WHERE user_id = user_uuid;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to get runway forecast (cash flow projection)
CREATE OR REPLACE FUNCTION get_runway_forecast(user_uuid UUID, months_ahead INTEGER DEFAULT 12)
RETURNS TABLE(
    current_balance_usd DECIMAL(15,2),
    monthly_income_avg DECIMAL(15,2),
    monthly_expenses_avg DECIMAL(15,2),
    monthly_burn_rate DECIMAL(15,2),
    runway_months DECIMAL(5,2),
    projected_balance_usd DECIMAL(15,2)
) AS $$
DECLARE
    current_bal DECIMAL(15,2);
    monthly_inc DECIMAL(15,2);
    monthly_exp DECIMAL(15,2);
    burn_rate DECIMAL(15,2);
    runway DECIMAL(5,2);
BEGIN
    -- Calculate current balance (cumulative cash flow)
    SELECT COALESCE(SUM(amount_signed_usd), 0) INTO current_bal
    FROM public.raw_events
    WHERE user_id = user_uuid AND affects_cash = true;
    
    -- Calculate average monthly income (last 6 months)
    SELECT COALESCE(AVG(monthly_total), 0) INTO monthly_inc
    FROM (
        SELECT SUM(amount_signed_usd) as monthly_total
        FROM public.raw_events
        WHERE user_id = user_uuid 
            AND transaction_type = 'income'
            AND transaction_date >= CURRENT_DATE - INTERVAL '6 months'
        GROUP BY DATE_TRUNC('month', transaction_date)
    ) monthly_income;
    
    -- Calculate average monthly expenses (last 6 months)
    SELECT COALESCE(ABS(AVG(monthly_total)), 0) INTO monthly_exp
    FROM (
        SELECT SUM(amount_signed_usd) as monthly_total
        FROM public.raw_events
        WHERE user_id = user_uuid 
            AND transaction_type = 'expense'
            AND transaction_date >= CURRENT_DATE - INTERVAL '6 months'
        GROUP BY DATE_TRUNC('month', transaction_date)
    ) monthly_expenses;
    
    -- Calculate burn rate (expenses - income)
    burn_rate := monthly_exp - monthly_inc;
    
    -- Calculate runway (months until cash runs out)
    IF burn_rate > 0 THEN
        runway := current_bal / burn_rate;
    ELSE
        runway := 999.99;  -- Positive cash flow = infinite runway
    END IF;
    
    RETURN QUERY
    SELECT 
        current_bal as current_balance_usd,
        monthly_inc as monthly_income_avg,
        monthly_exp as monthly_expenses_avg,
        burn_rate as monthly_burn_rate,
        runway as runway_months,
        (current_bal + (monthly_inc - monthly_exp) * months_ahead) as projected_balance_usd;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to get events requiring review
CREATE OR REPLACE FUNCTION get_events_for_review(user_uuid UUID, priority_filter TEXT DEFAULT NULL)
RETURNS TABLE(
    id UUID,
    kind TEXT,
    transaction_type TEXT,
    amount_signed_usd DECIMAL(15,2),
    vendor_standard TEXT,
    review_reason TEXT,
    review_priority TEXT,
    overall_confidence DECIMAL(3,2),
    validation_flags JSONB,
    transaction_date DATE,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        re.id,
        re.kind,
        re.transaction_type,
        re.amount_signed_usd,
        re.vendor_standard,
        re.review_reason,
        re.review_priority,
        re.overall_confidence,
        re.validation_flags,
        re.transaction_date,
        re.created_at
    FROM public.raw_events re
    WHERE re.user_id = user_uuid
        AND re.requires_review = true
        AND (priority_filter IS NULL OR re.review_priority = priority_filter)
    ORDER BY 
        CASE re.review_priority
            WHEN 'high' THEN 1
            WHEN 'medium' THEN 2
            WHEN 'low' THEN 3
            ELSE 4
        END,
        re.created_at DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to validate transaction semantics
CREATE OR REPLACE FUNCTION validate_transaction_semantics(
    p_transaction_type TEXT,
    p_amount_direction TEXT,
    p_amount_signed_usd DECIMAL(15,2),
    p_affects_cash BOOLEAN
)
RETURNS JSONB AS $$
DECLARE
    validation_result JSONB;
    errors TEXT[] := ARRAY[]::TEXT[];
BEGIN
    -- Validate income transactions
    IF p_transaction_type = 'income' THEN
        IF p_amount_direction != 'credit' THEN
            errors := array_append(errors, 'Income must have credit direction');
        END IF;
        IF p_amount_signed_usd < 0 THEN
            errors := array_append(errors, 'Income must have positive signed amount');
        END IF;
    END IF;
    
    -- Validate expense transactions
    IF p_transaction_type = 'expense' THEN
        IF p_amount_direction != 'debit' THEN
            errors := array_append(errors, 'Expense must have debit direction');
        END IF;
        IF p_amount_signed_usd > 0 THEN
            errors := array_append(errors, 'Expense must have negative signed amount');
        END IF;
    END IF;
    
    -- Validate transfer transactions
    IF p_transaction_type = 'transfer' THEN
        IF p_amount_direction != 'neutral' THEN
            errors := array_append(errors, 'Transfer must have neutral direction');
        END IF;
        IF p_affects_cash != false THEN
            errors := array_append(errors, 'Transfer should not affect cash (affects_cash = false)');
        END IF;
    END IF;
    
    -- Build result
    IF array_length(errors, 1) > 0 THEN
        validation_result := jsonb_build_object(
            'is_valid', false,
            'errors', to_jsonb(errors),
            'severity', 'error'
        );
    ELSE
        validation_result := jsonb_build_object(
            'is_valid', true,
            'errors', '[]'::jsonb,
            'severity', 'none'
        );
    END IF;
    
    RETURN validation_result;
END;
$$ LANGUAGE plpgsql;

-- Add comment explaining the accuracy enhancement
COMMENT ON COLUMN public.raw_events.transaction_type IS 'FIX #1: Transaction type (income, expense, transfer, refund) for accurate runway calculations';
COMMENT ON COLUMN public.raw_events.amount_signed_usd IS 'FIX #1: Signed amount in USD (negative for expenses, positive for income) for accurate cash flow';
COMMENT ON COLUMN public.raw_events.source_ts IS 'FIX #2: When transaction occurred in source system (for accurate time-series analysis)';
COMMENT ON COLUMN public.raw_events.validation_flags IS 'FIX #3: Data validation results (amount_valid, currency_valid, etc.)';
COMMENT ON COLUMN public.raw_events.vendor_canonical_id IS 'FIX #4: Canonical vendor ID for consistent entity aggregation';
COMMENT ON COLUMN public.raw_events.requires_review IS 'FIX #5: Flag for low-confidence transactions requiring manual review';

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Accuracy enhancement fields migration completed successfully';
    RAISE NOTICE 'Added 20+ new fields for production-grade financial data semantics';
    RAISE NOTICE 'Created 15+ indexes for optimized query performance';
    RAISE NOTICE 'Created 4 new functions for accuracy statistics and runway forecasting';
END $$;
