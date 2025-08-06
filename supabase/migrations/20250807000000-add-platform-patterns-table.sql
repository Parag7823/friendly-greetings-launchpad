-- Migration: Add platform patterns and discovered platforms tables
-- Date: 2025-08-07

-- Create platform_patterns table for storing learned platform patterns
CREATE TABLE IF NOT EXISTS public.platform_patterns (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    platform VARCHAR(100) NOT NULL,
    patterns JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, platform)
);

-- Create discovered_platforms table for storing new platform discoveries
CREATE TABLE IF NOT EXISTS public.discovered_platforms (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    platform_name VARCHAR(100) NOT NULL,
    discovery_reason TEXT,
    confidence_score DECIMAL(3,2) DEFAULT 0.5,
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_platform_patterns_user_id ON public.platform_patterns(user_id);
CREATE INDEX IF NOT EXISTS idx_platform_patterns_platform ON public.platform_patterns(platform);
CREATE INDEX IF NOT EXISTS idx_discovered_platforms_user_id ON public.discovered_platforms(user_id);
CREATE INDEX IF NOT EXISTS idx_discovered_platforms_platform_name ON public.discovered_platforms(platform_name);

-- Add RLS policies
ALTER TABLE public.platform_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.discovered_platforms ENABLE ROW LEVEL SECURITY;

-- Platform patterns policies
CREATE POLICY "Users can view their own platform patterns" ON public.platform_patterns
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own platform patterns" ON public.platform_patterns
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own platform patterns" ON public.platform_patterns
    FOR UPDATE USING (auth.uid() = user_id);

-- Discovered platforms policies
CREATE POLICY "Users can view their own discovered platforms" ON public.discovered_platforms
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own discovered platforms" ON public.discovered_platforms
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Function to get platform patterns for a user
CREATE OR REPLACE FUNCTION get_platform_patterns(p_user_id UUID)
RETURNS TABLE (
    platform VARCHAR(100),
    patterns JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pp.platform,
        pp.patterns,
        pp.created_at,
        pp.updated_at
    FROM public.platform_patterns pp
    WHERE pp.user_id = p_user_id
    ORDER BY pp.updated_at DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get discovered platforms for a user
CREATE OR REPLACE FUNCTION get_discovered_platforms(p_user_id UUID)
RETURNS TABLE (
    platform_name VARCHAR(100),
    discovery_reason TEXT,
    confidence_score DECIMAL(3,2),
    discovered_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        dp.platform_name,
        dp.discovery_reason,
        dp.confidence_score,
        dp.discovered_at
    FROM public.discovered_platforms dp
    WHERE dp.user_id = p_user_id
    ORDER BY dp.discovered_at DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get platform statistics
CREATE OR REPLACE FUNCTION get_platform_stats(p_user_id UUID)
RETURNS TABLE (
    total_platforms INTEGER,
    learned_platforms INTEGER,
    discovered_platforms INTEGER,
    most_used_platform VARCHAR(100),
    latest_discovery TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(DISTINCT pp.platform)::INTEGER as total_platforms,
        COUNT(DISTINCT pp.platform)::INTEGER as learned_platforms,
        COUNT(DISTINCT dp.platform_name)::INTEGER as discovered_platforms,
        (SELECT platform FROM public.platform_patterns 
         WHERE user_id = p_user_id 
         ORDER BY updated_at DESC LIMIT 1) as most_used_platform,
        (SELECT discovered_at FROM public.discovered_platforms 
         WHERE user_id = p_user_id 
         ORDER BY discovered_at DESC LIMIT 1) as latest_discovery
    FROM public.platform_patterns pp
    LEFT JOIN public.discovered_platforms dp ON dp.user_id = p_user_id
    WHERE pp.user_id = p_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER; 