-- Create tables for Finley's Agent-Native Ingestion System

-- 1. raw_records table - stores all ingested data
CREATE TABLE public.raw_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    content JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    file_name TEXT,
    file_size INTEGER,
    classification_status TEXT DEFAULT 'pending' CHECK (classification_status IN ('pending', 'processing', 'completed', 'failed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- 2. metrics table - stores computed financial metrics
CREATE TABLE public.metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    record_id UUID REFERENCES public.raw_records(id) ON DELETE CASCADE,
    metric_type TEXT NOT NULL, -- 'revenue', 'expense', 'headcount', etc.
    category TEXT,
    subcategory TEXT,
    amount DECIMAL(15,2),
    currency TEXT DEFAULT 'USD',
    date_recorded DATE,
    period_start DATE,
    period_end DATE,
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    classification_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- 3. ingestion_jobs table - tracks async processing jobs
CREATE TABLE public.ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    record_id UUID REFERENCES public.raw_records(id) ON DELETE CASCADE,
    job_type TEXT NOT NULL, -- 'classification', 'normalization', 'metric_extraction'
    status TEXT DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    error_message TEXT,
    result JSONB,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.raw_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ingestion_jobs ENABLE ROW LEVEL SECURITY;

-- RLS Policies for raw_records
CREATE POLICY "Users can view their own records" ON public.raw_records
FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own records" ON public.raw_records
FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own records" ON public.raw_records
FOR UPDATE USING (auth.uid() = user_id);

-- RLS Policies for metrics
CREATE POLICY "Users can view their own metrics" ON public.metrics
FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own metrics" ON public.metrics
FOR INSERT WITH CHECK (auth.uid() = user_id);

-- RLS Policies for ingestion_jobs
CREATE POLICY "Users can view their own jobs" ON public.ingestion_jobs
FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own jobs" ON public.ingestion_jobs
FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own jobs" ON public.ingestion_jobs
FOR UPDATE USING (auth.uid() = user_id);

-- Create indexes for better performance
CREATE INDEX idx_raw_records_user_id ON public.raw_records(user_id);
CREATE INDEX idx_raw_records_status ON public.raw_records(status);
CREATE INDEX idx_raw_records_created_at ON public.raw_records(created_at);
CREATE INDEX idx_metrics_user_id ON public.metrics(user_id);
CREATE INDEX idx_metrics_record_id ON public.metrics(record_id);
CREATE INDEX idx_metrics_type ON public.metrics(metric_type);
CREATE INDEX idx_metrics_date ON public.metrics(date_recorded);
CREATE INDEX idx_ingestion_jobs_user_id ON public.ingestion_jobs(user_id);
CREATE INDEX idx_ingestion_jobs_status ON public.ingestion_jobs(status);

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add triggers for updated_at
CREATE TRIGGER update_raw_records_updated_at
    BEFORE UPDATE ON public.raw_records
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_metrics_updated_at
    BEFORE UPDATE ON public.metrics
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

-- Create storage bucket for file uploads
INSERT INTO storage.buckets (id, name, public) 
VALUES ('finley-uploads', 'finley-uploads', false);

-- Storage policies for file uploads
CREATE POLICY "Users can upload their own files" ON storage.objects
FOR INSERT WITH CHECK (
    bucket_id = 'finley-uploads' AND 
    auth.uid()::text = (storage.foldername(name))[1]
);

CREATE POLICY "Users can view their own files" ON storage.objects
FOR SELECT USING (
    bucket_id = 'finley-uploads' AND 
    auth.uid()::text = (storage.foldername(name))[1]
);

CREATE POLICY "Users can update their own files" ON storage.objects
FOR UPDATE USING (
    bucket_id = 'finley-uploads' AND 
    auth.uid()::text = (storage.foldername(name))[1]
);

CREATE POLICY "Users can delete their own files" ON storage.objects
FOR DELETE USING (
    bucket_id = 'finley-uploads' AND 
    auth.uid()::text = (storage.foldername(name))[1]
);