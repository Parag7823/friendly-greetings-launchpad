export interface RawEvent {
  id: string;
  user_id?: string;
  file_id?: string;
  job_id?: string;
  provider: string;
  kind: string;
  source_platform?: string;
  payload: any;
  row_index: number;
  sheet_name?: string;
  source_filename: string;
  uploader?: string;
  ingest_ts?: string;
  processed_at?: string;
  status?: string;
  error_message?: string;
  confidence_score?: number;
  classification_metadata?: any;
  created_at?: string;
  updated_at?: string;
  category?: string;
  subcategory?: string;
  entities?: any;
  relationships?: any;
}

export interface DatabaseRow {
  [key: string]: any;
}

export interface FileProcessingEvent {
  row_data: DatabaseRow;
  row_index: number;
  sheet_name: string;
  file_name: string;
  classification?: {
    category?: string;
    subcategory?: string;
    confidence_score?: number;
    entities?: any;
  };
}