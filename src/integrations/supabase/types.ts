export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "12.2.3 (519615d)"
  }
  public: {
    Tables: {
      causal_relationships: {
        Row: {
          causal_direction: string | null
          causal_score: number | null
          consistency_score: number | null
          created_at: string | null
          criteria_details: Json | null
          dose_response_score: number | null
          id: string
          is_causal: boolean | null
          job_id: string | null
          plausibility_score: number | null
          relationship_id: string
          specificity_score: number | null
          strength_score: number | null
          temporal_precedence_score: number | null
          updated_at: string | null
          user_id: string
        }
        Insert: {
          causal_direction?: string | null
          causal_score?: number | null
          consistency_score?: number | null
          created_at?: string | null
          criteria_details?: Json | null
          dose_response_score?: number | null
          id?: string
          is_causal?: boolean | null
          job_id?: string | null
          plausibility_score?: number | null
          relationship_id: string
          specificity_score?: number | null
          strength_score?: number | null
          temporal_precedence_score?: number | null
          updated_at?: string | null
          user_id: string
        }
        Update: {
          causal_direction?: string | null
          causal_score?: number | null
          consistency_score?: number | null
          created_at?: string | null
          criteria_details?: Json | null
          dose_response_score?: number | null
          id?: string
          is_causal?: boolean | null
          job_id?: string | null
          plausibility_score?: number | null
          relationship_id?: string
          specificity_score?: number | null
          strength_score?: number | null
          temporal_precedence_score?: number | null
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "causal_relationships_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "causal_relationships_relationship_id_fkey"
            columns: ["relationship_id"]
            isOneToOne: true
            referencedRelation: "enriched_relationships"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "causal_relationships_relationship_id_fkey"
            columns: ["relationship_id"]
            isOneToOne: true
            referencedRelation: "relationship_instances"
            referencedColumns: ["id"]
          },
        ]
      }
      chat_messages: {
        Row: {
          chat_id: string
          chat_title: string | null
          confidence: number | null
          created_at: string | null
          id: string
          message: string
          metadata: Json | null
          question_type: string | null
          response: string | null
          role: string
          updated_at: string | null
          user_id: string | null
        }
        Insert: {
          chat_id: string
          chat_title?: string | null
          confidence?: number | null
          created_at?: string | null
          id?: string
          message: string
          metadata?: Json | null
          question_type?: string | null
          response?: string | null
          role: string
          updated_at?: string | null
          user_id?: string | null
        }
        Update: {
          chat_id?: string
          chat_title?: string | null
          confidence?: number | null
          created_at?: string | null
          id?: string
          message?: string
          metadata?: Json | null
          question_type?: string | null
          response?: string | null
          role?: string
          updated_at?: string | null
          user_id?: string | null
        }
        Relationships: []
      }
      connectors: {
        Row: {
          auth_type: string
          created_at: string | null
          enabled: boolean
          endpoints_needed: Json
          id: string
          integration_id: string
          job_id: string | null
          metadata: Json | null
          provider: string
          scopes: Json
          updated_at: string | null
        }
        Insert: {
          auth_type?: string
          created_at?: string | null
          enabled?: boolean
          endpoints_needed?: Json
          id?: string
          integration_id: string
          job_id?: string | null
          metadata?: Json | null
          provider: string
          scopes?: Json
          updated_at?: string | null
        }
        Update: {
          auth_type?: string
          created_at?: string | null
          enabled?: boolean
          endpoints_needed?: Json
          id?: string
          integration_id?: string
          job_id?: string | null
          metadata?: Json | null
          provider?: string
          scopes?: Json
          updated_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "connectors_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
        ]
      }
      counterfactual_analyses: {
        Row: {
          affected_event_count: number | null
          affected_events: Json | null
          analysis_metadata: Json | null
          counterfactual_value: Json
          created_at: string | null
          id: string
          intervention_event_id: string
          intervention_type: string
          job_id: string | null
          original_value: Json
          scenario_description: string | null
          scenario_name: string | null
          total_impact_delta_usd: number | null
          user_id: string
        }
        Insert: {
          affected_event_count?: number | null
          affected_events?: Json | null
          analysis_metadata?: Json | null
          counterfactual_value: Json
          created_at?: string | null
          id?: string
          intervention_event_id: string
          intervention_type: string
          job_id?: string | null
          original_value: Json
          scenario_description?: string | null
          scenario_name?: string | null
          total_impact_delta_usd?: number | null
          user_id: string
        }
        Update: {
          affected_event_count?: number | null
          affected_events?: Json | null
          analysis_metadata?: Json | null
          counterfactual_value?: Json
          created_at?: string | null
          id?: string
          intervention_event_id?: string
          intervention_type?: string
          job_id?: string | null
          original_value?: Json
          scenario_description?: string | null
          scenario_name?: string | null
          total_impact_delta_usd?: number | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "counterfactual_analyses_intervention_event_id_fkey"
            columns: ["intervention_event_id"]
            isOneToOne: false
            referencedRelation: "raw_events"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "counterfactual_analyses_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
        ]
      }
      cross_platform_relationships: {
        Row: {
          confidence_score: number | null
          created_at: string | null
          detection_method: string | null
          id: string
          job_id: string | null
          platform_compatibility: string | null
          relationship_type: string
          source_event_id: string
          source_platform: string | null
          target_event_id: string
          target_platform: string | null
          transaction_id: string | null
          user_id: string
        }
        Insert: {
          confidence_score?: number | null
          created_at?: string | null
          detection_method?: string | null
          id?: string
          job_id?: string | null
          platform_compatibility?: string | null
          relationship_type: string
          source_event_id: string
          source_platform?: string | null
          target_event_id: string
          target_platform?: string | null
          transaction_id?: string | null
          user_id: string
        }
        Update: {
          confidence_score?: number | null
          created_at?: string | null
          detection_method?: string | null
          id?: string
          job_id?: string | null
          platform_compatibility?: string | null
          relationship_type?: string
          source_event_id?: string
          source_platform?: string | null
          target_event_id?: string
          target_platform?: string | null
          transaction_id?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "cross_platform_relationships_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "cross_platform_relationships_source_event_id_fkey"
            columns: ["source_event_id"]
            isOneToOne: false
            referencedRelation: "raw_events"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "cross_platform_relationships_target_event_id_fkey"
            columns: ["target_event_id"]
            isOneToOne: false
            referencedRelation: "raw_events"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "cross_platform_relationships_transaction_id_fkey"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      discovered_platforms: {
        Row: {
          confidence_score: number | null
          discovered_at: string | null
          discovery_reason: string | null
          id: string
          job_id: string | null
          platform_name: string
          transaction_id: string | null
          user_id: string
        }
        Insert: {
          confidence_score?: number | null
          discovered_at?: string | null
          discovery_reason?: string | null
          id?: string
          job_id?: string | null
          platform_name: string
          transaction_id?: string | null
          user_id: string
        }
        Update: {
          confidence_score?: number | null
          discovered_at?: string | null
          discovery_reason?: string | null
          id?: string
          job_id?: string | null
          platform_name?: string
          transaction_id?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "discovered_platforms_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "discovered_platforms_transaction_id_fkey"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      entity_matches: {
        Row: {
          created_at: string | null
          id: string
          job_id: string | null
          match_confidence: number
          match_reason: string
          matched_fields: string[] | null
          normalized_entity_id: string | null
          similarity_score: number | null
          source_entity_name: string
          source_entity_type: string
          source_file: string
          source_platform: string
          source_row_id: string | null
          transaction_id: string | null
          user_id: string | null
        }
        Insert: {
          created_at?: string | null
          id?: string
          job_id?: string | null
          match_confidence: number
          match_reason: string
          matched_fields?: string[] | null
          normalized_entity_id?: string | null
          similarity_score?: number | null
          source_entity_name: string
          source_entity_type: string
          source_file: string
          source_platform: string
          source_row_id?: string | null
          transaction_id?: string | null
          user_id?: string | null
        }
        Update: {
          created_at?: string | null
          id?: string
          job_id?: string | null
          match_confidence?: number
          match_reason?: string
          matched_fields?: string[] | null
          normalized_entity_id?: string | null
          similarity_score?: number | null
          source_entity_name?: string
          source_entity_type?: string
          source_file?: string
          source_platform?: string
          source_row_id?: string | null
          transaction_id?: string | null
          user_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "entity_matches_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "entity_matches_normalized_entity_id_fkey"
            columns: ["normalized_entity_id"]
            isOneToOne: false
            referencedRelation: "normalized_entities"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "entity_matches_source_row_id_fkey"
            columns: ["source_row_id"]
            isOneToOne: false
            referencedRelation: "raw_events"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "entity_matches_transaction_fk"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      event_delta_logs: {
        Row: {
          confidence_score: number | null
          created_at: string | null
          delta_summary: Json
          events_included: Json
          existing_file_id: string
          id: string
          job_id: string | null
          merge_type: string | null
          new_file_id: string
          rows_added: number | null
          rows_skipped: number | null
          user_id: string
        }
        Insert: {
          confidence_score?: number | null
          created_at?: string | null
          delta_summary?: Json
          events_included?: Json
          existing_file_id: string
          id?: string
          job_id?: string | null
          merge_type?: string | null
          new_file_id: string
          rows_added?: number | null
          rows_skipped?: number | null
          user_id: string
        }
        Update: {
          confidence_score?: number | null
          created_at?: string | null
          delta_summary?: Json
          events_included?: Json
          existing_file_id?: string
          id?: string
          job_id?: string | null
          merge_type?: string | null
          new_file_id?: string
          rows_added?: number | null
          rows_skipped?: number | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "event_delta_logs_existing_file_id_fkey"
            columns: ["existing_file_id"]
            isOneToOne: false
            referencedRelation: "raw_records"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "event_delta_logs_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "event_delta_logs_new_file_id_fkey"
            columns: ["new_file_id"]
            isOneToOne: false
            referencedRelation: "raw_records"
            referencedColumns: ["id"]
          },
        ]
      }
      external_items: {
        Row: {
          created_at: string | null
          error: string | null
          hash: string | null
          id: string
          kind: string
          metadata: Json
          provider_id: string
          relevance_score: number | null
          source_ts: string | null
          status: string
          storage_path: string | null
          user_connection_id: string
          user_id: string
        }
        Insert: {
          created_at?: string | null
          error?: string | null
          hash?: string | null
          id?: string
          kind: string
          metadata?: Json
          provider_id: string
          relevance_score?: number | null
          source_ts?: string | null
          status?: string
          storage_path?: string | null
          user_connection_id: string
          user_id: string
        }
        Update: {
          created_at?: string | null
          error?: string | null
          hash?: string | null
          id?: string
          kind?: string
          metadata?: Json
          provider_id?: string
          relevance_score?: number | null
          source_ts?: string | null
          status?: string
          storage_path?: string | null
          user_connection_id?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "external_items_user_connection_id_fkey"
            columns: ["user_connection_id"]
            isOneToOne: false
            referencedRelation: "user_connections"
            referencedColumns: ["id"]
          },
        ]
      }
      field_mappings: {
        Row: {
          confidence: number | null
          created_at: string | null
          document_type: string | null
          filename_pattern: string | null
          id: string
          last_used_at: string | null
          mapping_source: string
          metadata: Json | null
          platform: string | null
          source_column: string
          success_count: number | null
          target_field: string
          transaction_id: string | null
          updated_at: string | null
          usage_count: number | null
          user_id: string | null
        }
        Insert: {
          confidence?: number | null
          created_at?: string | null
          document_type?: string | null
          filename_pattern?: string | null
          id?: string
          last_used_at?: string | null
          mapping_source: string
          metadata?: Json | null
          platform?: string | null
          source_column: string
          success_count?: number | null
          target_field: string
          transaction_id?: string | null
          updated_at?: string | null
          usage_count?: number | null
          user_id?: string | null
        }
        Update: {
          confidence?: number | null
          created_at?: string | null
          document_type?: string | null
          filename_pattern?: string | null
          id?: string
          last_used_at?: string | null
          mapping_source?: string
          metadata?: Json | null
          platform?: string | null
          source_column?: string
          success_count?: number | null
          target_field?: string
          transaction_id?: string | null
          updated_at?: string | null
          usage_count?: number | null
          user_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "field_mappings_transaction_id_fkey"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      ingestion_jobs: {
        Row: {
          ai_detection_summary: Json | null
          completed_at: string | null
          created_at: string | null
          duplicate_status: string | null
          error_details: string | null
          error_message: string | null
          errors_json: Json | null
          extracted_rows: number | null
          file_id: string | null
          file_size: number | null
          filename: string | null
          final_output: Json | null
          id: string
          job_type: string
          last_retry_at: string | null
          max_retries: number | null
          next_retry_at: string | null
          priority: string | null
          processing_stage: string | null
          progress: number | null
          progress_percentage: number | null
          record_id: string | null
          result: Json | null
          retry_count: number | null
          source: string
          started_at: string | null
          status: string | null
          status_message: string | null
          stream_offset: number | null
          total_rows: number | null
          transaction_id: string | null
          updated_at: string | null
          user_id: string | null
        }
        Insert: {
          ai_detection_summary?: Json | null
          completed_at?: string | null
          created_at?: string | null
          duplicate_status?: string | null
          error_details?: string | null
          error_message?: string | null
          errors_json?: Json | null
          extracted_rows?: number | null
          file_id?: string | null
          file_size?: number | null
          filename?: string | null
          final_output?: Json | null
          id?: string
          job_type: string
          last_retry_at?: string | null
          max_retries?: number | null
          next_retry_at?: string | null
          priority?: string | null
          processing_stage?: string | null
          progress?: number | null
          progress_percentage?: number | null
          record_id?: string | null
          result?: Json | null
          retry_count?: number | null
          source?: string
          started_at?: string | null
          status?: string | null
          status_message?: string | null
          stream_offset?: number | null
          total_rows?: number | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Update: {
          ai_detection_summary?: Json | null
          completed_at?: string | null
          created_at?: string | null
          duplicate_status?: string | null
          error_details?: string | null
          error_message?: string | null
          errors_json?: Json | null
          extracted_rows?: number | null
          file_id?: string | null
          file_size?: number | null
          filename?: string | null
          final_output?: Json | null
          id?: string
          job_type?: string
          last_retry_at?: string | null
          max_retries?: number | null
          next_retry_at?: string | null
          priority?: string | null
          processing_stage?: string | null
          progress?: number | null
          progress_percentage?: number | null
          record_id?: string | null
          result?: Json | null
          retry_count?: number | null
          source?: string
          started_at?: string | null
          status?: string | null
          status_message?: string | null
          stream_offset?: number | null
          total_rows?: number | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "ingestion_jobs_file_id_fkey"
            columns: ["file_id"]
            isOneToOne: false
            referencedRelation: "raw_records"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "ingestion_jobs_transaction_id_fkey"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      normalized_entities: {
        Row: {
          aliases: string[] | null
          bank_account: string | null
          canonical_name: string
          canonical_name_dmetaphone: string | null
          canonical_name_metaphone: string | null
          canonical_name_soundex: string | null
          confidence_score: number | null
          created_at: string | null
          email: string | null
          entity_type: string
          first_seen_at: string | null
          id: string
          is_deleted: boolean | null
          job_id: string | null
          last_seen_at: string | null
          lineage_path: Json | null
          phone: string | null
          platform_sources: string[] | null
          source_files: string[] | null
          tax_id: string | null
          transaction_id: string | null
          updated_at: string | null
          user_id: string | null
        }
        Insert: {
          aliases?: string[] | null
          bank_account?: string | null
          canonical_name: string
          canonical_name_dmetaphone?: string | null
          canonical_name_metaphone?: string | null
          canonical_name_soundex?: string | null
          confidence_score?: number | null
          created_at?: string | null
          email?: string | null
          entity_type: string
          first_seen_at?: string | null
          id?: string
          is_deleted?: boolean | null
          job_id?: string | null
          last_seen_at?: string | null
          lineage_path?: Json | null
          phone?: string | null
          platform_sources?: string[] | null
          source_files?: string[] | null
          tax_id?: string | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Update: {
          aliases?: string[] | null
          bank_account?: string | null
          canonical_name?: string
          canonical_name_dmetaphone?: string | null
          canonical_name_metaphone?: string | null
          canonical_name_soundex?: string | null
          confidence_score?: number | null
          created_at?: string | null
          email?: string | null
          entity_type?: string
          first_seen_at?: string | null
          id?: string
          is_deleted?: boolean | null
          job_id?: string | null
          last_seen_at?: string | null
          lineage_path?: Json | null
          phone?: string | null
          platform_sources?: string[] | null
          source_files?: string[] | null
          tax_id?: string | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "normalized_entities_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "normalized_entities_transaction_id_fkey"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      normalized_events: {
        Row: {
          causal_links: Json | null
          causal_reasoning: string | null
          causal_weight: number | null
          confidence_scores: Json | null
          created_at: string | null
          document_subtype: string | null
          document_type: string | null
          duplicate_group_id: string | null
          duplicate_hash: string | null
          final_platform: Json | null
          id: string
          job_id: string | null
          merge_strategy: string | null
          normalization_confidence: number | null
          normalization_method: string | null
          normalized_at: string | null
          normalized_payload: Json
          pattern_used_for_prediction: string | null
          platform_label: string | null
          prediction_confidence: number | null
          previous_versions: Json | null
          raw_event_id: string | null
          relationship_count: number | null
          relationship_evidence: Json | null
          requires_review: boolean | null
          resolved_entities: Json | null
          review_reason: string | null
          semantic_confidence: number | null
          semantic_links: Json | null
          temporal_confidence: number | null
          temporal_cycle_metadata: Json | null
          temporal_patterns: Json | null
          transaction_id: string | null
          updated_at: string | null
          user_id: string | null
          version_number: number | null
        }
        Insert: {
          causal_links?: Json | null
          causal_reasoning?: string | null
          causal_weight?: number | null
          confidence_scores?: Json | null
          created_at?: string | null
          document_subtype?: string | null
          document_type?: string | null
          duplicate_group_id?: string | null
          duplicate_hash?: string | null
          final_platform?: Json | null
          id?: string
          job_id?: string | null
          merge_strategy?: string | null
          normalization_confidence?: number | null
          normalization_method?: string | null
          normalized_at?: string | null
          normalized_payload?: Json
          pattern_used_for_prediction?: string | null
          platform_label?: string | null
          prediction_confidence?: number | null
          previous_versions?: Json | null
          raw_event_id?: string | null
          relationship_count?: number | null
          relationship_evidence?: Json | null
          requires_review?: boolean | null
          resolved_entities?: Json | null
          review_reason?: string | null
          semantic_confidence?: number | null
          semantic_links?: Json | null
          temporal_confidence?: number | null
          temporal_cycle_metadata?: Json | null
          temporal_patterns?: Json | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id?: string | null
          version_number?: number | null
        }
        Update: {
          causal_links?: Json | null
          causal_reasoning?: string | null
          causal_weight?: number | null
          confidence_scores?: Json | null
          created_at?: string | null
          document_subtype?: string | null
          document_type?: string | null
          duplicate_group_id?: string | null
          duplicate_hash?: string | null
          final_platform?: Json | null
          id?: string
          job_id?: string | null
          merge_strategy?: string | null
          normalization_confidence?: number | null
          normalization_method?: string | null
          normalized_at?: string | null
          normalized_payload?: Json
          pattern_used_for_prediction?: string | null
          platform_label?: string | null
          prediction_confidence?: number | null
          previous_versions?: Json | null
          raw_event_id?: string | null
          relationship_count?: number | null
          relationship_evidence?: Json | null
          requires_review?: boolean | null
          resolved_entities?: Json | null
          review_reason?: string | null
          semantic_confidence?: number | null
          semantic_links?: Json | null
          temporal_confidence?: number | null
          temporal_cycle_metadata?: Json | null
          temporal_patterns?: Json | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id?: string | null
          version_number?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "normalized_events_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "normalized_events_raw_event_id_fkey"
            columns: ["raw_event_id"]
            isOneToOne: false
            referencedRelation: "raw_events"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "normalized_events_transaction_id_fkey"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      platform_patterns: {
        Row: {
          confidence_score: number | null
          created_at: string | null
          detection_method: string | null
          id: string
          job_id: string | null
          pattern_data: Json | null
          pattern_type: string | null
          patterns: Json
          platform: string
          transaction_id: string | null
          updated_at: string | null
          user_id: string
        }
        Insert: {
          confidence_score?: number | null
          created_at?: string | null
          detection_method?: string | null
          id?: string
          job_id?: string | null
          pattern_data?: Json | null
          pattern_type?: string | null
          patterns?: Json
          platform: string
          transaction_id?: string | null
          updated_at?: string | null
          user_id: string
        }
        Update: {
          confidence_score?: number | null
          created_at?: string | null
          detection_method?: string | null
          id?: string
          job_id?: string | null
          pattern_data?: Json | null
          pattern_type?: string | null
          patterns?: Json
          platform?: string
          transaction_id?: string | null
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "platform_patterns_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "platform_patterns_transaction_id_fkey"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      predicted_relationships: {
        Row: {
          confidence_score: number | null
          created_at: string | null
          days_until_expected: number | null
          expected_date: string | null
          expected_date_range_end: string | null
          expected_date_range_start: string | null
          fulfilled_at: string | null
          fulfilled_by_event_id: string | null
          id: string
          job_id: string | null
          metadata: Json | null
          pattern_id: string | null
          predicted_at: string | null
          predicted_relationship_type: string | null
          predicted_target_type: string | null
          prediction_basis: Json | null
          prediction_method: string | null
          prediction_reasoning: string | null
          relationship_type: string
          source_entity_id: string | null
          source_event_id: string | null
          status: string | null
          target_entity_id: string | null
          temporal_pattern_id: string | null
          transaction_id: string | null
          updated_at: string | null
          user_id: string
        }
        Insert: {
          confidence_score?: number | null
          created_at?: string | null
          days_until_expected?: number | null
          expected_date?: string | null
          expected_date_range_end?: string | null
          expected_date_range_start?: string | null
          fulfilled_at?: string | null
          fulfilled_by_event_id?: string | null
          id?: string
          job_id?: string | null
          metadata?: Json | null
          pattern_id?: string | null
          predicted_at?: string | null
          predicted_relationship_type?: string | null
          predicted_target_type?: string | null
          prediction_basis?: Json | null
          prediction_method?: string | null
          prediction_reasoning?: string | null
          relationship_type: string
          source_entity_id?: string | null
          source_event_id?: string | null
          status?: string | null
          target_entity_id?: string | null
          temporal_pattern_id?: string | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id: string
        }
        Update: {
          confidence_score?: number | null
          created_at?: string | null
          days_until_expected?: number | null
          expected_date?: string | null
          expected_date_range_end?: string | null
          expected_date_range_start?: string | null
          fulfilled_at?: string | null
          fulfilled_by_event_id?: string | null
          id?: string
          job_id?: string | null
          metadata?: Json | null
          pattern_id?: string | null
          predicted_at?: string | null
          predicted_relationship_type?: string | null
          predicted_target_type?: string | null
          prediction_basis?: Json | null
          prediction_method?: string | null
          prediction_reasoning?: string | null
          relationship_type?: string
          source_entity_id?: string | null
          source_event_id?: string | null
          status?: string | null
          target_entity_id?: string | null
          temporal_pattern_id?: string | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "predicted_relationships_fulfilled_by_event_id_fkey"
            columns: ["fulfilled_by_event_id"]
            isOneToOne: false
            referencedRelation: "raw_events"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "predicted_relationships_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "predicted_relationships_pattern_fk"
            columns: ["pattern_id"]
            isOneToOne: false
            referencedRelation: "relationship_patterns"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "predicted_relationships_source_event_id_fkey"
            columns: ["source_event_id"]
            isOneToOne: false
            referencedRelation: "raw_events"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "predicted_relationships_temporal_pattern_id_fkey"
            columns: ["temporal_pattern_id"]
            isOneToOne: false
            referencedRelation: "temporal_patterns"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "predicted_relationships_transaction_fk"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      processing_locks: {
        Row: {
          acquired_at: string
          created_at: string | null
          expires_at: string
          id: string
          job_id: string | null
          lock_type: string
          metadata: Json | null
          resource_id: string
          status: string
          user_id: string | null
        }
        Insert: {
          acquired_at?: string
          created_at?: string | null
          expires_at: string
          id: string
          job_id?: string | null
          lock_type: string
          metadata?: Json | null
          resource_id: string
          status?: string
          user_id?: string | null
        }
        Update: {
          acquired_at?: string
          created_at?: string | null
          expires_at?: string
          id?: string
          job_id?: string | null
          lock_type?: string
          metadata?: Json | null
          resource_id?: string
          status?: string
          user_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "processing_locks_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
        ]
      }
      processing_transactions: {
        Row: {
          committed_at: string | null
          created_at: string | null
          end_time: string | null
          error_details: string | null
          failed_at: string | null
          file_id: string | null
          id: string
          inserted_ids: Json | null
          job_id: string | null
          metadata: Json | null
          operation_type: string
          rollback_data: Json | null
          rolled_back_at: string | null
          start_time: string | null
          started_at: string
          status: string
          updated_at: string | null
          updated_ids: Json | null
          user_id: string | null
        }
        Insert: {
          committed_at?: string | null
          created_at?: string | null
          end_time?: string | null
          error_details?: string | null
          failed_at?: string | null
          file_id?: string | null
          id?: string
          inserted_ids?: Json | null
          job_id?: string | null
          metadata?: Json | null
          operation_type: string
          rollback_data?: Json | null
          rolled_back_at?: string | null
          start_time?: string | null
          started_at: string
          status: string
          updated_at?: string | null
          updated_ids?: Json | null
          user_id?: string | null
        }
        Update: {
          committed_at?: string | null
          created_at?: string | null
          end_time?: string | null
          error_details?: string | null
          failed_at?: string | null
          file_id?: string | null
          id?: string
          inserted_ids?: Json | null
          job_id?: string | null
          metadata?: Json | null
          operation_type?: string
          rollback_data?: Json | null
          rolled_back_at?: string | null
          start_time?: string | null
          started_at?: string
          status?: string
          updated_at?: string | null
          updated_ids?: Json | null
          user_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "fk_processing_transactions_file_id"
            columns: ["file_id"]
            isOneToOne: false
            referencedRelation: "raw_records"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "fk_processing_transactions_job_id"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
        ]
      }
      raw_events: {
        Row: {
          accuracy_enhanced: boolean | null
          accuracy_version: string | null
          affects_cash: boolean | null
          ai_confidence: number | null
          ai_reasoning: string | null
          amount_direction: string | null
          amount_original: number | null
          amount_signed_usd: number | null
          amount_usd: number | null
          category: string | null
          classification_metadata: Json | null
          confidence_score: number | null
          created_at: string | null
          created_by: string | null
          currency: string | null
          document_confidence: number | null
          document_type: string | null
          entities: Json | null
          error_message: string | null
          exchange_date: string | null
          exchange_rate: number | null
          exchange_rate_date: string | null
          file_id: string | null
          id: string
          ingest_ts: string | null
          ingested_on: string | null
          ingested_ts: string | null
          is_deleted: boolean | null
          is_valid: boolean | null
          job_id: string | null
          kind: string
          last_relationship_check: string | null
          lineage_path: Json | null
          modified_at: string | null
          modified_by: string | null
          overall_confidence: number | null
          payload: Json
          platform_ids: Json | null
          processed_at: string | null
          processed_ts: string | null
          provider: string
          relationship_count: number | null
          relationships: Json | null
          requires_review: boolean | null
          review_priority: string | null
          review_reason: string | null
          row_hash: string | null
          row_index: number
          sheet_name: string | null
          source_filename: string
          source_platform: string | null
          source_ts: string | null
          standard_description: string | null
          status: string | null
          subcategory: string | null
          transaction_date: string | null
          transaction_id: string | null
          transaction_type: string | null
          updated_at: string | null
          uploader: string | null
          user_id: string | null
          validation_flags: Json | null
          vendor_alternatives: Json | null
          vendor_canonical_id: string | null
          vendor_cleaning_method: string | null
          vendor_confidence: number | null
          vendor_raw: string | null
          vendor_standard: string | null
          vendor_verified: boolean | null
        }
        Insert: {
          accuracy_enhanced?: boolean | null
          accuracy_version?: string | null
          affects_cash?: boolean | null
          ai_confidence?: number | null
          ai_reasoning?: string | null
          amount_direction?: string | null
          amount_original?: number | null
          amount_signed_usd?: number | null
          amount_usd?: number | null
          category?: string | null
          classification_metadata?: Json | null
          confidence_score?: number | null
          created_at?: string | null
          created_by?: string | null
          currency?: string | null
          document_confidence?: number | null
          document_type?: string | null
          entities?: Json | null
          error_message?: string | null
          exchange_date?: string | null
          exchange_rate?: number | null
          exchange_rate_date?: string | null
          file_id?: string | null
          id?: string
          ingest_ts?: string | null
          ingested_on?: string | null
          ingested_ts?: string | null
          is_deleted?: boolean | null
          is_valid?: boolean | null
          job_id?: string | null
          kind: string
          last_relationship_check?: string | null
          lineage_path?: Json | null
          modified_at?: string | null
          modified_by?: string | null
          overall_confidence?: number | null
          payload: Json
          platform_ids?: Json | null
          processed_at?: string | null
          processed_ts?: string | null
          provider: string
          relationship_count?: number | null
          relationships?: Json | null
          requires_review?: boolean | null
          review_priority?: string | null
          review_reason?: string | null
          row_hash?: string | null
          row_index: number
          sheet_name?: string | null
          source_filename: string
          source_platform?: string | null
          source_ts?: string | null
          standard_description?: string | null
          status?: string | null
          subcategory?: string | null
          transaction_date?: string | null
          transaction_id?: string | null
          transaction_type?: string | null
          updated_at?: string | null
          uploader?: string | null
          user_id?: string | null
          validation_flags?: Json | null
          vendor_alternatives?: Json | null
          vendor_canonical_id?: string | null
          vendor_cleaning_method?: string | null
          vendor_confidence?: number | null
          vendor_raw?: string | null
          vendor_standard?: string | null
          vendor_verified?: boolean | null
        }
        Update: {
          accuracy_enhanced?: boolean | null
          accuracy_version?: string | null
          affects_cash?: boolean | null
          ai_confidence?: number | null
          ai_reasoning?: string | null
          amount_direction?: string | null
          amount_original?: number | null
          amount_signed_usd?: number | null
          amount_usd?: number | null
          category?: string | null
          classification_metadata?: Json | null
          confidence_score?: number | null
          created_at?: string | null
          created_by?: string | null
          currency?: string | null
          document_confidence?: number | null
          document_type?: string | null
          entities?: Json | null
          error_message?: string | null
          exchange_date?: string | null
          exchange_rate?: number | null
          exchange_rate_date?: string | null
          file_id?: string | null
          id?: string
          ingest_ts?: string | null
          ingested_on?: string | null
          ingested_ts?: string | null
          is_deleted?: boolean | null
          is_valid?: boolean | null
          job_id?: string | null
          kind?: string
          last_relationship_check?: string | null
          lineage_path?: Json | null
          modified_at?: string | null
          modified_by?: string | null
          overall_confidence?: number | null
          payload?: Json
          platform_ids?: Json | null
          processed_at?: string | null
          processed_ts?: string | null
          provider?: string
          relationship_count?: number | null
          relationships?: Json | null
          requires_review?: boolean | null
          review_priority?: string | null
          review_reason?: string | null
          row_hash?: string | null
          row_index?: number
          sheet_name?: string | null
          source_filename?: string
          source_platform?: string | null
          source_ts?: string | null
          standard_description?: string | null
          status?: string | null
          subcategory?: string | null
          transaction_date?: string | null
          transaction_id?: string | null
          transaction_type?: string | null
          updated_at?: string | null
          uploader?: string | null
          user_id?: string | null
          validation_flags?: Json | null
          vendor_alternatives?: Json | null
          vendor_canonical_id?: string | null
          vendor_cleaning_method?: string | null
          vendor_confidence?: number | null
          vendor_raw?: string | null
          vendor_standard?: string | null
          vendor_verified?: boolean | null
        }
        Relationships: [
          {
            foreignKeyName: "raw_events_file_id_fkey"
            columns: ["file_id"]
            isOneToOne: false
            referencedRelation: "raw_records"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "raw_events_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "raw_events_transaction_id_fkey"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      raw_records: {
        Row: {
          classification_status: string | null
          content: Json
          created_at: string | null
          decision_metadata: Json | null
          decision_timestamp: string | null
          duplicate_decision: string | null
          duplicate_of: string | null
          external_item_id: string | null
          file_hash: string | null
          file_hash_verified: boolean | null
          file_name: string | null
          file_size: number | null
          id: string
          ingested_at: string | null
          integrity_check_at: string | null
          is_duplicate: boolean | null
          metadata: Json | null
          source: string
          status: string | null
          transaction_id: string | null
          updated_at: string | null
          user_id: string | null
        }
        Insert: {
          classification_status?: string | null
          content: Json
          created_at?: string | null
          decision_metadata?: Json | null
          decision_timestamp?: string | null
          duplicate_decision?: string | null
          duplicate_of?: string | null
          external_item_id?: string | null
          file_hash?: string | null
          file_hash_verified?: boolean | null
          file_name?: string | null
          file_size?: number | null
          id?: string
          ingested_at?: string | null
          integrity_check_at?: string | null
          is_duplicate?: boolean | null
          metadata?: Json | null
          source: string
          status?: string | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Update: {
          classification_status?: string | null
          content?: Json
          created_at?: string | null
          decision_metadata?: Json | null
          decision_timestamp?: string | null
          duplicate_decision?: string | null
          duplicate_of?: string | null
          external_item_id?: string | null
          file_hash?: string | null
          file_hash_verified?: boolean | null
          file_name?: string | null
          file_size?: number | null
          id?: string
          ingested_at?: string | null
          integrity_check_at?: string | null
          is_duplicate?: boolean | null
          metadata?: Json | null
          source?: string
          status?: string | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "fk_raw_records_external_item"
            columns: ["external_item_id"]
            isOneToOne: false
            referencedRelation: "external_items"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "raw_records_duplicate_of_fkey"
            columns: ["duplicate_of"]
            isOneToOne: false
            referencedRelation: "raw_records"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "raw_records_transaction_id_fkey"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      relationship_instances: {
        Row: {
          business_logic: string | null
          confidence_score: number | null
          created_at: string | null
          detection_method: string | null
          duplicate_confidence: number | null
          id: string
          is_deleted: boolean | null
          is_duplicate: boolean | null
          job_id: string | null
          key_factors: Json | null
          metadata: Json | null
          pattern_id: string | null
          reasoning: string | null
          relationship_embedding: string | null
          relationship_type: string
          semantic_description: string | null
          source_event_id: string
          target_event_id: string
          temporal_causality: string | null
          transaction_id: string | null
          updated_at: string | null
          user_id: string
        }
        Insert: {
          business_logic?: string | null
          confidence_score?: number | null
          created_at?: string | null
          detection_method?: string | null
          duplicate_confidence?: number | null
          id?: string
          is_deleted?: boolean | null
          is_duplicate?: boolean | null
          job_id?: string | null
          key_factors?: Json | null
          metadata?: Json | null
          pattern_id?: string | null
          reasoning?: string | null
          relationship_embedding?: string | null
          relationship_type: string
          semantic_description?: string | null
          source_event_id: string
          target_event_id: string
          temporal_causality?: string | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id: string
        }
        Update: {
          business_logic?: string | null
          confidence_score?: number | null
          created_at?: string | null
          detection_method?: string | null
          duplicate_confidence?: number | null
          id?: string
          is_deleted?: boolean | null
          is_duplicate?: boolean | null
          job_id?: string | null
          key_factors?: Json | null
          metadata?: Json | null
          pattern_id?: string | null
          reasoning?: string | null
          relationship_embedding?: string | null
          relationship_type?: string
          semantic_description?: string | null
          source_event_id?: string
          target_event_id?: string
          temporal_causality?: string | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "relationship_instances_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "relationship_instances_pattern_id_fkey"
            columns: ["pattern_id"]
            isOneToOne: false
            referencedRelation: "relationship_patterns"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "relationship_instances_source_event_id_fkey"
            columns: ["source_event_id"]
            isOneToOne: false
            referencedRelation: "raw_events"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "relationship_instances_target_event_id_fkey"
            columns: ["target_event_id"]
            isOneToOne: false
            referencedRelation: "raw_events"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "relationship_instances_transaction_id_fkey"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      relationship_patterns: {
        Row: {
          created_at: string | null
          id: string
          pattern_data: Json
          relationship_type: string
          updated_at: string | null
          user_id: string
        }
        Insert: {
          created_at?: string | null
          id?: string
          pattern_data?: Json
          relationship_type: string
          updated_at?: string | null
          user_id: string
        }
        Update: {
          created_at?: string | null
          id?: string
          pattern_data?: Json
          relationship_type?: string
          updated_at?: string | null
          user_id?: string
        }
        Relationships: []
      }
      resolution_log: {
        Row: {
          cache_hit: boolean | null
          confidence: number | null
          correct_entity_id: string | null
          correction_timestamp: string | null
          created_at: string | null
          entity_name: string
          entity_type: string
          id: string
          identifier_similarity: number | null
          identifiers: Json | null
          metadata: Json | null
          name_similarity: number | null
          phonetic_match: boolean | null
          platform: string | null
          processing_time_ms: number | null
          resolution_id: string
          resolution_method: string
          resolved_at: string | null
          resolved_entity_id: string | null
          resolved_name: string | null
          row_id: string | null
          source_file: string | null
          transaction_id: string | null
          user_corrected: boolean | null
          user_id: string | null
        }
        Insert: {
          cache_hit?: boolean | null
          confidence?: number | null
          correct_entity_id?: string | null
          correction_timestamp?: string | null
          created_at?: string | null
          entity_name: string
          entity_type: string
          id?: string
          identifier_similarity?: number | null
          identifiers?: Json | null
          metadata?: Json | null
          name_similarity?: number | null
          phonetic_match?: boolean | null
          platform?: string | null
          processing_time_ms?: number | null
          resolution_id: string
          resolution_method: string
          resolved_at?: string | null
          resolved_entity_id?: string | null
          resolved_name?: string | null
          row_id?: string | null
          source_file?: string | null
          transaction_id?: string | null
          user_corrected?: boolean | null
          user_id?: string | null
        }
        Update: {
          cache_hit?: boolean | null
          confidence?: number | null
          correct_entity_id?: string | null
          correction_timestamp?: string | null
          created_at?: string | null
          entity_name?: string
          entity_type?: string
          id?: string
          identifier_similarity?: number | null
          identifiers?: Json | null
          metadata?: Json | null
          name_similarity?: number | null
          phonetic_match?: boolean | null
          platform?: string | null
          processing_time_ms?: number | null
          resolution_id?: string
          resolution_method?: string
          resolved_at?: string | null
          resolved_entity_id?: string | null
          resolved_name?: string | null
          row_id?: string | null
          source_file?: string | null
          transaction_id?: string | null
          user_corrected?: boolean | null
          user_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "resolution_log_correct_entity_id_fkey"
            columns: ["correct_entity_id"]
            isOneToOne: false
            referencedRelation: "normalized_entities"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "resolution_log_resolved_entity_id_fkey"
            columns: ["resolved_entity_id"]
            isOneToOne: false
            referencedRelation: "normalized_entities"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "resolution_log_transaction_id_fkey"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      root_cause_analyses: {
        Row: {
          affected_event_count: number | null
          affected_event_ids: Json | null
          analysis_metadata: Json | null
          causal_path: Json
          confidence_score: number | null
          created_at: string | null
          id: string
          job_id: string | null
          path_length: number | null
          problem_event_id: string
          root_cause_description: string | null
          root_cause_type: string | null
          root_event_id: string
          total_impact_usd: number | null
          user_id: string
        }
        Insert: {
          affected_event_count?: number | null
          affected_event_ids?: Json | null
          analysis_metadata?: Json | null
          causal_path?: Json
          confidence_score?: number | null
          created_at?: string | null
          id?: string
          job_id?: string | null
          path_length?: number | null
          problem_event_id: string
          root_cause_description?: string | null
          root_cause_type?: string | null
          root_event_id: string
          total_impact_usd?: number | null
          user_id: string
        }
        Update: {
          affected_event_count?: number | null
          affected_event_ids?: Json | null
          analysis_metadata?: Json | null
          causal_path?: Json
          confidence_score?: number | null
          created_at?: string | null
          id?: string
          job_id?: string | null
          path_length?: number | null
          problem_event_id?: string
          root_cause_description?: string | null
          root_cause_type?: string | null
          root_event_id?: string
          total_impact_usd?: number | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "root_cause_analyses_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "root_cause_analyses_problem_event_id_fkey"
            columns: ["problem_event_id"]
            isOneToOne: false
            referencedRelation: "raw_events"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "root_cause_analyses_root_event_id_fkey"
            columns: ["root_event_id"]
            isOneToOne: false
            referencedRelation: "raw_events"
            referencedColumns: ["id"]
          },
        ]
      }
      sync_cursors: {
        Row: {
          cursor_type: string
          id: string
          resource: string
          updated_at: string | null
          user_connection_id: string
          user_id: string
          value: string
        }
        Insert: {
          cursor_type: string
          id?: string
          resource: string
          updated_at?: string | null
          user_connection_id: string
          user_id: string
          value: string
        }
        Update: {
          cursor_type?: string
          id?: string
          resource?: string
          updated_at?: string | null
          user_connection_id?: string
          user_id?: string
          value?: string
        }
        Relationships: [
          {
            foreignKeyName: "sync_cursors_user_connection_id_fkey"
            columns: ["user_connection_id"]
            isOneToOne: false
            referencedRelation: "user_connections"
            referencedColumns: ["id"]
          },
        ]
      }
      sync_runs: {
        Row: {
          error: string | null
          finished_at: string | null
          id: string
          started_at: string | null
          stats: Json
          status: string
          type: string
          user_connection_id: string
          user_id: string
        }
        Insert: {
          error?: string | null
          finished_at?: string | null
          id?: string
          started_at?: string | null
          stats?: Json
          status?: string
          type: string
          user_connection_id: string
          user_id: string
        }
        Update: {
          error?: string | null
          finished_at?: string | null
          id?: string
          started_at?: string | null
          stats?: Json
          status?: string
          type?: string
          user_connection_id?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "sync_runs_user_connection_id_fkey"
            columns: ["user_connection_id"]
            isOneToOne: false
            referencedRelation: "user_connections"
            referencedColumns: ["id"]
          },
        ]
      }
      temporal_patterns: {
        Row: {
          anomalies: Json | null
          avg_days_between: number
          business_logic: string | null
          confidence_score: number | null
          created_at: string | null
          forecast_data: Json | null
          forecast_expires_at: string | null
          forecast_generated_at: string | null
          has_seasonal_pattern: boolean | null
          id: string
          job_id: string | null
          last_validated_at: string | null
          learned_from_relationship_ids: Json | null
          max_days: number | null
          median_days: number | null
          min_days: number | null
          pattern_description: string | null
          relationship_type: string
          sample_count: number | null
          seasonal_amplitude: number | null
          seasonal_data: Json | null
          seasonal_period_days: number | null
          std_dev_days: number | null
          transaction_id: string | null
          updated_at: string | null
          user_id: string
        }
        Insert: {
          anomalies?: Json | null
          avg_days_between: number
          business_logic?: string | null
          confidence_score?: number | null
          created_at?: string | null
          forecast_data?: Json | null
          forecast_expires_at?: string | null
          forecast_generated_at?: string | null
          has_seasonal_pattern?: boolean | null
          id?: string
          job_id?: string | null
          last_validated_at?: string | null
          learned_from_relationship_ids?: Json | null
          max_days?: number | null
          median_days?: number | null
          min_days?: number | null
          pattern_description?: string | null
          relationship_type: string
          sample_count?: number | null
          seasonal_amplitude?: number | null
          seasonal_data?: Json | null
          seasonal_period_days?: number | null
          std_dev_days?: number | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id: string
        }
        Update: {
          anomalies?: Json | null
          avg_days_between?: number
          business_logic?: string | null
          confidence_score?: number | null
          created_at?: string | null
          forecast_data?: Json | null
          forecast_expires_at?: string | null
          forecast_generated_at?: string | null
          has_seasonal_pattern?: boolean | null
          id?: string
          job_id?: string | null
          last_validated_at?: string | null
          learned_from_relationship_ids?: Json | null
          max_days?: number | null
          median_days?: number | null
          min_days?: number | null
          pattern_description?: string | null
          relationship_type?: string
          sample_count?: number | null
          seasonal_amplitude?: number | null
          seasonal_data?: Json | null
          seasonal_period_days?: number | null
          std_dev_days?: number | null
          transaction_id?: string | null
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "temporal_patterns_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "ingestion_jobs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "temporal_patterns_transaction_id_fkey"
            columns: ["transaction_id"]
            isOneToOne: false
            referencedRelation: "processing_transactions"
            referencedColumns: ["id"]
          },
        ]
      }
      user_connections: {
        Row: {
          connector_id: string | null
          created_at: string | null
          id: string
          integration_id: string | null
          last_synced_at: string | null
          metadata: Json | null
          nango_connection_id: string
          provider: string | null
          provider_account_id: string | null
          status: string
          sync_frequency_minutes: number | null
          sync_mode: string
          updated_at: string | null
          user_id: string
        }
        Insert: {
          connector_id?: string | null
          created_at?: string | null
          id?: string
          integration_id?: string | null
          last_synced_at?: string | null
          metadata?: Json | null
          nango_connection_id: string
          provider?: string | null
          provider_account_id?: string | null
          status?: string
          sync_frequency_minutes?: number | null
          sync_mode?: string
          updated_at?: string | null
          user_id: string
        }
        Update: {
          connector_id?: string | null
          created_at?: string | null
          id?: string
          integration_id?: string | null
          last_synced_at?: string | null
          metadata?: Json | null
          nango_connection_id?: string
          provider?: string | null
          provider_account_id?: string | null
          status?: string
          sync_frequency_minutes?: number | null
          sync_mode?: string
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "user_connections_connector_id_fkey"
            columns: ["connector_id"]
            isOneToOne: false
            referencedRelation: "connectors"
            referencedColumns: ["id"]
          },
        ]
      }
      webhook_events: {
        Row: {
          error: string | null
          event_id: string | null
          event_type: string | null
          id: string
          payload: Json
          processed_at: string | null
          received_at: string | null
          signature_valid: boolean
          status: string
          user_connection_id: string | null
          user_id: string
        }
        Insert: {
          error?: string | null
          event_id?: string | null
          event_type?: string | null
          id?: string
          payload?: Json
          processed_at?: string | null
          received_at?: string | null
          signature_valid?: boolean
          status?: string
          user_connection_id?: string | null
          user_id: string
        }
        Update: {
          error?: string | null
          event_id?: string | null
          event_type?: string | null
          id?: string
          payload?: Json
          processed_at?: string | null
          received_at?: string | null
          signature_valid?: boolean
          status?: string
          user_connection_id?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "webhook_events_user_connection_id_fkey"
            columns: ["user_connection_id"]
            isOneToOne: false
            referencedRelation: "user_connections"
            referencedColumns: ["id"]
          },
        ]
      }
    }
    Views: {
      enriched_relationships: {
        Row: {
          business_logic: string | null
          confidence_score: number | null
          created_at: string | null
          days_between: number | null
          detection_method: string | null
          id: string | null
          is_causal: boolean | null
          key_factors: Json | null
          metadata: Json | null
          reasoning: string | null
          relationship_type: string | null
          semantic_description: string | null
          source_amount: number | null
          source_date: string | null
          source_document_type: string | null
          source_event_id: string | null
          source_platform: string | null
          source_vendor: string | null
          target_amount: number | null
          target_date: string | null
          target_document_type: string | null
          target_event_id: string | null
          target_platform: string | null
          target_vendor: string | null
          temporal_causality: string | null
          updated_at: string | null
          user_id: string | null
        }
        Relationships: [
          {
            foreignKeyName: "relationship_instances_source_event_id_fkey"
            columns: ["source_event_id"]
            isOneToOne: false
            referencedRelation: "raw_events"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "relationship_instances_target_event_id_fkey"
            columns: ["target_event_id"]
            isOneToOne: false
            referencedRelation: "raw_events"
            referencedColumns: ["id"]
          },
        ]
      }
      user_dashboard_metrics: {
        Row: {
          active_days: number | null
          avg_confidence: number | null
          failed_events: number | null
          last_activity: string | null
          processed_events: number | null
          total_events: number | null
          unique_files: number | null
          unique_platforms: number | null
          user_id: string | null
        }
        Relationships: []
      }
    }
    Functions: {
      analyze_detection_patterns: {
        Args: {
          p_detection_type: string
          p_min_confidence?: number
          p_min_occurrences?: number
          p_user_id: string
        }
        Returns: {
          avg_confidence: number
          common_indicators: Json
          detected_value: string
          occurrence_count: number
          suggested_pattern: Json
        }[]
      }
      analyze_query_performance: {
        Args: never
        Returns: {
          avg_duration_ms: number
          query_type: string
          recommendation: string
          total_calls: number
        }[]
      }
      analyze_resolution_patterns: {
        Args: { p_entity_type?: string; p_user_id: string }
        Returns: {
          avg_confidence: number
          occurrence_count: number
          pattern_type: string
          pattern_value: string
          success_rate: number
        }[]
      }
      append_lineage_step: {
        Args: {
          p_existing_path: Json
          p_metadata?: Json
          p_operation: string
          p_step: string
        }
        Returns: Json
      }
      build_lineage_step: {
        Args: { p_metadata?: Json; p_operation: string; p_step: string }
        Returns: Json
      }
      calculate_bradford_hill_scores: {
        Args: {
          p_relationship_id: string
          p_source_event_id: string
          p_target_event_id: string
          p_user_id: string
        }
        Returns: Json
      }
      calculate_entity_similarity: {
        Args: { name1: string; name2: string }
        Returns: number
      }
      calculate_row_hash: {
        Args: {
          p_payload: Json
          p_row_index: number
          p_source_filename: string
        }
        Returns: string
      }
      cleanup_expired_locks: { Args: never; Returns: number }
      cleanup_expired_processing_locks: { Args: never; Returns: number }
      cleanup_old_detection_logs: {
        Args: { days_to_keep?: number }
        Returns: number
      }
      cleanup_old_transactions: {
        Args: { days_to_keep?: number }
        Returns: number
      }
      daitch_mokotoff: { Args: { "": string }; Returns: string[] }
      detect_temporal_anomalies: {
        Args: { p_threshold_std_dev?: number; p_user_id: string }
        Returns: {
          actual_days: number
          deviation_days: number
          expected_days: number
          relationship_id: string
          relationship_type: string
          severity: string
        }[]
      }
      dmetaphone: { Args: { "": string }; Returns: string }
      dmetaphone_alt: { Args: { "": string }; Returns: string }
      find_causal_chain: {
        Args: {
          max_depth?: number
          start_event_id: string
          user_id_param?: string
        }
        Returns: {
          chain_depth: number
          chain_description: string
          event_path: string[]
          relationship_path: string[]
          total_confidence: number
        }[]
      }
      find_cross_document_relationships: {
        Args: {
          p_amount_tolerance?: number
          p_date_range_days?: number
          p_max_results?: number
          p_relationship_type: string
          p_source_document_type: string
          p_target_document_type: string
          p_user_id: string
        }
        Returns: {
          amount_match: boolean
          confidence: number
          date_match: boolean
          entity_match: boolean
          metadata: Json
          relationship_type: string
          source_event_id: string
          target_event_id: string
        }[]
      }
      find_fuzzy_entity_matches: {
        Args: {
          p_entity_name: string
          p_entity_type: string
          p_max_results?: number
          p_similarity_threshold?: number
          p_user_id: string
        }
        Returns: {
          bank_account: string
          canonical_name: string
          email: string
          entity_id: string
          match_type: string
          phone: string
          similarity_score: number
          tax_id: string
        }[]
      }
      find_or_create_entity: {
        Args: {
          p_bank_account?: string
          p_email?: string
          p_entity_name: string
          p_entity_type: string
          p_phone?: string
          p_platform: string
          p_source_file?: string
          p_tax_id?: string
          p_user_id: string
        }
        Returns: string
      }
      find_orphaned_events: {
        Args: { p_cutoff_time: string; p_user_id: string }
        Returns: {
          created_at: string
          file_id: string
          id: string
          job_id: string
        }[]
      }
      find_orphaned_records: {
        Args: { p_cutoff_time: string; p_user_id: string }
        Returns: {
          created_at: string
          file_name: string
          id: string
        }[]
      }
      find_phonetic_entity_matches: {
        Args: {
          p_entity_name: string
          p_entity_type: string
          p_max_results?: number
          p_user_id: string
        }
        Returns: {
          bank_account: string
          canonical_name: string
          email: string
          entity_id: string
          match_method: string
          phone: string
          tax_id: string
        }[]
      }
      find_root_causes: {
        Args: {
          p_max_depth?: number
          p_problem_event_id: string
          p_user_id: string
        }
        Returns: {
          causal_path: Json
          path_length: number
          root_event_id: string
          total_causal_score: number
        }[]
      }
      find_within_document_relationships: {
        Args: {
          p_file_id: string
          p_max_results?: number
          p_relationship_type: string
          p_user_id: string
        }
        Returns: {
          confidence: number
          metadata: Json
          relationship_type: string
          source_event_id: string
          target_event_id: string
        }[]
      }
      get_accuracy_stats: {
        Args: { user_uuid: string }
        Returns: {
          accuracy_enhanced_events: number
          avg_overall_confidence: number
          events_requiring_review: number
          expense_events: number
          high_priority_reviews: number
          income_events: number
          net_cash_flow_usd: number
          total_events: number
          total_expenses_usd: number
          total_income_usd: number
          transfer_events: number
          validation_error_rate: number
        }[]
      }
      get_causal_graph_stats: { Args: { p_user_id: string }; Returns: Json }
      get_currency_summary: {
        Args: { user_uuid: string }
        Returns: {
          avg_exchange_rate: number
          currency: string
          total_original_amount: number
          total_usd_amount: number
          transaction_count: number
        }[]
      }
      get_delta_merge_history: {
        Args: { p_file_id: string; p_user_id: string }
        Returns: {
          merge_date: string
          merge_id: string
          merge_type: string
          rows_added: number
          rows_skipped: number
          source_file_name: string
          target_file_name: string
        }[]
      }
      get_detection_statistics: { Args: { p_user_id: string }; Returns: Json }
      get_discovered_platforms: {
        Args: { p_user_id: string }
        Returns: {
          confidence_score: number
          discovered_at: string
          discovery_reason: string
          platform_name: string
        }[]
      }
      get_document_type_stats: {
        Args: { user_uuid: string }
        Returns: {
          avg_confidence: number
          document_type: string
          total_amount_usd: number
          total_count: number
          unique_platforms: string[]
        }[]
      }
      get_duplicate_candidates: {
        Args: { p_file_hash: string; p_limit?: number; p_user_id: string }
        Returns: {
          created_at: string
          file_hash: string
          file_name: string
          file_size: number
          id: string
        }[]
      }
      get_enhanced_job_stats: {
        Args: { days_back?: number; user_uuid: string }
        Returns: {
          avg_processing_time_minutes: number
          completed_jobs: number
          failed_jobs: number
          pending_jobs: number
          priority_distribution: Json
          processing_jobs: number
          retry_rate: number
          success_rate: number
          total_events_processed: number
          total_files_processed: number
          total_jobs: number
        }[]
      }
      get_enrichment_stats: {
        Args: { user_uuid: string }
        Returns: {
          avg_exchange_rate: number
          currency_breakdown: Json
          events_with_currency_conversion: number
          events_with_platform_ids: number
          events_with_vendor_standardization: number
          total_amount_usd: number
          total_events: number
          vendor_standardization_accuracy: number
        }[]
      }
      get_entity_details: {
        Args: { entity_id: string; user_uuid: string }
        Returns: {
          entity_info: Json
          match_history: Json
          related_events: Json
        }[]
      }
      get_entity_resolution_stats: {
        Args: { user_uuid: string }
        Returns: {
          avg_confidence: number
          bank_matches: number
          customers_count: number
          email_matches: number
          employees_count: number
          exact_matches: number
          fuzzy_matches: number
          new_entities: number
          projects_count: number
          total_entities: number
          total_matches: number
          vendors_count: number
        }[]
      }
      get_entity_stats: {
        Args: { user_uuid: string }
        Returns: {
          customer_names: string[]
          employee_names: string[]
          project_names: string[]
          total_customers: number
          total_employees: number
          total_projects: number
          total_vendors: number
          vendor_names: string[]
        }[]
      }
      get_event_provenance: {
        Args: { p_event_id: string; p_user_id: string }
        Returns: {
          audit_trail: Json
          entity_links: Json
          event_id: string
          lineage_path: Json
          row_hash: string
          source_info: Json
          transformation_summary: Json
        }[]
      }
      get_events_by_platform_id: {
        Args: {
          p_id_type: string
          p_id_value: string
          p_platform: string
          p_user_id: string
        }
        Returns: {
          confidence_score: number
          created_at: string
          id: string
          kind: string
          payload: Json
          platform_ids: Json
          source_platform: string
        }[]
      }
      get_events_for_review: {
        Args: { priority_filter?: string; user_uuid: string }
        Returns: {
          amount_signed_usd: number
          created_at: string
          id: string
          kind: string
          overall_confidence: number
          review_priority: string
          review_reason: string
          transaction_date: string
          transaction_type: string
          validation_flags: Json
          vendor_standard: string
        }[]
      }
      get_field_mapping: {
        Args: {
          p_document_type?: string
          p_platform?: string
          p_source_column: string
          p_user_id: string
        }
        Returns: {
          confidence: number
          mapping_source: string
          metadata: Json
          target_field: string
        }[]
      }
      get_lineage_summary: {
        Args: { p_event_id: string; p_user_id: string }
        Returns: {
          metadata: Json
          operation: string
          step_name: string
          step_number: number
          step_timestamp: string
        }[]
      }
      get_next_job_for_processing: {
        Args: never
        Returns: {
          created_at: string
          file_size: number
          filename: string
          job_id: string
          priority: string
          user_id: string
        }[]
      }
      get_normalization_stats: {
        Args: { user_uuid: string }
        Returns: {
          avg_confidence: number
          by_document_type: Json
          by_platform: Json
          duplicate_groups: number
          requires_review_count: number
          total_normalized: number
        }[]
      }
      get_platform_id_stats: {
        Args: { p_user_id: string }
        Returns: {
          avg_confidence: number
          id_type: string
          most_common_ids: Json
          platform: string
          total_count: number
          unique_count: number
        }[]
      }
      get_platform_patterns: {
        Args: { p_user_id: string }
        Returns: {
          created_at: string
          patterns: Json
          platform: string
          updated_at: string
        }[]
      }
      get_platform_stats: {
        Args: { p_user_id: string }
        Returns: {
          discovered_platforms: number
          latest_discovery: string
          learned_platforms: number
          most_used_platform: string
          total_platforms: number
        }[]
      }
      get_raw_events_stats: {
        Args: { user_uuid: string }
        Returns: {
          category_breakdown: Json
          failed_events: number
          kind_breakdown: Json
          pending_events: number
          processed_events: number
          total_events: number
          unique_files: number
          unique_platforms: string[]
        }[]
      }
      get_relationship_statistics: {
        Args: { p_days?: number; p_user_id: string }
        Returns: {
          avg_confidence: number
          by_type: Json
          cross_file_relationships: number
          total_relationships: number
          within_file_relationships: number
        }[]
      }
      get_relationship_stats: { Args: { user_id_param: string }; Returns: Json }
      get_resolution_statistics: {
        Args: { p_days?: number; p_entity_type?: string; p_user_id: string }
        Returns: {
          avg_confidence: number
          avg_processing_time_ms: number
          cache_hit_rate: number
          exact_matches: number
          fuzzy_matches: number
          new_entities: number
          total_resolutions: number
          user_corrections: number
        }[]
      }
      get_runway_forecast: {
        Args: { months_ahead?: number; user_uuid: string }
        Returns: {
          current_balance_usd: number
          monthly_burn_rate: number
          monthly_expenses_avg: number
          monthly_income_avg: number
          projected_balance_usd: number
          runway_months: number
        }[]
      }
      get_semantic_relationship_stats: {
        Args: { user_id_param: string }
        Returns: Json
      }
      get_system_health_metrics: {
        Args: never
        Returns: {
          metric_name: string
          metric_unit: string
          metric_value: number
          recorded_at: string
        }[]
      }
      get_temporal_pattern_stats: { Args: { p_user_id: string }; Returns: Json }
      get_transaction_rollback_data: {
        Args: { p_transaction_id: string }
        Returns: {
          operation_type: string
          record_ids: string[]
          table_name: string
        }[]
      }
      get_transaction_stats: {
        Args: { user_uuid: string }
        Returns: {
          active_transactions: number
          avg_processing_time_seconds: number
          committed_transactions: number
          failed_transactions: number
          rolled_back_transactions: number
          total_transactions: number
        }[]
      }
      get_user_events_optimized:
        | {
            Args: {
              p_file_id?: string
              p_job_id?: string
              p_kind?: string
              p_limit?: number
              p_offset?: number
              p_source_platform?: string
              p_status?: string
              p_user_id: string
            }
            Returns: {
              confidence_score: number
              created_at: string
              id: string
              kind: string
              payload: Json
              processed_at: string
              row_index: number
              source_filename: string
              source_platform: string
              status: string
              total_count: number
            }[]
          }
        | {
            Args: {
              p_kind: string
              p_limit: number
              p_offset: number
              p_source_platform: string
              p_status: string
              p_user_id: string
            }
            Returns: {
              confidence_score: number
              created_at: string
              id: string
              kind: string
              payload: Json
              processed_at: string
              row_index: number
              source_filename: string
              source_platform: string
              status: string
              total_count: number
            }[]
          }
      get_user_field_mappings: {
        Args: { p_platform?: string; p_user_id: string }
        Returns: {
          confidence: number
          document_type: string
          id: string
          last_used_at: string
          mapping_source: string
          platform: string
          source_column: string
          success_count: number
          target_field: string
          usage_count: number
        }[]
      }
      get_user_statistics_fast: { Args: { p_user_id: string }; Returns: Json }
      learn_temporal_pattern: {
        Args: { p_relationship_type: string; p_user_id: string }
        Returns: Json
      }
      predict_missing_relationships: {
        Args: { p_lookback_days?: number; p_user_id: string }
        Returns: {
          confidence_score: number
          days_overdue: number
          expected_date: string
          predicted_target_type: string
          relationship_type: string
          source_event_id: string
        }[]
      }
      predict_next_transactions: {
        Args: {
          p_lookback_days?: number
          p_max_predictions?: number
          p_user_id: string
        }
        Returns: {
          confidence: number
          pattern_type: string
          predicted_amount: number
          predicted_date: string
          predicted_vendor: string
          supporting_events: number
        }[]
      }
      record_entity_correction: {
        Args: { p_correct_entity_id: string; p_resolution_log_id: string }
        Returns: undefined
      }
      record_mapping_success: {
        Args: { p_mapping_id: string }
        Returns: undefined
      }
      record_performance_metric: {
        Args: {
          p_job_id?: string
          p_metadata?: Json
          p_metric_name: string
          p_metric_unit?: string
          p_metric_value: number
          p_user_id?: string
        }
        Returns: string
      }
      refresh_user_dashboard_metrics: { Args: never; Returns: undefined }
      search_entities_by_name: {
        Args: { p_entity_type?: string; search_term: string; user_uuid: string }
        Returns: {
          aliases: string[]
          canonical_name: string
          confidence_score: number
          email: string
          entity_type: string
          id: string
          platform_sources: string[]
          similarity_score: number
        }[]
      }
      search_events_by_document_type: {
        Args: { doc_type?: string; min_confidence?: number; user_uuid: string }
        Returns: {
          amount_usd: number
          category: string
          created_at: string
          document_confidence: number
          document_type: string
          id: string
          kind: string
          source_platform: string
          subcategory: string
          vendor_standard: string
        }[]
      }
      search_events_by_entity: {
        Args: { entity_name: string; entity_type: string; user_uuid: string }
        Returns: {
          category: string
          classification_metadata: Json
          created_at: string
          entities: Json
          id: string
          kind: string
          payload: Json
          source_platform: string
          subcategory: string
        }[]
      }
      search_events_by_vendor: {
        Args: { user_uuid: string; vendor_name: string }
        Returns: {
          amount_usd: number
          category: string
          created_at: string
          currency: string
          id: string
          kind: string
          platform: string
          standard_description: string
          subcategory: string
          vendor_standard: string
        }[]
      }
      search_normalized_events: {
        Args: {
          document_type_filter?: string
          min_confidence?: number
          platform_filter?: string
          search_text?: string
          user_uuid: string
        }
        Returns: {
          document_type: string
          id: string
          normalization_confidence: number
          normalized_at: string
          normalized_payload: Json
          platform_label: string
          raw_event_id: string
          requires_review: boolean
          resolved_entities: Json
        }[]
      }
      search_relationships: {
        Args: {
          min_confidence?: number
          relationship_type_param?: string
          search_term?: string
          user_id_param: string
        }
        Returns: {
          confidence_score: number
          created_at: string
          detection_method: string
          id: string
          reasoning: string
          relationship_type: string
          source_event_id: string
          target_event_id: string
        }[]
      }
      search_similar_relationships: {
        Args: {
          max_results?: number
          query_embedding: string
          similarity_threshold?: number
          user_id_param?: string
        }
        Returns: {
          business_logic: string
          confidence_score: number
          created_at: string
          id: string
          relationship_type: string
          semantic_description: string
          similarity_score: number
          source_event_id: string
          target_event_id: string
          temporal_causality: string
        }[]
      }
      show_limit: { Args: never; Returns: number }
      show_trgm: { Args: { "": string }; Returns: string[] }
      soundex: { Args: { "": string }; Returns: string }
      text_soundex: { Args: { "": string }; Returns: string }
      upsert_field_mapping: {
        Args: {
          p_confidence?: number
          p_document_type?: string
          p_mapping_source?: string
          p_metadata?: Json
          p_platform?: string
          p_source_column: string
          p_target_field: string
          p_user_id: string
        }
        Returns: string
      }
      validate_event_record_consistency: {
        Args: { p_user_id: string }
        Returns: {
          file_id: string
          id: string
          job_id: string
        }[]
      }
      validate_platform_id_pattern: {
        Args: { p_id_type: string; p_id_value: string; p_platform: string }
        Returns: Json
      }
      validate_transaction_consistency: {
        Args: { p_user_id: string }
        Returns: {
          committed_at: string
          id: string
          status: string
        }[]
      }
      validate_transaction_semantics: {
        Args: {
          p_affects_cash: boolean
          p_amount_direction: string
          p_amount_signed_usd: number
          p_transaction_type: string
        }
        Returns: Json
      }
      verify_row_integrity: {
        Args: { p_event_id: string; p_expected_hash: string; p_user_id: string }
        Returns: {
          expected_hash: string
          is_valid: boolean
          message: string
          stored_hash: string
        }[]
      }
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
