export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instanciate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "12.2.3 (519615d)"
  }
  public: {
    Tables: {
      cross_platform_relationships: {
        Row: {
          confidence_score: number | null
          created_at: string | null
          detection_method: string | null
          id: string
          platform_compatibility: string | null
          relationship_type: string
          source_event_id: string
          source_platform: string | null
          target_event_id: string
          target_platform: string | null
          user_id: string
        }
        Insert: {
          confidence_score?: number | null
          created_at?: string | null
          detection_method?: string | null
          id?: string
          platform_compatibility?: string | null
          relationship_type: string
          source_event_id: string
          source_platform?: string | null
          target_event_id: string
          target_platform?: string | null
          user_id: string
        }
        Update: {
          confidence_score?: number | null
          created_at?: string | null
          detection_method?: string | null
          id?: string
          platform_compatibility?: string | null
          relationship_type?: string
          source_event_id?: string
          source_platform?: string | null
          target_event_id?: string
          target_platform?: string | null
          user_id?: string
        }
        Relationships: [
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
      discovered_platforms: {
        Row: {
          confidence_score: number | null
          discovered_at: string | null
          discovery_reason: string | null
          id: string
          platform_name: string
          user_id: string
        }
        Insert: {
          confidence_score?: number | null
          discovered_at?: string | null
          discovery_reason?: string | null
          id?: string
          platform_name: string
          user_id: string
        }
        Update: {
          confidence_score?: number | null
          discovered_at?: string | null
          discovery_reason?: string | null
          id?: string
          platform_name?: string
          user_id?: string
        }
        Relationships: []
      }
      entity_matches: {
        Row: {
          created_at: string | null
          id: string
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
          user_id: string | null
        }
        Insert: {
          created_at?: string | null
          id?: string
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
          user_id?: string | null
        }
        Update: {
          created_at?: string | null
          id?: string
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
          user_id?: string | null
        }
        Relationships: [
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
        ]
      }
      ingestion_jobs: {
        Row: {
          completed_at: string | null
          created_at: string | null
          error_message: string | null
          file_id: string | null
          id: string
          job_type: string
          progress: number | null
          record_id: string | null
          result: Json | null
          started_at: string | null
          status: string | null
          updated_at: string | null
          user_id: string | null
        }
        Insert: {
          completed_at?: string | null
          created_at?: string | null
          error_message?: string | null
          file_id?: string | null
          id?: string
          job_type: string
          progress?: number | null
          record_id?: string | null
          result?: Json | null
          started_at?: string | null
          status?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Update: {
          completed_at?: string | null
          created_at?: string | null
          error_message?: string | null
          file_id?: string | null
          id?: string
          job_type?: string
          progress?: number | null
          record_id?: string | null
          result?: Json | null
          started_at?: string | null
          status?: string | null
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
        ]
      }
      integration_test_logs: {
        Row: {
          created_at: string
          id: string
          message: string
          source: string
          timestamp: string
        }
        Insert: {
          created_at?: string
          id?: string
          message: string
          source?: string
          timestamp?: string
        }
        Update: {
          created_at?: string
          id?: string
          message?: string
          source?: string
          timestamp?: string
        }
        Relationships: []
      }
      metrics: {
        Row: {
          amount: number | null
          category: string | null
          classification_metadata: Json | null
          confidence_score: number | null
          created_at: string | null
          currency: string | null
          date_recorded: string | null
          id: string
          metric_type: string
          period_end: string | null
          period_start: string | null
          record_id: string | null
          subcategory: string | null
          updated_at: string | null
          user_id: string | null
        }
        Insert: {
          amount?: number | null
          category?: string | null
          classification_metadata?: Json | null
          confidence_score?: number | null
          created_at?: string | null
          currency?: string | null
          date_recorded?: string | null
          id?: string
          metric_type: string
          period_end?: string | null
          period_start?: string | null
          record_id?: string | null
          subcategory?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Update: {
          amount?: number | null
          category?: string | null
          classification_metadata?: Json | null
          confidence_score?: number | null
          created_at?: string | null
          currency?: string | null
          date_recorded?: string | null
          id?: string
          metric_type?: string
          period_end?: string | null
          period_start?: string | null
          record_id?: string | null
          subcategory?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "metrics_record_id_fkey"
            columns: ["record_id"]
            isOneToOne: false
            referencedRelation: "raw_records"
            referencedColumns: ["id"]
          },
        ]
      }
      normalized_entities: {
        Row: {
          aliases: string[] | null
          bank_account: string | null
          canonical_name: string
          confidence_score: number | null
          created_at: string | null
          email: string | null
          entity_type: string
          first_seen_at: string | null
          id: string
          last_seen_at: string | null
          phone: string | null
          platform_sources: string[] | null
          source_files: string[] | null
          tax_id: string | null
          updated_at: string | null
          user_id: string | null
        }
        Insert: {
          aliases?: string[] | null
          bank_account?: string | null
          canonical_name: string
          confidence_score?: number | null
          created_at?: string | null
          email?: string | null
          entity_type: string
          first_seen_at?: string | null
          id?: string
          last_seen_at?: string | null
          phone?: string | null
          platform_sources?: string[] | null
          source_files?: string[] | null
          tax_id?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Update: {
          aliases?: string[] | null
          bank_account?: string | null
          canonical_name?: string
          confidence_score?: number | null
          created_at?: string | null
          email?: string | null
          entity_type?: string
          first_seen_at?: string | null
          id?: string
          last_seen_at?: string | null
          phone?: string | null
          platform_sources?: string[] | null
          source_files?: string[] | null
          tax_id?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Relationships: []
      }
      platform_patterns: {
        Row: {
          created_at: string | null
          id: string
          patterns: Json
          platform: string
          updated_at: string | null
          user_id: string
        }
        Insert: {
          created_at?: string | null
          id?: string
          patterns?: Json
          platform: string
          updated_at?: string | null
          user_id: string
        }
        Update: {
          created_at?: string | null
          id?: string
          patterns?: Json
          platform?: string
          updated_at?: string | null
          user_id?: string
        }
        Relationships: []
      }
      raw_events: {
        Row: {
          amount_original: number | null
          amount_usd: number | null
          category: string | null
          classification_metadata: Json | null
          confidence_score: number | null
          created_at: string | null
          currency: string | null
          entities: Json | null
          error_message: string | null
          exchange_date: string | null
          exchange_rate: number | null
          file_id: string | null
          id: string
          ingest_ts: string | null
          ingested_on: string | null
          job_id: string | null
          kind: string
          last_relationship_check: string | null
          payload: Json
          platform_ids: Json | null
          processed_at: string | null
          provider: string
          relationship_count: number | null
          relationships: Json | null
          row_index: number
          sheet_name: string | null
          source_filename: string
          source_platform: string | null
          standard_description: string | null
          status: string | null
          subcategory: string | null
          updated_at: string | null
          uploader: string | null
          user_id: string | null
          vendor_cleaning_method: string | null
          vendor_confidence: number | null
          vendor_raw: string | null
          vendor_standard: string | null
        }
        Insert: {
          amount_original?: number | null
          amount_usd?: number | null
          category?: string | null
          classification_metadata?: Json | null
          confidence_score?: number | null
          created_at?: string | null
          currency?: string | null
          entities?: Json | null
          error_message?: string | null
          exchange_date?: string | null
          exchange_rate?: number | null
          file_id?: string | null
          id?: string
          ingest_ts?: string | null
          ingested_on?: string | null
          job_id?: string | null
          kind: string
          last_relationship_check?: string | null
          payload: Json
          platform_ids?: Json | null
          processed_at?: string | null
          provider: string
          relationship_count?: number | null
          relationships?: Json | null
          row_index: number
          sheet_name?: string | null
          source_filename: string
          source_platform?: string | null
          standard_description?: string | null
          status?: string | null
          subcategory?: string | null
          updated_at?: string | null
          uploader?: string | null
          user_id?: string | null
          vendor_cleaning_method?: string | null
          vendor_confidence?: number | null
          vendor_raw?: string | null
          vendor_standard?: string | null
        }
        Update: {
          amount_original?: number | null
          amount_usd?: number | null
          category?: string | null
          classification_metadata?: Json | null
          confidence_score?: number | null
          created_at?: string | null
          currency?: string | null
          entities?: Json | null
          error_message?: string | null
          exchange_date?: string | null
          exchange_rate?: number | null
          file_id?: string | null
          id?: string
          ingest_ts?: string | null
          ingested_on?: string | null
          job_id?: string | null
          kind?: string
          last_relationship_check?: string | null
          payload?: Json
          platform_ids?: Json | null
          processed_at?: string | null
          provider?: string
          relationship_count?: number | null
          relationships?: Json | null
          row_index?: number
          sheet_name?: string | null
          source_filename?: string
          source_platform?: string | null
          standard_description?: string | null
          status?: string | null
          subcategory?: string | null
          updated_at?: string | null
          uploader?: string | null
          user_id?: string | null
          vendor_cleaning_method?: string | null
          vendor_confidence?: number | null
          vendor_raw?: string | null
          vendor_standard?: string | null
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
        ]
      }
      raw_records: {
        Row: {
          classification_status: string | null
          content: Json
          created_at: string | null
          file_name: string | null
          file_size: number | null
          file_hash: string | null
          id: string
          ingested_at: string | null
          metadata: Json | null
          source: string
          status: string | null
          updated_at: string | null
          user_id: string | null
        }
        Insert: {
          classification_status?: string | null
          content: Json
          created_at?: string | null
          file_name?: string | null
          file_size?: number | null
          file_hash?: string | null
          id?: string
          ingested_at?: string | null
          metadata?: Json | null
          source: string
          status?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Update: {
          classification_status?: string | null
          content?: Json
          created_at?: string | null
          file_name?: string | null
          file_size?: number | null
          file_hash?: string | null
          id?: string
          ingested_at?: string | null
          metadata?: Json | null
          source?: string
          status?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Relationships: []
      }
      relationship_instances: {
        Row: {
          confidence_score: number | null
          created_at: string | null
          detection_method: string | null
          id: string
          pattern_id: string | null
          reasoning: string | null
          relationship_type: string
          source_event_id: string
          target_event_id: string
          user_id: string
        }
        Insert: {
          confidence_score?: number | null
          created_at?: string | null
          detection_method?: string | null
          id?: string
          pattern_id?: string | null
          reasoning?: string | null
          relationship_type: string
          source_event_id: string
          target_event_id: string
          user_id: string
        }
        Update: {
          confidence_score?: number | null
          created_at?: string | null
          detection_method?: string | null
          id?: string
          pattern_id?: string | null
          reasoning?: string | null
          relationship_type?: string
          source_event_id?: string
          target_event_id?: string
          user_id?: string
        }
        Relationships: [
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
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      find_or_create_entity: {
        Args: {
          p_user_id: string
          p_entity_name: string
          p_entity_type: string
          p_platform: string
          p_email?: string
          p_bank_account?: string
          p_phone?: string
          p_tax_id?: string
          p_source_file?: string
        }
        Returns: string
      }
      get_currency_summary: {
        Args: { user_uuid: string }
        Returns: {
          currency: string
          total_original_amount: number
          total_usd_amount: number
          transaction_count: number
          avg_exchange_rate: number
        }[]
      }
      get_discovered_platforms: {
        Args: { p_user_id: string }
        Returns: {
          platform_name: string
          discovery_reason: string
          confidence_score: number
          discovered_at: string
        }[]
      }
      get_enrichment_stats: {
        Args: { user_uuid: string }
        Returns: {
          total_events: number
          events_with_currency_conversion: number
          events_with_vendor_standardization: number
          events_with_platform_ids: number
          total_amount_usd: number
          currency_breakdown: Json
          vendor_standardization_accuracy: number
          avg_exchange_rate: number
        }[]
      }
      get_entity_details: {
        Args: { user_uuid: string; entity_id: string }
        Returns: {
          entity_info: Json
          related_events: Json
          match_history: Json
        }[]
      }
      get_entity_resolution_stats: {
        Args: { user_uuid: string }
        Returns: {
          total_entities: number
          employees_count: number
          vendors_count: number
          customers_count: number
          projects_count: number
          total_matches: number
          exact_matches: number
          fuzzy_matches: number
          email_matches: number
          bank_matches: number
          new_entities: number
          avg_confidence: number
        }[]
      }
      get_entity_stats: {
        Args: { user_uuid: string }
        Returns: {
          total_employees: number
          total_vendors: number
          total_customers: number
          total_projects: number
          employee_names: string[]
          vendor_names: string[]
          customer_names: string[]
          project_names: string[]
        }[]
      }
      get_platform_patterns: {
        Args: { p_user_id: string }
        Returns: {
          platform: string
          patterns: Json
          created_at: string
          updated_at: string
        }[]
      }
      get_platform_stats: {
        Args: { p_user_id: string }
        Returns: {
          total_platforms: number
          learned_platforms: number
          discovered_platforms: number
          most_used_platform: string
          latest_discovery: string
        }[]
      }
      get_raw_events_stats: {
        Args: { user_uuid: string }
        Returns: {
          total_events: number
          processed_events: number
          failed_events: number
          pending_events: number
          unique_files: number
          unique_platforms: string[]
          category_breakdown: Json
          kind_breakdown: Json
        }[]
      }
      get_relationship_stats: {
        Args: { user_id_param: string }
        Returns: Json
      }
      search_entities_by_name: {
        Args: { user_uuid: string; search_term: string; p_entity_type?: string }
        Returns: {
          id: string
          entity_type: string
          canonical_name: string
          aliases: string[]
          email: string
          platform_sources: string[]
          confidence_score: number
          similarity_score: number
        }[]
      }
      search_events_by_entity: {
        Args: { user_uuid: string; entity_type: string; entity_name: string }
        Returns: {
          id: string
          kind: string
          category: string
          subcategory: string
          source_platform: string
          payload: Json
          classification_metadata: Json
          entities: Json
          created_at: string
        }[]
      }
      search_events_by_vendor: {
        Args: { user_uuid: string; vendor_name: string }
        Returns: {
          id: string
          kind: string
          category: string
          subcategory: string
          amount_usd: number
          currency: string
          vendor_standard: string
          platform: string
          standard_description: string
          created_at: string
        }[]
      }
      search_relationships: {
        Args: {
          user_id_param: string
          search_term?: string
          relationship_type_param?: string
          min_confidence?: number
        }
        Returns: {
          id: string
          source_event_id: string
          target_event_id: string
          relationship_type: string
          confidence_score: number
          detection_method: string
          reasoning: string
          created_at: string
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
