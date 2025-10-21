-- Migration: Seed connectors table with all supported providers
-- Date: 2025-10-20
-- Purpose: Populate connectors table for OAuth integrations

-- Insert all 9 supported providers
INSERT INTO public.connectors (provider, integration_id, auth_type, scopes, endpoints_needed, enabled) VALUES
-- Email Providers
('google-mail', 'google-mail', 'OAUTH2', 
 '["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/gmail.modify"]'::jsonb,
 '["/gmail/v1/users/me/messages", "/gmail/v1/users/me/history"]'::jsonb,
 true),

('zoho-mail', 'zoho-mail', 'OAUTH2',
 '["ZohoMail.messages.READ", "ZohoMail.folders.READ"]'::jsonb,
 '["/api/accounts", "/api/messages"]'::jsonb,
 true),

-- Storage Providers
('dropbox', 'dropbox', 'OAUTH2',
 '["files.metadata.read", "files.content.read"]'::jsonb,
 '["/2/files/list_folder", "/2/files/download"]'::jsonb,
 true),

('google-drive', 'google-drive', 'OAUTH2',
 '["https://www.googleapis.com/auth/drive.readonly"]'::jsonb,
 '["/drive/v3/files"]'::jsonb,
 true),

-- Accounting Providers
('zoho-books', 'zoho-books', 'OAUTH2',
 '["ZohoBooks.fullaccess.all"]'::jsonb,
 '["/api/v3/invoices", "/api/v3/bills", "/api/v3/contacts"]'::jsonb,
 true),

('quickbooks-sandbox', 'quickbooks-sandbox', 'OAUTH2',
 '["com.intuit.quickbooks.accounting"]'::jsonb,
 '["/v3/company/{realmId}/query", "/v3/company/{realmId}/invoice"]'::jsonb,
 true),

('xero', 'xero', 'OAUTH2',
 '["accounting.transactions.read", "accounting.contacts.read"]'::jsonb,
 '["/api.xro/2.0/Invoices", "/api.xro/2.0/Contacts"]'::jsonb,
 true),

-- Payment Providers
('stripe', 'stripe', 'OAUTH2',
 '["read_only"]'::jsonb,
 '["/v1/charges", "/v1/invoices", "/v1/customers", "/v1/payment_intents"]'::jsonb,
 true),

('razorpay', 'razorpay', 'OAUTH2',
 '["read_only"]'::jsonb,
 '["/v1/payments", "/v1/orders", "/v1/customers"]'::jsonb,
 true)

ON CONFLICT (provider) DO UPDATE SET
  integration_id = EXCLUDED.integration_id,
  auth_type = EXCLUDED.auth_type,
  scopes = EXCLUDED.scopes,
  endpoints_needed = EXCLUDED.endpoints_needed,
  enabled = EXCLUDED.enabled,
  updated_at = NOW();

-- Verify insertion
DO $$
DECLARE
  connector_count INTEGER;
BEGIN
  SELECT COUNT(*) INTO connector_count FROM public.connectors;
  RAISE NOTICE 'Total connectors in table: %', connector_count;
  
  IF connector_count < 9 THEN
    RAISE WARNING 'Expected 9 connectors, but found %', connector_count;
  ELSE
    RAISE NOTICE '✅ All 9 connectors successfully seeded';
  END IF;
END $$;
