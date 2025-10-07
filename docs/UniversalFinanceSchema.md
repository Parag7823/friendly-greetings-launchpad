# Universal Finance Schema (v1)

This schema defines a single, consistent shape for any financial row/transaction, regardless of source (CSV, Excel, PDF, API, connector).

- One record per transaction/row: stored as JSON in `universal_records.universal`.
- Parties (vendor/customer/employee) are canonicalized in `universal_parties`.
- Links (invoice→payment, revenue→bank) are in `universal_relationships`.

## Top-level fields

- `id` (UUID): Universal record ID (database primary key, not in the JSON payload).
- `user_id` (UUID): Owner of the record.
- `record_type` (string): invoice | payment | expense | revenue | transfer | journal | other.
- `amount` (object):
  - `value` (number)
  - `currency` (string, ISO 4217)
- `dates` (object):
  - `issued` (ISO8601)
  - `posted` (ISO8601)
  - `due` (ISO8601)
- `identifiers` (object):
  - `invoice_id`, `payment_id`, `external_id`, `reference`
- `parties` (object):
  - `payer`, `payee`, `customer`, `vendor` (each: `{ name?: string, party_id?: string, identifiers?: object }`)
- `line_items` (array): items with `{ description, quantity, unit_price, amount, tax }`
- `classification` (object): `{ document_type, platform, confidence, methods }`
- `source` (object): `{ platform, filename, row_index, event_id, file_id, ingest_ts }`
- `context` (object): `{ description, memo, notes }`
- `taxes` (object): `{ tax_code, tax_amount }`
- `metadata` (object): free-form additions

## Example

```json
{
  "user_id": "9b9f3a5b-2f78-4a1d-9c1b-0f5efc3b3e1a",
  "record_type": "invoice",
  "amount": { "value": 1234.56, "currency": "USD" },
  "dates": { "issued": "2025-01-15T10:00:00Z", "due": "2025-02-14T23:59:59Z" },
  "identifiers": { "invoice_id": "INV-001", "external_id": "stripe:inv_abc" },
  "parties": {
    "vendor": { "name": "Acme Corp", "party_id": "e3b3...", "identifiers": {"tax_id": "AB123"} },
    "customer": { "name": "Beta LLC" }
  },
  "line_items": [
    { "description": "Subscription", "quantity": 1, "unit_price": 1200.00, "amount": 1200.00, "tax": 34.56 }
  ],
  "classification": { "document_type": "invoice", "platform": "stripe", "confidence": 0.93, "methods": ["ai", "pattern"] },
  "source": { "platform": "stripe", "filename": "invoices.csv", "row_index": 42, "event_id": "...", "file_id": "...", "ingest_ts": "2025-01-15T10:01:00Z" },
  "context": { "description": "January subscription" },
  "taxes": { "tax_code": "US-CA", "tax_amount": 34.56 },
  "metadata": { "region": "NA" }
}
```

## Storage

- `universal_records` (new table) stores the JSON blob + common searchable columns.
- GIN JSONB indexes provide fast filtering (e.g., by `amount.currency`, `identifiers.invoice_id`).

## Backward compatibility

- During rollout we dual-write to legacy tables plus universal tables.
- Later, compatibility views can serve legacy readers from `universal_*`.

## Versioning

- Schema versions are encoded in file name and `metadata.schema_version` if needed.
- Breaking changes create a new `v2` JSON schema file and additive DB columns if required.
