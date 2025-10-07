from typing import Dict, Any, Optional, List
from datetime import datetime

class UniversalFinanceMapper:
    """
    Maps legacy/raw event structures and extractor outputs to the Universal Finance Schema (v1).

    Input candidates:
    - raw_event row (payload + metadata)
    - enrichment/classification outputs (category, platform, ai reasoning)
    - resolved entities (vendor/customer)
    """

    @staticmethod
    def from_raw_event(event: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        payload = event.get('payload') or {}
        meta = event.get('classification_metadata') or {}
        src_platform = event.get('source_platform') or meta.get('platform') or payload.get('platform')

        amount_val = payload.get('amount_usd') or payload.get('amount') or payload.get('total')
        currency = payload.get('currency') or 'USD'

        parties: Dict[str, Any] = {}
        entities = payload.get('entities') or {}
        if isinstance(entities, dict):
            vendor = entities.get('vendor') or entities.get('merchant')
            if vendor:
                parties['vendor'] = {'name': vendor}
            customer = entities.get('customer')
            if customer:
                parties['customer'] = {'name': customer}

        record = {
            'user_id': user_id or event.get('user_id'),
            'record_type': meta.get('category') or payload.get('kind') or 'transaction',
            'amount': {
                'value': _safe_number(amount_val),
                'currency': currency
            },
            'dates': {
                'issued': _parse_dt(payload.get('date') or payload.get('created_at')),
                'posted': _parse_dt(event.get('created_at')),
                'due': _parse_dt(payload.get('due_date'))
            },
            'identifiers': {
                'invoice_id': payload.get('invoice_id') or payload.get('reference'),
                'payment_id': payload.get('payment_id'),
                'external_id': payload.get('external_id') or payload.get('id'),
                'reference': payload.get('reference')
            },
            'parties': parties or None,
            'line_items': payload.get('line_items') if isinstance(payload.get('line_items'), list) else None,
            'classification': {
                'document_type': meta.get('subcategory') or payload.get('row_type'),
                'platform': src_platform or 'unknown',
                'confidence': event.get('confidence_score') or meta.get('confidence')
            },
            'source': {
                'platform': src_platform or 'unknown',
                'filename': event.get('source_filename'),
                'row_index': event.get('row_index'),
                'event_id': event.get('id'),
                'file_id': event.get('file_id'),
                'ingest_ts': event.get('ingest_ts')
            },
            'context': {
                'description': payload.get('standard_description') or payload.get('description') or '',
                'memo': payload.get('memo')
            },
            'taxes': {
                'tax_code': payload.get('tax_code'),
                'tax_amount': _safe_number(payload.get('tax_amount'))
            },
            'metadata': {
                'schema_version': 'v1',
                'mapping': 'from_raw_event'
            }
        }
        # Remove empty optional sections
        for k in ['parties', 'line_items']:
            if not record.get(k):
                record[k] = None
        return record


def _parse_dt(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    try:
        # accept ISO strings; return ISO
        return datetime.fromisoformat(str(val).replace('Z', '+00:00')).isoformat()
    except Exception:
        return None


def _safe_number(v: Any) -> Optional[float]:
    try:
        if v is None or v == '':
            return None
        return float(v)
    except Exception:
        return None
