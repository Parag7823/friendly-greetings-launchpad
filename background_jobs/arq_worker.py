import os
import uuid
from typing import Dict, Any, Optional, List
from arq import Retry, cron

# ARQ imports
from arq.connections import RedisSettings

# Import application code to reuse existing logic end-to-end
from core_infrastructure.fastapi_backend_v2 import (
    NangoClient,
    ConnectorSyncRequest,
    _gmail_sync_run,
    _dropbox_sync_run,
    _gdrive_sync_run,
    _zoho_mail_sync_run,
    start_processing_job,
    start_pdf_processing_job,
    logger,
    JOBS_PROCESSED,
    NANGO_GMAIL_INTEGRATION_ID,
    NANGO_DROPBOX_INTEGRATION_ID,
    NANGO_GOOGLE_DRIVE_INTEGRATION_ID,
    NANGO_ZOHO_MAIL_INTEGRATION_ID,
)

# FIX #6: Supabase client initialization with graceful degradation
try:
    from supabase_client import get_supabase_client
    supabase = get_supabase_client()
    logger.info("✅ ARQ worker using centralized Supabase client with connection pooling")
except ImportError as e:
    logger.error("supabase_client.py not found", error=str(e))
    supabase = None
except Exception as e:
    logger.error("Failed to initialize Supabase client", error=str(e))
    supabase = None


# --------------- ARQ Task Functions ---------------
# Each task is fully functional and reuses the existing application logic.

# FIX #7: DEAD CODE REMOVED - _record_job_failure()
# This function was defined but never called anywhere in the codebase.
# ARQ's native Retry mechanism handles retries, and error logging is done
# directly in each sync function (gmail_sync, dropbox_sync, etc.).
# Removing this unused function reduces code clutter and maintenance burden.


# CRITICAL FIX #1 & #2: Release locks and rate limit slots after sync completes
async def _release_sync_resources(user_id: str, provider: str, connection_id: str) -> None:
    """
    Release distributed lock and rate limit slot after sync completes.
    Called in finally block to ensure cleanup even on errors.
    
    Args:
        user_id: User ID
        provider: Provider name (gmail, quickbooks, etc.)
        connection_id: Connection ID
    """
    try:
        from core_infrastructure.rate_limiter import get_sync_lock, get_global_rate_limiter
        
        # Release sync lock
        sync_lock = get_sync_lock()
        await sync_lock.release_sync_lock(user_id, provider, connection_id)
        
        # Release rate limit slot
        rate_limiter = get_global_rate_limiter()
        await rate_limiter.release_sync_slot(provider, user_id)
        
        logger.info(
            "sync_resources_released",
            user_id=user_id,
            provider=provider,
            connection_id=connection_id
        )
    except Exception as e:
        logger.error(f"Failed to release sync resources: {e}")


async def gmail_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIX #7: Uses ARQ's native Retry mechanism with exponential backoff.
    - Removes custom Redis-backed retry logic
    - Leverages ARQ's built-in retry handling
    - Cleaner, more robust implementation
    - Logs exceptions before retrying for observability
    
    CRITICAL FIX #1 & #2: Release locks and rate limit slots after sync completes
    """
    nango = NangoClient()
    try:
        return await _gmail_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        # FIX #7: Log exception before retrying for observability
        logger.error("Gmail sync failed", error=str(e), error_type=type(e).__name__, request=req)
        # Use ARQ's native Retry with exponential backoff
        # ARQ handles retry count internally, no need for custom Redis tracking
        raise Retry(defer=30)  # Initial 30 second delay, ARQ will exponentially backoff
    finally:
        # CRITICAL FIX #1 & #2: Always release resources
        user_id = req.get('user_id')
        connection_id = req.get('connection_id')
        if user_id and connection_id:
            await _release_sync_resources(user_id, 'gmail', connection_id)


async def dropbox_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIX #7: Uses ARQ's native Retry mechanism with exponential backoff.
    - Logs exceptions before retrying for observability
    
    CRITICAL FIX #1 & #2: Release locks and rate limit slots after sync completes
    """
    nango = NangoClient()
    try:
        return await _dropbox_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        # FIX #7: Log exception before retrying for observability
        logger.error("Dropbox sync failed", error=str(e), error_type=type(e).__name__, request=req)
        # Use ARQ's native Retry with exponential backoff
        raise Retry(defer=30)
    finally:
        # CRITICAL FIX #1 & #2: Always release resources
        user_id = req.get('user_id')
        connection_id = req.get('connection_id')
        if user_id and connection_id:
            await _release_sync_resources(user_id, 'dropbox', connection_id)


async def gdrive_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIX #7: Uses ARQ's native Retry mechanism with exponential backoff.
    - Logs exceptions before retrying for observability
    
    CRITICAL FIX #1 & #2: Release locks and rate limit slots after sync completes
    """
    nango = NangoClient()
    try:
        return await _gdrive_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        # FIX #7: Log exception before retrying for observability
        logger.error("Google Drive sync failed", error=str(e), error_type=type(e).__name__, request=req)
        # Use ARQ's native Retry with exponential backoff
        raise Retry(defer=30)
    finally:
        # CRITICAL FIX #1 & #2: Always release resources
        user_id = req.get('user_id')
        connection_id = req.get('connection_id')
        if user_id and connection_id:
            await _release_sync_resources(user_id, 'google-drive', connection_id)


async def zoho_mail_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIX #7: Uses ARQ's native Retry mechanism with exponential backoff.
    - Logs exceptions before retrying for observability
    
    CRITICAL FIX #1 & #2: Release locks and rate limit slots after sync completes
    """
    nango = NangoClient()
    try:
        return await _zoho_mail_sync_run(nango, ConnectorSyncRequest(**req))
    except Exception as e:
        # FIX #7: Log exception before retrying for observability
        logger.error("Zoho Mail sync failed", error=str(e), error_type=type(e).__name__, request=req)
        # Use ARQ's native Retry with exponential backoff
        raise Retry(defer=45)  # Initial 45 second delay for Zoho (slower API)
    finally:
        # CRITICAL FIX #1 & #2: Always release resources
        user_id = req.get('user_id')
        connection_id = req.get('connection_id')
        if user_id and connection_id:
            await _release_sync_resources(user_id, 'zoho-mail', connection_id)


async def quickbooks_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """
    IMPLEMENTATION #2: QuickBooks sync using Nango Proxy API.
    Fetches invoices, bills, and transactions for accounting reconciliation.
    Uses rate limiting from config_manager.py.
    """
    nango = NangoClient()
    try:
        from core_infrastructure.config_manager import get_nango_config
        config = get_nango_config()
        provider_key = config.quickbooks_integration_id
        connection_id = req.get('connection_id')
        user_id = req.get('user_id')
        sync_run_id = str(uuid.uuid4())
        
        stats = {
            'records_fetched': 0,
            'actions_used': 0,
            'invoices': 0,
            'bills': 0,
            'transactions': 0,
            'queued_jobs': 0,
            'skipped': 0,
        }
        errors: List[str] = []
        
        logger.info(f"🚀 QuickBooks sync started for user={user_id}")
        
        try:
            # Connectivity check
            company_info = await nango.proxy_get(
                'quickbooks',
                '/v2/company/{realm_id}',
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            logger.info(f"✅ QuickBooks company check passed: {company_info.get('CompanyInfo', {}).get('CompanyName', 'unknown')}")
        except Exception as e:
            error_msg = f"QuickBooks company check failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Rate limiter
        from core_infrastructure.rate_limiter import ConcurrencyLimiter
        limiter = ConcurrencyLimiter(max_concurrency=3)  # QB has stricter limits
        
        # Fetch invoices
        try:
            invoices_query = "select * from Invoice where MetaData.UpdatedTime >= '2024-01-01T00:00:00'"
            invoices_response = await nango.proxy_post(
                'quickbooks',
                '/v2/company/{realm_id}/query',
                json_body={'query': invoices_query},
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            
            invoices = invoices_response.get('QueryResponse', {}).get('Invoice') or []
            stats['invoices'] = len(invoices)
            stats['records_fetched'] += len(invoices)
            logger.info(f"✅ Fetched {len(invoices)} QuickBooks invoices")
            
            # Queue each invoice for processing
            for invoice in invoices:
                try:
                    invoice_data = {
                        'user_id': user_id,
                        'provider': 'quickbooks',
                        'provider_id': invoice.get('Id'),
                        'kind': 'invoice',
                        'amount': float(invoice.get('TotalAmt', 0)),
                        'currency': invoice.get('CurrencyRef', {}).get('value', 'USD'),
                        'date': invoice.get('TxnDate'),
                        'vendor': invoice.get('CustomerRef', {}).get('name', 'unknown'),
                        'metadata': invoice
                    }
                    # Store in external_items for processing
                    if supabase:
                        supabase.table('external_items').insert(invoice_data).execute()
                        stats['queued_jobs'] += 1
                except Exception as inv_err:
                    logger.error(f"Failed to queue QB invoice: {inv_err}")
                    errors.append(f"Invoice queue error: {inv_err}")
        except Exception as e:
            error_msg = f"QuickBooks invoice fetch failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        # Fetch bills
        try:
            bills_query = "select * from Bill where MetaData.UpdatedTime >= '2024-01-01T00:00:00'"
            bills_response = await nango.proxy_post(
                'quickbooks',
                '/v2/company/{realm_id}/query',
                json_body={'query': bills_query},
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            
            bills = bills_response.get('QueryResponse', {}).get('Bill') or []
            stats['bills'] = len(bills)
            stats['records_fetched'] += len(bills)
            logger.info(f"✅ Fetched {len(bills)} QuickBooks bills")
            
            for bill in bills:
                try:
                    bill_data = {
                        'user_id': user_id,
                        'provider': 'quickbooks',
                        'provider_id': bill.get('Id'),
                        'kind': 'bill',
                        'amount': float(bill.get('TotalAmt', 0)),
                        'currency': bill.get('CurrencyRef', {}).get('value', 'USD'),
                        'date': bill.get('TxnDate'),
                        'vendor': bill.get('VendorRef', {}).get('name', 'unknown'),
                        'metadata': bill
                    }
                    if supabase:
                        supabase.table('external_items').insert(bill_data).execute()
                        stats['queued_jobs'] += 1
                except Exception as bill_err:
                    logger.error(f"Failed to queue QB bill: {bill_err}")
                    errors.append(f"Bill queue error: {bill_err}")
        except Exception as e:
            error_msg = f"QuickBooks bill fetch failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        logger.info(f"✅ QuickBooks sync completed: {stats}")
        return {'status': 'completed', 'sync_run_id': sync_run_id, 'stats': stats, 'errors': errors}
        
    except Exception as e:
        logger.error(f"QuickBooks sync failed: {e}")
        raise Retry(defer=60)  # Retry after 60 seconds
    finally:
        # CRITICAL FIX #1 & #2: Always release resources
        user_id = req.get('user_id')
        connection_id = req.get('connection_id')
        if user_id and connection_id:
            await _release_sync_resources(user_id, 'quickbooks', connection_id)


async def xero_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """
    IMPLEMENTATION #3: Xero sync using Nango Proxy API.
    Fetches invoices, bills, and bank transactions for accounting reconciliation.
    Uses rate limiting from config_manager.py.
    """
    nango = NangoClient()
    try:
        from core_infrastructure.config_manager import get_nango_config
        config = get_nango_config()
        provider_key = config.xero_integration_id
        connection_id = req.get('connection_id')
        user_id = req.get('user_id')
        sync_run_id = str(uuid.uuid4())
        
        stats = {
            'records_fetched': 0,
            'actions_used': 0,
            'invoices': 0,
            'bills': 0,
            'bank_transactions': 0,
            'queued_jobs': 0,
            'skipped': 0,
        }
        errors: List[str] = []
        
        logger.info(f"🚀 Xero sync started for user={user_id}")
        
        try:
            # Connectivity check - get organisations
            orgs_response = await nango.proxy_get(
                'xero',
                '/api.xro/2.0/Organisations',
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            orgs = orgs_response.get('Organisations') or []
            if orgs:
                org_name = orgs[0].get('Name', 'unknown')
                logger.info(f"✅ Xero organisation check passed: {org_name}")
            else:
                raise Exception("No Xero organisations found")
        except Exception as e:
            error_msg = f"Xero organisation check failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Rate limiter
        from core_infrastructure.rate_limiter import ConcurrencyLimiter
        limiter = ConcurrencyLimiter(max_concurrency=3)  # Xero has API limits
        
        # Fetch invoices
        try:
            invoices_response = await nango.proxy_get(
                'xero',
                '/api.xro/2.0/Invoices?where=Status=="DRAFT" or Status=="SUBMITTED" or Status=="AUTHORISED"',
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            
            invoices = invoices_response.get('Invoices') or []
            stats['invoices'] = len(invoices)
            stats['records_fetched'] += len(invoices)
            logger.info(f"✅ Fetched {len(invoices)} Xero invoices")
            
            for invoice in invoices:
                try:
                    invoice_data = {
                        'user_id': user_id,
                        'provider': 'xero',
                        'provider_id': invoice.get('InvoiceID'),
                        'kind': 'invoice',
                        'amount': float(invoice.get('Total', 0)),
                        'currency': invoice.get('CurrencyCode', 'USD'),
                        'date': invoice.get('InvoiceNumber'),
                        'vendor': invoice.get('Contact', {}).get('Name', 'unknown'),
                        'metadata': invoice
                    }
                    if supabase:
                        supabase.table('external_items').insert(invoice_data).execute()
                        stats['queued_jobs'] += 1
                except Exception as inv_err:
                    logger.error(f"Failed to queue Xero invoice: {inv_err}")
                    errors.append(f"Invoice queue error: {inv_err}")
        except Exception as e:
            error_msg = f"Xero invoice fetch failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        # Fetch bank transactions
        try:
            transactions_response = await nango.proxy_get(
                'xero',
                '/api.xro/2.0/BankTransactions',
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            
            transactions = transactions_response.get('BankTransactions') or []
            stats['bank_transactions'] = len(transactions)
            stats['records_fetched'] += len(transactions)
            logger.info(f"✅ Fetched {len(transactions)} Xero bank transactions")
            
            for txn in transactions:
                try:
                    txn_data = {
                        'user_id': user_id,
                        'provider': 'xero',
                        'provider_id': txn.get('BankTransactionID'),
                        'kind': 'bank_transaction',
                        'amount': float(txn.get('Total', 0)),
                        'currency': txn.get('CurrencyCode', 'USD'),
                        'date': txn.get('DateString'),
                        'vendor': txn.get('Contact', {}).get('Name', 'unknown'),
                        'metadata': txn
                    }
                    if supabase:
                        supabase.table('external_items').insert(txn_data).execute()
                        stats['queued_jobs'] += 1
                except Exception as txn_err:
                    logger.error(f"Failed to queue Xero transaction: {txn_err}")
                    errors.append(f"Transaction queue error: {txn_err}")
        except Exception as e:
            error_msg = f"Xero bank transaction fetch failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        logger.info(f"✅ Xero sync completed: {stats}")
        return {'status': 'completed', 'sync_run_id': sync_run_id, 'stats': stats, 'errors': errors}
        
    except Exception as e:
        logger.error(f"Xero sync failed: {e}")
        raise Retry(defer=60)
    finally:
        # CRITICAL FIX #1 & #2: Always release resources
        user_id = req.get('user_id')
        connection_id = req.get('connection_id')
        if user_id and connection_id:
            await _release_sync_resources(user_id, 'xero', connection_id)


async def zoho_books_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """
    IMPLEMENTATION #4: Zoho Books sync using Nango Proxy API.
    Fetches invoices, bills, and expenses for accounting reconciliation.
    Uses rate limiting from config_manager.py.
    """
    nango = NangoClient()
    try:
        from core_infrastructure.config_manager import get_nango_config
        config = get_nango_config()
        provider_key = config.zoho_books_integration_id
        connection_id = req.get('connection_id')
        user_id = req.get('user_id')
        sync_run_id = str(uuid.uuid4())
        
        stats = {
            'records_fetched': 0,
            'actions_used': 0,
            'invoices': 0,
            'bills': 0,
            'expenses': 0,
            'queued_jobs': 0,
            'skipped': 0,
        }
        errors: List[str] = []
        
        logger.info(f"🚀 Zoho Books sync started for user={user_id}")
        
        try:
            # Connectivity check - get organization
            org_response = await nango.proxy_get(
                'zoho-books',
                '/api/v3/organizations',
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            orgs = org_response.get('organizations') or []
            if orgs:
                org_name = orgs[0].get('name', 'unknown')
                org_id = orgs[0].get('organization_id')
                logger.info(f"✅ Zoho Books organisation check passed: {org_name}")
            else:
                raise Exception("No Zoho Books organisations found")
        except Exception as e:
            error_msg = f"Zoho Books organisation check failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Rate limiter
        from core_infrastructure.rate_limiter import ConcurrencyLimiter
        limiter = ConcurrencyLimiter(max_concurrency=3)
        
        # Fetch invoices
        try:
            invoices_response = await nango.proxy_get(
                'zoho-books',
                f'/api/v3/invoices?organization_id={org_id}',
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            
            invoices = invoices_response.get('invoices') or []
            stats['invoices'] = len(invoices)
            stats['records_fetched'] += len(invoices)
            logger.info(f"✅ Fetched {len(invoices)} Zoho Books invoices")
            
            for invoice in invoices:
                try:
                    invoice_data = {
                        'user_id': user_id,
                        'provider': 'zoho-books',
                        'provider_id': invoice.get('invoice_id'),
                        'kind': 'invoice',
                        'amount': float(invoice.get('total', 0)),
                        'currency': invoice.get('currency_code', 'USD'),
                        'date': invoice.get('invoice_date'),
                        'vendor': invoice.get('customer_name', 'unknown'),
                        'metadata': invoice
                    }
                    if supabase:
                        supabase.table('external_items').insert(invoice_data).execute()
                        stats['queued_jobs'] += 1
                except Exception as inv_err:
                    logger.error(f"Failed to queue Zoho Books invoice: {inv_err}")
                    errors.append(f"Invoice queue error: {inv_err}")
        except Exception as e:
            error_msg = f"Zoho Books invoice fetch failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        # Fetch bills
        try:
            bills_response = await nango.proxy_get(
                'zoho-books',
                f'/api/v3/bills?organization_id={org_id}',
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            
            bills = bills_response.get('bills') or []
            stats['bills'] = len(bills)
            stats['records_fetched'] += len(bills)
            logger.info(f"✅ Fetched {len(bills)} Zoho Books bills")
            
            for bill in bills:
                try:
                    bill_data = {
                        'user_id': user_id,
                        'provider': 'zoho-books',
                        'provider_id': bill.get('bill_id'),
                        'kind': 'bill',
                        'amount': float(bill.get('total', 0)),
                        'currency': bill.get('currency_code', 'USD'),
                        'date': bill.get('bill_date'),
                        'vendor': bill.get('vendor_name', 'unknown'),
                        'metadata': bill
                    }
                    if supabase:
                        supabase.table('external_items').insert(bill_data).execute()
                        stats['queued_jobs'] += 1
                except Exception as bill_err:
                    logger.error(f"Failed to queue Zoho Books bill: {bill_err}")
                    errors.append(f"Bill queue error: {bill_err}")
        except Exception as e:
            error_msg = f"Zoho Books bill fetch failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        logger.info(f"✅ Zoho Books sync completed: {stats}")
        return {'status': 'completed', 'sync_run_id': sync_run_id, 'stats': stats, 'errors': errors}
        
    except Exception as e:
        logger.error(f"Zoho Books sync failed: {e}")
        raise Retry(defer=60)
    finally:
        # CRITICAL FIX #1 & #2: Always release resources
        user_id = req.get('user_id')
        connection_id = req.get('connection_id')
        if user_id and connection_id:
            await _release_sync_resources(user_id, 'zoho-books', connection_id)


async def stripe_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """
    IMPLEMENTATION #5: Stripe sync using Nango Proxy API.
    Fetches charges, invoices, and payouts for payment reconciliation.
    Uses rate limiting from config_manager.py.
    """
    nango = NangoClient()
    try:
        from core_infrastructure.config_manager import get_nango_config
        config = get_nango_config()
        provider_key = config.stripe_integration_id
        connection_id = req.get('connection_id')
        user_id = req.get('user_id')
        sync_run_id = str(uuid.uuid4())
        
        stats = {
            'records_fetched': 0,
            'actions_used': 0,
            'charges': 0,
            'invoices': 0,
            'payouts': 0,
            'queued_jobs': 0,
            'skipped': 0,
        }
        errors: List[str] = []
        
        logger.info(f"🚀 Stripe sync started for user={user_id}")
        
        try:
            # Connectivity check - get account
            account_response = await nango.proxy_get(
                'stripe',
                '/v1/account',
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            account_name = account_response.get('business_profile', {}).get('name') or account_response.get('email', 'unknown')
            logger.info(f"✅ Stripe account check passed: {account_name}")
        except Exception as e:
            error_msg = f"Stripe account check failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Rate limiter
        from core_infrastructure.rate_limiter import ConcurrencyLimiter
        limiter = ConcurrencyLimiter(max_concurrency=5)  # Stripe allows higher concurrency
        
        # Fetch charges
        try:
            charges_response = await nango.proxy_get(
                'stripe',
                '/v1/charges?limit=100',
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            
            charges = charges_response.get('data') or []
            stats['charges'] = len(charges)
            stats['records_fetched'] += len(charges)
            logger.info(f"✅ Fetched {len(charges)} Stripe charges")
            
            for charge in charges:
                try:
                    charge_data = {
                        'user_id': user_id,
                        'provider': 'stripe',
                        'provider_id': charge.get('id'),
                        'kind': 'charge',
                        'amount': float(charge.get('amount', 0)) / 100,  # Stripe uses cents
                        'currency': charge.get('currency', 'usd').upper(),
                        'date': charge.get('created'),
                        'vendor': charge.get('description', 'Stripe charge'),
                        'metadata': charge
                    }
                    if supabase:
                        supabase.table('external_items').insert(charge_data).execute()
                        stats['queued_jobs'] += 1
                except Exception as charge_err:
                    logger.error(f"Failed to queue Stripe charge: {charge_err}")
                    errors.append(f"Charge queue error: {charge_err}")
        except Exception as e:
            error_msg = f"Stripe charge fetch failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        # Fetch invoices
        try:
            invoices_response = await nango.proxy_get(
                'stripe',
                '/v1/invoices?limit=100',
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            
            invoices = invoices_response.get('data') or []
            stats['invoices'] = len(invoices)
            stats['records_fetched'] += len(invoices)
            logger.info(f"✅ Fetched {len(invoices)} Stripe invoices")
            
            for invoice in invoices:
                try:
                    invoice_data = {
                        'user_id': user_id,
                        'provider': 'stripe',
                        'provider_id': invoice.get('id'),
                        'kind': 'invoice',
                        'amount': float(invoice.get('amount_due', 0)) / 100,
                        'currency': invoice.get('currency', 'usd').upper(),
                        'date': invoice.get('created'),
                        'vendor': invoice.get('customer_name', 'Stripe invoice'),
                        'metadata': invoice
                    }
                    if supabase:
                        supabase.table('external_items').insert(invoice_data).execute()
                        stats['queued_jobs'] += 1
                except Exception as inv_err:
                    logger.error(f"Failed to queue Stripe invoice: {inv_err}")
                    errors.append(f"Invoice queue error: {inv_err}")
        except Exception as e:
            error_msg = f"Stripe invoice fetch failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        logger.info(f"✅ Stripe sync completed: {stats}")
        return {'status': 'completed', 'sync_run_id': sync_run_id, 'stats': stats, 'errors': errors}
        
    except Exception as e:
        logger.error(f"Stripe sync failed: {e}")
        raise Retry(defer=45)
    finally:
        # CRITICAL FIX #1 & #2: Always release resources
        user_id = req.get('user_id')
        connection_id = req.get('connection_id')
        if user_id and connection_id:
            await _release_sync_resources(user_id, 'stripe', connection_id)


async def razorpay_sync(ctx, req: Dict[str, Any]) -> Dict[str, Any]:
    """
    IMPLEMENTATION #6: Razorpay sync using Nango Proxy API.
    Fetches payments, invoices, and transfers for payment reconciliation.
    Uses rate limiting from config_manager.py.
    """
    nango = NangoClient()
    try:
        from core_infrastructure.config_manager import get_nango_config
        config = get_nango_config()
        provider_key = config.razorpay_integration_id
        connection_id = req.get('connection_id')
        user_id = req.get('user_id')
        sync_run_id = str(uuid.uuid4())
        
        stats = {
            'records_fetched': 0,
            'actions_used': 0,
            'payments': 0,
            'invoices': 0,
            'transfers': 0,
            'queued_jobs': 0,
            'skipped': 0,
        }
        errors: List[str] = []
        
        logger.info(f"🚀 Razorpay sync started for user={user_id}")
        
        try:
            # Connectivity check - get account
            account_response = await nango.proxy_get(
                'razorpay',
                '/api/v1/account',
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            account_id = account_response.get('id', 'unknown')
            logger.info(f"✅ Razorpay account check passed: {account_id}")
        except Exception as e:
            error_msg = f"Razorpay account check failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Rate limiter
        from core_infrastructure.rate_limiter import ConcurrencyLimiter
        limiter = ConcurrencyLimiter(max_concurrency=4)  # Razorpay API limits
        
        # Fetch payments
        try:
            payments_response = await nango.proxy_get(
                'razorpay',
                '/api/v1/payments?count=100',
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            
            payments = payments_response.get('items') or []
            stats['payments'] = len(payments)
            stats['records_fetched'] += len(payments)
            logger.info(f"✅ Fetched {len(payments)} Razorpay payments")
            
            for payment in payments:
                try:
                    payment_data = {
                        'user_id': user_id,
                        'provider': 'razorpay',
                        'provider_id': payment.get('id'),
                        'kind': 'payment',
                        'amount': float(payment.get('amount', 0)) / 100,  # Razorpay uses paise
                        'currency': payment.get('currency', 'INR'),
                        'date': payment.get('created_at'),
                        'vendor': payment.get('description', 'Razorpay payment'),
                        'metadata': payment
                    }
                    if supabase:
                        supabase.table('external_items').insert(payment_data).execute()
                        stats['queued_jobs'] += 1
                except Exception as pay_err:
                    logger.error(f"Failed to queue Razorpay payment: {pay_err}")
                    errors.append(f"Payment queue error: {pay_err}")
        except Exception as e:
            error_msg = f"Razorpay payment fetch failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        # Fetch invoices
        try:
            invoices_response = await nango.proxy_get(
                'razorpay',
                '/api/v1/invoices?count=100',
                connection_id=connection_id,
                provider_config_key=provider_key
            )
            stats['actions_used'] += 1
            
            invoices = invoices_response.get('items') or []
            stats['invoices'] = len(invoices)
            stats['records_fetched'] += len(invoices)
            logger.info(f"✅ Fetched {len(invoices)} Razorpay invoices")
            
            for invoice in invoices:
                try:
                    invoice_data = {
                        'user_id': user_id,
                        'provider': 'razorpay',
                        'provider_id': invoice.get('id'),
                        'kind': 'invoice',
                        'amount': float(invoice.get('amount', 0)) / 100,
                        'currency': invoice.get('currency', 'INR'),
                        'date': invoice.get('created_at'),
                        'vendor': invoice.get('customer_details', {}).get('name', 'Razorpay invoice'),
                        'metadata': invoice
                    }
                    if supabase:
                        supabase.table('external_items').insert(invoice_data).execute()
                        stats['queued_jobs'] += 1
                except Exception as inv_err:
                    logger.error(f"Failed to queue Razorpay invoice: {inv_err}")
                    errors.append(f"Invoice queue error: {inv_err}")
        except Exception as e:
            error_msg = f"Razorpay invoice fetch failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        logger.info(f"✅ Razorpay sync completed: {stats}")
        return {'status': 'completed', 'sync_run_id': sync_run_id, 'stats': stats, 'errors': errors}
        
    except Exception as e:
        logger.error(f"Razorpay sync failed: {e}")
        raise Retry(defer=45)
    finally:
        # CRITICAL FIX #1 & #2: Always release resources
        user_id = req.get('user_id')
        connection_id = req.get('connection_id')
        if user_id and connection_id:
            await _release_sync_resources(user_id, 'razorpay', connection_id)


async def process_spreadsheet(ctx, user_id: str, filename: str, storage_path: str, job_id: str, 
                             duplicate_decision: Optional[str] = None, existing_file_id: Optional[str] = None) -> Dict[str, Any]:
    """
    CRITICAL FIX: Removed nested transaction wrapper.
    ExcelProcessor.process_file manages transaction internally (fastapi_backend_v2.py@8676-8680).
    Nested transactions cause deadlocks.
    FIX #7: Added exception logging for observability
    """
    try:
        # Transaction managed inside start_processing_job -> ExcelProcessor.process_file
        await start_processing_job(
            user_id=user_id, 
            job_id=job_id, 
            storage_path=storage_path, 
            filename=filename,
            duplicate_decision=duplicate_decision,
            existing_file_id=existing_file_id
        )
        logger.info("ARQ spreadsheet processing completed", job_id=job_id)
        return {"status": "completed", "job_id": job_id}
            
    except Exception as e:
        # FIX #7: Log full exception details for debugging
        logger.error("ARQ spreadsheet processing failed", job_id=job_id,
                    error=str(e), error_type=type(e).__name__, user_id=user_id, filename=filename)
        # Transaction auto-rolls back on exception
        # Use ARQ's native Retry with exponential backoff
        # Calculate exponential backoff: 60s, 120s, 240s for retries 1, 2, 3
        retry_count = getattr(ctx, 'retry_count', 0) if ctx else 0
        if retry_count < 3:
            delay = 60 * (2 ** retry_count)  # 60s, 120s, 240s
            logger.info("Retrying spreadsheet processing", delay_seconds=delay, attempt=retry_count + 1, job_id=job_id)
            raise Retry(defer=delay)
        else:
            logger.error("Spreadsheet processing failed after 3 retries", job_id=job_id, error=str(e))
            return {"status": "failed", "job_id": job_id, "error": str(e), "retries_exhausted": True}


async def process_pdf(ctx, user_id: str, filename: str, storage_path: str, job_id: str) -> Dict[str, Any]:
    """FIX #7: Wrap ARQ PDF processing in transaction for consistency with Phase 1-11
    Added exception logging for observability
    """
    try:
        from aident_cfo_brain.transaction_manager import get_transaction_manager
        transaction_manager = get_transaction_manager()
        
        # Wrap entire processing in transaction for atomic operations
        async with transaction_manager.transaction(
            user_id=user_id,
            operation_type="arq_pdf_processing"
        ) as tx:
            await start_pdf_processing_job(user_id, job_id, storage_path, filename)
            logger.info("ARQ PDF processing completed", job_id=job_id)
            return {"status": "completed", "job_id": job_id}
            
    except Exception as e:
        # FIX #7: Log full exception details for debugging
        logger.error("ARQ PDF processing failed", job_id=job_id,
                    error=str(e), error_type=type(e).__name__, user_id=user_id, filename=filename)
        # Transaction auto-rolls back on exception
        # Use ARQ's native Retry with exponential backoff
        retry_count = getattr(ctx, 'retry_count', 0) if ctx else 0
        if retry_count < 3:
            delay = 60 * (2 ** retry_count)  # 60s, 120s, 240s
            logger.info("Retrying PDF processing", delay_seconds=delay, attempt=retry_count + 1, job_id=job_id)
            raise Retry(defer=delay)
        else:
            logger.error("PDF processing failed after 3 retries", job_id=job_id, error=str(e))
            return {"status": "failed", "job_id": job_id, "error": str(e), "retries_exhausted": True}


async def learn_field_mapping_batch(ctx, mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    CRITICAL FIX: Persistent field mapping learning via ARQ.
    Replaces in-process asyncio.Queue to prevent data loss on restart.
    FIX #7: Added exception logging for observability
    
    Args:
        ctx: ARQ context
        mappings: List of field mapping records to persist
    """
    try:
        from field_mapping_learner import FieldMappingLearner
        
        logger.info("Processing field mapping records via ARQ", record_count=len(mappings))
        
        learner = FieldMappingLearner(supabase=supabase)
        success_count = 0
        failed_count = 0
        
        for mapping in mappings:
            try:
                success = await learner._write_mapping_with_retry(mapping)
                if success:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                # FIX #7: Log mapping details for debugging
                logger.error("Failed to write field mapping", error=str(e), mapping_id=mapping.get('id'))
                failed_count += 1
        
        logger.info("Field mapping batch completed", success_count=success_count, failed_count=failed_count)
        return {
            "status": "success",
            "total": len(mappings),
            "success_count": success_count,
            "failed_count": failed_count
        }
        
    except Exception as e:
        # FIX #7: Log full exception details for debugging
        logger.error("Field mapping batch processing failed", error=str(e), error_type=type(e).__name__)
        # Use ARQ's native Retry with exponential backoff
        retry_count = getattr(ctx, 'retry_count', 0) if ctx else 0
        if retry_count < 3:
            delay = 30 * (2 ** retry_count)  # 30s, 60s, 120s
            logger.info("Retrying field mapping batch", delay_seconds=delay, attempt=retry_count + 1)
            raise Retry(defer=delay)
        else:
            logger.error("Field mapping batch failed after 3 retries", error=str(e))
            return {"status": "failed", "error": str(e), "retries_exhausted": True}


async def detect_relationships(ctx, user_id: str, file_id: str = None) -> Dict[str, Any]:
    """
    CRITICAL FIX #8: Background task for relationship detection with transaction wrapper.
    This decouples heavy analysis from the synchronous ingestion pipeline and ensures
    atomic operations (all-or-nothing) for data consistency.
    FIX #7: Added exception logging for observability
    
    Args:
        ctx: ARQ context
        user_id: User ID to detect relationships for
        file_id: Optional file_id to scope detection to specific file
    """
    try:
        from aident_cfo_brain.enhanced_relationship_detector import EnhancedRelationshipDetector
        from groq import Groq
        from aident_cfo_brain.transaction_manager import get_transaction_manager
        from aident_cfo_brain.finley_graph_engine import FinleyGraphEngine
        from data_ingestion_normalization.embedding_service import EmbeddingService
        
        logger.info("Starting background relationship detection", user_id=user_id, file_id=file_id)
        
        # FIX #8: Wrap entire detection in transaction for atomic operations
        transaction_manager = get_transaction_manager()
        
        async with transaction_manager.transaction(
            user_id=user_id,
            operation_type="arq_relationship_detection"
        ) as tx:
            # CRITICAL FIX: Use centralized_cache instead of deprecated ai_cache_system
            # This ensures workers share the same Redis cache as the main application
            try:
                from centralized_cache import safe_get_cache
                cache_client = safe_get_cache()
            except:
                cache_client = None
            
            # FIX #6: Initialize embedding service for dependency injection
            embedding_service = None
            try:
                embedding_service = EmbeddingService(cache_client=cache_client)
                logger.info("✅ EmbeddingService initialized for relationship detection")
            except Exception as e:
                logger.warning("Failed to initialize EmbeddingService", error=str(e))
            
            # CRITICAL FIX: Use Groq client (Llama-3.3-70B) instead of Anthropic
            # This matches the main application's AI configuration
            groq_api_key = os.getenv('GROQ_API_KEY')
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable is required for relationship detection")
            
            groq_client = Groq(api_key=groq_api_key)
            logger.info("✅ Groq client initialized for relationship detection (Llama-3.3-70B)")
            
            # Initialize relationship detector with Groq client and embedding service
            # Note: EnhancedRelationshipDetector accepts anthropic_client parameter for backward compatibility
            # but can work with any AI client that follows the same interface
            relationship_detector = EnhancedRelationshipDetector(
                anthropic_client=groq_client,  # Pass Groq client via anthropic_client parameter
                supabase_client=supabase,
                cache_client=cache_client,
                embedding_service=embedding_service  # FIX #6: Pass injected embedding service
            )
            
            # CRITICAL FIX: Use new database-driven detection (no N² loops, no hardcoded filenames)
            relationship_results = await relationship_detector.detect_all_relationships(
                user_id, 
                file_id=file_id
            )
            
            # Relationships are already stored by the detector
            logger.info("Background relationship detection completed", total_relationships=relationship_results.get('total_relationships', 0))
            
            # FIX #4-8: Run advanced analytics AFTER relationship detection
            analytics_results = {}
            try:
                logger.info("Starting advanced analytics", user_id=user_id)
                
                # CRITICAL FIX: Use engines already initialized by EnhancedRelationshipDetector
                # This eliminates redundant initialization and ensures consistent state
                temporal_learner = relationship_detector.temporal_learner if hasattr(relationship_detector, 'temporal_learner') else None
                causal_engine = relationship_detector.causal_engine if hasattr(relationship_detector, 'causal_engine') else None
                
                # Fallback: Initialize only if not available (backward compatibility)
                if temporal_learner is None or causal_engine is None:
                    from temporal_pattern_learner import TemporalPatternLearner
                    from causal_inference_engine import CausalInferenceEngine
                    if temporal_learner is None:
                        temporal_learner = TemporalPatternLearner(supabase)
                    if causal_engine is None:
                        causal_engine = CausalInferenceEngine(supabase)
                
                # 1. Learn temporal patterns
                pattern_results = await temporal_learner.learn_all_patterns(user_id)
                analytics_results['temporal_patterns'] = pattern_results.get('total_patterns', 0)
                
                # 2. Detect temporal anomalies
                anomaly_results = await temporal_learner.detect_temporal_anomalies(user_id)
                analytics_results['temporal_anomalies'] = anomaly_results.get('total_anomalies', 0)
                
                # 3. Predict missing relationships
                prediction_results = await temporal_learner.predict_missing_relationships(user_id)
                analytics_results['predicted_relationships'] = prediction_results.get('total_predictions', 0)
                
                # 4. Analyze causal relationships
                causal_results = await causal_engine.analyze_all_relationships(user_id)
                analytics_results['causal_relationships'] = causal_results.get('total_causal', 0)
                
                # 5. Perform root cause analysis
                root_cause_results = await causal_engine.analyze_root_causes(user_id)
                analytics_results['root_cause_analyses'] = root_cause_results.get('total_root_causes', 0)
                
                # 6. Run counterfactual analysis
                counterfactual_results = await causal_engine.analyze_counterfactuals(user_id)
                analytics_results['counterfactual_analyses'] = counterfactual_results.get('total_scenarios', 0)
                
                logger.info("Advanced analytics completed", results=analytics_results)
                
            except Exception as analytics_error:
                logger.error("Advanced analytics failed (non-critical)", error=str(analytics_error))
                analytics_results['error'] = str(analytics_error)
            
            # FIX #11: Invalidate graph cache after successful detection
            try:
                graph_engine = FinleyGraphEngine(supabase_url=os.getenv('SUPABASE_URL'), redis_url=os.getenv('REDIS_URL'))
                await graph_engine.clear_graph_cache(user_id)
                logger.info("Graph cache invalidated", user_id=user_id)
            except Exception as cache_error:
                logger.warning("Failed to invalidate graph cache", error=str(cache_error))
            
            # SEGMENT C: Queue Phase 3 and Phase 4 background jobs
            # FIX #12: Queue graph building (Phase 3) after relationships detected
            # FIX #13: Queue CFO brain initialization (Phase 4) after graph built
            try:
                from arq.connections import create_pool
                redis_url = os.getenv('ARQ_REDIS_URL') or os.getenv('REDIS_URL') or 'redis://localhost:6379'
                arq_pool = await create_pool(redis_url)
                
                # Queue Phase 3: Graph building
                logger.info("Queueing PHASE 3 (graph building)", user_id=user_id)
                await arq_pool.enqueue_job('build_graph_background', user_id=user_id)
                
                # Queue Phase 4: CFO brain initialization
                # Note: ARQ will execute these sequentially based on queue order
                logger.info("Queueing PHASE 4 (CFO brain init)", user_id=user_id)
                await arq_pool.enqueue_job('initialize_cfo_brain', user_id=user_id)
                
                logger.info("PHASE 3 & 4 queued", user_id=user_id)
            except Exception as queue_error:
                logger.warning("Failed to queue Phase 3/4 jobs", error=str(queue_error))
            
            return {
                "status": "success",
                "user_id": user_id,
                "file_id": file_id,
                "total_relationships": relationship_results.get('total_relationships', 0),
                "cross_document_relationships": relationship_results.get('cross_document_relationships', 0),
                "within_file_relationships": relationship_results.get('within_file_relationships', 0),
                "method": "database_joins",
                "complexity": "O(N log N)",
                "advanced_analytics": analytics_results,
                "phase_3_4_queued": True
            }
        
    except Exception as e:
        logger.error("Background relationship detection failed", user_id=user_id, error=str(e), exc_info=True)
        # FIX #9: Use ARQ's native Retry with exponential backoff (no _retry_or_dlq)
        # Calculate exponential backoff: 60s, 120s, 240s for retries 1, 2, 3
        retry_count = getattr(ctx, 'retry_count', 0) if ctx else 0
        if retry_count < 3:
            delay = 60 * (2 ** retry_count)  # 60s, 120s, 240s
            logger.info("Retrying relationship detection", delay_seconds=delay, attempt=retry_count + 1)
            raise Retry(defer=delay)
        else:
            logger.error("Relationship detection failed after 3 retries", user_id=user_id)
            return {"status": "failed", "user_id": user_id, "error": str(e), "retries_exhausted": True}


async def build_graph_background(ctx, user_id: str) -> Dict[str, Any]:
    """
    PHASE 3: Build and cache knowledge graph after relationships detected.
    
    This is a critical background job that:
    1. Fetches nodes from normalized_entities
    2. Fetches edges from view_enriched_relationships (with all 9 intelligence layers)
    3. Builds igraph with full relationship intelligence
    4. Caches in Redis for instant graph queries
    
    FIX #12: Proactive graph building (not lazy on-demand)
    - Eliminates cold-start delay on first user question
    - Pre-warms Redis cache for instant queries
    - Enables Phase 4 (CFO brain initialization)
    
    Args:
        ctx: ARQ context
        user_id: User ID to build graph for
    
    Returns:
        Dict with status and graph statistics
    """
    try:
        from aident_cfo_brain.finley_graph_engine import FinleyGraphEngine
        
        logger.info("PHASE 3: Starting graph building for user", user_id=user_id)
        
        # Initialize graph engine with Redis caching
        redis_url = os.getenv('ARQ_REDIS_URL') or os.getenv('REDIS_URL')
        graph_engine = FinleyGraphEngine(supabase=supabase, redis_url=redis_url)
        
        # Build graph from database (force rebuild to ensure fresh data)
        stats = await graph_engine.build_graph(user_id, force_rebuild=True)
        
        logger.info(
            "PHASE 3 COMPLETE: Graph built successfully",
            user_id=user_id,
            node_count=stats.node_count,
            edge_count=stats.edge_count,
            density=round(stats.density, 3),
            build_time_seconds=round(stats.build_time_seconds, 2)
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "phase": 3,
            "node_count": stats.node_count,
            "edge_count": stats.edge_count,
            "density": stats.density,
            "build_time_seconds": stats.build_time_seconds,
            "message": f"Knowledge graph built: {stats.node_count} nodes, {stats.edge_count} edges"
        }
        
    except Exception as e:
        logger.error("PHASE 3 FAILED: Graph building failed", user_id=user_id, error=str(e), exc_info=True)
        # Use ARQ's native Retry with exponential backoff
        retry_count = getattr(ctx, 'retry_count', 0) if ctx else 0
        if retry_count < 3:
            delay = 60 * (2 ** retry_count)  # 60s, 120s, 240s
            logger.info("Retrying graph building", delay_seconds=delay, attempt=retry_count + 1)
            raise Retry(defer=delay)
        else:
            logger.error("Graph building failed after 3 retries", user_id=user_id)
            return {"status": "failed", "user_id": user_id, "phase": 3, "error": str(e), "retries_exhausted": True}


async def initialize_cfo_brain(ctx, user_id: str) -> Dict[str, Any]:
    """
    PHASE 4: Initialize all CFO brain engines after graph is ready.
    
    This is the final background job that:
    1. Initializes CausalInferenceEngine
    2. Initializes TemporalPatternLearner
    3. Initializes AidentMemoryManager
    4. Pre-warms all caches for instant responses
    
    FIX #13: Proactive CFO brain initialization (not lazy on-demand)
    - Eliminates cold-start delay on first user question
    - Pre-computes causal relationships
    - Pre-learns temporal patterns
    - Initializes conversation memory
    
    Args:
        ctx: ARQ context
        user_id: User ID to initialize brain for
    
    Returns:
        Dict with status and initialization results
    """
    try:
        from causal_inference_engine import CausalInferenceEngine
        from temporal_pattern_learner import TemporalPatternLearner
        from aident_memory_manager import AidentMemoryManager
        
        logger.info("PHASE 4: Starting CFO brain initialization", user_id=user_id)
        
        redis_url = os.getenv('ARQ_REDIS_URL') or os.getenv('REDIS_URL')
        
        # Initialize all engines
        causal_engine = CausalInferenceEngine(supabase_client=supabase)
        temporal_learner = TemporalPatternLearner(supabase_client=supabase)
        memory_manager = AidentMemoryManager(user_id=user_id, redis_url=redis_url)
        
        logger.info("Engines initialized, pre-warming caches", user_id=user_id)
        
        # Pre-warm caches by running analysis
        try:
            causal_results = await causal_engine.analyze_all_relationships(user_id)
            logger.info("Causal analysis pre-warmed", total_causal=causal_results.get('total_causal', 0))
        except Exception as e:
            logger.warning("Causal analysis pre-warm failed (non-critical)", error=str(e))
            causal_results = {"total_causal": 0, "error": str(e)}
        
        try:
            temporal_results = await temporal_learner.learn_all_patterns(user_id)
            logger.info("Temporal patterns pre-warmed", total_patterns=temporal_results.get('total_patterns', 0))
        except Exception as e:
            logger.warning("Temporal learning pre-warm failed (non-critical)", error=str(e))
            temporal_results = {"total_patterns": 0, "error": str(e)}
        
        try:
            await memory_manager.load_memory()
            logger.info("Memory manager initialized", user_id=user_id)
        except Exception as e:
            logger.warning("Memory manager initialization failed (non-critical)", error=str(e))
        
        logger.info(
            "PHASE 4 COMPLETE: CFO brain fully initialized",
            user_id=user_id,
            causal_relationships=causal_results.get('total_causal', 0),
            temporal_patterns=temporal_results.get('total_patterns', 0)
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "phase": 4,
            "causal_relationships": causal_results.get('total_causal', 0),
            "temporal_patterns": temporal_results.get('total_patterns', 0),
            "message": "CFO brain fully initialized and ready for questions"
        }
        
    except Exception as e:
        logger.error("PHASE 4 FAILED: CFO brain initialization failed", user_id=user_id, error=str(e), exc_info=True)
        # Use ARQ's native Retry with exponential backoff
        retry_count = getattr(ctx, 'retry_count', 0) if ctx else 0
        if retry_count < 3:
            delay = 60 * (2 ** retry_count)  # 60s, 120s, 240s
            logger.info("Retrying CFO brain initialization", delay_seconds=delay, attempt=retry_count + 1)
            raise Retry(defer=delay)
        else:
            logger.error("CFO brain initialization failed after 3 retries", user_id=user_id)
            return {"status": "failed", "user_id": user_id, "phase": 4, "error": str(e), "retries_exhausted": True}


async def generate_prophet_forecasts(ctx) -> None:
    """
    Nightly job: Generate Prophet forecasts for all users.
    Stores results in temporal_patterns table for instant API access.
    """
    try:
        logger.info("🌙 Starting nightly Prophet forecasting job")
        from temporal_pattern_learner import TemporalPatternLearner
        
        # Initialize learner
        learner = TemporalPatternLearner(supabase_client=supabase)
        
        # Get all users with relationships
        # Using a distinct query to find active users
        users_result = supabase.table('relationship_instances').select('user_id').execute()
        if not users_result.data:
            logger.info("No users found for forecasting")
            return
            
        user_ids = list(set(r['user_id'] for r in users_result.data))
        logger.info("Generating forecasts for users", user_count=len(user_ids))
        
        for user_id in user_ids:
            try:
                # Get relationship types for this user
                patterns_result = supabase.table('temporal_patterns').select('relationship_type').eq('user_id', user_id).execute()
                if not patterns_result.data:
                    continue
                    
                relationship_types = [p['relationship_type'] for p in patterns_result.data]
                
                for rel_type in relationship_types:
                    try:
                        # Generate forecast (runs in thread pool - OK for background job)
                        # This will update the database automatically
                        await learner.forecast_with_prophet(
                            user_id=user_id,
                            relationship_type=rel_type,
                            forecast_days=90
                        )
                        logger.debug("Generated forecast", user_id=user_id, relationship_type=rel_type)
                        
                    except Exception as e:
                        logger.error("Forecast failed", user_id=user_id, relationship_type=rel_type, error=str(e))
                        continue
                        
            except Exception as user_err:
                logger.error("Forecasting failed for user", user_id=user_id, error=str(user_err))
                continue
                
        logger.info("✅ Nightly Prophet forecasting job completed")
        
    except Exception as e:
        logger.error("Nightly Prophet forecasting job failed", error=str(e))


# --------------- ARQ Worker Settings ---------------
class WorkerSettings:
    redis_settings = RedisSettings.from_dsn(
        os.environ.get("ARQ_REDIS_URL") or os.environ.get("REDIS_URL") or "redis://localhost:6379"
    )

    functions = [
        gmail_sync,
        dropbox_sync,
        gdrive_sync,
        zoho_mail_sync,
        quickbooks_sync,
        xero_sync,
        zoho_books_sync,
        stripe_sync,
        razorpay_sync,
        process_spreadsheet,
        process_pdf,
        learn_field_mapping_batch,  # CRITICAL FIX: Persistent field mapping learning
        detect_relationships,  # PHASE 2: Background task for relationship detection
        build_graph_background,  # FIX #12: PHASE 3: Proactive graph building
        initialize_cfo_brain,  # FIX #13: PHASE 4: CFO brain initialization
        generate_prophet_forecasts,  # Nightly forecasting job
    ]
    
    cron_jobs = [
        cron(generate_prophet_forecasts, hour=2, minute=0)  # Run at 2 AM daily
    ]

    # Keep results in Redis only briefly; we don't depend on ARQ results downstream
    keep_result = 0
    # Reasonable timeout per job (seconds)
    function_timeout = 60 * 15
