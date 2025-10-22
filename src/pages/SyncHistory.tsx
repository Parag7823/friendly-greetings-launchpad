import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/components/ui/use-toast";
import { Button } from "@/components/ui/button";
import { config } from "@/config";

interface RunItem {
  id: string;
  type?: string;
  status?: string;
  started_at?: string;
  finished_at?: string | null;
  stats?: Record<string, any> | null;
  error?: any;
}

export default function SyncHistory() {
  const { connectionId } = useParams<{ connectionId: string }>();
  const [search, setSearch] = useSearchParams();
  const page = Math.max(1, Number(search.get("page") || 1));
  const pageSize = Math.max(1, Math.min(100, Number(search.get("page_size") || 20)));
  const [runs, setRuns] = useState<RunItem[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();
  const [header, setHeader] = useState<{ status?: string; last_synced_at?: string; integration_id?: string } | null>(null);

  const brandIcon = (slug: string, colorHex: string, alt: string) => (
    <img src={`https://cdn.simpleicons.org/${slug}/${colorHex.replace('#', '')}`} alt={alt} className="w-6 h-6" />
  );

  const titleMeta = useMemo(() => {
    const integ = header?.integration_id || "";
    const map: Record<string, { slug: string; color: string; name: string }> = {
      "google-mail": { slug: "gmail", color: "EA4335", name: "Gmail" },
      "zoho-mail": { slug: "zoho", color: "C8202F", name: "Zoho Mail" },
      "zoho-books": { slug: "zoho", color: "C8202F", name: "Zoho Books" },
      dropbox: { slug: "dropbox", color: "0061FF", name: "Dropbox" },
      "google-drive": { slug: "googledrive", color: "1A73E8", name: "Google Drive" },
      quickbooks: { slug: "intuitquickbooks", color: "2CA01C", name: "QuickBooks" },
      "quickbooks-sandbox": { slug: "intuitquickbooks", color: "2CA01C", name: "QuickBooks (Sandbox)" },
      xero: { slug: "xero", color: "13B5EA", name: "Xero" },
      stripe: { slug: "stripe", color: "635BFF", name: "Stripe" },
      razorpay: { slug: "razorpay", color: "0C2451", name: "Razorpay" },
    };
    return map[integ] || { slug: "cloud", color: "6B7280", name: integ || "Connection" };
  }, [header]);

  const renderCount = (stats?: Record<string, any> | null) => {
    if (!stats) return "-";
    const keys = ["transactions", "records", "items", "count", "synced", "success_count", "total"];
    for (const k of keys) if (typeof stats[k] === "number") return String(stats[k]);
    try {
      const sum = Object.values(stats).filter(v => typeof v === "number").reduce((a: number, b: number) => a + (b as number), 0);
      return sum ? String(sum) : "-";
    } catch { return "-"; }
  };

  const fetchPage = async () => {
    if (!connectionId) return;
    try {
      setLoading(true);
      const { data: sessionData } = await supabase.auth.getSession();
      const sessionToken = sessionData?.session?.access_token;
      const user = sessionData?.session?.user;
      const params = new URLSearchParams({
        connection_id: connectionId,
        user_id: user?.id || "",
        page: String(page),
        page_size: String(pageSize),
      });
      if (sessionToken) params.set("session_token", sessionToken);
      const resp = await fetch(`${config.apiUrl}/api/connectors/history?${params.toString()}`, {
        headers: {
          ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
        }
      });
      if (!resp.ok) throw new Error("Failed to load history");
      const data = await resp.json();
      setRuns(Array.isArray(data?.runs) ? data.runs : []);
      setHasMore(Boolean(data?.has_more));

      // Also fetch header details from status endpoint
      const params2 = new URLSearchParams({ connection_id: connectionId, user_id: user?.id || "" });
      if (sessionToken) params2.set("session_token", sessionToken);
      const resp2 = await fetch(`${config.apiUrl}/api/connectors/status?${params2.toString()}`, {
        headers: {
          ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
        }
      });
      if (resp2.ok) {
        const s = await resp2.json();
        setHeader(s?.connection || null);
      }
    } catch (e: any) {
      console.error(e);
      toast({ title: "Failed to load history", description: e?.message || "Please try again.", variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchPage(); /* eslint-disable-next-line */ }, [connectionId, page, pageSize]);

  const goPage = (p: number) => {
    const next = new URLSearchParams(search);
    next.set("page", String(Math.max(1, p)));
    next.set("page_size", String(pageSize));
    setSearch(next, { replace: true });
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="sticky top-0 z-10 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            {brandIcon(titleMeta.slug, titleMeta.color, titleMeta.name)}
            <h1 className="text-xl font-semibold text-foreground">Sync History · {titleMeta.name}</h1>
          </div>
          <Button variant="secondary" onClick={() => navigate('/')}>Back to Home</Button>
        </div>
      </div>

      <div className="max-w-4xl mx-auto p-6 space-y-6">
        <div className="rounded-md border border-border p-3 text-xs text-muted-foreground">
          Status: <span className={header?.status === 'active' ? 'text-emerald-500' : 'text-destructive'}>{header?.status || 'unknown'}</span>
          <span className="mx-2">·</span>
          Last sync: {header?.last_synced_at ? new Date(header.last_synced_at).toLocaleString() : '—'}
        </div>

        <div className="space-y-2">
          {runs.map(r => (
            <div key={r.id} className="flex items-center justify-between bg-muted/30 rounded-md px-3 py-2 text-sm">
              <div className="truncate">
                <span className="text-foreground mr-2">{r.started_at ? new Date(r.started_at).toLocaleString() : '—'}</span>
              </div>
              <div className="flex items-center gap-4">
                <span className={r.status === 'success' ? 'text-emerald-500' : r.status === 'partial' ? 'text-amber-500' : 'text-destructive'}>
                  {r.status || 'unknown'}
                </span>
                <span className="text-muted-foreground">{renderCount(r.stats)} transactions</span>
              </div>
            </div>
          ))}
          {runs.length === 0 && !loading && (
            <div className="text-xs text-muted-foreground">No runs found.</div>
          )}
        </div>

        <div className="flex items-center justify-between pt-2">
          <Button variant="secondary" onClick={() => goPage(Math.max(1, page - 1))} disabled={page <= 1 || loading}>Previous</Button>
          <div className="text-xs text-muted-foreground">Page {page}</div>
          <Button onClick={() => goPage(page + 1)} disabled={!hasMore || loading}>Next</Button>
        </div>
      </div>
    </div>
  );
}
