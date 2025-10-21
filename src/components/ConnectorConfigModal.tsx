import { useEffect, useMemo, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, ChevronUp, ExternalLink } from "lucide-react";
import { Link } from "react-router-dom";
import { config } from "@/config";

interface ConnectorConfigModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  connectionId: string | null;
  userId: string | undefined;
}

interface SyncRun {
  id: string;
  type?: string;
  status?: string;
  started_at?: string;
  finished_at?: string | null;
  stats?: Record<string, any> | null;
  error?: any;
}

export default function ConnectorConfigModal({ open, onOpenChange, connectionId, userId }: ConnectorConfigModalProps) {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [details, setDetails] = useState<any | null>(null);
  const [runs, setRuns] = useState<SyncRun[]>([]);
  const [expanded, setExpanded] = useState(false);
  const [savingFreq, setSavingFreq] = useState(false);
  const [freqValue, setFreqValue] = useState<string>("");
  const [advOpen, setAdvOpen] = useState(false);
  const [advSaving, setAdvSaving] = useState(false);
  const [realmId, setRealmId] = useState<string>("");
  const [tenantId, setTenantId] = useState<string>("");

  const integrationId = details?.connection?.integration_id as string | undefined;
  const titleName = useMemo(() => {
    const uc = details?.connection || {};
    // Prefer metadata display name, else integration id
    return (
      uc?.metadata?.display_name || uc?.metadata?.company_name || integrationId || "Connection"
    );
  }, [details, integrationId]);

  const fetchStatus = async () => {
    if (!open || !connectionId || !userId) return;
    try {
      setLoading(true);
      const { data: sessionData } = await supabase.auth.getSession();
      const sessionToken = sessionData?.session?.access_token;
      const params = new URLSearchParams({ connection_id: connectionId, user_id: userId });
      if (sessionToken) params.set("session_token", sessionToken);
      const resp = await fetch(`${config.apiUrl}/api/connectors/status?${params.toString()}`, {
        headers: {
          ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
        }
      });
      if (!resp.ok) throw new Error("Failed to load connection status");
      const data = await resp.json();
      setDetails(data || null);
      setRuns(Array.isArray(data?.recent_runs) ? data.recent_runs : []);
      const uc = data?.connection || {};
      const m = uc?.metadata || {};
      const fmin = typeof uc?.sync_frequency_minutes === "number" ? uc.sync_frequency_minutes : null;
      setFreqValue(fmin === 0 ? "0" : fmin === 60 ? "60" : fmin === 1440 ? "1440" : fmin ? String(fmin) : "");
      setRealmId(m?.realmId || m?.realm_id || "");
      setTenantId(m?.tenantId || m?.tenant_id || "");
    } catch (e: any) {
      console.error("Config load failed", e);
      toast({ title: "Failed to load details", description: e?.message || "Please try again.", variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, connectionId, userId]);

  const reconnect = async () => {
    if (!integrationId) return;
    try {
      setReconnecting(true);
      const { data: sessionData } = await supabase.auth.getSession();
      const sessionToken = sessionData?.session?.access_token;
      // Reuse initiate endpoint to start reauth for same integration
      const resp = await fetch(`${config.apiUrl}/api/connectors/initiate`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
        },
        body: JSON.stringify({ provider: integrationId, user_id: userId || "", session_token: sessionToken })
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err?.detail || "Failed to start reconnect");
      }
      const data = await resp.json();
      const s = data?.connect_session || {};
      const url = s.connect_url || s.url || s.authorization_url || s.hosted_url || (s.data && (s.data.connect_url || s.data.url));
      if (url) {
        window.open(url as string, "_blank", "noopener,noreferrer");
        toast({ title: "Reconnect", description: "Opened provider window to re-authorize." });
      } else {
        toast({ title: "Reconnect", description: "Session created but no URL returned.", variant: "destructive" });
      }
    } catch (e: any) {
      console.error("Reconnect failed", e);
      toast({ title: "Reconnect failed", description: e?.message || "Please try again.", variant: "destructive" });
    } finally {
      setReconnecting(false);
    }
  };

  const uc = details?.connection || {};
  const email = uc?.metadata?.email || uc?.metadata?.connected_email || uc?.provider_account_id || "";
  const company = uc?.metadata?.company_name || uc?.metadata?.tenant_name || "";
  const freqMin = typeof uc?.sync_frequency_minutes === "number" ? uc.sync_frequency_minutes : null;
  const frequencyLabel = freqMin === 0 ? "real-time" : freqMin === 60 ? "hourly" : freqMin === 1440 ? "daily" : (freqMin ? `${freqMin} min` : "unknown");

  const renderCount = (stats: Record<string, any> | null | undefined) => {
    if (!stats) return "-";
    const keys = ["transactions", "records", "items", "count", "synced", "success_count", "total"];
    for (const k of keys) {
      if (typeof stats[k] === "number") return String(stats[k]);
    }
    // Sum numeric values as fallback
    try {
      const sum = Object.values(stats).filter(v => typeof v === "number").reduce((a: number, b: number) => a + (b as number), 0);
      return sum ? String(sum) : "-";
    } catch {
      return "-";
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-3xl p-0 border-0 bg-transparent">
        <div className="p-[1.25px] rounded-2xl bg-gradient-to-r from-sky-500 via-fuchsia-500 to-rose-500">
          <div className="rounded-2xl bg-background" aria-busy={loading}>
            <div className="p-5 border-b border-border">
              <DialogHeader>
                <DialogTitle className="flex items-center justify-between">
                  <span>Configure: {titleName}</span>
                </DialogTitle>
              </DialogHeader>
            </div>

            <div className="p-5 space-y-6">
              {/* Connection Details */}
              <section>
                <h3 className="text-sm font-medium text-muted-foreground mb-2">Connection Details</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
                  <div className="flex items-center justify-between bg-muted/30 rounded-md px-3 py-2">
                    <span className="text-muted-foreground">Status</span>
                    <span className={uc?.status === 'active' ? 'text-emerald-500' : 'text-destructive'}>
                      {uc?.status === 'active' ? 'Connected' : 'Disconnected'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between bg-muted/30 rounded-md px-3 py-2">
                    <span className="text-muted-foreground">Connected as</span>
                    <span className="truncate max-w-[60%] text-foreground" title={email}>{email || '—'}</span>
                  </div>
                  <div className="flex items-center justify-between bg-muted/30 rounded-md px-3 py-2">
                    <span className="text-muted-foreground">Company</span>
                    <span className="truncate max-w-[60%] text-foreground" title={company}>{company || '—'}</span>
                  </div>
                  <div className="flex items-center justify-between bg-muted/30 rounded-md px-3 py-2">
                    <span className="text-muted-foreground">Last sync</span>
                    <span className="text-foreground">{uc?.last_synced_at ? new Date(uc.last_synced_at).toLocaleString() : '—'}</span>
                  </div>
                  <div className="flex items-center justify-between bg-muted/30 rounded-md px-3 py-2 gap-3">
                    <span className="text-muted-foreground">Frequency</span>
                    <div className="flex items-center gap-2">
                      <Select value={freqValue} onValueChange={setFreqValue}>
                        <SelectTrigger className="w-40" aria-label="Sync frequency"><SelectValue placeholder={frequencyLabel} /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0">Real-time</SelectItem>
                          <SelectItem value="60">Hourly</SelectItem>
                          <SelectItem value="1440">Daily</SelectItem>
                        </SelectContent>
                      </Select>
                      <Button
                        size="sm"
                        onClick={async () => {
                          if (!connectionId || !userId) return;
                          try {
                            setSavingFreq(true);
                            const { data: sessionData } = await supabase.auth.getSession();
                            const sessionToken = sessionData?.session?.access_token;
                            const resp = await fetch(`${config.apiUrl}/api/connectors/frequency`, {
                              method: 'POST',
                              headers: { 
                                'Content-Type': 'application/json',
                                ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
                              },
                              body: JSON.stringify({ user_id: userId, connection_id: connectionId, minutes: Number(freqValue || 60), session_token: sessionToken })
                            });
                            if (!resp.ok) throw new Error('Failed to update frequency');
                            toast({ title: 'Frequency updated', description: 'New schedule saved.' });
                            await fetchStatus();
                          } catch (e: any) {
                            toast({ title: 'Update failed', description: e?.message || 'Please try again.', variant: 'destructive' });
                          } finally {
                            setSavingFreq(false);
                          }
                        }}
                        disabled={savingFreq || !freqValue}
                        aria-label="Save frequency"
                      >
                        {savingFreq ? 'Saving…' : 'Save'}
                      </Button>
                    </div>
                  </div>
                </div>
                <div className="mt-3 flex gap-2">
                  <Button onClick={reconnect} disabled={!integrationId || reconnecting} aria-label="Reconnect provider">
                    {reconnecting ? 'Reconnecting…' : 'Reconnect'}
                  </Button>
                  {/* Placeholder for Test Connection (not required now) */}
                </div>
              </section>

              {/* Advanced */}
              <section>
                <Collapsible open={advOpen} onOpenChange={setAdvOpen}>
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-medium text-muted-foreground">Advanced</h3>
                    <CollapsibleTrigger className="text-xs text-muted-foreground flex items-center gap-1">
                      {advOpen ? (<><ChevronUp className="w-4 h-4" />Hide</>) : (<><ChevronDown className="w-4 h-4" />Show</>)}
                    </CollapsibleTrigger>
                  </div>
                  <CollapsibleContent>
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
                      {/* QuickBooks realmId */}
                      <div className="bg-muted/30 rounded-md px-3 py-2">
                        <label htmlFor="realmId" className="text-muted-foreground mb-1 block">QuickBooks Realm ID</label>
                        <input id="realmId" aria-label="QuickBooks Realm ID" className="w-full bg-background border border-border rounded-md px-2 py-1 text-sm" value={realmId} onChange={(e) => setRealmId(e.target.value)} placeholder="realmId" />
                      </div>
                      {/* Xero tenantId */}
                      <div className="bg-muted/30 rounded-md px-3 py-2">
                        <label htmlFor="tenantId" className="text-muted-foreground mb-1 block">Xero Tenant ID</label>
                        <input id="tenantId" aria-label="Xero Tenant ID" className="w-full bg-background border border-border rounded-md px-2 py-1 text-sm" value={tenantId} onChange={(e) => setTenantId(e.target.value)} placeholder="tenantId" />
                      </div>
                    </div>
                    <div className="mt-3">
                      <Button
                        onClick={async () => {
                          if (!connectionId || !userId) return;
                          try {
                            setAdvSaving(true);
                            const { data: sessionData } = await supabase.auth.getSession();
                            const sessionToken = sessionData?.session?.access_token;
                            const updates: any = {};
                            if (realmId !== undefined) updates.realmId = realmId;
                            if (tenantId !== undefined) updates.tenantId = tenantId;
                            const resp = await fetch(`${config.apiUrl}/api/connectors/metadata`, {
                              method: 'POST',
                              headers: { 
                                'Content-Type': 'application/json',
                                ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
                              },
                              body: JSON.stringify({ user_id: userId, connection_id: connectionId, updates, session_token: sessionToken })
                            });
                            if (!resp.ok) throw new Error('Failed to save metadata');
                            toast({ title: 'Saved', description: 'Advanced settings updated.' });
                            await fetchStatus();
                          } catch (e: any) {
                            toast({ title: 'Save failed', description: e?.message || 'Please try again.', variant: 'destructive' });
                          } finally {
                            setAdvSaving(false);
                          }
                        }}
                        disabled={advSaving}
                        aria-label="Save advanced settings"
                      >
                        {advSaving ? 'Saving…' : 'Save Advanced Settings'}
                      </Button>
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </section>

              {/* Sync History */}
              <section>
                <h3 className="text-sm font-medium text-muted-foreground mb-2">Sync History</h3>
                <div className="space-y-2">
                  {(expanded ? runs : runs.slice(0, 10)).map((r) => (
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
                  {runs.length === 0 && (
                    <div className="text-xs text-muted-foreground">No runs yet.</div>
                  )}
                </div>
                <div className="mt-3">
                  <div className="flex items-center gap-2">
                    {runs.length > 10 && (
                      <Button variant="secondary" onClick={() => setExpanded((v) => !v)} aria-label="Toggle history">
                        {expanded ? 'Show less' : 'Expand' }
                      </Button>
                    )}
                    {connectionId && (
                      <Link to={`/connectors/${connectionId}/history`} className="inline-flex items-center text-sm underline" aria-label="View full history">
                        View full history <ExternalLink className="w-3.5 h-3.5 ml-1" />
                      </Link>
                    )}
                  </div>
                </div>
              </section>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
