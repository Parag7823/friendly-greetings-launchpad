import { MessageCircle, Send, Upload, Plug, FileSpreadsheet, Receipt, Database } from 'lucide-react';
import { useState, useEffect } from 'react';
import { EnhancedFileUpload } from './EnhancedFileUpload';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { useAuth } from './AuthProvider';
import IntegrationCard from './IntegrationCard';
import { supabase } from '@/integrations/supabase/client';

interface ChatInterfaceProps {
  currentView?: string;
  onNavigate?: (view: string) => void;
}

export const ChatInterface = ({ currentView = 'chat', onNavigate }: ChatInterfaceProps) => {
  const { user } = useAuth();
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Array<{ id: string; text: string; isUser: boolean; timestamp: Date }>>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [isNewChat, setIsNewChat] = useState(true);
  const [providers, setProviders] = useState<any[]>([]);
  const [loadingProviders, setLoadingProviders] = useState(false);
  const [connecting, setConnecting] = useState<string | null>(null);
  const [connections, setConnections] = useState<any[]>([]);
  const [loadingConnections, setLoadingConnections] = useState(false);
  const [syncing, setSyncing] = useState<string | null>(null);

  // Load connector providers when opening marketplace
  useEffect(() => {
    const loadProviders = async () => {
      if (currentView !== 'marketplace') return;
      try {
        setLoadingProviders(true);
        const { data: sessionData } = await supabase.auth.getSession();
        const sessionToken = sessionData?.session?.access_token;
        const resp = await fetch('/api/connectors/providers', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: user?.id || '', session_token: sessionToken })
        });
        if (resp.ok) {
          const data = await resp.json();
          setProviders(data?.providers || []);
        }
      } catch (e) {
        console.error('Failed to load providers', e);
      } finally {
        setLoadingProviders(false);
      }
    };
    loadProviders();
  }, [currentView, user?.id]);

  const handleConnect = async (providerKey: string) => {
    try {
      setConnecting(providerKey);
      const { data: sessionData } = await supabase.auth.getSession();
      const sessionToken = sessionData?.session?.access_token;
      const resp = await fetch('/api/connectors/initiate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: providerKey, user_id: user?.id || '', session_token: sessionToken })
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err?.detail || 'Failed to initiate connector');
      }
      const data = await resp.json();
      const s = data?.connect_session || {};
      // Try common URL fields returned by Nango
      const url = s.connect_url || s.url || s.authorization_url || s.hosted_url || (s.data && (s.data.connect_url || s.data.url));
      if (url) {
        window.open(url as string, '_blank', 'noopener,noreferrer');
      } else {
        alert('Connect session created but no URL was provided. Please check backend logs.');
      }
    } catch (e) {
      console.error('Connect failed', e);
      alert('Unable to start the connector authorization flow.');
    } finally {
      setConnecting(null);
    }
  };

  // Helpers for marketplace rendering
  const providerSet = new Set((providers || []).map((p: any) => p.provider));
  const isAvailable = (slug: string) => providerSet.size === 0 || providerSet.has(slug);
  const brandIcon = (slug: string, colorHex: string, alt: string) => (
    <img
      src={`https://cdn.simpleicons.org/${slug}/${colorHex.replace('#', '')}`}
      alt={alt}
      className="w-8 h-8"
      loading="lazy"
      width={32}
      height={32}
    />
  );

  // Fetch user's existing connections for the marketplace
  const fetchConnections = async () => {
    try {
      setLoadingConnections(true);
      const { data: sessionData } = await supabase.auth.getSession();
      const sessionToken = sessionData?.session?.access_token;
      const resp = await fetch('/api/connectors/user-connections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: user?.id || '', session_token: sessionToken }),
      });
      if (resp.ok) {
        const data = await resp.json();
        setConnections(Array.isArray(data?.connections) ? data.connections : []);
      }
    } catch (e) {
      console.error('Failed to load user connections', e);
    } finally {
      setLoadingConnections(false);
    }
  };

  useEffect(() => {
    if (currentView !== 'marketplace') return;
    fetchConnections();
    const id = window.setInterval(fetchConnections, 15000);
    return () => window.clearInterval(id);
  }, [currentView, user?.id]);

  const handleSyncNow = async (integrationId: string | null | undefined, connectionId: string) => {
    if (!integrationId) return;
    try {
      setSyncing(connectionId);
      const { data: sessionData } = await supabase.auth.getSession();
      const sessionToken = sessionData?.session?.access_token;
      const resp = await fetch('/api/connectors/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user?.id || '',
          connection_id: connectionId,
          integration_id: integrationId,
          mode: 'incremental',
          session_token: sessionToken,
        }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err?.detail || 'Failed to trigger sync');
      }
      // Refresh connections to update last_synced_at/status
      await fetchConnections();
      alert('Sync started. It may take a moment to appear in Recent Runs.');
    } catch (e) {
      console.error('Sync failed', e);
      alert('Unable to start sync. Please try again.');
    } finally {
      setSyncing(null);
    }
  };

  // Function to reset chat for new conversation
  const resetChat = () => {
    setMessages([]);
    setCurrentChatId(null);
    setIsNewChat(true);
  };

  // Listen for new chat events
  useEffect(() => {
    const handleNewChat = () => {
      resetChat();
    };

    window.addEventListener('new-chat-requested', handleNewChat);
    return () => window.removeEventListener('new-chat-requested', handleNewChat);
  }, []);

  const handleSendMessage = async () => {
    if (message.trim()) {
      const userMessage = {
        id: `msg-${Date.now()}`,
        text: message,
        isUser: true,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, userMessage]);
      const currentMessage = message;
      setMessage('');
      
      try {
        let chatId = currentChatId;
        
        // If this is a new chat, generate a title and create chat entry
        if (isNewChat) {
          try {
            const titleResponse = await fetch('/generate-chat-title', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                message: currentMessage,
                user_id: user?.id || 'anonymous'
              })
            });
            
            if (titleResponse.ok) {
              const titleData = await titleResponse.json();
              chatId = titleData.chat_id;
              setCurrentChatId(chatId);
              setIsNewChat(false);
              
              // Notify parent component about new chat
              if (onNavigate) {
                // Trigger a custom event to update sidebar
                window.dispatchEvent(new CustomEvent('new-chat-created', {
                  detail: {
                    chatId: chatId,
                    title: titleData.title,
                    timestamp: new Date()
                  }
                }));
              }
            }
          } catch (titleError) {
            console.error('Title generation error:', titleError);
            // Continue with chat even if title generation fails
            chatId = `chat-${Date.now()}`;
            setCurrentChatId(chatId);
            setIsNewChat(false);
          }
        }
        
        // Send message to backend
        const response = await fetch('/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: currentMessage,
            user_id: user?.id || 'anonymous',
            chat_id: chatId
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          
          const aiMessage = {
            id: `msg-${Date.now()}-ai`,
            text: data.response,
            isUser: false,
            timestamp: new Date(data.timestamp)
          };
          
          setMessages(prev => [...prev, aiMessage]);
        } else {
          throw new Error('Failed to get response from AI');
        }
      } catch (error) {
        console.error('Chat error:', error);
        
        const errorMessage = {
          id: `msg-${Date.now()}-error`,
          text: 'Sorry, I encountered an error. Please try again.',
          isUser: false,
          timestamp: new Date()
        };
        
        setMessages(prev => [...prev, errorMessage]);
      }
    }
  };

  const renderCurrentView = () => {
    switch (currentView) {
      case 'upload':
        return (
          <div className="h-full overflow-y-auto">
            <div className="p-4">
              <div className="mb-4">
                <h1 className="text-xl font-semibold text-foreground mb-1">Upload Financial Documents</h1>
                <p className="text-sm text-muted-foreground">Upload your Excel, CSV, or other financial files for AI analysis</p>
              </div>
              <EnhancedFileUpload />
            </div>
          </div>
        );
      
      case 'marketplace':
        return (
          <div className="h-full overflow-y-auto">
            <div className="px-6 pt-6">
              <h1 className="text-2xl font-semibold text-primary tracking-tight">
                Autonomous, zero-effort ingestion from anywhere.
              </h1>
              <p className="text-sm text-muted-foreground mt-1">Connect cloud storage, email, and accounting platforms in seconds.</p>
            </div>

            <div className="p-6 space-y-8">
              <div className="finley-card rounded-md border border-border bg-card text-card-foreground p-3 text-xs text-muted-foreground">
                Click Connect. A secure window opens to authorize the provider. After completing authorization, sync starts automatically (handled by the backend webhook).
              </div>
              {/* My Connections */}
              <section>
                <h2 className="text-sm font-medium text-muted-foreground mb-3">My Connections</h2>
                {loadingConnections && (
                  <div className="text-xs text-muted-foreground">Loading your connections…</div>
                )}
                {!loadingConnections && connections.length === 0 && (
                  <div className="text-xs text-muted-foreground">No connections yet — connect a provider below to get started.</div>
                )}
                {connections.length > 0 && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {connections.map((c) => {
                      const integ = (c.integration_id || '').toString();
                      const map: Record<string, { slug: string; color: string; name: string }> = {
                        'google-mail': { slug: 'gmail', color: 'EA4335', name: 'Gmail' },
                        'zoho-mail': { slug: 'zoho', color: 'C8202F', name: 'Zoho Mail' },
                        'dropbox': { slug: 'dropbox', color: '0061FF', name: 'Dropbox' },
                        'google-drive': { slug: 'googledrive', color: '1A73E8', name: 'Google Drive' },
                        'quickbooks': { slug: 'intuitquickbooks', color: '2CA01C', name: 'QuickBooks' },
                        'quickbooks-sandbox': { slug: 'intuitquickbooks', color: '2CA01C', name: 'QuickBooks (Sandbox)' },
                        'xero': { slug: 'xero', color: '13B5EA', name: 'Xero' },
                      };
                      const meta = map[integ] || { slug: 'cloud', color: '6B7280', name: integ || 'Connection' };
                      const last = c.last_synced_at ? new Date(c.last_synced_at).toLocaleString() : 'Never';
                      return (
                        <IntegrationCard
                          key={c.connection_id}
                          icon={brandIcon(meta.slug, meta.color, meta.name)}
                          title={meta.name}
                          description={`Status: ${c.status || 'unknown'} • Last synced: ${last}`}
                          actionLabel={syncing === c.connection_id ? 'Syncing…' : 'Sync Now'}
                          onAction={() => handleSyncNow(integ, c.connection_id)}
                          disabled={!!(syncing === c.connection_id)}
                        />
                      );
                    })}
                  </div>
                )}
              </section>
              {/* Cloud & Storage */}
              <section>
                <h2 className="text-sm font-medium text-muted-foreground mb-3">Cloud & Storage</h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  <IntegrationCard
                    icon={brandIcon('googledrive', '1A73E8', 'Google Drive')}
                    title="Google Drive"
                    description="Ingest spreadsheets and documents directly from your Drive."
                    actionLabel={connecting === 'google-drive' ? 'Connecting…' : (loadingProviders ? 'Loading…' : 'Connect')}
                    onAction={() => handleConnect('google-drive')}
                    disabled={loadingProviders || !isAvailable('google-drive') || connecting === 'google-drive'}
                  />
                  <IntegrationCard
                    icon={brandIcon('dropbox', '0061FF', 'Dropbox')}
                    title="Dropbox"
                    description="Sync shared folders and financial files at scale."
                    actionLabel={connecting === 'dropbox' ? 'Connecting…' : (loadingProviders ? 'Loading…' : 'Connect')}
                    onAction={() => handleConnect('dropbox')}
                    disabled={loadingProviders || !isAvailable('dropbox') || connecting === 'dropbox'}
                  />
                  <IntegrationCard
                    icon={brandIcon('gmail', 'EA4335', 'Gmail')}
                    title="Gmail"
                    description="Autofetch statements and invoices from email attachments."
                    actionLabel={connecting === 'google-mail' ? 'Connecting…' : (loadingProviders ? 'Loading…' : 'Connect')}
                    onAction={() => handleConnect('google-mail')}
                    disabled={loadingProviders || !isAvailable('google-mail') || connecting === 'google-mail'}
                  />
                  <IntegrationCard
                    icon={brandIcon('zoho', 'C8202F', 'Zoho Mail')}
                    title="Zoho Mail"
                    description="Pull financial documents from Zoho Mail attachments."
                    actionLabel={connecting === 'zoho-mail' ? 'Connecting…' : (loadingProviders ? 'Loading…' : 'Connect')}
                    onAction={() => handleConnect('zoho-mail')}
                    disabled={loadingProviders || !isAvailable('zoho-mail') || connecting === 'zoho-mail'}
                  />
                </div>
              </section>

              {/* Accounting Platforms */}
              <section>
                <h2 className="text-sm font-medium text-muted-foreground mb-3">Accounting Platforms</h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  <IntegrationCard
                    icon={brandIcon('intuitquickbooks', '2CA01C', 'QuickBooks')}
                    title="QuickBooks (Sandbox)"
                    description="Historic and incremental sync of accounting data."
                    actionLabel={connecting === 'quickbooks-sandbox' ? 'Connecting…' : (loadingProviders ? 'Loading…' : 'Connect')}
                    onAction={() => handleConnect('quickbooks-sandbox')}
                    disabled={loadingProviders || !isAvailable('quickbooks-sandbox') || connecting === 'quickbooks-sandbox'}
                  />
                  <IntegrationCard
                    icon={brandIcon('xero', '13B5EA', 'Xero')}
                    title="Xero"
                    description="Sync contacts, invoices, and payments securely."
                    actionLabel={connecting === 'xero' ? 'Connecting…' : (loadingProviders ? 'Loading…' : 'Connect')}
                    onAction={() => handleConnect('xero')}
                    disabled={loadingProviders || !isAvailable('xero') || connecting === 'xero'}
                  />
                </div>
              </section>

              {/* Future Categories */}
              <section>
                <h2 className="text-sm font-medium text-muted-foreground mb-3">Coming Next</h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  <IntegrationCard
                    icon={<Database className="w-8 h-8 text-muted-foreground" aria-hidden />}
                    title="Banking APIs"
                    description="Direct bank feeds and transaction sync."
                    statusLabel="Coming Soon"
                    disabled
                  />
                </div>
              </section>
            </div>
          </div>
        );
      
      case 'chat':
      default:
  return (
    <div className="finley-chat flex flex-col h-full">
      {/* Chat Messages Area */}
            <div className="flex-1 overflow-y-auto p-4">
              {messages.length === 0 ? (
                <div className="h-full flex items-center justify-center">
        <div className="text-center">
                    <h1 className="text-2xl font-semibold text-foreground tracking-tight mb-2">
                      Finance Meets Intelligence
          </h1>
                    <p className="text-muted-foreground text-base">
                      Ask me anything about your financial data
                    </p>
                  </div>
                </div>
              ) : (
                <div className="max-w-5xl mx-auto space-y-3">
                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[80%] rounded-xl px-3 py-2 ${
                          msg.isUser
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted text-foreground'
                        }`}
                      >
                        <p className="text-xs">{msg.text}</p>
                        <p className="text-[10px] opacity-70 mt-1">
                          {msg.timestamp.toLocaleTimeString()}
                        </p>
        </div>
                    </div>
                  ))}
                </div>
              )}
      </div>
      
      {/* Chat Input Area - Fixed at bottom */}
      <div className="border-t border-border p-3 bg-background">
        <div className="w-full">
          <div className="relative">
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                    placeholder="Ask anything about your financial data..."
                    className="w-full bg-card border border-border rounded-lg px-3 py-2 pr-10 text-sm text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring transition-all duration-200"
            />
            
            <button
              onClick={handleSendMessage}
              disabled={!message.trim()}
                    className="absolute right-2 top-1/2 -translate-y-1/2 w-7 h-7 bg-primary text-primary-foreground rounded-md flex items-center justify-center transition-all duration-200 hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
    }
  };

  return renderCurrentView();
};