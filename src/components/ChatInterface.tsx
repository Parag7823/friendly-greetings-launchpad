import { MessageCircle, Send, Upload, Plug, FileSpreadsheet, Receipt, Database, Layers, Paperclip, Image as ImageIcon } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import { EnhancedFileUpload } from './EnhancedFileUpload';
import { InlineUploadZone } from './InlineUploadZone';
import { DataSourcesPanel } from './DataSourcesPanel';
import { MarkdownMessage } from './MarkdownMessage';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { StarBorder } from './ui/star-border';
import { useAuth } from './AuthProvider';
import { motion, AnimatePresence } from 'framer-motion';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/components/ui/use-toast';
import ConnectorConfigModal from './ConnectorConfigModal';
import { useSearchParams } from 'react-router-dom';
import { config } from '@/config';
import { useFastAPIProcessor } from './FastAPIProcessor';

interface ChatInterfaceProps {
  currentView?: string;
  onNavigate?: (view: string) => void;
}

export const ChatInterface = ({ currentView = 'chat', onNavigate }: ChatInterfaceProps) => {
  const { user } = useAuth();
  const { toast } = useToast();
  const { processFileWithFastAPI } = useFastAPIProcessor();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [searchParams, setSearchParams] = useSearchParams();
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Array<{ id: string; text: string; isUser: boolean; timestamp: Date }>>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [isNewChat, setIsNewChat] = useState(true);
  const [providers, setProviders] = useState<any[]>([]);
  const [loadingProviders, setLoadingProviders] = useState(false);
  const [connecting, setConnecting] = useState<string | null>(null);
  const [connections, setConnections] = useState<any[]>([]);
  const [uploadingFile, setUploadingFile] = useState(false);
  const [loadingConnections, setLoadingConnections] = useState(false);
  const [syncing, setSyncing] = useState<string | null>(null);
  const [configOpen, setConfigOpen] = useState(false);
  const [configConnId, setConfigConnId] = useState<string | null>(null);
  const [showDataSources, setShowDataSources] = useState(false);
  const [showInlineUpload, setShowInlineUpload] = useState(false);
  const [uploadingFiles, setUploadingFiles] = useState<File[]>([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);

  // Sample questions showcasing platform capabilities
  const sampleQuestions = [
    "What were my top expenses last quarter?",
    "Show me revenue trends for the past 6 months",
    "Which vendors am I spending the most with?",
    "Analyze my cash flow patterns",
    "Compare Q1 vs Q2 profitability"
  ];

  // Initialize chat ID from URL or create new one
  useEffect(() => {
    if (!user?.id) return;
    
    const chatIdFromUrl = searchParams.get('chat_id');
    if (chatIdFromUrl) {
      setCurrentChatId(chatIdFromUrl);
      setIsNewChat(false); // Existing chat
    } else {
      // Generate new chat ID and persist in URL
      const newChatId = `chat_${Date.now()}`;
      setCurrentChatId(newChatId);
      setIsNewChat(true); // New chat
      setSearchParams({ chat_id: newChatId }, { replace: true }); // Replace to avoid history pollution
    }
  }, [user?.id, searchParams]);

  // Load chat history on mount
  useEffect(() => {
    const loadChatHistory = async () => {
      if (!user?.id || !currentChatId) return;
      
      try {
        const { data, error } = await supabase
          .from('chat_messages')
          .select('*')
          .eq('user_id', user.id)
          .eq('chat_id', currentChatId)
          .order('created_at', { ascending: true });

        if (error) throw error;

        if (data && data.length > 0) {
          const loadedMessages = data.map((msg: any) => ({
            id: msg.id,
            text: msg.message,
            isUser: msg.role === 'user',
            timestamp: new Date(msg.created_at)
          }));
          setMessages(loadedMessages);
          setIsNewChat(false);
        }
      } catch (error) {
        console.error('Failed to load chat history:', error);
      }
    };

    loadChatHistory();
  }, [user?.id, currentChatId]);

  // Load connector providers when opening marketplace
  useEffect(() => {
    const loadProviders = async () => {
      if (currentView !== 'marketplace') return;
      try {
        setLoadingProviders(true);
        const { data: sessionData } = await supabase.auth.getSession();
        const sessionToken = sessionData?.session?.access_token;
        const resp = await fetch(`${config.apiUrl}/api/connectors/providers`, {
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
      const resp = await fetch(`${config.apiUrl}/api/connectors/initiate`, {
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
        toast({ title: 'Connection created', description: 'Session created but no URL returned. Check backend logs.', variant: 'destructive' });
      }
    } catch (e) {
      console.error('Connect failed', e);
      toast({ title: 'Connect failed', description: 'Unable to start the connector authorization flow.', variant: 'destructive' });
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
      const resp = await fetch(`${config.apiUrl}/api/connectors/user-connections`, {
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

  // Open modal from query param for persistence on refresh
  useEffect(() => {
    if (currentView !== 'marketplace') return;
    const q = searchParams.get('connection');
    if (q) {
      setConfigConnId(q);
      setConfigOpen(true);
    }
  }, [currentView, searchParams]);

  const openConfig = (connectionId: string) => {
    setConfigConnId(connectionId);
    setConfigOpen(true);
    const next = new URLSearchParams(searchParams);
    next.set('connection', connectionId);
    setSearchParams(next, { replace: true });
  };

  const closeConfig = (open: boolean) => {
    setConfigOpen(open);
    if (!open) {
      const next = new URLSearchParams(searchParams);
      next.delete('connection');
      setSearchParams(next, { replace: true });
      // Refresh connections to reflect any changes after closing
      fetchConnections();
    }
  };

  const handleSyncNow = async (integrationId: string | null | undefined, connectionId: string) => {
    if (!integrationId) return;
    try {
      setSyncing(connectionId);
      const { data: sessionData } = await supabase.auth.getSession();
      const sessionToken = sessionData?.session?.access_token;
      const resp = await fetch(`${config.apiUrl}/api/connectors/sync`, {
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
      toast({ title: 'Sync started', description: 'It may take a moment to appear in Recent Runs.' });
    } catch (e) {
      console.error('Sync failed', e);
      toast({ title: 'Sync failed', description: 'Unable to start sync. Please try again.', variant: 'destructive' });
    } finally {
      setSyncing(null);
    }
  };

  // Function to reset chat for new conversation
  const resetChat = () => {
    setMessages([]);
    const newChatId = `chat_${Date.now()}`;
    setCurrentChatId(newChatId);
    setSearchParams({ chat_id: newChatId });
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

  // Rotate sample questions
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentQuestionIndex((prev) => (prev + 1) % sampleQuestions.length);
    }, 3000); // Change every 3 seconds

    return () => clearInterval(interval);
  }, [sampleQuestions.length]);

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
            const { data: sessionData } = await supabase.auth.getSession();
            const sessionToken = sessionData?.session?.access_token;
            
            const titleResponse = await fetch(`${config.apiUrl}/generate-chat-title`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
              },
              body: JSON.stringify({
                message: currentMessage,
                user_id: user?.id || 'anonymous',
                session_token: sessionToken
              })
            });
            
            if (titleResponse.ok) {
              const titleData = await titleResponse.json();
              chatId = titleData.chat_id;
              setCurrentChatId(chatId);
              setIsNewChat(false);
              
              // Update URL with the chat ID from backend
              setSearchParams({ chat_id: chatId }, { replace: true });
              
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
        const { data: sessionData } = await supabase.auth.getSession();
        const sessionToken = sessionData?.session?.access_token;
        
        const response = await fetch(`${config.apiUrl}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
          },
          body: JSON.stringify({
            message: currentMessage,
            user_id: user?.id || 'anonymous',
            chat_id: chatId,
            session_token: sessionToken
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
          // Get error details from backend
          let errorDetail = 'Failed to get response from AI';
          try {
            const errorData = await response.json();
            errorDetail = errorData.detail || errorDetail;
          } catch {
            // If can't parse JSON, use status text
            errorDetail = `Server error (${response.status}): ${response.statusText}`;
          }
          throw new Error(errorDetail);
        }
      } catch (error) {
        console.error('Chat error:', error);
        
        // Show actual error message from backend
        const errorText = error instanceof Error ? error.message : 'Sorry, I encountered an error. Please try again.';
        
        const errorMessage = {
          id: `msg-${Date.now()}-error`,
          text: errorText,
          isUser: false,
          timestamp: new Date()
        };
        
        setMessages(prev => [...prev, errorMessage]);
      }
    }
  };

  const handleInlineFilesSelected = (files: File[]) => {
    // Open Data Sources panel immediately (non-blocking)
    setShowDataSources(true);
    
    // Trigger file upload event for Data Sources panel to handle
    // No chat message needed - user can see progress in Data Sources panel
    window.dispatchEvent(new CustomEvent('files-selected-for-upload', {
      detail: { files }
    }));
  };

  const handleFileUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const fileArray = Array.from(files);
    setUploadingFile(true);

    toast({
      title: 'Uploading Files',
      description: `Processing ${fileArray.length} file(s)...`
    });

    // Process each file
    for (const file of fileArray) {
      try {
        await processFileWithFastAPI(file);
        
        // Add a system message to chat
        const systemMessage = {
          id: `msg-${Date.now()}-${file.name}`,
          text: `✅ File uploaded: **${file.name}**`,
          isUser: false,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, systemMessage]);
      } catch (error) {
        console.error('File upload failed:', error);
        toast({
          title: 'Upload Failed',
          description: `Failed to upload ${file.name}`,
          variant: 'destructive'
        });
      }
    }

    setUploadingFile(false);
    
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

    // Open Data Sources panel to show uploaded files
    setShowDataSources(true);
  };

  const renderCurrentView = () => {
    switch (currentView) {
      case 'chat':
      default:
        return (
          <div className="h-full flex bg-background relative">
            {/* Main Chat Area - Responsive to Data Sources panel */}
            <motion.div 
              className="flex-1 flex flex-col min-w-0"
              animate={{ 
                marginRight: showDataSources ? '500px' : '0px' 
              }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            >
              {/* Data Sources Button - Fixed top right */}
              <div className="absolute top-4 right-4 z-10">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowDataSources(true)}
                  className="shadow-lg"
                >
                  <Layers className="w-4 h-4 mr-2" />
                  Data Sources
                </Button>
              </div>

              {/* Chat Messages Area */}
              <div className={`flex-1 overflow-y-auto p-4 transition-all duration-300 ${showDataSources ? 'pr-8' : ''}`}>
              {messages.length === 0 ? (
                <div className="h-full flex items-center justify-center">
                  <div className="max-w-2xl w-full space-y-6 px-4">
                    <div className="text-center">
                      <h1 className="text-2xl font-semibold text-foreground tracking-tight mb-2">
                        Finance Meets Intelligence
                      </h1>
                      <p className="text-muted-foreground text-base mb-6">
                        I can help you understand your financial data. Let's get started!
                      </p>
                    </div>
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
                        className={`max-w-[80%] rounded-md px-3 py-2 border ${
                          msg.isUser
                            ? 'bg-primary text-primary-foreground border-primary'
                            : 'bg-[#1a1a1a]/90 backdrop-blur-sm text-white border-white/10'
                        }`}
                      >
                        {msg.isUser ? (
                          <p className="text-xs">{msg.text}</p>
                        ) : (
                          <MarkdownMessage content={msg.text} />
                        )}
                        <p className="text-[10px] opacity-70 mt-1">
                          {msg.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
              {/* Chat Input Area - Rounded with animated questions */}
              <div className="border-t border-border/50 p-4 bg-background/80 backdrop-blur-md">
                <div className="max-w-4xl mx-auto">
                  <div className="relative rounded-[20px]">
                    {/* Animated gradient border line */}
                    <div
                      className="absolute inset-0 opacity-75 animate-border-slide pointer-events-none z-0"
                      style={{
                        background: `linear-gradient(90deg, 
                          transparent 0%, 
                          transparent 40%, 
                          hsl(var(--foreground)) 50%, 
                          transparent 60%, 
                          transparent 100%)`,
                        backgroundSize: '200% 100%',
                        animationDuration: '5s',
                        padding: '1px',
                        borderRadius: '20px',
                        WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
                        WebkitMaskComposite: 'xor',
                        maskComposite: 'exclude',
                      }}
                    />
                    
                    {/* Input wrapper with background */}
                    <div className="relative z-10 border rounded-[20px] bg-gradient-to-b from-background via-background to-muted/50 border-border/60 dark:from-zinc-900 dark:via-zinc-900 dark:to-zinc-800 dark:border-zinc-700 shadow-sm">
                      <div className="relative">
                        <input
                          type="text"
                          value={message}
                          onChange={(e) => setMessage(e.target.value)}
                          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              e.preventDefault();
                              handleSendMessage();
                            }
                          }}
                          placeholder={sampleQuestions[currentQuestionIndex]}
                          className="w-full bg-transparent border-none pl-14 pr-14 py-4 text-sm text-foreground placeholder-muted-foreground focus:outline-none focus:ring-0"
                          autoComplete="off"
                          spellCheck="false"
                        />
                        
                        {/* File Upload Button - Left side */}
                        <button
                          onClick={handleFileUploadClick}
                          disabled={uploadingFile}
                          className="absolute left-4 top-1/2 -translate-y-1/2 w-9 h-9 text-muted-foreground hover:text-foreground rounded-full flex items-center justify-center transition-all duration-200 hover:bg-muted/50 disabled:opacity-50 disabled:cursor-not-allowed"
                          title="Upload files or images"
                        >
                          {uploadingFile ? (
                            <Loader2 className="w-5 h-5 animate-spin" />
                          ) : (
                            <Paperclip className="w-5 h-5" />
                          )}
                        </button>
                        
                        {/* Hidden file input */}
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept=".xlsx,.xls,.csv,.pdf,image/*"
                          multiple
                          onChange={handleFileSelect}
                          className="hidden"
                        />
                        
                        {/* Send Button - Right side */}
                        <button
                          onClick={handleSendMessage}
                          disabled={!message.trim()}
                          className="absolute right-4 top-1/2 -translate-y-1/2 w-9 h-9 bg-primary text-primary-foreground rounded-full flex items-center justify-center transition-all duration-200 hover:bg-primary/90 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                        >
                          <Send className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Data Sources Panel - Positioned absolutely */}
            <DataSourcesPanel 
              isOpen={showDataSources} 
              onClose={() => setShowDataSources(false)} 
            />
          </div>
        );
    }
  };

  return renderCurrentView();
};