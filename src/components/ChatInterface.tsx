import { Send, FileSpreadsheet, Paperclip, Loader2, X } from 'lucide-react';
import { useState, useEffect, useRef, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { MarkdownMessage } from './MarkdownMessage';
import { ThinkingShimmer } from './ThinkingShimmer';
import { useAuth } from './AuthProvider';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';
import { useSearchParams } from 'react-router-dom';
import { config } from '@/config';
import { useFastAPIProcessor } from './FastAPIProcessor';
import { ChatInputMicroInteractions } from './ChatInputMicroInteractions';
import { useFileUpload } from '@/hooks/useFileUpload';
import { getSessionToken } from '@/utils/authHelpers';
import { useStandardToasts } from '@/hooks/useStandardToasts';

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
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [pastedImages, setPastedImages] = useState<File[]>([]);
  const [isThinking, setIsThinking] = useState(false);

  // IMPROVEMENT: Cleanup Object URLs to prevent memory leaks
  useEffect(() => {
    // Create URLs for preview
    const urls = pastedImages.map(f => URL.createObjectURL(f));

    // Cleanup function to revoke URLs when component unmounts or files change
    return () => {
      urls.forEach(url => URL.revokeObjectURL(url));
    };
  }, [pastedImages]);

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

  // CRITICAL FIX: Load chat history on mount or when chat_id changes
  useEffect(() => {
    const loadChatHistory = async () => {
      if (!user?.id || !currentChatId) return;

      try {
        // Load chat messages from backend
        const sessionToken = await getSessionToken();
        const { data, error } = await supabase
          .from('chat_messages')
          .select('*')
          .eq('user_id', user.id)
          .eq('chat_id', currentChatId)
          .order('created_at', { ascending: true });

        if (error) {
          console.error('Failed to load chat history:', error);
          return;
        }

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
        const sessionToken = await getSessionToken();
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

  // PERF FIX #2: Wrap handleConnect in useCallback to prevent unnecessary re-renders
  const handleConnect = useCallback(async (providerKey: string) => {
    try {
      setConnecting(providerKey);
      const sessionToken = await getSessionToken();
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
  }, [user?.id, toast]);

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
      const sessionToken = await getSessionToken();
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
    // BOTTLENECK #1 FIX: Polling interval optimized
    // CRITICAL FIX: Reduce polling frequency to prevent database overload
    // With 1000 concurrent users, 15-second polling = 4,000 req/min
    // Changed to 60 seconds = 1,000 req/min (4x reduction)
    // Connections don't change frequently, so 1-minute polling is sufficient
    // 
    // FUTURE OPTIMIZATION: Replace with WebSocket events
    // Backend could emit 'connection_updated' events when connections change
    // This would eliminate polling entirely and provide real-time updates
    const id = window.setInterval(fetchConnections, 60000);
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

  // PERF FIX #3: Wrap handleSyncNow in useCallback to prevent unnecessary re-renders
  const handleSyncNow = useCallback(async (integrationId: string | null | undefined, connectionId: string) => {
    if (!integrationId) return;
    try {
      setSyncing(connectionId);
      const sessionToken = await getSessionToken();
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
  }, [user?.id, toast]);

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
    }, 5000); // Change every 5 seconds

    return () => clearInterval(interval);
  }, [sampleQuestions.length]);

  // PERF FIX #1: Wrap handleSendMessage in useCallback to prevent unnecessary re-renders
  // This prevents child components from re-rendering when parent state changes
  const handleSendMessage = useCallback(async () => {
    if (message.trim() || pastedImages.length > 0) {
      // CRITICAL FIX: Process pasted images before sending message
      if (pastedImages.length > 0) {
        // Show user message with file attachments
        const fileNames = pastedImages.map(f => f.name).join(', ');
        const attachmentMessage = {
          id: `msg-${Date.now()}-attachments`,
          text: `📎 Uploading ${pastedImages.length} file(s): ${fileNames}`,
          isUser: true,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, attachmentMessage]);

        // Upload pasted images immediately
        await processFiles(pastedImages);
        setPastedImages([]); // Clear pasted images after processing
      }

      // Only send text message if there's text content
      if (!message.trim()) {
        return; // Images are already being processed, no need to send empty message
      }

      const userMessage = {
        id: `msg-${Date.now()}`,
        text: message,
        isUser: true,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, userMessage]);
      const currentMessage = message;
      setMessage(''); // Clear input immediately
      setIsThinking(true);

      try {
        let chatId = currentChatId;

        // If this is a new chat, generate a title and create chat entry
        if (isNewChat) {
          try {
            const sessionToken = await getSessionToken();

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
              const generatedTitle = titleData.title || 'New Chat';
              
              setCurrentChatId(chatId);
              setIsNewChat(false);

              // Update URL with the chat ID from backend
              setSearchParams({ chat_id: chatId }, { replace: true });

              // Trigger a custom event to update sidebar with generated title
              window.dispatchEvent(new CustomEvent('new-chat-created', {
                detail: {
                  chatId: chatId,
                  title: generatedTitle,
                  timestamp: new Date()
                }
              }));

              console.log('✅ Chat title generated:', generatedTitle);
            } else {
              console.warn('Title generation response not ok:', titleResponse.status);
              // Continue with chat even if title generation fails
              chatId = `chat_${Date.now()}`;
              setCurrentChatId(chatId);
              setIsNewChat(false);
            }
          } catch (titleError) {
            console.error('Title generation error:', titleError);
            // Continue with chat even if title generation fails
            chatId = `chat_${Date.now()}`;
            setCurrentChatId(chatId);
            setIsNewChat(false);
          }
        }

        // Create placeholder AI message for streaming
        const aiMessageId = `msg-${Date.now()}-ai`;
        const aiMessage = {
          id: aiMessageId,
          text: '',
          isUser: false,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, aiMessage]);

        // Send message to backend with streaming
        const sessionToken = await getSessionToken();
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

        if (!response.ok) {
          throw new Error(`Server error (${response.status}): ${response.statusText}`);
        }

        if (!response.body) {
          throw new Error('Response body is not available');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              console.log('✅ Stream ended');
              break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const jsonStr = line.slice(6); // Remove 'data: ' prefix
                  if (!jsonStr.trim()) continue;
                  
                  const chunk = JSON.parse(jsonStr);
                  console.log('📨 Received chunk:', chunk.type);

                  if (chunk.error) {
                    throw new Error(chunk.error);
                  }

                  if (chunk.type === 'thinking') {
                    // Show thinking indicator
                    setMessages(prev =>
                      prev.map(msg =>
                        msg.id === aiMessageId
                          ? { ...msg, text: 'thinking' } // Special marker for thinking state
                          : msg
                      )
                    );
                  } else if (chunk.type === 'chunk') {
                    // Update message with streamed content
                    setMessages(prev =>
                      prev.map(msg =>
                        msg.id === aiMessageId
                          ? { ...msg, text: chunk.content }
                          : msg
                      )
                    );
                  } else if (chunk.type === 'complete') {
                    // Stream complete, update with metadata
                    setMessages(prev =>
                      prev.map(msg =>
                        msg.id === aiMessageId
                          ? {
                              ...msg,
                              timestamp: new Date(chunk.timestamp)
                            }
                          : msg
                      )
                    );
                    console.log('✅ Chat response complete');
                  }
                } catch (parseError) {
                  console.error('Error parsing SSE chunk:', parseError, 'Line:', line);
                }
              }
            }
          }
        } finally {
          reader.cancel();
        }

        setIsThinking(false);
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
        setIsThinking(false);
      } finally {
        // Scroll to bottom after message sent
        if (messagesEndRef.current) {
          messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
      }
    }
  }, [message, pastedImages, currentChatId, isNewChat, user?.id, onNavigate, setSearchParams]);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setPastedImages(prev => [...prev, ...acceptedFiles]);
      toast({
        title: 'Files Attached',
        description: `${acceptedFiles.length} file(s) ready to send`
      });
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    noClick: true,
    noKeyboard: true,
    accept: {
      'image/*': [],
      'application/pdf': ['.pdf'],
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    }
  });

  const handleInlineFilesSelected = (files: File[]) => {
    // Trigger file upload event for Data Sources panel to handle
    // No chat message needed - user can see progress in Data Sources panel
    window.dispatchEvent(new CustomEvent('files-selected-for-upload', {
      detail: { files }
    }));
  };

  const processFiles = async (files: File[]) => {
    if (files.length === 0) return;

    setUploadingFile(true);

    toast({
      title: 'Uploading Files',
      description: `Processing ${files.length} file(s) in parallel...`
    });

    // Process all files in parallel using Promise.allSettled
    const uploadPromises = files.map(async (file) => {
      try {
        await processFileWithFastAPI(file);

        // Add a system message to chat
        const systemMessage = {
          id: `msg-${Date.now()}-${Math.random()}-${file.name}`,
          text: `✅ File uploaded: **${file.name}**`,
          isUser: false,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, systemMessage]);

        return { status: 'success', file: file.name };
      } catch (error) {
        console.error('File upload failed:', error);
        toast({
          title: 'Upload Failed',
          description: `Failed to upload ${file.name}`,
          variant: 'destructive'
        });

        return { status: 'failed', file: file.name, error };
      }
    });

    // Wait for all uploads to complete
    const results = await Promise.allSettled(uploadPromises);

    // Count successes and failures
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;

    // Show final toast
    toast({
      title: 'Upload Complete',
      description: `${successful} file(s) uploaded successfully${failed > 0 ? `, ${failed} failed` : ''}`,
      variant: failed > 0 ? 'destructive' : 'default'
    });

    setUploadingFile(false);
  };



  const handlePaste = async (event: React.ClipboardEvent) => {
    const items = event.clipboardData?.items;
    if (!items) return;

    const files: File[] = [];

    for (let i = 0; i < items.length; i++) {
      const item = items[i];

      // Handle images from clipboard
      if (item.type.indexOf('image') !== -1) {
        const file = item.getAsFile();
        if (file) {
          // Rename with timestamp for clarity
          const renamedFile = new File([file], `pasted-image-${Date.now()}.png`, { type: file.type });
          files.push(renamedFile);
        }
      }
      // Handle files from clipboard
      else if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file) {
          files.push(file);
        }
      }
    }

    if (files.length > 0) {
      event.preventDefault();
      // Store pasted images for preview instead of auto-uploading
      setPastedImages(prev => [...prev, ...files]);
      toast({
        title: 'Image Attached',
        description: `${files.length} image(s) ready to send`
      });
    }
  };



  const renderCurrentView = () => {
    switch (currentView) {
      case 'chat':
      default:
        return (
          <div className="h-full flex flex-col finley-dynamic-bg">
            {/* Chat Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 pt-2">
              {messages.length === 0 ? (
                <div className="flex items-center justify-center min-h-full">
                  <div className="max-w-2xl w-full space-y-6 px-4">
                    <div className="text-center">
                      <h1 className="text-lg font-semibold text-foreground tracking-tight mb-2">
                        Finance Meets Intelligence
                      </h1>
                      <p className="text-muted-foreground text-sm mb-6">
                        I can help you understand your financial data. Let's get started!
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="max-w-5xl mx-auto space-y-4 pb-4">
                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}
                    >
                      {msg.isUser ? (
                        // User message: Keep in rectangular box
                        <div
                          className="max-w-[80%] rounded-md px-4 py-3 border bg-gradient-to-b from-background via-background to-muted/50 border-border/60 dark:from-zinc-900 dark:via-zinc-900 dark:to-zinc-800 dark:border-zinc-700 text-foreground"
                        >
                          <p className="chat-message-user">{msg.text}</p>
                          <p className="chat-message-timestamp">
                            {msg.timestamp.toLocaleTimeString()}
                          </p>
                        </div>
                      ) : (
                        // AI response: No box, plain text
                        <div className="max-w-[80%] text-foreground">
                          {msg.text === 'thinking' ? (
                            <ThinkingShimmer children="AI is thinking" />
                          ) : (
                            <MarkdownMessage content={msg.text} />
                          )}
                          <p className="chat-message-timestamp mt-2">
                            {msg.timestamp.toLocaleTimeString()}
                          </p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Chat Input Area with Micro-Interactions */}
            <div className="border-t border-border/50 p-4 finley-dynamic-bg">
              <div className="max-w-4xl mx-auto">
                {/* Attached Files Preview - Above input */}
                {pastedImages.length > 0 && (
                  <div className="mb-3 flex flex-wrap gap-2 p-2 glass-card rounded-lg">
                    {pastedImages.map((file, index) => {
                      const isImage = file.type.startsWith('image/');
                      return (
                        <div key={index} className="relative group">
                          {isImage ? (
                            <img
                              src={URL.createObjectURL(file)}
                              alt={file.name}
                              className="w-16 h-16 object-cover rounded border border-border"
                            />
                          ) : (
                            <div className="w-16 h-16 flex flex-col items-center justify-center rounded border border-border bg-muted text-center p-1">
                              <FileSpreadsheet className="w-6 h-6 text-muted-foreground mb-1" />
                              <span className="text-[8px] text-muted-foreground truncate w-full px-1">
                                {file.name.split('.').pop()?.toUpperCase()}
                              </span>
                            </div>
                          )}
                          <button
                            onClick={() => setPastedImages(prev => prev.filter((_, i) => i !== index))}
                            className="absolute -top-1 -right-1 w-5 h-5 bg-destructive text-destructive-foreground rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                            title={`Remove ${file.name}`}
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                      );
                    })}
                    <div className="flex items-center text-xs text-muted-foreground px-2">
                      {pastedImages.length} file(s) ready to send
                    </div>
                  </div>
                )}

                {/* Drag and Drop Wrapper */}
                <div
                  {...getRootProps()}
                  className={`relative rounded-lg transition-all duration-200 ${isDragActive ? 'ring-2 ring-primary/50' : ''}`}
                >
                  <input {...getInputProps()} />

                  {/* Drag Overlay */}
                  {isDragActive && (
                    <div className="absolute inset-0 z-50 rounded-lg bg-primary/10 backdrop-blur-sm flex items-center justify-center border-2 border-dashed border-primary">
                      <div className="text-center">
                        <Paperclip className="w-8 h-8 text-primary mx-auto mb-2 animate-bounce" />
                        <p className="text-sm font-medium text-primary">Drop files here</p>
                      </div>
                    </div>
                  )}

                  {/* Chat Input Micro-Interactions Component */}
                  <ChatInputMicroInteractions
                    value={message}
                    onChange={setMessage}
                    onSend={handleSendMessage}
                    isLoading={isThinking}
                    placeholder={isDragActive ? "Drop files to attach..." : sampleQuestions[currentQuestionIndex]}
                    onFileClick={open}
                  />
                </div>
              </div>
            </div>
          </div>
        );
    }
  };

  return renderCurrentView();
};