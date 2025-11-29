import { useEffect, useState } from 'react';
import { 
  MessageSquarePlus, 
  Plug, 
  MessageSquare, 
  X,
  Check,
  X as XIcon
} from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { ChatContextMenu } from './ChatContextMenu';
import { ShareModal } from './ShareModal';
import { useAuth } from './AuthProvider';
import { supabase } from '@/integrations/supabase/client';
import { config } from '../config';
import { getSessionToken } from '@/utils/authHelpers';

interface ChatHistory {
  id: string;
  title: string;
  timestamp: Date;
  messages: any[];
}

interface FinleySidebarProps {
  onClose?: () => void;
  onNavigate?: (view: string) => void;
  currentView?: string;
  isCollapsed?: boolean;
  currentChatId?: string | null;
}

export const FinleySidebar = ({ onClose, onNavigate, currentView = 'chat', isCollapsed = false, currentChatId = null }: FinleySidebarProps) => {
  const { user } = useAuth();
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([]);
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState<string>('');
  const [shareModal, setShareModal] = useState<{ isOpen: boolean; chatId: string; title: string }>({
    isOpen: false,
    chatId: '',
    title: ''
  });

  // FIX #13: Load chat history from database first, fallback to localStorage only if DB fails
  // This prevents 2 API calls and race conditions
  // FIX #12: Fixed dependency array to use [user?.id] instead of [user]
  useEffect(() => {
    const loadChatHistory = async () => {
      if (!user?.id) return; // Wait for user to be available

      try {
        // FIX #9: Use centralized getSessionToken helper instead of inline call
        const sessionToken = await getSessionToken();
        
        const response = await fetch(`${config.apiUrl}/chat-history/${user.id}`, {
          headers: {
            ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
          }
        });
        if (response.ok) {
          const data = await response.json();
          if (data.chats && data.chats.length > 0) {
            const dbHistory = data.chats.map((chat: any) => ({
              id: chat.id,
              title: chat.title || 'New Chat',
              timestamp: new Date(chat.created_at),
              messages: chat.messages || []
            }));
            
            setChatHistory(dbHistory.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()));
            // Update localStorage with DB data
            localStorage.setItem('finley-chat-history', JSON.stringify(dbHistory));
            return; // Success - don't fallback to localStorage
          }
        }
      } catch (error) {
        console.error('Failed to load chat history from database:', error);
      }
      
      // Fallback: Load from localStorage only if DB fails or returns no data
      try {
        const savedHistory = localStorage.getItem('finley-chat-history');
        if (savedHistory) {
          const parsed = JSON.parse(savedHistory);
          setChatHistory(parsed.map((chat: any) => ({
            ...chat,
            timestamp: new Date(chat.timestamp)
          })));
        }
      } catch (error) {
        console.error('Failed to load chat history from localStorage:', error);
      }
    };

    loadChatHistory();
  }, [user?.id]);

  // Save chat history to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('finley-chat-history', JSON.stringify(chatHistory));
  }, [chatHistory]);

  // Listen for new chat creation events
  useEffect(() => {
    const handleNewChatCreated = (event: CustomEvent) => {
      const { chatId, title, timestamp } = event.detail;
      const newChat: ChatHistory = {
        id: chatId,
        title: title,
        timestamp: timestamp,
        messages: []
      };
      setChatHistory(prev => [newChat, ...prev]);
    };

    window.addEventListener('new-chat-created', handleNewChatCreated as EventListener);
    return () => window.removeEventListener('new-chat-created', handleNewChatCreated as EventListener);
  }, []);

  const handleNewChat = () => {
    // Reset the chat interface without creating a new sidebar entry
    window.dispatchEvent(new CustomEvent('new-chat-requested'));
    onNavigate?.('chat');
    onClose?.();
  };

  const handleConnectorMarketplace = () => {
    onNavigate?.('marketplace');
    onClose?.();
  };

  const handleChatSelect = (chatId: string) => {
    // Load the selected chat
    window.dispatchEvent(new CustomEvent('chat-selected', {
      detail: { chatId }
    }));
    onNavigate?.('chat');
    onClose?.();
  };

  const handleRename = (chatId: string) => {
    const chat = chatHistory.find(c => c.id === chatId);
    if (chat) {
      setEditingChatId(chatId);
      setEditingTitle(chat.title);
    }
  };

  const handleRenameSave = async () => {
    if (!editingChatId || !editingTitle.trim()) return;

    const newTitle = editingTitle.trim();
    
    try {
      // FIX #9: Use centralized getSessionToken helper
      const sessionToken = await getSessionToken();
      
      const response = await fetch(`${config.apiUrl}/chat/rename`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
        },
        body: JSON.stringify({
          chat_id: editingChatId,
          title: newTitle,
          user_id: user?.id || 'anonymous'
        })
      });

      if (response.ok) {
        const result = await response.json();
        // Chat rename successful
        
        // Update the chat history in state
        const updatedHistory = chatHistory.map(chat => 
          chat.id === editingChatId 
            ? { ...chat, title: newTitle }
            : chat
        );
        
        setChatHistory(updatedHistory);
        
        // Update localStorage
        localStorage.setItem('finley-chat-history', JSON.stringify(updatedHistory));
        
        setEditingChatId(null);
        setEditingTitle('');
        
        // Chat renamed successfully
      } else {
        const errorData = await response.json();
        console.error('Rename failed:', errorData);
        throw new Error(errorData.detail || 'Failed to rename chat');
      }
    } catch (error) {
      console.error('Rename error:', error);
      // Revert the title change
      setEditingTitle(chatHistory.find(c => c.id === editingChatId)?.title || '');
      // Show error to user
      alert('Failed to rename chat. Please try again.');
    }
  };

  const handleRenameCancel = () => {
    setEditingChatId(null);
    setEditingTitle('');
  };

  const handleDelete = async (chatId: string) => {
    try {
      // FIX #9: Use centralized getSessionToken helper
      const sessionToken = await getSessionToken();
      
      const response = await fetch(`${config.apiUrl}/chat/delete`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
        },
        body: JSON.stringify({
          chat_id: chatId,
          user_id: user?.id || 'anonymous'
        })
      });

      if (response.ok) {
        const result = await response.json();
        // Chat delete successful
        
        // Update the chat history in state
        const updatedHistory = chatHistory.filter(chat => chat.id !== chatId);
        setChatHistory(updatedHistory);
        
        // Update localStorage
        localStorage.setItem('finley-chat-history', JSON.stringify(updatedHistory));
        
        // If the deleted chat was the current one, reset the current chat
        if (currentChatId === chatId) {
          window.dispatchEvent(new CustomEvent('new-chat-requested'));
        }
        
        // Chat deleted successfully
      } else {
        const errorData = await response.json();
        console.error('Delete failed:', errorData);
        throw new Error(errorData.detail || 'Failed to delete chat');
      }
    } catch (error) {
      console.error('Delete error:', error);
      // Show error to user
      alert('Failed to delete chat. Please try again.');
    }
  };

  const handleShare = (chatId: string) => {
    const chat = chatHistory.find(c => c.id === chatId);
    if (chat) {
      setShareModal({
        isOpen: true,
        chatId: chatId,
        title: chat.title
      });
    }
  };

  const truncateTitle = (title: string, maxLength: number = 20) => {
    return title.length > maxLength ? title.substring(0, maxLength) + '...' : title;
  };

  return (
    <TooltipProvider>
      <div className="finley-sidebar flex flex-col h-full overflow-y-auto bg-muted/30">
      {/* Header */}
        <div className={`mb-4 ${isCollapsed ? 'p-3' : 'p-4'}`}>
          <div className="flex items-center justify-between">
            {!isCollapsed && (
              <div>
        <h1 className="text-xl font-semibold text-foreground tracking-tight">
          Finley AI
        </h1>
        <p className="text-xs text-muted-foreground mt-0.5">
          Intelligent Financial Analyst
        </p>
      </div>
            )}
            {isCollapsed && (
              <div className="w-7 h-7 bg-primary rounded-md flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-xs">F</span>
              </div>
            )}
            {/* Close button for mobile */}
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden"
              onClick={onClose}
            >
              <X className="h-3.5 w-3.5" />
            </Button>
      {/* Navigation Items */}
      <div className={`flex-1 space-y-1.5 ${isCollapsed ? 'px-1.5' : 'px-4'}`}>
        {/* New Chat */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              className={`w-full h-9 rounded-lg border-2 border-primary hover:bg-primary/10 transition-all duration-200 ${isCollapsed ? 'justify-center px-0' : 'justify-start px-2'} gradient-copper-border`}
              onClick={handleNewChat}
            >
              <MessageSquarePlus className="w-4 h-4" />
              {!isCollapsed && <span className="font-medium text-sm ml-2">New Chat</span>}
            </Button>
          </TooltipTrigger>
          {isCollapsed && <TooltipContent side="right"><p>New Chat</p></TooltipContent>}
        </Tooltip>

        {/* Chat History Section */}
        {chatHistory.length > 0 && (
          <div className="mt-4">
            {!isCollapsed && (
              <h3 className="text-xs font-medium text-muted-foreground mb-2 px-2">
                Chat History
              </h3>
            )}
            <div className="space-y-1">
              {chatHistory.map((chat) => (
                <div key={chat.id} className="group relative">
                  {editingChatId === chat.id ? (
                    // Inline editing mode
                    <div className="flex items-center space-x-2 p-1.5">
                      <Input
                        value={editingTitle}
                        onChange={(e) => setEditingTitle(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            handleRenameSave();
                          } else if (e.key === 'Escape') {
                            handleRenameCancel();
                          }
                        }}
                        onBlur={handleRenameSave}
                        className="flex-1 h-7 text-xs"
                        autoFocus
                      />
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-5 w-5"
                        onClick={handleRenameSave}
                      >
                        <Check className="h-3 w-3" />
                      </Button>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-5 w-5"
                        onClick={handleRenameCancel}
                      >
                        <XIcon className="h-3 w-3" />
                      </Button>
                    </div>
                  ) : (
                    // Normal display mode
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant={currentChatId === chat.id ? "secondary" : "ghost"}
                          className={`w-full h-8 rounded-lg text-left group ${isCollapsed ? 'justify-center px-0' : 'justify-start px-2'}`}
                          onClick={() => handleChatSelect(chat.id)}
                        >
                          <MessageSquare className="w-3.5 h-3.5 flex-shrink-0" />
                          {!isCollapsed && (
                            <div className="flex-1 min-w-0 ml-3">
                              <div className="text-xs font-medium truncate">
                                {truncateTitle(chat.title)}
                              </div>
                              <div className="text-[10px] text-muted-foreground">
                                {chat.timestamp.toLocaleDateString()}
                              </div>
                            </div>
                          )}
                          {!isCollapsed && (
                            <ChatContextMenu
                              chatId={chat.id}
                              onRename={handleRename}
                              onDelete={handleDelete}
                              onShare={handleShare}
                              isCollapsed={isCollapsed}
                            />
                          )}
                        </Button>
                      </TooltipTrigger>
                      {isCollapsed && (
                        <TooltipContent side="right">
                          <div>
                            <p className="font-medium">{chat.title}</p>
                            <p className="text-xs text-muted-foreground">
                              {chat.timestamp.toLocaleDateString()}
                </p>
              </div>
                        </TooltipContent>
                      )}
                    </Tooltip>
                  )}
                </div>
              ))}
        </div>
        </div>
        )}
      </div>
      
      {/* Footer */}
      <div className={`mt-auto pt-6 border-t border-border ${isCollapsed ? 'px-2' : 'px-6'}`}>
        {/* Footer space reserved for future use */}
      </div>
      
      {/* Share Modal */}
      <ShareModal
        isOpen={shareModal.isOpen}
        onClose={() => setShareModal({ isOpen: false, chatId: '', title: '' })}
        chatId={shareModal.chatId}
        chatTitle={shareModal.title}
      />
      </div>
    </TooltipProvider>
  );
};