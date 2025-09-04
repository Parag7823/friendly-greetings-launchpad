import { useEffect, useState } from 'react';
import { 
  MessageSquarePlus, 
  Plug, 
  Upload, 
  MessageSquare, 
  X 
} from 'lucide-react';
import { Button } from './ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';

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
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([]);

  // Load chat history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('finley-chat-history');
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        setChatHistory(parsed.map((chat: any) => ({
          ...chat,
          timestamp: new Date(chat.timestamp)
        })));
      } catch (error) {
        console.error('Failed to load chat history:', error);
      }
    }
  }, []);

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

  const handleUploadFile = () => {
    onNavigate?.('upload');
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

  const truncateTitle = (title: string, maxLength: number = 20) => {
    return title.length > maxLength ? title.substring(0, maxLength) + '...' : title;
  };

  return (
    <TooltipProvider>
      <div className="finley-sidebar flex flex-col h-full overflow-y-auto bg-muted/30">
      {/* Header */}
        <div className={`mb-8 ${isCollapsed ? 'p-4' : 'p-6'}`}>
          <div className="flex items-center justify-between">
            {!isCollapsed && (
              <div>
        <h1 className="text-2xl font-bold text-foreground tracking-tight">
          Finley AI
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Intelligent Financial Analyst
        </p>
      </div>
            )}
            {isCollapsed && (
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-sm">F</span>
              </div>
            )}
            {/* Close button for mobile */}
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden"
              onClick={onClose}
            >
              <X className="h-4 w-4" />
            </Button>
                  </div>
                </div>
                
      {/* Navigation Items */}
      <div className={`flex-1 space-y-2 ${isCollapsed ? 'px-2' : 'px-6'}`}>
        {/* New Chat */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={currentView === 'chat' ? 'secondary' : 'ghost'}
              className={`w-full h-12 rounded-2xl ${isCollapsed ? 'justify-center px-0' : 'justify-start px-3'}`}
              onClick={handleNewChat}
            >
              <MessageSquarePlus className="w-5 h-5" />
              {!isCollapsed && <span className="font-medium ml-3">New Chat</span>}
            </Button>
          </TooltipTrigger>
          {isCollapsed && <TooltipContent side="right"><p>New Chat</p></TooltipContent>}
        </Tooltip>

        {/* Connector Marketplace */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={currentView === 'marketplace' ? 'secondary' : 'ghost'}
              className={`w-full h-12 rounded-2xl ${isCollapsed ? 'justify-center px-0' : 'justify-start px-3'}`}
              onClick={handleConnectorMarketplace}
            >
              <Plug className="w-5 h-5" />
              {!isCollapsed && <span className="font-medium ml-3">Connector Marketplace</span>}
            </Button>
          </TooltipTrigger>
          {isCollapsed && <TooltipContent side="right"><p>Connector Marketplace</p></TooltipContent>}
        </Tooltip>

        {/* Upload File */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={currentView === 'upload' ? 'secondary' : 'ghost'}
              className={`w-full h-12 rounded-2xl ${isCollapsed ? 'justify-center px-0' : 'justify-start px-3'}`}
              onClick={handleUploadFile}
            >
              <Upload className="w-5 h-5" />
              {!isCollapsed && <span className="font-medium ml-3">Upload File</span>}
            </Button>
          </TooltipTrigger>
          {isCollapsed && <TooltipContent side="right"><p>Upload File</p></TooltipContent>}
        </Tooltip>

        {/* Chat History Section */}
        {chatHistory.length > 0 && (
          <div className="mt-8">
            {!isCollapsed && (
              <h3 className="text-sm font-medium text-muted-foreground mb-3 px-3">
                Chat History
              </h3>
            )}
            <div className="space-y-1">
              {chatHistory.map((chat) => (
                <Tooltip key={chat.id}>
                  <TooltipTrigger asChild>
                    <Button
                      variant={currentChatId === chat.id ? "secondary" : "ghost"}
                      className={`w-full h-10 rounded-xl text-left ${isCollapsed ? 'justify-center px-0' : 'justify-start px-3'}`}
                      onClick={() => handleChatSelect(chat.id)}
                    >
                      <MessageSquare className="w-4 h-4 flex-shrink-0" />
                      {!isCollapsed && (
                        <div className="flex-1 min-w-0 ml-3">
                          <div className="text-sm font-medium truncate">
                            {truncateTitle(chat.title)}
                  </div>
                          <div className="text-xs text-muted-foreground">
                            {chat.timestamp.toLocaleDateString()}
                  </div>
                </div>
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
              ))}
        </div>
        </div>
        )}
      </div>
      
      {/* Footer */}
      <div className={`mt-auto pt-6 border-t border-border ${isCollapsed ? 'px-2' : 'px-6'}`}>
        {!isCollapsed && (
          <p className="text-xs text-muted-foreground text-center">
            Powered by Finley AI
          </p>
        )}
      </div>
      </div>
    </TooltipProvider>
  );
};