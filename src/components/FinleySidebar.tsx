import { useEffect, useState } from 'react';
import { 
  MessageSquarePlus, 
  Plug, 
  Upload, 
  MessageSquare, 
  X 
} from 'lucide-react';
import { Button } from './ui/button';

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
}

export const FinleySidebar = ({ onClose, onNavigate, currentView = 'chat' }: FinleySidebarProps) => {
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

  const handleNewChat = () => {
    const newChat: ChatHistory = {
      id: `chat-${Date.now()}`,
      title: 'New Chat',
      timestamp: new Date(),
      messages: []
    };
    setChatHistory(prev => [newChat, ...prev]);
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
    onNavigate?.('chat');
    onClose?.();
    // TODO: Load the selected chat
  };

  const truncateTitle = (title: string, maxLength: number = 20) => {
    return title.length > maxLength ? title.substring(0, maxLength) + '...' : title;
  };

  return (
    <div className="finley-sidebar flex flex-col h-full p-6 overflow-y-auto bg-muted/30">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground tracking-tight">
              Finley AI
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              Intelligent Financial Analyst
            </p>
          </div>
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
      <div className="flex-1 space-y-2">
        {/* New Chat */}
        <Button
          variant={currentView === 'chat' ? 'secondary' : 'ghost'}
          className="w-full justify-start h-12 px-3 rounded-2xl"
          onClick={handleNewChat}
        >
          <MessageSquarePlus className="w-5 h-5 mr-3" />
          <span className="font-medium">New Chat</span>
        </Button>

        {/* Connector Marketplace */}
        <Button
          variant={currentView === 'marketplace' ? 'secondary' : 'ghost'}
          className="w-full justify-start h-12 px-3 rounded-2xl"
          onClick={handleConnectorMarketplace}
        >
          <Plug className="w-5 h-5 mr-3" />
          <span className="font-medium">Connector Marketplace</span>
        </Button>

        {/* Upload File */}
        <Button
          variant={currentView === 'upload' ? 'secondary' : 'ghost'}
          className="w-full justify-start h-12 px-3 rounded-2xl"
          onClick={handleUploadFile}
        >
          <Upload className="w-5 h-5 mr-3" />
          <span className="font-medium">Upload File</span>
        </Button>

        {/* Chat History Section */}
        {chatHistory.length > 0 && (
          <div className="mt-8">
            <h3 className="text-sm font-medium text-muted-foreground mb-3 px-3">
              Chat History
            </h3>
            <div className="space-y-1">
              {chatHistory.map((chat) => (
                <Button
                  key={chat.id}
                  variant="ghost"
                  className="w-full justify-start h-10 px-3 rounded-xl text-left"
                  onClick={() => handleChatSelect(chat.id)}
                >
                  <MessageSquare className="w-4 h-4 mr-3 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate">
                      {truncateTitle(chat.title)}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {chat.timestamp.toLocaleDateString()}
                    </div>
                  </div>
                </Button>
              ))}
            </div>
          </div>
        )}
      </div>
      
      {/* Footer */}
      <div className="mt-auto pt-6 border-t border-border">
        <p className="text-xs text-muted-foreground text-center">
          Powered by Finley AI
        </p>
      </div>
    </div>
  );
};