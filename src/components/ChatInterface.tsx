import { MessageCircle, Send, Upload, Plug } from 'lucide-react';
import { useState, useEffect } from 'react';
import { EnhancedExcelUpload } from './EnhancedExcelUpload';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';

interface ChatInterfaceProps {
  currentView?: string;
  onNavigate?: (view: string) => void;
}

export const ChatInterface = ({ currentView = 'chat', onNavigate }: ChatInterfaceProps) => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Array<{ id: string; text: string; isUser: boolean; timestamp: Date }>>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [isNewChat, setIsNewChat] = useState(true);

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
                user_id: 'current-user-id' // Replace with actual user ID
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
            user_id: 'current-user-id', // Replace with actual user ID
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
            <div className="p-6">
              <div className="mb-6">
                <h1 className="text-2xl font-bold text-foreground mb-2">Upload Financial Documents</h1>
                <p className="text-muted-foreground">Upload your Excel, CSV, or other financial files for AI analysis</p>
              </div>
              <EnhancedExcelUpload />
            </div>
          </div>
        );
      
      case 'marketplace':
        return (
          <div className="h-full flex items-center justify-center p-8">
            <Card className="max-w-md w-full">
              <CardHeader className="text-center">
                <div className="mx-auto w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center mb-4">
                  <Plug className="w-6 h-6 text-primary" />
                </div>
                <CardTitle>Connector Marketplace</CardTitle>
                <CardDescription>
                  Connect with your favorite financial platforms and services
                </CardDescription>
              </CardHeader>
              <CardContent className="text-center">
                <p className="text-muted-foreground mb-4">
                  Coming Soon! We're working on integrations with popular financial platforms.
                </p>
                <div className="space-y-2 text-sm text-muted-foreground">
                  <p>• QuickBooks Integration</p>
                  <p>• Stripe Payment Processing</p>
                  <p>• Bank API Connections</p>
                  <p>• And many more...</p>
                </div>
              </CardContent>
            </Card>
          </div>
        );
      
      case 'chat':
      default:
        return (
          <div className="finley-chat flex flex-col h-full">
            {/* Chat Messages Area */}
            <div className="flex-1 overflow-y-auto p-8">
              {messages.length === 0 ? (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center">
                    <h1 className="text-3xl font-bold text-foreground tracking-tight mb-4">
                      Finance Meets Intelligence
                    </h1>
                    <p className="text-muted-foreground text-lg">
                      Ask me anything about your financial data
                    </p>
                  </div>
                </div>
              ) : (
                <div className="max-w-3xl mx-auto space-y-4">
                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                          msg.isUser
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted text-foreground'
                        }`}
                      >
                        <p className="text-sm">{msg.text}</p>
                        <p className="text-xs opacity-70 mt-1">
                          {msg.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            {/* Chat Input Area - Fixed at bottom */}
            <div className="border-t border-border p-6 bg-background">
              <div className="max-w-3xl mx-auto">
                <div className="relative">
                  <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                    placeholder="Ask anything about your financial data..."
                    className="w-full bg-card border border-border rounded-2xl px-4 py-3 pr-12 text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring transition-all duration-200"
                  />
                  
                  <button
                    onClick={handleSendMessage}
                    disabled={!message.trim()}
                    className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 bg-primary text-primary-foreground rounded-xl flex items-center justify-center transition-all duration-200 hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Send className="w-4 h-4" />
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