import { MessageCircle, Send } from 'lucide-react';
import { useState } from 'react';

export const ChatInterface = () => {
  const [message, setMessage] = useState('');

  const handleSendMessage = () => {
    if (message.trim()) {
      // TODO: Implement chat functionality
      console.log('Sending message:', message);
      setMessage('');
    }
  };

  return (
    <div className="finley-chat flex flex-col h-full">
      {/* Chat Messages Area */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-foreground tracking-tight">
            Finley AI
          </h1>
        </div>
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
              placeholder="Ask anything..."
              className="w-full bg-card border border-border rounded-lg px-4 py-3 pr-12 text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring transition-all duration-200"
            />
            
            <button
              onClick={handleSendMessage}
              disabled={!message.trim()}
              className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 bg-primary text-primary-foreground rounded-md flex items-center justify-center transition-all duration-200 hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};