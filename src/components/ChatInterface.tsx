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
        <div className="text-center max-w-md animate-fade-in">
          <div className="w-16 h-16 bg-card rounded-full flex items-center justify-center mx-auto mb-6 hover-lift">
            <MessageCircle className="w-8 h-8 text-finley-accent" />
          </div>
          
          <h2 className="text-2xl font-semibold text-foreground mb-3">
            Welcome to Finley AI
          </h2>
          
          <p className="text-muted-foreground mb-8 leading-relaxed">
            Your intelligent financial analyst is ready to help you transform data into actionable insights. 
            Start by uploading an Excel file or ask me anything about your financial data.
          </p>
          
          <div className="text-sm text-finley-accent bg-accent/20 rounded-lg p-3 border border-border">
            💡 Try uploading an Excel file to begin your financial analysis journey
          </div>
        </div>
      </div>
      
      {/* Chat Input Area */}
      <div className="border-t border-border p-6">
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
          
          <p className="text-xs text-muted-foreground mt-2 text-center">
            Finley AI can help you analyze financial data, generate insights, and answer questions about your spreadsheets.
          </p>
        </div>
      </div>
    </div>
  );
};