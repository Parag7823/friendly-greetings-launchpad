import { Send, Loader2, AlertCircle, CheckCircle, TrendingUp, TrendingDown, Network, FileText, Upload as UploadIcon } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import { useAuth } from './AuthProvider';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { cn } from '@/lib/utils';
import { useFastAPIProcessor } from './FastAPIProcessor';
import { useToast } from '@/hooks/use-toast';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  questionType?: string;
  confidence?: number;
  data?: any;
  actions?: Action[];
  visualizations?: Visualization[];
  followUpQuestions?: string[];
}

interface Action {
  type: string;
  label: string;
  data?: any;
}

interface Visualization {
  type: string;
  title: string;
  data: any;
}

export const IntelligentChatInterface = () => {
  const { user } = useAuth();
  const { toast } = useToast();
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Initialize file processor for uploads
  const { processFileWithFastAPI } = useFastAPIProcessor();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Welcome message
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([{
        id: 'welcome',
        text: "👋 Hi! I'm Finley, your intelligent financial assistant. I can help you:\n\n• Analyze your financial data\n• Find causal relationships (\"Why did revenue drop?\")\n• Predict patterns (\"When will customer pay?\")\n• Explore relationships (\"Show vendor connections\")\n• Run what-if scenarios\n• Upload and process files\n• Connect integrations\n\nWhat would you like to do?",
        isUser: false,
        timestamp: new Date(),
        followUpQuestions: [
          "Upload my financial data",
          "Connect QuickBooks",
          "Why did my expenses increase?",
          "Show me my top vendors"
        ]
      }]);
    }
  }, []);

  const handleSendMessage = async (messageText?: string) => {
    const textToSend = messageText || message.trim();
    
    if (!textToSend) return;

    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      text: textToSend,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setMessage('');
    setIsLoading(true);

    try {
      // Check if this is a file upload request
      if (textToSend.toLowerCase().includes('upload') || textToSend.toLowerCase().includes('file')) {
        // Trigger file upload
        fileInputRef.current?.click();
        setIsLoading(false);
        return;
      }

      // Check if this is a connector request
      if (textToSend.toLowerCase().includes('connect') && 
          (textToSend.toLowerCase().includes('quickbooks') || 
           textToSend.toLowerCase().includes('xero') || 
           textToSend.toLowerCase().includes('gmail'))) {
        // Handle connector request
        await handleConnectorRequest(textToSend);
        return;
      }

      // Send to intelligent chat orchestrator
      const response = await fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: textToSend,
          user_id: user?.id || 'anonymous',
          chat_id: currentChatId
        })
      });

      if (response.ok) {
        const data = await response.json();

        const aiMessage: Message = {
          id: `msg-${Date.now()}-ai`,
          text: data.response,
          isUser: false,
          timestamp: new Date(data.timestamp),
          questionType: data.question_type,
          confidence: data.confidence,
          data: data.data,
          actions: data.actions,
          visualizations: data.visualizations,
          followUpQuestions: data.follow_up_questions
        };

        setMessages(prev => [...prev, aiMessage]);
      } else {
        throw new Error('Failed to get response');
      }
    } catch (error) {
      console.error('Chat error:', error);
      
      const errorMessage: Message = {
        id: `msg-${Date.now()}-error`,
        text: 'Sorry, I encountered an error. Please try again.',
        isUser: false,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleConnectorRequest = async (messageText: string) => {
    // Extract provider from message
    let provider = '';
    if (messageText.toLowerCase().includes('quickbooks')) provider = 'quickbooks';
    else if (messageText.toLowerCase().includes('xero')) provider = 'xero';
    else if (messageText.toLowerCase().includes('gmail')) provider = 'gmail';

    const connectingMessage: Message = {
      id: `msg-${Date.now()}-connecting`,
      text: `🔄 Opening ${provider} authorization window...`,
      isUser: false,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, connectingMessage]);
    setIsLoading(false);

    // Implement actual connector logic here
    // This would call your existing connector endpoints
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    
    const uploadingMessage: Message = {
      id: `msg-${Date.now()}-uploading`,
      text: `📤 Uploading ${file.name}... I'll analyze it and learn from your data.`,
      isUser: false,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, uploadingMessage]);
    setIsLoading(true);

    try {
      // Process file with FastAPI
      await processFileWithFastAPI(
        file,
        undefined, // No custom prompt
        (progress) => {
          // Update progress
          setUploadProgress(progress.progress || 0);
          
          // Add progress message
          if (progress.status === 'processing' && progress.message) {
            const progressMsg: Message = {
              id: `msg-${Date.now()}-progress`,
              text: `⚙️ ${progress.message}`,
              isUser: false,
              timestamp: new Date()
            };
            setMessages(prev => [...prev.slice(0, -1), progressMsg]);
          }
        }
      );

      // Success message
      const successMessage: Message = {
        id: `msg-${Date.now()}-success`,
        text: `✅ Successfully processed ${file.name}! I've learned from your data. You can now ask me questions about it.`,
        isUser: false,
        timestamp: new Date(),
        followUpQuestions: [
          "Analyze my financial data",
          "Show me spending patterns",
          "Find any anomalies"
        ]
      };
      
      setMessages(prev => [...prev.slice(0, -1), successMessage]);
      
      toast({
        title: "File processed successfully",
        description: `${file.name} has been analyzed and added to your knowledge base.`
      });
      
    } catch (error) {
      console.error('File upload error:', error);
      
      const errorMessage: Message = {
        id: `msg-${Date.now()}-error`,
        text: `❌ Failed to process ${file.name}. Please try again.`,
        isUser: false,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev.slice(0, -1), errorMessage]);
      
      toast({
        title: "Upload failed",
        description: "There was an error processing your file.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
      setUploadProgress(0);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleFollowUpClick = (question: string) => {
    handleSendMessage(question);
  };

  const handleActionClick = (action: Action) => {
    // Handle different action types
    switch (action.type) {
      case 'view_details':
        // Show detailed analysis
        break;
      case 'run_whatif':
        // Automatically run what-if scenario
        handleSendMessage(`Run what-if scenario based on previous analysis`);
        break;
      case 'view_graph':
        // Show relationship graph
        break;
      case 'export':
        // Export data
        break;
      default:
        console.log('Unknown action:', action);
    }
  };

  const renderMessage = (msg: Message) => {
    return (
      <div
        key={msg.id}
        className={cn(
          "flex w-full mb-4",
          msg.isUser ? "justify-end" : "justify-start"
        )}
      >
        <div
          className={cn(
            "max-w-[80%] rounded-lg px-4 py-3",
            msg.isUser
              ? "bg-primary text-primary-foreground"
              : "bg-[#0A0A0A] border border-white/10 text-white"
          )}
        >
          {/* Message text */}
          <div className="whitespace-pre-wrap text-sm">{msg.text}</div>

          {/* Confidence badge */}
          {msg.confidence !== undefined && (
            <div className="mt-2">
              <Badge variant="outline" className="text-xs">
                Confidence: {(msg.confidence * 100).toFixed(0)}%
              </Badge>
            </div>
          )}

          {/* Question type indicator */}
          {msg.questionType && (
            <div className="mt-2 flex items-center gap-2 text-xs text-white/60">
              {msg.questionType === 'causal' && <TrendingDown className="w-3 h-3" />}
              {msg.questionType === 'temporal' && <TrendingUp className="w-3 h-3" />}
              {msg.questionType === 'relationship' && <Network className="w-3 h-3" />}
              <span className="capitalize">{msg.questionType} Analysis</span>
            </div>
          )}

          {/* Actions */}
          {msg.actions && msg.actions.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {msg.actions.map((action, idx) => (
                <Button
                  key={idx}
                  size="sm"
                  variant="outline"
                  className="text-xs h-7"
                  onClick={() => handleActionClick(action)}
                >
                  {action.label}
                </Button>
              ))}
            </div>
          )}

          {/* Follow-up questions */}
          {msg.followUpQuestions && msg.followUpQuestions.length > 0 && (
            <div className="mt-3 space-y-2">
              <div className="text-xs text-white/60">💡 Suggested questions:</div>
              {msg.followUpQuestions.map((question, idx) => (
                <button
                  key={idx}
                  className="block w-full text-left text-xs px-3 py-2 rounded bg-white/5 hover:bg-white/10 transition-colors border border-white/10"
                  onClick={() => handleFollowUpClick(question)}
                >
                  {question}
                </button>
              ))}
            </div>
          )}

          {/* Timestamp */}
          <div className="mt-2 text-[10px] opacity-60">
            {msg.timestamp.toLocaleTimeString()}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="border-b border-border px-6 py-4">
        <h1 className="text-xl font-semibold text-foreground">💬 Finley AI</h1>
        <p className="text-sm text-muted-foreground">Your Intelligent Financial Brain</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.map(renderMessage)}
        
        {/* Loading indicator */}
        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-[#0A0A0A] border border-white/10 rounded-lg px-4 py-3">
              <div className="flex items-center gap-2 text-white/60">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-sm">Thinking...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border px-6 py-4">
        <div className="flex gap-2">
          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
            placeholder="Ask me anything about your finances..."
            className="flex-1"
            disabled={isLoading}
          />
          <Button
            onClick={() => handleSendMessage()}
            disabled={isLoading || !message.trim()}
            size="icon"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
        
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".xlsx,.xls,.csv"
          onChange={handleFileUpload}
          className="hidden"
        />
        
        {/* Quick actions */}
        <div className="mt-3 flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            className="text-xs h-7"
            onClick={() => fileInputRef.current?.click()}
          >
            <UploadIcon className="w-3 h-3 mr-1" />
            Upload File
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="text-xs h-7"
            onClick={() => handleSendMessage("Connect QuickBooks")}
          >
            Connect QuickBooks
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="text-xs h-7"
            onClick={() => handleSendMessage("Show me my financial summary")}
          >
            Financial Summary
          </Button>
        </div>
      </div>
    </div>
  );
};
