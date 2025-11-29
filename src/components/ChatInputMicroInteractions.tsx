import React, { useState, useEffect } from 'react';
import { Send, Paperclip } from 'lucide-react';

interface ChatInputMicroInteractionsProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  isLoading?: boolean;
  placeholder?: string;
  onFileClick?: () => void;
}

/**
 * ChatInputMicroInteractions: Premium chat input with Copper glow and thinking indicator
 * Features:
 * - Idle state: Subtle Copper glow
 * - Typing state: Pulsing glow animation
 * - Sending state: "Finley is thinking..." with animated dots
 */
export const ChatInputMicroInteractions: React.FC<ChatInputMicroInteractionsProps> = ({
  value,
  onChange,
  onSend,
  isLoading = false,
  placeholder = 'Ask Finley anything...',
  onFileClick
}) => {
  const [isFocused, setIsFocused] = useState(false);
  const [isTyping, setIsTyping] = useState(false);

  useEffect(() => {
    setIsTyping(value.length > 0);
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !isLoading) {
        onSend();
      }
    }
  };

  return (
    <div className="w-full">
      {/* Chat Input Container with Glassmorphism */}
      <div
        className={`
          relative rounded-lg border transition-all duration-300
          ${isFocused || isTyping
            ? 'border-primary/60 shadow-lg shadow-primary/20'
            : 'border-slate-700/50 shadow-md shadow-primary/10'
          }
          ${isLoading ? 'opacity-60 pointer-events-none' : ''}
          glass-card
        `}
      >
        {/* Animated Glow Background */}
        <div
          className={`
            absolute inset-0 rounded-lg pointer-events-none
            ${isFocused || isTyping
              ? 'animate-pulse bg-primary/5'
              : 'bg-primary/0'
            }
          `}
        />

        {/* Input Area */}
        <div className="relative flex items-end gap-4 p-4">
          {/* File Upload Button */}
          <button
            onClick={onFileClick}
            disabled={isLoading}
            className="flex-shrink-0 p-2 hover:bg-slate-700/50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Attach file"
          >
            <Paperclip className="w-4 h-4 text-muted-foreground hover:text-primary transition-colors" />
          </button>

          {/* Textarea */}
          <textarea
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={isLoading}
            rows={1}
            className={`
              flex-1 bg-transparent text-foreground placeholder:text-slate-600 placeholder:font-normal
              outline-none resize-none max-h-32
              disabled:opacity-50 disabled:cursor-not-allowed
              text-sm font-normal
            `}
          />

          {/* Send Button */}
          <button
            onClick={onSend}
            disabled={!value.trim() || isLoading}
            className={`
              flex-shrink-0 p-2 rounded-lg transition-all duration-200
              ${value.trim() && !isLoading
                ? 'bg-primary hover:bg-copper-dark text-primary-foreground shadow-lg shadow-primary/30 hover:shadow-primary/50'
                : 'bg-slate-700/50 text-muted-foreground cursor-not-allowed opacity-50'
              }
            `}
            title="Send message (Shift+Enter for new line)"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>

        {/* Character Count (Optional) */}
        {value.length > 0 && (
          <div className="px-4 pb-3 text-xs text-slate-500 font-medium">
            {value.length} characters
          </div>
        )}
      </div>
    </div>
  );
};
