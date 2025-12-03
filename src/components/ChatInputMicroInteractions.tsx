import React, { useState, useEffect, useMemo } from 'react';
import { Send, Paperclip } from 'lucide-react';
import TextareaAutosize from 'react-textarea-autosize';
import debounce from 'lodash.debounce';

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

  // Debounce typing state to prevent excessive re-renders (300ms delay)
  const debouncedSetTyping = useMemo(
    () => debounce((val: boolean) => setIsTyping(val), 300),
    []
  );

  useEffect(() => {
    debouncedSetTyping(value.length > 0);
    
    // Cleanup debounced function on unmount
    return () => {
      debouncedSetTyping.cancel();
    };
  }, [value, debouncedSetTyping]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Only prevent default if there's content AND user pressed Enter without Shift
    // This allows natural newline behavior when input is empty
    if (e.key === 'Enter' && !e.shiftKey && value.trim() && !isLoading) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div
      className={`
        w-full relative flex flex-col gap-2 p-2 border border-border/40 rounded-md
        bg-transparent transition-all duration-300
        ${isFocused || isTyping
          ? 'border-slate-600/60'
          : 'border-slate-700/30'
        }
        ${isLoading ? 'opacity-60 pointer-events-none' : ''}
      `}
    >
      {/* Top Row - Send Button on Right */}
      <div className="flex items-center justify-end">
        <button
          onClick={onSend}
          disabled={!value.trim() || isLoading}
          className={`
            flex-shrink-0 p-1.5 rounded transition-all duration-200
            ${value.trim() && !isLoading
              ? 'bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg shadow-primary/30 hover:shadow-primary/50'
              : 'bg-transparent text-muted-foreground cursor-not-allowed opacity-30'
            }
          `}
          title="Send message (Enter to send)"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>

      {/* Input Row */}
      <div className="flex items-end gap-2">
        {/* File Upload Button */}
        <button
          onClick={onFileClick}
          disabled={isLoading}
          className="flex-shrink-0 p-1.5 hover:bg-slate-700/50 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title="Attach file"
        >
          <Paperclip className="w-4 h-4 text-muted-foreground hover:text-primary transition-colors" />
        </button>

        {/* Textarea with Auto-Resize */}
        <TextareaAutosize
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isLoading}
          minRows={1}
          maxRows={4}
          aria-label="Chat message input"
          aria-describedby="input-help"
          aria-busy={isLoading}
          className={`
            flex-1 bg-transparent text-foreground placeholder:text-slate-600 placeholder:font-normal
            outline-none resize-none
            disabled:opacity-50 disabled:cursor-not-allowed
            text-sm font-normal
          `}
        />
        {/* Hidden help text for screen readers */}
        <span id="input-help" className="sr-only">
          Press Enter to send message, Shift+Enter for new line
        </span>
      </div>
    </div>
  );
};
