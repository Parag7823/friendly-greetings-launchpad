import { History, Plus } from 'lucide-react';
import { Button } from './ui/button';
import { cn } from '@/lib/utils';

interface ChatTitleBarProps {
  title: string;
  onHistoryClick: () => void;
  onNewChat?: () => void;
  isLoading?: boolean;
}

export const ChatTitleBar = ({
  title,
  onHistoryClick,
  onNewChat,
  isLoading = false,
}: ChatTitleBarProps) => {
  return (
    <div className="border-b border-border bg-background/80 backdrop-blur-sm px-4 py-3">
      <div className="flex items-center justify-between">
        {/* Title Section */}
        <div className="flex-1 min-w-0">
          <h2 className={cn(
            "text-sm font-semibold text-foreground truncate transition-opacity",
            isLoading && "opacity-60"
          )}>
            {title || "New Chat"}
          </h2>
          <p className="text-xs text-muted-foreground">
            {isLoading ? "Loading..." : "Chat"}
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2 flex-shrink-0 ml-4">
          {onNewChat && (
            <Button
              size="sm"
              variant="ghost"
              className="h-8 w-8 p-0"
              onClick={onNewChat}
              title="New Chat"
            >
              <Plus className="w-4 h-4" />
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            className="h-8 w-8 p-0"
            onClick={onHistoryClick}
            title="Chat History"
          >
            <History className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};
