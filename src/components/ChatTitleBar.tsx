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
            "text-[10px] font-medium text-foreground truncate transition-opacity",
            isLoading && "opacity-60"
          )}>
            {title || "New Chat"}
          </h2>
          <p className="text-[9px] text-muted-foreground">
            {isLoading ? "Loading..." : "Chat"}
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2 flex-shrink-0 ml-4">
          {onNewChat && (
            <Button
              size="sm"
              variant="ghost"
              className="h-6 w-6 p-0 hover:bg-slate-700/50"
              onClick={onNewChat}
              title="New Chat"
            >
              <Plus className="w-3 h-3 text-foreground" />
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0 hover:bg-slate-700/50"
            onClick={onHistoryClick}
            title="Chat History"
          >
            <History className="w-3 h-3 text-foreground" />
          </Button>
        </div>
      </div>
    </div>
  );
};
