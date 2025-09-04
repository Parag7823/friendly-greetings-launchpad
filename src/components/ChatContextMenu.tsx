import { useState, useRef, useEffect } from 'react';
import { MoreHorizontal, Share, Edit2, Trash2 } from 'lucide-react';
import { Button } from './ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from './ui/dropdown-menu';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from './ui/alert-dialog';
import { Input } from './ui/input';

interface ChatContextMenuProps {
  chatId: string;
  currentTitle: string;
  onRename: (chatId: string, newTitle: string) => void;
  onDelete: (chatId: string) => void;
  onShare: (chatId: string) => void;
  isCollapsed?: boolean;
}

export const ChatContextMenu = ({
  chatId,
  currentTitle,
  onRename,
  onDelete,
  onShare,
  isCollapsed = false
}: ChatContextMenuProps) => {
  const [isRenaming, setIsRenaming] = useState(false);
  const [newTitle, setNewTitle] = useState(currentTitle);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input when renaming starts
  useEffect(() => {
    if (isRenaming && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isRenaming]);

  const handleRenameStart = () => {
    setIsRenaming(true);
    setNewTitle(currentTitle);
  };

  const handleRenameSave = () => {
    if (newTitle.trim() && newTitle.trim() !== currentTitle) {
      onRename(chatId, newTitle.trim());
    }
    setIsRenaming(false);
  };

  const handleRenameCancel = () => {
    setNewTitle(currentTitle);
    setIsRenaming(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleRenameSave();
    } else if (e.key === 'Escape') {
      handleRenameCancel();
    }
  };

  const handleDeleteConfirm = () => {
    onDelete(chatId);
    setShowDeleteDialog(false);
  };

  if (isRenaming) {
    return (
      <div className="flex-1 min-w-0">
        <Input
          ref={inputRef}
          value={newTitle}
          onChange={(e) => setNewTitle(e.target.value)}
          onBlur={handleRenameSave}
          onKeyDown={handleKeyPress}
          className="h-8 text-sm"
          maxLength={50}
        />
      </div>
    );
  }

  return (
    <>
      <div className="flex items-center justify-between w-full group">
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium truncate">
            {currentTitle}
          </div>
        </div>
        
        {!isCollapsed && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={(e) => e.stopPropagation()}
              >
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem onClick={() => onShare(chatId)}>
                <Share className="h-4 w-4 mr-2" />
                Share
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleRenameStart}>
                <Edit2 className="h-4 w-4 mr-2" />
                Rename
              </DropdownMenuItem>
              <DropdownMenuItem 
                onClick={() => setShowDeleteDialog(true)}
                className="text-destructive focus:text-destructive"
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>

      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Chat</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this chat? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteConfirm}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
};
