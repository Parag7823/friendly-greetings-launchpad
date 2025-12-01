import { useState, useEffect } from 'react';
import { X, MessageSquare, Trash2, Edit2 } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';
import { useAuth } from './AuthProvider';
import { config } from '@/config';
import { getSessionToken } from '@/utils/authHelpers';
import { cn } from '@/lib/utils';

interface ChatHistory {
  id: string;
  title: string;
  created_at: string;
  updated_at?: string;
}

interface ChatHistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectChat: (chatId: string, title: string) => void;
  currentChatId?: string | null;
}

export const ChatHistoryModal = ({
  isOpen,
  onClose,
  onSelectChat,
  currentChatId,
}: ChatHistoryModalProps) => {
  const { user } = useAuth();
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([]);
  const [loading, setLoading] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');

  // Load chat history
  useEffect(() => {
    if (!isOpen || !user?.id) return;

    const loadHistory = async () => {
      setLoading(true);
      try {
        const sessionToken = await getSessionToken();
        const response = await fetch(`${config.apiUrl}/chat-history/${user.id}`, {
          headers: {
            ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
          }
        });

        if (response.ok) {
          const data = await response.json();
          if (data.chats) {
            setChatHistory(data.chats.sort((a: any, b: any) => 
              new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
            ));
          }
        }
      } catch (error) {
        console.error('Failed to load chat history:', error);
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, [isOpen, user?.id]);

  const handleSelectChat = (chatId: string, title: string) => {
    onSelectChat(chatId, title);
    onClose();
  };

  const handleUpdateTitle = async (chatId: string, newTitle: string) => {
    if (!newTitle.trim()) return;

    try {
      const sessionToken = await getSessionToken();
      const response = await fetch(`${config.apiUrl}/chat/${chatId}/title`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
        },
        body: JSON.stringify({ title: newTitle })
      });

      if (response.ok) {
        setChatHistory(prev =>
          prev.map(chat =>
            chat.id === chatId ? { ...chat, title: newTitle } : chat
          )
        );
        setEditingId(null);
      }
    } catch (error) {
      console.error('Failed to update chat title:', error);
    }
  };

  const handleDeleteChat = async (chatId: string) => {
    try {
      const sessionToken = await getSessionToken();
      const response = await fetch(`${config.apiUrl}/chat/${chatId}`, {
        method: 'DELETE',
        headers: {
          ...(sessionToken && { 'Authorization': `Bearer ${sessionToken}` })
        }
      });

      if (response.ok) {
        setChatHistory(prev => prev.filter(chat => chat.id !== chatId));
      }
    } catch (error) {
      console.error('Failed to delete chat:', error);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-md max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5" />
            Chat History
          </DialogTitle>
        </DialogHeader>

        {/* Chat List */}
        <div className="flex-1 overflow-y-auto space-y-2">
          {loading ? (
            <div className="text-center py-8 text-muted-foreground">
              Loading chats...
            </div>
          ) : chatHistory.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No chats yet
            </div>
          ) : (
            chatHistory.map(chat => (
              <div
                key={chat.id}
                className={cn(
                  "p-3 rounded-lg border transition-colors cursor-pointer group",
                  currentChatId === chat.id
                    ? "bg-primary/10 border-primary"
                    : "border-border hover:bg-muted/50"
                )}
              >
                {editingId === chat.id ? (
                  <div className="flex gap-2">
                    <Input
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      className="text-sm"
                      autoFocus
                    />
                    <Button
                      size="sm"
                      onClick={() => handleUpdateTitle(chat.id, editingTitle)}
                      className="px-2"
                    >
                      Save
                    </Button>
                  </div>
                ) : (
                  <div className="flex items-start justify-between gap-2">
                    <div
                      className="flex-1 min-w-0"
                      onClick={() => handleSelectChat(chat.id, chat.title)}
                    >
                      <p className="text-sm font-medium truncate">{chat.title}</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(chat.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 w-6 p-0"
                        onClick={() => {
                          setEditingId(chat.id);
                          setEditingTitle(chat.title);
                        }}
                      >
                        <Edit2 className="w-3 h-3" />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 w-6 p-0 hover:text-destructive"
                        onClick={() => handleDeleteChat(chat.id)}
                      >
                        <Trash2 className="w-3 h-3" />
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};
