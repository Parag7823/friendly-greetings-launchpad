import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu } from 'lucide-react';
import { FinleySidebar } from './FinleySidebar';
import { ThreePanelLayout } from './ThreePanelLayout';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from './AuthProvider';
import { Button } from './ui/button';

export const FinleyLayout = () => {
  const { user, loading, signInAnonymously } = useAuth();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(true);
  const [currentView, setCurrentView] = useState('chat');
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const location = useLocation();
  const navigate = useNavigate();

  // Listen for chat events
  useEffect(() => {
    const handleNewChatCreated = (event: CustomEvent) => {
      setCurrentChatId(event.detail.chatId);
    };

    const handleChatSelected = (event: CustomEvent) => {
      setCurrentChatId(event.detail.chatId);
    };

    const handleNewChatRequested = () => {
      setCurrentChatId(null);
    };

    window.addEventListener('new-chat-created', handleNewChatCreated as EventListener);
    window.addEventListener('chat-selected', handleChatSelected as EventListener);
    window.addEventListener('new-chat-requested', handleNewChatRequested);

    return () => {
      window.removeEventListener('new-chat-created', handleNewChatCreated as EventListener);
      window.removeEventListener('chat-selected', handleChatSelected as EventListener);
      window.removeEventListener('new-chat-requested', handleNewChatRequested);
    };
  }, []);

  // Sync view with route path
  useEffect(() => {
    // All routes now default to chat view
    setCurrentView('chat');
  }, [location.pathname]);

  const handleNavigate = (view: string) => {
    setCurrentView(view);
    // Navigate to chat for all views
    if (location.pathname !== '/chat' && location.pathname !== '/') {
      navigate('/chat');
    }
  };

  if (loading) {
    return (
      <div className="h-screen w-full bg-background flex items-center justify-center">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="h-screen w-full bg-background flex items-center justify-center">
        <div className="text-center space-y-4">
          <h1 className="text-2xl font-semibold">Welcome to Finley AI</h1>
          <p className="text-muted-foreground">Sign in to start analyzing your financial documents</p>
          <Button onClick={signInAnonymously}>Get Started</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-full bg-background flex overflow-hidden relative">
      {/* Sidebar Toggle Button */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-3 left-3 z-50"
        onClick={() => {
          if (window.innerWidth < 1024) {
            setIsSidebarOpen(!isSidebarOpen);
          } else {
            setIsSidebarCollapsed(!isSidebarCollapsed);
          }
        }}
      >
        <Menu className="h-4 w-4" />
      </Button>

      {/* Mobile Overlay */}
      <AnimatePresence>
        {isSidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-[#1a1a1a]/50 z-40 lg:hidden"
            onClick={() => setIsSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Mobile Sidebar */}
      <AnimatePresence>
        {isSidebarOpen && (
          <motion.div
            initial={{ x: -320 }}
            animate={{ x: 0 }}
            exit={{ x: -320 }}
            transition={{ type: "spring", damping: 20, stiffness: 300 }}
            className="lg:hidden w-80 flex-shrink-0 bg-muted/30 border-r border-border z-50 fixed left-0 top-0 h-full"
          >
            <FinleySidebar 
              onClose={() => setIsSidebarOpen(false)} 
              onNavigate={handleNavigate}
              currentView={currentView}
              currentChatId={currentChatId}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Desktop Sidebar - Collapsible on large screens */}
      <motion.div
        className="hidden lg:block bg-muted/30 border-r border-border flex-shrink-0"
        animate={{ width: isSidebarCollapsed ? "72px" : "280px" }}
        transition={{ type: "spring", damping: 20, stiffness: 300 }}
      >
        <FinleySidebar 
          onNavigate={handleNavigate}
          currentView={currentView}
          isCollapsed={isSidebarCollapsed}
          currentChatId={currentChatId}
        />
      </motion.div>
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* 3-Panel Layout */}
        <div className="flex-1 min-h-0">
          <ThreePanelLayout 
            currentView={currentView}
            onNavigate={handleNavigate}
          />
        </div>
      </div>
    </div>
  );
};