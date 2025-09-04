import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X } from 'lucide-react';
import { FinleySidebar } from './FinleySidebar';
import { ChatInterface } from './ChatInterface';
import { useAuth } from './AuthProvider';
import { Button } from './ui/button';

export const FinleyLayout = () => {
  const { user, loading, signInAnonymously } = useAuth();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [currentView, setCurrentView] = useState('chat');

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
        className="fixed top-4 left-4 z-50 lg:hidden"
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
      >
        {isSidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </Button>

      {/* Mobile Overlay */}
      <AnimatePresence>
        {isSidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-40 lg:hidden"
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
              onNavigate={setCurrentView}
              currentView={currentView}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Desktop Sidebar - Always visible on large screens */}
      <div className="hidden lg:block w-80 flex-shrink-0 bg-muted/30 border-r border-border">
        <FinleySidebar 
          onNavigate={setCurrentView}
          currentView={currentView}
        />
      </div>
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Chat Interface */}
        <div className="flex-1">
          <ChatInterface 
            currentView={currentView}
            onNavigate={setCurrentView}
          />
        </div>
      </div>
    </div>
  );
};