import { FinleySidebar } from './FinleySidebar';
import { ChatInterface } from './ChatInterface';
import { ExcelUpload } from './ExcelUpload';
import { useAuth } from './AuthProvider';
import { Button } from './ui/button';

export const FinleyLayout = () => {
  const { user, loading, signInAnonymously } = useAuth();

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
    <div className="h-screen w-full bg-background flex overflow-hidden">
      {/* Left Sidebar */}
      <div className="w-80 flex-shrink-0">
        <FinleySidebar />
      </div>
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Excel Upload Section */}
        <div className="p-6 border-b">
          <ExcelUpload />
        </div>
        
        {/* Chat Interface */}
        <div className="flex-1">
          <ChatInterface />
        </div>
      </div>
    </div>
  );
};