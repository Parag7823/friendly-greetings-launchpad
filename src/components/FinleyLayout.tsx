import { FinleySidebar } from './FinleySidebar';
import { ChatInterface } from './ChatInterface';

export const FinleyLayout = () => {
  return (
    <div className="h-screen w-full bg-background flex overflow-hidden">
      {/* Left Sidebar */}
      <div className="w-80 flex-shrink-0">
        <FinleySidebar />
      </div>
      
      {/* Main Chat Interface */}
      <div className="flex-1 flex flex-col">
        <ChatInterface />
      </div>
    </div>
  );
};