import { ThreePanelLayout } from '@/components/ThreePanelLayout';
import { DataUniverse } from '@/components/DataUniverse';
import { ChatInterface } from '@/components/ChatInterface';

export const MainWorkspace = () => {
  return (
    <ThreePanelLayout
      leftPanel={<DataUniverse />}
      centerPanel={
        <ChatInterface 
          currentView="chat"
          onNavigate={() => {}}
        />
      }
      rightPanel={
        <div className="h-full p-4">
          <div className="text-sm font-semibold text-foreground mb-2">Properties</div>
          <div className="text-xs text-muted-foreground">
            Select an item to view details
          </div>
        </div>
      }
      defaultLeftSize={20}
      defaultCenterSize={60}
      defaultRightSize={20}
    />
  );
};

export default MainWorkspace;
