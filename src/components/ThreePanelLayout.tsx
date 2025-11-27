import { useState, useRef } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { GripVertical, Database } from 'lucide-react';
import { ChatInterface } from './ChatInterface';
import { TabbedFilePreview } from './TabbedFilePreview';
import { DataSourcesPanel } from './DataSourcesPanel';
import { Button } from './ui/button';
import { cn } from '@/lib/utils';

interface ThreePanelLayoutProps {
  currentView?: string;
  onNavigate?: (view: string) => void;
}

export const ThreePanelLayout = ({ currentView = 'chat', onNavigate }: ThreePanelLayoutProps) => {
  const [openFiles, setOpenFiles] = useState<any[]>([]);
  const [activeFileId, setActiveFileId] = useState<string | null>(null);
  const rightPanelRef = useRef<any>(null);
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(true);

  // AUDIT FIX #1: Removed broken useEffect that ran after paint
  // The Panel component now uses defaultCollapsed={true} to start hidden
  // This prevents visual flashing and ensures the panel is hidden before first render

  // Handle file click from Data Sources
  const handleFileClick = (fileId: string, filename: string, fileData: any) => {
    // Add to open files if not already open
    if (!openFiles.find(f => f.id === fileId)) {
      setOpenFiles(prev => [...prev, { id: fileId, filename, ...fileData }]);
    }
    // Set as active file
    setActiveFileId(fileId);
  };

  const handleFileClose = (fileId: string) => {
    setOpenFiles(prev => prev.filter(f => f.id !== fileId));
    // If closing active file, switch to another
    if (fileId === activeFileId) {
      const remaining = openFiles.filter(f => f.id !== fileId);
      setActiveFileId(remaining.length > 0 ? remaining[0].id : null);
    }
  };

  const toggleRightPanel = () => {
    if (rightPanelRef.current) {
      if (isPanelCollapsed) {
        rightPanelRef.current?.expand?.();
      } else {
        rightPanelRef.current?.collapse?.();
      }
    }
  };

  return (
    <div className="h-full w-full finley-dynamic-bg flex flex-col">
      {/* Header with toggle button (AUDIT FIX #4) */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/30">
        <div className="flex-1" />
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleRightPanel}
          title={isPanelCollapsed ? 'Show Data Sources' : 'Hide Data Sources'}
          className="h-8 w-8 p-0"
        >
          <Database className="w-4 h-4" />
        </Button>
      </div>

      <PanelGroup direction="horizontal" className="h-full flex-1">
        {/* Chat Panel - 30% default */}
        <Panel defaultSize={30} minSize={20} maxSize={50} className="relative">
          <ChatInterface currentView={currentView} onNavigate={onNavigate} />
        </Panel>

        {/* Resize Handle */}
        <PanelResizeHandle className="w-0.5 bg-border hover:bg-primary transition-colors relative group">
          <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-3 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
            <GripVertical className="w-3 h-3 text-muted-foreground" />
          </div>
        </PanelResizeHandle>

        {/* File Preview Panel - 45% default */}
        <Panel defaultSize={45} minSize={30} className="relative">
          <TabbedFilePreview
            openFiles={openFiles}
            activeFileId={activeFileId}
            onFileSelect={setActiveFileId}
            onFileClose={handleFileClose}
          />
        </Panel>

        {/* Resize Handle */}
        <PanelResizeHandle className="w-0.5 bg-border hover:bg-primary transition-colors relative group">
          <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-3 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
            <GripVertical className="w-3 h-3 text-muted-foreground" />
          </div>
        </PanelResizeHandle>

        {/* Data Sources Panel - Hidden by default, collapsible (AUDIT FIX #2) */}
        <Panel
          ref={rightPanelRef}
          defaultSize={0}         // AUDIT FIX #2: Start collapsed (hidden)
          defaultCollapsed={true}  // AUDIT FIX #2: Start in collapsed state
          collapsedSize={0}       // AUDIT FIX #2: Fully hidden when collapsed
          minSize={15}
          maxSize={40}
          collapsible
          className="relative"
          onCollapse={() => {
            setIsPanelCollapsed(true);
            localStorage.setItem('rightPanelExpanded', 'false');
          }}
          onExpand={() => {
            setIsPanelCollapsed(false);
            localStorage.setItem('rightPanelExpanded', 'true');
          }}
        >
          <div className="h-full border-l border-border">
            <DataSourcesPanel
              isOpen={!isPanelCollapsed}  // AUDIT FIX #3: Dynamic based on panel state
              onClose={() => rightPanelRef.current?.collapse()}  // AUDIT FIX #3: Actually collapse on close
              onFilePreview={handleFileClick}
            />
          </div>
        </Panel>
      </PanelGroup>
    </div>
  );
};
