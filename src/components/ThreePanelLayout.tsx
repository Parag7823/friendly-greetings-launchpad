import { useState, useRef } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { ChatInterface } from './ChatInterface';
import { TabbedFilePreview } from './TabbedFilePreview';
import { DataSourcesPanel } from './DataSourcesPanel';
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

  // AUDIT FIX #5: Expose toggleRightPanel to TabbedFilePreview via callback
  const toggleRightPanel = () => {
    if (rightPanelRef.current) {
      if (isPanelCollapsed) {
        rightPanelRef.current?.expand?.();
      } else {
        rightPanelRef.current?.collapse?.();
      }
    }
  };

  // AUDIT FIX #1: Removed broken useEffect that ran after paint
  // The Panel component now uses defaultCollapsed={true} to start hidden
  // This prevents visual flashing and ensures the panel is hidden before first render

  // Handle file click from Data Sources (from DataSourcesPanel)
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

  return (
    <div className="h-full w-full finley-dynamic-bg flex flex-col">
      {/* AUDIT FIX #5: Removed wasteful 48px header bar - Database button moved to TabbedFilePreview */}
      <PanelGroup direction="horizontal" className="h-full flex-1">
        {/* Chat Panel - 30% default (resizable 15-50%) */}
        <Panel defaultSize={30} minSize={15} maxSize={50} className="relative">
          <ChatInterface currentView={currentView} onNavigate={onNavigate} />
        </Panel>

        {/* Resize Handle - Gradient from file preview header */}
        <PanelResizeHandle className="w-0.5 hover:w-1 transition-all" style={{backgroundImage: 'linear-gradient(to bottom, transparent 0%, rgba(37, 99, 235, 0.3) 20%, rgba(37, 99, 235, 1) 50%, rgba(37, 99, 235, 0.3) 80%, transparent 100%)'}} />

        {/* File Preview Panel - 70% default (resizable 50-85%) */}
        <Panel defaultSize={70} minSize={50} maxSize={85} className="relative p-4 overflow-hidden">
          <div className="h-full w-full rounded-xl overflow-hidden bg-background shadow-lg">
            <TabbedFilePreview
              openFiles={openFiles}
              activeFileId={activeFileId}
              onFileSelect={setActiveFileId}
              onFileClose={handleFileClose}
              onToggleDataSources={toggleRightPanel}
              isDataSourcesCollapsed={isPanelCollapsed}
            />
          </div>
        </Panel>

        {/* Resize Handle - Gradient from file preview header */}
        <PanelResizeHandle className="w-0.5 hover:w-1 transition-all" style={{backgroundImage: 'linear-gradient(to bottom, transparent 0%, rgba(37, 99, 235, 0.3) 20%, rgba(37, 99, 235, 1) 50%, rgba(37, 99, 235, 0.3) 80%, transparent 100%)'}} />

        {/* Data Sources Panel - Hidden by default, collapsible (AUDIT FIX #2) */}
        <Panel
          ref={rightPanelRef}
          defaultSize={0}         // AUDIT FIX #2: Start collapsed (hidden)
          collapsedSize={0}       // AUDIT FIX #2: Fully hidden when collapsed
          minSize={30}            // Minimum 30% when expanded (increased from 20%)
          maxSize={50}            // Maximum 50% of screen (increased from 45%)
          collapsible
          className="relative transition-all duration-500 overflow-hidden"
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
