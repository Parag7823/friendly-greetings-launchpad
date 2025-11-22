import { useState } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { GripVertical } from 'lucide-react';
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

  return (
    <div className="h-full w-full finley-dynamic-bg">
      <PanelGroup direction="horizontal" className="h-full">
        {/* Chat Panel - 30% default */}
        <Panel defaultSize={30} minSize={20} maxSize={50} className="relative">
          <ChatInterface currentView={currentView} onNavigate={onNavigate} />
        </Panel>

        {/* Resize Handle */}
        <PanelResizeHandle className="w-1 bg-border hover:bg-primary transition-colors relative group">
          <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-4 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
            <GripVertical className="w-4 h-4 text-muted-foreground" />
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
        <PanelResizeHandle className="w-1 bg-border hover:bg-primary transition-colors relative group">
          <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-4 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
            <GripVertical className="w-4 h-4 text-muted-foreground" />
          </div>
        </PanelResizeHandle>

        {/* Data Sources Panel - 25% default */}
        <Panel defaultSize={25} minSize={15} maxSize={40} className="relative">
          <div className="h-full border-l border-border">
            <DataSourcesPanel
              isOpen={true}
              onClose={() => { }} // Always visible in 3-panel layout
              onFilePreview={handleFileClick}
            />
          </div>
        </Panel>
      </PanelGroup>
    </div>
  );
};
