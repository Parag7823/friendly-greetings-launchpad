import { useState } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { GripVertical } from 'lucide-react';

interface ThreePanelLayoutProps {
  leftPanel: React.ReactNode;
  centerPanel: React.ReactNode;
  rightPanel: React.ReactNode;
  defaultLeftSize?: number;
  defaultCenterSize?: number;
  defaultRightSize?: number;
  minLeftSize?: number;
  minCenterSize?: number;
  minRightSize?: number;
}

export const ThreePanelLayout = ({
  leftPanel,
  centerPanel,
  rightPanel,
  defaultLeftSize = 20,
  defaultCenterSize = 60,
  defaultRightSize = 20,
  minLeftSize = 15,
  minCenterSize = 30,
  minRightSize = 15,
}: ThreePanelLayoutProps) => {
  return (
    <div className="h-screen w-full bg-background overflow-hidden">
      <PanelGroup direction="horizontal" className="h-full">
        {/* LEFT PANEL - Data Universe */}
        <Panel
          defaultSize={defaultLeftSize}
          minSize={minLeftSize}
          maxSize={40}
          className="bg-card"
        >
          <div className="h-full overflow-hidden">
            {leftPanel}
          </div>
        </Panel>

        {/* Resize Handle - Left/Center */}
        <PanelResizeHandle className="w-1 bg-border hover:bg-primary transition-colors duration-200 relative group">
          <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-4 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
            <GripVertical className="h-4 w-4 text-muted-foreground" />
          </div>
        </PanelResizeHandle>

        {/* CENTER PANEL - Main Content */}
        <Panel
          defaultSize={defaultCenterSize}
          minSize={minCenterSize}
          className="bg-background"
        >
          <div className="h-full overflow-hidden">
            {centerPanel}
          </div>
        </Panel>

        {/* Resize Handle - Center/Right */}
        <PanelResizeHandle className="w-1 bg-border hover:bg-primary transition-colors duration-200 relative group">
          <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-4 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
            <GripVertical className="h-4 w-4 text-muted-foreground" />
          </div>
        </PanelResizeHandle>

        {/* RIGHT PANEL - Details/Properties */}
        <Panel
          defaultSize={defaultRightSize}
          minSize={minRightSize}
          maxSize={40}
          className="bg-card"
        >
          <div className="h-full overflow-hidden">
            {rightPanel}
          </div>
        </Panel>
      </PanelGroup>
    </div>
  );
};
