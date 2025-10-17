// src/App.tsx
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import Sidebar from './components/Sidebar';
import ContextIntelligence from './components/ContextIntelligence';
import ReasoningCanvas from './components/ReasoningCanvas';
import { Toaster } from "@/components/ui/toaster";

function App() {
  return (
    <>
      <div className="h-screen w-screen bg-background text-foreground flex">
        <PanelGroup direction="horizontal">
          {/* Left Panel (Sidebar) */}
          <Panel defaultSize={20} minSize={15} maxSize={30}>
            <Sidebar />
          </Panel>
          <PanelResizeHandle className="w-1 bg-border hover:bg-primary transition-colors" />

          {/* Center Panel (Context Intelligence) */}
          <Panel defaultSize={50} minSize={30}>
            <ContextIntelligence />
          </Panel>
          <PanelResizeHandle className="w-1 bg-border hover:bg-primary transition-colors" />

          {/* Right Panel (Reasoning Canvas) */}
          <Panel defaultSize={30} minSize={20}>
            <ReasoningCanvas />
          </Panel>
        </PanelGroup>
      </div>
      <Toaster />
    </>
  );
}

export default App;
