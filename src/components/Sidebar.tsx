// src/components/Sidebar.tsx
import React from 'react';
import DataUniverse from './DataUniverse';
import Notifications from './Notifications'; // Import the new component
import { ScrollArea } from './ui/scroll-area';

const Sidebar: React.FC = () => {
  return (
    <div className="h-full flex flex-col border-r">
      <div className="p-4 border-b">
        <h1 className="text-lg font-bold text-primary">Tcore</h1>
      </div>
      <div className="p-2 border-b">
        <Notifications />
      </div>
      <ScrollArea className="flex-1">
        <DataUniverse />
      </ScrollArea>
      <div className="p-4 border-t">
        <p className="text-xs text-muted-foreground">User profile</p>
      </div>
    </div>
  );
};

export default Sidebar;
