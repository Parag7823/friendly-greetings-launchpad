// src/components/DataUniverse.tsx
import React from 'react';
import { Database, FileText, Cloud } from 'lucide-react';

const DataUniverse: React.FC = () => {
  return (
    <div className="p-4 space-y-4">
      <div>
        <h3 className="text-sm font-semibold mb-2">Data Sources</h3>
        <div className="space-y-2">
          <div className="flex items-center gap-2 p-2 rounded-md hover:bg-muted/50 transition-colors cursor-pointer">
            <Database className="h-4 w-4 text-primary" />
            <span className="text-sm">Local Files</span>
          </div>
          <div className="flex items-center gap-2 p-2 rounded-md hover:bg-muted/50 transition-colors cursor-pointer">
            <Cloud className="h-4 w-4 text-primary" />
            <span className="text-sm">Cloud Storage</span>
          </div>
          <div className="flex items-center gap-2 p-2 rounded-md hover:bg-muted/50 transition-colors cursor-pointer">
            <FileText className="h-4 w-4 text-primary" />
            <span className="text-sm">Recent Documents</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataUniverse;
