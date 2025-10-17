// src/components/ContextIntelligence.tsx
import React from 'react';
import { EnhancedFileUpload } from './EnhancedFileUpload'; // Import the component

const ContextIntelligence: React.FC = () => {
  return (
    <div className="h-full w-full flex flex-col">
      {/* We can add a header or tabs here in the future */}
      <div className="flex-1 p-4 overflow-auto">
        <EnhancedFileUpload />
      </div>
    </div>
  );
};

export default ContextIntelligence;
