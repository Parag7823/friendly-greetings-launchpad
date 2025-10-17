// src/components/ReasoningCanvas.tsx
import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

const ReasoningCanvas: React.FC = () => {
  return (
    <div className="h-full w-full p-4">
      <Card className="h-full">
        <CardHeader>
          <CardTitle>Reasoning Canvas</CardTitle>
        </CardHeader>
        <CardContent>
          <p>This panel will serve as the new chat and reasoning interface.</p>
        </CardContent>
      </Card>
    </div>
  );
};

export default ReasoningCanvas;
