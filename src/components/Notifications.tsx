// src/components/Notifications.tsx
import React from 'react';
import { Bell } from 'lucide-react';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

const Notifications: React.FC = () => {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button className="w-full flex items-center gap-2 p-2 rounded-md hover:bg-muted/50 transition-colors text-sm font-medium">
          <Bell className="h-4 w-4" />
          <span>Notifications</span>
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-80">
        <div className="p-4">
          <h4 className="font-medium leading-none">Notifications</h4>
          <p className="text-sm text-muted-foreground mt-1">
            You have no new notifications.
          </p>
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default Notifications;
