import { Bell } from 'lucide-react';

export const Notifications = () => {
  return (
    <div className="p-4">
      <div className="flex items-center gap-2 mb-4">
        <Bell className="h-5 w-5 text-primary" />
        <h2 className="text-sm font-semibold text-foreground">Notifications</h2>
      </div>
      <div className="text-center text-xs text-muted-foreground">
        <p>No new notifications.</p>
      </div>
    </div>
  );
};
