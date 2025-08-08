import { ReactNode } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

type IntegrationCardProps = {
  icon: ReactNode;
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
  statusLabel?: string;
  disabled?: boolean;
};

export const IntegrationCard = ({ icon, title, description, actionLabel, onAction, statusLabel, disabled }: IntegrationCardProps) => {
  return (
    <Card
      className={`rounded-lg shadow-sm ${disabled ? 'opacity-70 cursor-not-allowed' : 'transition-transform duration-200 hover:shadow-md hover:scale-[1.01]'} `}
      aria-disabled={disabled}
    >
      <CardHeader>
        <div className="mb-2 flex items-center justify-start text-foreground">
          {icon}
        </div>
        <CardTitle className="text-xl font-semibold">{title}</CardTitle>
        <CardDescription className="line-clamp-2">{description}</CardDescription>
      </CardHeader>
      <CardContent>
        {/* reserved for future details */}
      </CardContent>
      <CardFooter className="pt-0">
        {actionLabel && !disabled ? (
          <Button onClick={onAction} className="min-w-36">
            {actionLabel}
          </Button>
        ) : statusLabel ? (
          <Badge variant="secondary" className="mx-auto">{statusLabel}</Badge>
        ) : null}
      </CardFooter>
    </Card>
  );
};

export default IntegrationCard;
