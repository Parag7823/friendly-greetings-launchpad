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
  variant?: 'grid' | 'list';
  className?: string;
  actionAriaLabel?: string;
  secondaryActionLabel?: string;
  onSecondaryAction?: () => void;
  secondaryAriaLabel?: string;
};

export const IntegrationCard = ({ icon, title, description, actionLabel, onAction, statusLabel, disabled, variant = 'grid', className = '', actionAriaLabel, secondaryActionLabel, onSecondaryAction, secondaryAriaLabel }: IntegrationCardProps) => {
  if (variant === 'list') {
    return (
      <Card
        className={`w-full rounded-md shadow-sm bg-[#0A0A0A] border border-white/10 hover:border-white/20 transition-colors ${disabled ? 'opacity-70 cursor-not-allowed' : ''} ${className}`}
        aria-disabled={disabled}
      >
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 p-4">
          <div className="shrink-0 w-10 h-10 rounded-full bg-white/5 flex items-center justify-center overflow-hidden">
            {icon}
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-base font-semibold text-white truncate">{title}</div>
            <div className="text-sm text-white/60 leading-snug mt-0.5 line-clamp-2">{description}</div>
          </div>
          <div className="sm:ml-auto flex items-center gap-2">
            {secondaryActionLabel && onSecondaryAction && (
              <Button variant="secondary" onClick={onSecondaryAction} aria-label={secondaryAriaLabel || secondaryActionLabel}>
                {secondaryActionLabel}
              </Button>
            )}
            {actionLabel && !disabled ? (
              <Button onClick={onAction} className="min-w-28" aria-label={actionAriaLabel || actionLabel}>
                {actionLabel}
              </Button>
            ) : statusLabel ? (
              <Badge variant="secondary">{statusLabel}</Badge>
            ) : null}
          </div>
        </div>
      </Card>
    );
  }

  // default grid card for backward compatibility
  return (
    <Card
      className={`rounded-lg shadow-sm ${disabled ? 'opacity-70 cursor-not-allowed' : 'transition-transform duration-200 hover:shadow-md hover:scale-[1.01]'} ${className}`}
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
      <CardFooter className="pt-0 flex items-center gap-2">
        {secondaryActionLabel && onSecondaryAction && (
          <Button variant="secondary" onClick={onSecondaryAction} aria-label={secondaryAriaLabel || secondaryActionLabel}>
            {secondaryActionLabel}
          </Button>
        )}
        {actionLabel && !disabled ? (
          <Button onClick={onAction} className="min-w-36" aria-label={actionAriaLabel || actionLabel}>
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
