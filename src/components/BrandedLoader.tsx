import React from 'react';
import { Loader2 } from 'lucide-react';

interface BrandedLoaderProps {
  text?: string;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'spinner' | 'pulse' | 'dots';
}

/**
 * BrandedLoader: Premium loading indicator with Copper accent
 * Replaces generic gray spinners with branded, contextual loading states
 */
export const BrandedLoader: React.FC<BrandedLoaderProps> = ({
  text = 'Loading...',
  size = 'md',
  variant = 'spinner'
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8'
  };

  const textSizeClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base'
  };

  if (variant === 'dots') {
    return (
      <div className="flex flex-col items-center justify-center gap-3">
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
          <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
          <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
        {text && <p className={`${textSizeClasses[size]} text-muted-foreground font-medium`}>{text}</p>}
      </div>
    );
  }

  if (variant === 'pulse') {
    return (
      <div className="flex flex-col items-center justify-center gap-3">
        <div className={`${sizeClasses[size]} rounded-full border-2 border-primary/30 border-t-primary animate-spin`} />
        {text && <p className={`${textSizeClasses[size]} text-muted-foreground font-medium`}>{text}</p>}
      </div>
    );
  }

  // Default spinner variant
  return (
    <div className="flex flex-col items-center justify-center gap-3">
      <Loader2 className={`${sizeClasses[size]} text-primary animate-spin`} />
      {text && <p className={`${textSizeClasses[size]} text-muted-foreground font-medium`}>{text}</p>}
    </div>
  );
};
