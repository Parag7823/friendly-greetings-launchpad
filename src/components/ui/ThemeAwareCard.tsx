import React from 'react';
import { cn } from '@/lib/utils';

interface ThemeAwareCardProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'default' | 'danger' | 'success' | 'warning';
  onClick?: () => void;
}

export const ThemeAwareCard = ({ 
  children, 
  className = '', 
  variant = 'default',
  onClick 
}: ThemeAwareCardProps) => {
  const borderColors = {
    default: 'border-white/10 hover:border-white/20',
    danger: 'border-red-500/30 hover:border-red-500/40',
    success: 'border-green-500/30 hover:border-green-500/40',
    warning: 'border-yellow-500/30 hover:border-yellow-500/40'
  };

  return (
    <div 
      className={cn(
        'bg-[#1a1a1a] border rounded-md text-white transition-colors',
        borderColors[variant],
        onClick && 'cursor-pointer',
        className
      )}
      onClick={onClick}
    >
      {children}
    </div>
  );
};
