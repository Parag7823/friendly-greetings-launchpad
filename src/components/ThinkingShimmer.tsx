'use client';
import React, { useMemo, type JSX } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface ThinkingShimmerProps {
  children?: string;
  className?: string;
  duration?: number;
  spread?: number;
}

/**
 * ThinkingShimmer Component
 * Displays an animated shimmer effect while AI is thinking
 * Uses branding colors (primary color) for the shimmer gradient
 */
export function ThinkingShimmer({
  children = 'AI is thinking',
  className,
  duration = 2,
  spread = 2,
}: ThinkingShimmerProps) {
  const MotionComponent = motion('p' as keyof JSX.IntrinsicElements);

  const dynamicSpread = useMemo(() => {
    return children.length * spread;
  }, [children, spread]);

  return (
    <MotionComponent
      className={cn(
        'relative inline-block bg-[length:250%_100%,auto] bg-clip-text',
        'text-transparent',
        // Light mode: uses muted color for base, primary for shimmer
        '[--base-color:hsl(var(--muted-foreground))]',
        '[--base-gradient-color:hsl(var(--primary))]',
        // Dark mode: uses muted color for base, primary for shimmer
        'dark:[--base-color:hsl(var(--muted-foreground))]',
        'dark:[--base-gradient-color:hsl(var(--primary))]',
        '[--bg:linear-gradient(90deg,#0000_calc(50%-var(--spread)),var(--base-gradient-color),#0000_calc(50%+var(--spread)))]',
        'dark:[--bg:linear-gradient(90deg,#0000_calc(50%-var(--spread)),var(--base-gradient-color),#0000_calc(50%+var(--spread)))]',
        className
      )}
      initial={{ backgroundPosition: '100% center' }}
      animate={{ backgroundPosition: '0% center' }}
      transition={{
        repeat: Infinity,
        duration,
        ease: 'linear',
      }}
      style={
        {
          '--spread': `${dynamicSpread}px`,
          backgroundImage: `var(--bg), linear-gradient(var(--base-color), var(--base-color))`,
        } as React.CSSProperties
      }
    >
      {children}
    </MotionComponent>
  );
}
