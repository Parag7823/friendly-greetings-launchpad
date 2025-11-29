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
 * FIX #3: Uses metallic grey color for motion, reduced font size, smooth animation
 */
export function ThinkingShimmer({
  children = 'Thinking',
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
        'text-transparent text-xs font-medium',
        // Light mode: uses muted color for base, metallic grey for shimmer
        '[--base-color:hsl(var(--muted-foreground))]',
        '[--base-gradient-color:#9ca3af]', // Metallic grey (gray-400)
        // Dark mode: uses muted color for base, metallic grey for shimmer
        'dark:[--base-color:hsl(var(--muted-foreground))]',
        'dark:[--base-gradient-color:#6b7280]', // Metallic grey (gray-500) for dark mode
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
