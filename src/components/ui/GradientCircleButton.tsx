import React from 'react';

interface GradientCircleButtonProps {
  state: 'connect' | 'connected';
  label?: string; // visible text displayed to the right if desired by parent; inside circle we show icon/short
  onClick?: () => void;
  disabled?: boolean;
  ariaLabel: string;
  title?: string;
  className?: string;
}

/**
 * Circular button with animated gradient ring and dark inner fill.
 * - Uses a conic-gradient ring rotated slowly (CSS animation) for a flowing effect.
 * - Electric blue glow on hover/focus for premium feel and accessibility.
 */
export const GradientCircleButton: React.FC<GradientCircleButtonProps> = ({
  state,
  onClick,
  disabled,
  ariaLabel,
  title,
  className = '',
}) => {
  const isConnected = state === 'connected';
  return (
    <button
      type={onClick ? 'button' : 'button'}
      onClick={isConnected ? undefined : onClick}
      className={`relative inline-flex items-center justify-center w-10 h-10 rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 ${
        disabled || isConnected ? 'opacity-90 cursor-default' : 'cursor-pointer'
      } ${className}`}
      aria-label={ariaLabel}
      title={title || ariaLabel}
      disabled={disabled || isConnected}
    >
      {/* Animated gradient ring */}
      <span
        className="absolute inset-0 rounded-full p-[2px] animate-spin-slow glow-electric"
        aria-hidden
        style={{
          background:
            'conic-gradient(from 0deg, #00FFFF, #60A5FA, #8B5CF6, #00FFFF)',
          WebkitMask:
            'linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0)',
          WebkitMaskComposite: 'xor' as any,
          maskComposite: 'exclude' as any,
        }}
      >
        <span className="block w-full h-full rounded-full bg-background" />
      </span>

      {/* Inner content */}
      <span className="relative z-10 flex items-center justify-center w-9 h-9 rounded-full bg-secondary text-foreground">
        {isConnected ? (
          // Check icon (inline SVG) for connected state
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="w-5 h-5 text-cyan-300"
            aria-hidden
          >
            <path
              fillRule="evenodd"
              d="M10.28 15.72a.75.75 0 01-1.06 0l-3-3a.75.75 0 111.06-1.06l2.47 2.47 6.47-6.47a.75.75 0 111.06 1.06l-7 7z"
              clipRule="evenodd"
            />
          </svg>
        ) : (
          // Plug icon (inline SVG) for connect state
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="w-5 h-5 text-cyan-200"
            aria-hidden
          >
            <path d="M7 3a1 1 0 000 2h1v2a4 4 0 004 4h0a4 4 0 004-4V5h1a1 1 0 100-2H7z" />
            <path d="M9 19a3 3 0 006 0v-2H9v2z" />
          </svg>
        )}
      </span>
    </button>
  );
};

export default GradientCircleButton;
