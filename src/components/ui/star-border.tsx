import { cn } from "@/lib/utils"
import { ElementType, ComponentPropsWithoutRef } from "react"

interface StarBorderProps<T extends ElementType> {
  as?: T
  color?: string
  speed?: string
  className?: string
  children: React.ReactNode
}

export function StarBorder<T extends ElementType = "button">({
  as,
  className,
  color,
  speed = "6s",
  children,
  ...props
}: StarBorderProps<T> & Omit<ComponentPropsWithoutRef<T>, keyof StarBorderProps<T>>) {
  const Component = as || "button"
  const defaultColor = color || "hsl(var(--foreground))"

  return (
    <Component 
      className={cn(
        "relative inline-block overflow-hidden rounded-[20px]",
        className
      )} 
      {...props}
    >
      {/* Animated gradient border line */}
      <div
        className="absolute inset-0 rounded-[20px] opacity-75 animate-border-slide"
        style={{
          background: `linear-gradient(90deg, 
            transparent 0%, 
            transparent 40%, 
            ${defaultColor} 50%, 
            transparent 60%, 
            transparent 100%)`,
          backgroundSize: '200% 100%',
          animationDuration: speed,
          padding: '1px',
          WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
          WebkitMaskComposite: 'xor',
          maskComposite: 'exclude',
        }}
      />
      
      {/* Button content */}
      <div className={cn(
        "relative z-10 border text-foreground text-center py-2 px-4 rounded-[20px]",
        "bg-gradient-to-b from-background via-background to-muted/50 border-border/60",
        "dark:from-zinc-900 dark:via-zinc-900 dark:to-zinc-800 dark:border-zinc-700",
        "transition-all duration-200 hover:from-muted/20 hover:to-muted/40",
        "shadow-sm hover:shadow-md"
      )}>
        {children}
      </div>
    </Component>
  )
}
