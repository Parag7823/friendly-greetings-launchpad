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
        "relative inline-block py-[1px] overflow-hidden rounded-[20px]",
        className
      )} 
      {...props}
    >
      <div
        className={cn(
          "absolute w-[200%] h-[40%] bottom-[-8px] right-[-150%] rounded-full animate-star-movement-bottom z-0",
          "opacity-50 dark:opacity-90 blur-[3px]" 
        )}
        style={{
          background: `radial-gradient(circle, ${defaultColor}, transparent 20%)`,
          animationDuration: speed,
        }}
      />
      <div
        className={cn(
          "absolute w-[200%] h-[40%] top-[-8px] left-[-150%] rounded-full animate-star-movement-top z-0",
          "opacity-50 dark:opacity-90 blur-[3px]"
        )}
        style={{
          background: `radial-gradient(circle, ${defaultColor}, transparent 20%)`,
          animationDuration: speed,
        }}
      />
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
