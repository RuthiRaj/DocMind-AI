import React from 'react';
import { cn } from '@/lib/cn';

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'secondary' | 'success' | 'destructive' | 'outline' | 'info' | 'warning';
}

export default function Badge({
  className,
  variant = 'default',
  ...props
}: BadgeProps) {
  const baseStyles = 'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2';
  
  const variants = {
    default: 'border-transparent bg-primary text-primary-foreground',
    secondary: 'border-transparent bg-secondary text-secondary-foreground',
    success: 'border-transparent bg-emerald-500/15 text-emerald-600 dark:text-emerald-400',
    destructive: 'border-transparent bg-rose-500/15 text-rose-600 dark:text-rose-400',
    outline: 'border-border text-foreground',
    info: 'border-transparent bg-blue-500/15 text-blue-600 dark:text-blue-400',
    warning: 'border-transparent bg-amber-500/15 text-amber-600 dark:text-amber-400',
  };

  return (
    <div className={cn(baseStyles, variants[variant], className)} {...props} />
  );
}
