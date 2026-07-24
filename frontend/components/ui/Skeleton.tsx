import React from 'react';
import { cn } from '@/lib/cn';

export default function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('animate-pulse rounded-md bg-muted/65', className)}
      {...props}
    />
  );
}
