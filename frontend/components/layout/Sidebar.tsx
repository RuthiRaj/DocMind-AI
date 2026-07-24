'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, UploadCloud, FileText, Activity, Settings, ChevronRight, Database } from 'lucide-react';
import { cn } from '@/lib/cn';
import { ROUTES } from '@/constants/routes';

interface SidebarProps {
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

export default function Sidebar({ isOpen, setIsOpen }: SidebarProps) {
  const pathname = usePathname();

  const menuItems = [
    { label: 'Dashboard', path: ROUTES.DASHBOARD, icon: LayoutDashboard },
    { label: 'Upload PDF', path: ROUTES.UPLOAD, icon: UploadCloud },
    { label: 'Documents', path: ROUTES.DOCUMENTS, icon: FileText },
    { label: 'Health Status', path: ROUTES.HEALTH, icon: Activity },
    { label: 'Maintenance', path: ROUTES.MAINTENANCE, icon: Settings },
  ];

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          onClick={() => setIsOpen(false)}
          className="fixed inset-0 z-30 bg-black/50 md:hidden transition-opacity"
        />
      )}

      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 flex flex-col w-64 border-r border-border bg-card text-card-foreground transform transition-transform duration-300 md:translate-x-0 md:static shrink-0',
          isOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        {/* Brand Header */}
        <div className="flex items-center gap-2.5 h-16 px-6 border-b border-border">
          <Database className="w-6 h-6 text-primary" />
          <span className="font-bold text-xl tracking-tight bg-gradient-to-r from-primary to-violet-500 bg-clip-text text-transparent">
            DocMind AI
          </span>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 px-4 py-6 space-y-1.5 overflow-y-auto">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.path || (item.path !== '/' && pathname.startsWith(item.path));

            return (
              <Link
                key={item.label}
                href={item.path}
                onClick={() => setIsOpen(false)}
                className={cn(
                  'flex items-center justify-between px-4 py-3 rounded-lg text-sm font-medium transition-colors group cursor-pointer',
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <div className="flex items-center gap-3">
                  <Icon className="w-5 h-5 shrink-0" />
                  <span>{item.label}</span>
                </div>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity',
                    isActive && 'opacity-100 text-primary-foreground'
                  )}
                />
              </Link>
            );
          })}
        </nav>

        {/* Footer info */}
        <div className="p-4 border-t border-border bg-muted/40">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>DocMind Version</span>
            <span className="font-mono bg-muted px-1.5 py-0.5 rounded">1.0.0</span>
          </div>
        </div>
      </aside>
    </>
  );
}
