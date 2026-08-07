'use client';

import React from 'react';
import { usePathname } from 'next/navigation';
import { Menu, Activity } from 'lucide-react';
import ThemeToggle from '@/components/common/ThemeToggle';
import { ROUTES } from '@/constants/routes';

interface NavbarProps {
  setSidebarOpen: (open: boolean) => void;
}

export default function Navbar({ setSidebarOpen }: NavbarProps) {
  const pathname = usePathname();

  // Deduce title based on pathname
  const getPageTitle = () => {
    if (pathname === ROUTES.LANDING) return 'Welcome';
    if (pathname.startsWith(ROUTES.DASHBOARD)) return 'System Dashboard';
    if (pathname.startsWith(ROUTES.UPLOAD)) return 'Ingest PDF Documents';
    if (pathname.startsWith('/documents') && pathname.includes('/pipeline')) return 'Pipeline Execution Progress';
    if (pathname.startsWith('/documents')) return 'Document Lifecycle Management';
    if (pathname.startsWith('/retrieve')) return 'Similarity Search Playground';
    if (pathname.startsWith('/chat')) return 'Grounded AI Chat (RAG)';
    if (pathname.startsWith(ROUTES.HEALTH)) return 'System Health Diagnostics';
    if (pathname.startsWith(ROUTES.MAINTENANCE)) return 'System Maintenance Dashboard';
    return 'DocMind AI';
  };

  return (
    <header className="flex items-center justify-between h-16 px-6 border-b border-border bg-card text-card-foreground shrink-0 sticky top-0 z-30">
      <div className="flex items-center gap-4">
        {/* Toggle mobile sidebar */}
        <button
          onClick={() => setSidebarOpen(true)}
          type="button"
          className="p-2 -ml-2 rounded-lg hover:bg-muted text-muted-foreground hover:text-foreground md:hidden cursor-pointer"
          aria-label="Open sidebar menu"
        >
          <Menu className="w-6 h-6" />
        </button>

        <h1 className="font-semibold text-lg md:text-xl text-foreground truncate">
          {getPageTitle()}
        </h1>
      </div>

      <div className="flex items-center gap-4">
        {/* Simple System Online indicator */}
        <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-semibold rounded-full border border-emerald-500/25">
          <Activity className="w-3.5 h-3.5 animate-pulse" />
          <span>System Online</span>
        </div>

        <ThemeToggle />
      </div>
    </header>
  );
}
