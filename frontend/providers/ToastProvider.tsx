'use client';

import React, { createContext, useContext, useState, useCallback } from 'react';
import { X, CheckCircle, AlertCircle, Info, AlertTriangle } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface ToastItem {
  id: string;
  type: ToastType;
  title: string;
  message: string;
}

interface ToastContextType {
  toast: (type: ToastType, title: string, message: string) => void;
  success: (title: string, message: string) => void;
  error: (title: string, message: string) => void;
  info: (title: string, message: string) => void;
  warning: (title: string, message: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const toast = useCallback((type: ToastType, title: string, message: string) => {
    const id = Math.random().toString(36).substring(2, 9);
    setToasts((prev) => [...prev, { id, type, title, message }]);
    
    // Auto-remove after 4 seconds
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const success = useCallback((title: string, message: string) => toast('success', title, message), [toast]);
  const error = useCallback((title: string, message: string) => toast('error', title, message), [toast]);
  const info = useCallback((title: string, message: string) => toast('info', title, message), [toast]);
  const warning = useCallback((title: string, message: string) => toast('warning', title, message), [toast]);

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  return (
    <ToastContext.Provider value={{ toast, success, error, info, warning }}>
      {children}
      
      {/* Toast container */}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 w-full max-w-sm">
        {toasts.map((t) => {
          let Icon = Info;
          let iconColor = 'text-blue-500';
          let borderClass = 'border-blue-500';
          
          if (t.type === 'success') {
            Icon = CheckCircle;
            iconColor = 'text-emerald-500';
            borderClass = 'border-emerald-500';
          } else if (t.type === 'error') {
            Icon = AlertCircle;
            iconColor = 'text-rose-500';
            borderClass = 'border-rose-500';
          } else if (t.type === 'warning') {
            Icon = AlertTriangle;
            iconColor = 'text-amber-500';
            borderClass = 'border-amber-500';
          }

          return (
            <div
              key={t.id}
              className={`flex items-start gap-3 p-4 bg-card text-card-foreground border-l-4 ${borderClass} rounded-lg shadow-lg fade-in`}
              role="alert"
            >
              <Icon className={`w-5 h-5 mt-0.5 shrink-0 ${iconColor}`} />
              <div className="flex-1 min-w-0">
                <p className="font-semibold text-sm truncate">{t.title}</p>
                <p className="text-xs text-muted-foreground mt-0.5">{t.message}</p>
              </div>
              <button
                onClick={() => removeToast(t.id)}
                type="button"
                className="text-muted-foreground hover:text-foreground shrink-0 focus:outline-none"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}
