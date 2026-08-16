'use client';

import { useQuery } from '@tanstack/react-query';
import { getSystemHealth } from '@/services/health';
import { handleApiError } from '@/services/api';
import { useToast } from '@/providers/ToastProvider';
import { useEffect } from 'react';

export function useHealth() {
  const { error: toastError } = useToast();
  
  const query = useQuery({
    queryKey: ['system-health'],
    queryFn: getSystemHealth,
    refetchInterval: 10000, // Refresh health status check every 10 seconds
  });

  const isError = query.isError;
  const rawError = query.error;

  useEffect(() => {
    if (isError && rawError) {
      const parsed = handleApiError(rawError);
      toastError('System Diagnostics Offline', parsed.message);
    }
  }, [isError, rawError, toastError]);

  const apiError = query.error ? handleApiError(query.error) : null;

  return {
    health: query.data || null,
    isLoading: query.isLoading,
    isError: query.isError,
    error: apiError,
    refetch: query.refetch,
  };
}
