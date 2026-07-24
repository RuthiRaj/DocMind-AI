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

  const apiError = query.error ? handleApiError(query.error) : null;

  useEffect(() => {
    if (apiError) {
      toastError('System Diagnostics Offline', apiError.message);
    }
  }, [apiError, toastError]);

  return {
    health: query.data || null,
    isLoading: query.isLoading,
    isError: query.isError,
    error: apiError,
    refetch: query.refetch,
  };
}
