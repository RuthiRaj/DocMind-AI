'use client';

import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { runMaintenanceCleanup } from '@/services/maintenance';
import { handleApiError } from '@/services/api';
import { CleanupResponse } from '@/types/Api';
import { useToast } from '@/providers/ToastProvider';

export function useMaintenance() {
  const queryClient = useQueryClient();
  const { success: toastSuccess, error: toastError } = useToast();

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<CleanupResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const cleanup = async () => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await runMaintenanceCleanup();
      setResult(data);
      toastSuccess('Maintenance Success', data.message);
      
      // Invalidate queries to refresh lists and dashboard sizes
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['storage-statistics'] });
    } catch (err: unknown) {
      const parsed = handleApiError(err);
      setError(parsed.message);
      toastError('Cleanup Failed', parsed.message);
    } finally {
      setIsLoading(false);
    }
  };

  return {
    cleanup,
    result,
    isLoading,
    error,
  };
}
