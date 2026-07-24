'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  listDocuments,
  getDocumentDetails,
  getDocumentStatus,
  deleteDocument,
  getStorageStatistics,
} from '@/services/documents';
import { handleApiError } from '@/services/api';
import { useToast } from '@/providers/ToastProvider';

export function useDocuments(params?: {
  skip?: number;
  limit?: number;
  sort_by?: string;
  descending?: boolean;
  status_filter?: string | null;
}) {
  const queryClient = useQueryClient();
  const { success: toastSuccess, error: toastError } = useToast();

  // 1. Documents list query
  const listQuery = useQuery({
    queryKey: ['documents', params],
    queryFn: () => listDocuments(params),
  });

  // 2. Storage statistics query
  const statsQuery = useQuery({
    queryKey: ['storage-statistics'],
    queryFn: getStorageStatistics,
  });

  // 3. Document details query wrapper (conditional)
  const useDetails = (id: string) => {
    return useQuery({
      queryKey: ['document-details', id],
      queryFn: () => getDocumentDetails(id),
      enabled: !!id,
    });
  };

  // 4. Document pipeline status query wrapper (conditional)
  const useStatus = (id: string, autoRefresh: boolean = false) => {
    return useQuery({
      queryKey: ['document-status', id],
      queryFn: () => getDocumentStatus(id),
      enabled: !!id,
      refetchInterval: autoRefresh ? 3000 : false, // Poll status every 3 seconds if active
    });
  };

  // 5. Document delete mutation
  const deleteMutation = useMutation({
    mutationFn: deleteDocument,
    onSuccess: (data, id) => {
      toastSuccess('Document Deleted', `Document ${id} successfully removed from storage.`);
      // Invalidate queries to refresh lists and dashboard
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['storage-statistics'] });
    },
    onError: (err) => {
      const parsed = handleApiError(err);
      toastError('Deletion Failed', parsed.message);
    },
  });

  return {
    documents: listQuery.data?.documents || [],
    totalCount: listQuery.data?.total_count || 0,
    listLoading: listQuery.isLoading,
    listError: listQuery.error ? handleApiError(listQuery.error) : null,
    refetchList: listQuery.refetch,

    stats: statsQuery.data || null,
    statsLoading: statsQuery.isLoading,
    statsError: statsQuery.error ? handleApiError(statsQuery.error) : null,
    refetchStats: statsQuery.refetch,

    useDetails,
    useStatus,

    deleteDoc: deleteMutation.mutateAsync,
    isDeleting: deleteMutation.isPending,
  };
}
