'use client';

import { useState } from 'react';
import { queryRetrieval } from '@/services/retrieval';
import { handleApiError } from '@/services/api';
import { RetrievalResponse } from '@/types/Retrieval';
import { useToast } from '@/providers/ToastProvider';

export function useRetrieval(documentId: string) {
  const { error: toastError } = useToast();

  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<RetrievalResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const search = async (query: string, topK?: number) => {
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    try {
      const data = await queryRetrieval(documentId, { query, top_k: topK });
      setResults(data);
    } catch (err: unknown) {
      const parsed = handleApiError(err);
      setError(parsed.message);
      toastError('Retrieval Failed', parsed.message);
    } finally {
      setIsLoading(false);
    }
  };

  return {
    search,
    results: results?.results || [],
    retrievalTime: results?.retrieval_time_seconds || 0,
    totalResults: results?.total_results || 0,
    isLoading,
    error,
  };
}
