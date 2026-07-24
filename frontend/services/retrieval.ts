import apiClient from './api';
import { ENDPOINTS } from '@/constants/api';
import { RetrievalResponse } from '@/types/Retrieval';

export async function queryRetrieval(
  id: string,
  payload: { query: string; top_k?: number }
): Promise<RetrievalResponse> {
  const res = await apiClient.post<RetrievalResponse>(ENDPOINTS.RETRIEVAL(id), payload);
  return res.data;
}
