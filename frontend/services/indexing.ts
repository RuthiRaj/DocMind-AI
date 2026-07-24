import apiClient from './api';

export interface IndexingResponse {
  document_id: string;
  total_indexed_vectors: number;
  message: string;
}

export async function indexDocument(id: string, force: boolean = false): Promise<IndexingResponse> {
  const res = await apiClient.post<IndexingResponse>(`/index/${id}`, null, {
    params: { force },
  });
  return res.data;
}
