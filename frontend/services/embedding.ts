import apiClient from './api';

export interface EmbeddingResponse {
  document_id: string;
  embedding_count: number;
  message: string;
}

export async function embedDocument(id: string, force: boolean = false): Promise<EmbeddingResponse> {
  const res = await apiClient.post<EmbeddingResponse>(`/embed/${id}`, null, {
    params: { force },
  });
  return res.data;
}
