import apiClient from './api';

export interface ChunkingResponse {
  document_id: string;
  total_chunks: number;
  message: string;
}

export async function chunkDocument(id: string, force: boolean = false): Promise<ChunkingResponse> {
  const res = await apiClient.post<ChunkingResponse>(`/chunk/${id}`, null, {
    params: { force },
  });
  return res.data;
}
