import apiClient from './api';
import { ENDPOINTS } from '@/constants/api';
import { ChatResponse } from '@/types/Chat';

interface BackendSource {
  chunk_id: string;
  chunk_index: number;
  last_chunk_index?: number | null;
  score: number;
  start_page: number;
  end_page: number;
  text?: string;
}

interface BackendChatResponse {
  answer: string;
  document_id: string;
  sources?: BackendSource[];
}

export async function sendChatMessage(
  id: string,
  payload: { question: string }
): Promise<ChatResponse> {
  const res = await apiClient.post<BackendChatResponse>(ENDPOINTS.CHAT(id), payload);
  const data = res.data;
  return {
    answer: data.answer,
    citations: (data.sources || []).map((source: BackendSource) => ({
      chunk_id: source.chunk_id,
      document_id: data.document_id,
      page_number: source.start_page,
      text: source.text || '',
      similarity_score: source.score,
      chunk_index: source.chunk_index,
    })),
  };
}
