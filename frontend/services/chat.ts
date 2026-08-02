import apiClient from './api';
import { ENDPOINTS } from '@/constants/api';
import { ChatResponse } from '@/types/Chat';

export async function sendChatMessage(
  id: string,
  payload: { question: string }
): Promise<ChatResponse> {
  const res = await apiClient.post<any>(ENDPOINTS.CHAT(id), payload);
  const data = res.data;
  return {
    answer: data.answer,
    citations: (data.sources || []).map((source: any) => ({
      chunk_id: source.chunk_id,
      document_id: data.document_id,
      page_number: source.start_page,
      text: source.text || '',
      similarity_score: source.score,
      chunk_index: source.chunk_index,
    })),
  };
}
