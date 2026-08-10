import apiClient from './api';
import { ENDPOINTS } from '@/constants/api';
import { ChatResponse } from '@/types/Chat';
import { getSessionId, setSessionId } from '@/lib/session';

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
  session_id?: string;
  context_truncated?: boolean;
}

export async function sendChatMessage(
  id: string,
  payload: { question: string }
): Promise<ChatResponse> {
  // Include session_id for conversation memory scoping
  const sessionId = getSessionId(id);
  const requestPayload = {
    ...payload,
    session_id: sessionId,
  };

  const res = await apiClient.post<BackendChatResponse>(ENDPOINTS.CHAT(id), requestPayload);
  const data = res.data;

  // Persist the session_id returned by the server (for first-time requests)
  if (data.session_id) {
    setSessionId(id, data.session_id);
  }

  return {
    answer: data.answer,
    contextTruncated: data.context_truncated ?? false,
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
