import apiClient from './api';
import { ENDPOINTS } from '@/constants/api';
import { ChatResponse } from '@/types/Chat';

export async function sendChatMessage(
  id: string,
  payload: { query: string }
): Promise<ChatResponse> {
  const res = await apiClient.post<ChatResponse>(ENDPOINTS.CHAT(id), payload);
  return res.data;
}
