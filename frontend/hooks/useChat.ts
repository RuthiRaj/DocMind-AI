'use client';

import { useState } from 'react';
import { sendChatMessage } from '@/services/chat';
import { handleApiError } from '@/services/api';
import { Message } from '@/types/Chat';
import { useToast } from '@/providers/ToastProvider';

export function useChat(documentId: string) {
  const { error: toastError } = useToast();

  const [messages, setMessages] = useState<Message[]>([]);
  const [isSending, setIsSending] = useState(false);

  const sendMessage = async (text: string) => {
    if (!text.trim()) return;

    const userMessage: Message = {
      id: Math.random().toString(36).substring(2, 9),
      sender: 'user',
      text,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsSending(true);

    try {
      const response = await sendChatMessage(documentId, { question: text });
      
      const assistantMessage: Message = {
        id: Math.random().toString(36).substring(2, 9),
        sender: 'assistant',
        text: response.answer,
        timestamp: new Date().toISOString(),
        citations: response.citations,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      const parsed = handleApiError(err);
      
      const errorMessage: Message = {
        id: Math.random().toString(36).substring(2, 9),
        sender: 'assistant',
        text: parsed.message || 'An error occurred while answering your request.',
        timestamp: new Date().toISOString(),
        isError: true,
      };

      setMessages((prev) => [...prev, errorMessage]);
      toastError('Chat Error', parsed.message);
    } finally {
      setIsSending(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return {
    messages,
    sendMessage,
    isSending,
    clearChat,
  };
}
