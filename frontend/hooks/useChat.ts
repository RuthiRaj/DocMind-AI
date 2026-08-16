'use client';

import { useState, useRef } from 'react';
import { sendChatMessage } from '@/services/chat';
import { handleApiError } from '@/services/api';
import { Message } from '@/types/Chat';
import { useToast } from '@/providers/ToastProvider';
import { clearSessionId } from '@/lib/session';

export function useChat(documentId: string) {
  const { error: toastError } = useToast();

  const [messages, setMessages] = useState<Message[]>([]);
  const [isSending, setIsSending] = useState(false);

  // Synchronous ref guard — prevents duplicate POST /chat requests from React
  // StrictMode double-invocations, rapid clicks, or suggestion chip double-taps.
  // Unlike setState, ref mutation is synchronous and takes effect immediately,
  // so the second caller sees isSendingRef.current === true before the first
  // await resolves.
  const isSendingRef = useRef(false);

  const sendMessage = async (text: string) => {
    if (!text.trim()) return;

    // Synchronous guard — checked and set before any await.
    // Prevents a second concurrent invocation (e.g. React StrictMode
    // double-effect, rapid click, or suggestion chip double-tap) from
    // dispatching a second POST /chat/{document_id} request.
    if (isSendingRef.current) return;
    isSendingRef.current = true;

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
    } catch (err: unknown) {
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
      // Reset both the ref and the state so the next user action is accepted.
      isSendingRef.current = false;
      setIsSending(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    clearSessionId(documentId);
  };

  return {
    messages,
    sendMessage,
    isSending,
    clearChat,
  };
}
