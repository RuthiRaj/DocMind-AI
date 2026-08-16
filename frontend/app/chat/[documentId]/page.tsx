'use client';

import React, { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useChat } from '@/hooks/useChat';
import { useDocuments } from '@/hooks/useDocuments';
import MainLayout from '@/components/layout/MainLayout';
import { Card, CardContent } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Dialog from '@/components/ui/Dialog';
import Skeleton from '@/components/ui/Skeleton';
import { copyToClipboard } from '@/lib/copy';
import { useToast } from '@/providers/ToastProvider';
import { CHAT_SUGGESTIONS } from '@/constants/messages';
import {
  ArrowLeft,
  Send,
  MessageSquare,
  Sparkles,
  Copy,
  CheckCircle2,
  Bookmark,
  FileText,
  Trash2,
  Compass,
  AlertCircle,
  PlayCircle
} from 'lucide-react';

interface ChatPageProps {
  params: Promise<{ documentId: string }>;
}

export default function ChatPage({ params }: ChatPageProps) {
  // Unwrap Next 15 params promise
  const { documentId } = React.use(params);

  const { success: toastSuccess } = useToast();
  const { messages, sendMessage, isSending, clearChat } = useChat(documentId);
  const { useStatus } = useDocuments();
  
  // Fetch status of the document to enforce pipeline completion guard
  const { data: statusData, isLoading: isStatusLoading } = useStatus(documentId);

  const [input, setInput] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  
  // Active citation details modal state
  const [activeCitation, setActiveCitation] = useState<{
    text: string;
    startPage: number;
    endPage: number;
    score: number;
    chunkId: string;
  } | null>(null);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isSending) return;
    const textToSend = input;
    setInput('');
    await sendMessage(textToSend);
  };

  const handleSuggestionClick = async (suggestion: string) => {
    setInput('');
    await sendMessage(suggestion);
  };

  const handleCopyMessage = async (text: string, id: string) => {
    const success = await copyToClipboard(text);
    if (success) {
      setCopiedId(id);
      toastSuccess('Copied Answer', 'Answer content successfully copied.');
      setTimeout(() => setCopiedId(null), 2000);
    }
  };

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSending]);

  // If status is loading
  if (isStatusLoading) {
    return (
      <MainLayout>
        <div className="max-w-4xl mx-auto space-y-6 flex flex-col h-[calc(100vh-130px)]">
          <Skeleton className="w-full h-8" />
          <Skeleton className="w-full h-full" />
        </div>
      </MainLayout>
    );
  }

  // Enforce pipeline completion guard
  if (!statusData || statusData.chat_ready !== true) {
    return (
      <MainLayout>
        <div className="max-w-md mx-auto py-12 space-y-6 text-center animate-fadeIn">
          <Card className="border-rose-500/20 bg-rose-500/5">
            <CardContent className="p-8 flex flex-col items-center justify-center space-y-4">
              <AlertCircle className="w-12 h-12 text-rose-500" />
              <h3 className="text-lg font-bold text-foreground">Pipeline Incomplete</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                This document is not yet ready for chat. You must run and complete all document processing pipeline stages first.
              </p>
              <div className="flex gap-2 pt-2">
                <Link href="/documents">
                  <Button variant="outline" size="sm">Back to Documents</Button>
                </Link>
                <Link href={`/documents/${documentId}/pipeline`}>
                  <Button size="sm" className="flex items-center gap-1.5 cursor-pointer">
                    <PlayCircle className="w-4 h-4" />
                    <span>Run Pipeline</span>
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      </MainLayout>
    );
  }

  return (
    <MainLayout>
      <div className="max-w-4xl mx-auto space-y-6 flex flex-col h-[calc(100vh-130px)]">
        {/* Navigation Breadcrumb */}
        <div className="flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Link href="/dashboard" className="hover:text-foreground">Dashboard</Link>
            <span>/</span>
            <Link href="/documents" className="hover:text-foreground">Documents</Link>
            <span>/</span>
            <span className="text-foreground">AI Chat</span>
          </div>

          <div className="flex gap-2">
            <Link href="/documents">
              <Button variant="outline" size="sm" className="flex items-center gap-2 cursor-pointer">
                <ArrowLeft className="w-4 h-4" />
                <span>Back</span>
              </Button>
            </Link>
            {messages.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                className="flex items-center gap-1.5 text-rose-500 hover:bg-rose-500/10 cursor-pointer"
                onClick={clearChat}
              >
                <Trash2 className="w-4 h-4" />
                <span>Clear Chat</span>
              </Button>
            )}
          </div>
        </div>

        {/* Conversation Box */}
        <Card className="flex-1 flex flex-col overflow-hidden min-h-[400px]">
          <CardContent className="flex-1 overflow-y-auto p-6 space-y-6 min-w-0">
            {messages.length === 0 ? (
              /* Empty state suggestions */
              <div className="flex flex-col items-center justify-center h-full text-center max-w-lg mx-auto space-y-6 fade-in">
                <div className="p-4 bg-primary/10 text-primary rounded-full">
                  <MessageSquare className="w-10 h-10 animate-bounce" />
                </div>
                <div className="space-y-2">
                  <h3 className="font-bold text-lg text-foreground">Query Grounded AI Chat</h3>
                  <p className="text-sm text-muted-foreground">
                    Ask questions about this document. The system retrieves relevant chunks using FAISS cosine similarity and compiles grounded answers containing citation sources.
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-2.5 w-full pt-4 text-left">
                  <span className="text-xs font-semibold text-muted-foreground flex items-center gap-1.5 px-1 select-none">
                    <Compass className="w-3.5 h-3.5 text-primary" />
                    Suggested Questions:
                  </span>
                  {CHAT_SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => handleSuggestionClick(suggestion)}
                      type="button"
                      disabled={isSending}
                      className="p-3 bg-muted/40 hover:bg-muted text-xs text-foreground font-semibold border border-border/40 rounded-xl text-left transition-colors cursor-pointer disabled:opacity-50 disabled:pointer-events-none"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              /* Message logs list */
              <div className="space-y-6">
                {messages.map((m) => {
                  const isAssistant = m.sender === 'assistant';
                  return (
                    <div
                      key={m.id}
                      className={`flex gap-4 fade-in ${
                        isAssistant ? 'justify-start' : 'justify-end'
                      }`}
                    >
                      {/* Avatar icon */}
                      {isAssistant && (
                        <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/15 text-primary shrink-0 font-bold select-none">
                          <Sparkles className="w-4 h-4" />
                        </div>
                      )}

                      <div className="flex flex-col gap-2 max-w-[85%]">
                        {/* Bubble content */}
                        <div
                          className={`p-4 rounded-2xl border text-sm ${
                            isAssistant
                              ? m.isError
                                ? 'bg-rose-500/10 border-rose-500/20 text-rose-700 dark:text-rose-400'
                                : 'bg-card border-border text-foreground'
                              : 'bg-primary text-primary-foreground border-transparent'
                          }`}
                        >
                          {isAssistant && !m.isError ? (
                            <div className="prose dark:prose-invert max-w-none text-sm leading-relaxed prose-p:leading-relaxed prose-pre:bg-muted prose-pre:p-3 prose-pre:rounded-lg">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {m.text}
                              </ReactMarkdown>
                            </div>
                          ) : (
                            <p className="whitespace-pre-wrap leading-relaxed">{m.text}</p>
                          )}
                        </div>

                        {/* Citation cards and Copy triggers */}
                        {isAssistant && !m.isError && (
                          <div className="flex flex-wrap gap-2.5 items-center pl-1">
                            {/* Copy button */}
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 text-xs px-2 cursor-pointer flex items-center gap-1 text-muted-foreground hover:text-foreground"
                              onClick={() => handleCopyMessage(m.text, m.id)}
                            >
                              {copiedId === m.id ? (
                                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                              ) : (
                                <Copy className="w-3.5 h-3.5" />
                              )}
                              <span>{copiedId === m.id ? 'Copied' : 'Copy'}</span>
                            </Button>

                            {/* Citations badges */}
                            {m.citations && m.citations.length > 0 && (
                              <div className="flex flex-wrap gap-1.5 items-center">
                                <span className="text-xs text-muted-foreground flex items-center gap-1 font-semibold select-none mr-1">
                                  <Bookmark className="w-3 h-3 text-primary" />
                                  Sources:
                                </span>
                                {m.citations.map((c) => {
                                  const pageLabel = c.start_page === c.end_page ? `Page ${c.start_page}` : `Page ${c.start_page}-${c.end_page}`;
                                  return (
                                    <button
                                      key={c.chunk_id}
                                      onClick={() =>
                                        setActiveCitation({
                                          text: c.text,
                                          startPage: c.start_page,
                                          endPage: c.end_page,
                                          score: c.similarity_score,
                                          chunkId: c.chunk_id,
                                        })
                                      }
                                      type="button"
                                      className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border border-primary/25 bg-primary/5 hover:bg-primary/10 text-primary text-xs font-semibold cursor-pointer transition-colors"
                                    >
                                      <FileText className="w-3 h-3" />
                                      <span>{pageLabel}</span>
                                    </button>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}

                {/* Loading typing indicator */}
                {isSending && (
                  <div className="flex gap-4 fade-in justify-start">
                    <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/15 text-primary shrink-0 font-bold select-none">
                      <Sparkles className="w-4 h-4" />
                    </div>
                    <div className="p-4 rounded-2xl bg-card border border-border/80 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce delay-75" />
                      <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce delay-150" />
                      <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce delay-300" />
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
            )}
          </CardContent>

          {/* Prompt footer entry forms */}
          <div className="p-4 border-t border-border shrink-0 bg-muted/20">
            <form onSubmit={handleSend} className="flex gap-2">
              <Input
                placeholder={isSending ? 'AI is thinking...' : 'Ask a question about the document...'}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isSending}
                required
              />
              <Button type="submit" loading={isSending} className="cursor-pointer">
                {!isSending && <Send className="w-4 h-4" />}
              </Button>
            </form>
          </div>
        </Card>

        {/* Citation source detail viewer modal */}
        <Dialog
          isOpen={activeCitation !== null}
          onClose={() => setActiveCitation(null)}
          title={`Citation Source - ${activeCitation ? (activeCitation.startPage === activeCitation.endPage ? `Page ${activeCitation.startPage}` : `Pages ${activeCitation.startPage}–${activeCitation.endPage}`) : ''}`}
        >
          {activeCitation && (
            <div className="space-y-4">
              <div className="flex items-center justify-between text-xs text-muted-foreground bg-muted p-2 rounded border border-border/30">
                <span className="font-mono">Chunk: {activeCitation.chunkId.slice(0, 8)}...</span>
                <span>Similarity Score: <strong className="text-foreground font-mono">{activeCitation.score.toFixed(4)}</strong></span>
              </div>
              <div className="p-4 bg-muted/20 border border-border/40 rounded-xl max-h-[40vh] overflow-y-auto">
                <p className="text-sm leading-relaxed text-foreground whitespace-pre-wrap">
                  {activeCitation.text}
                </p>
              </div>
              <div className="flex justify-end pt-1">
                <Button size="sm" onClick={() => setActiveCitation(null)}>
                  Close
                </Button>
              </div>
            </div>
          )}
        </Dialog>
      </div>
    </MainLayout>
  );
}
