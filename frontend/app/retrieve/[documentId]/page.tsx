'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRetrieval } from '@/hooks/useRetrieval';
import { useDocuments } from '@/hooks/useDocuments';
import MainLayout from '@/components/layout/MainLayout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import Input from '@/components/ui/Input';
import Skeleton from '@/components/ui/Skeleton';
import { copyToClipboard } from '@/lib/copy';
import { useToast } from '@/providers/ToastProvider';
import {
  ArrowLeft,
  Search,
  Sliders,
  Copy,
  CheckCircle2,
  FileText,
  Clock,
  Sparkles,
  Info,
  AlertCircle,
  PlayCircle
} from 'lucide-react';

interface RetrievalPageProps {
  params: Promise<{ documentId: string }>;
}

export default function RetrievalPlaygroundPage({ params }: RetrievalPageProps) {
  // Unwrap Next 15 params promise
  const { documentId } = React.use(params);

  const { success: toastSuccess } = useToast();
  const { search, results, retrievalTime, totalResults, isLoading } = useRetrieval(documentId);
  const { useStatus } = useDocuments();

  // Fetch status of the document to enforce pipeline completion guard
  const { data: statusData, isLoading: isStatusLoading } = useStatus(documentId);

  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    search(query, topK);
  };

  const handleCopy = async (text: string, id: string) => {
    const success = await copyToClipboard(text);
    if (success) {
      setCopiedId(id);
      toastSuccess('Copied to Clipboard', 'Text segment copied successfully.');
      setTimeout(() => setCopiedId(null), 2000);
    }
  };

  // If status is loading
  if (isStatusLoading) {
    return (
      <MainLayout>
        <div className="max-w-4xl mx-auto space-y-6">
          <Skeleton className="w-full h-8" />
          <Skeleton className="w-full h-40" />
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
                This document is not yet ready for semantic retrieval. You must run and complete all document processing pipeline stages first.
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
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Navigation Breadcrumb */}
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Link href="/dashboard" className="hover:text-foreground">Dashboard</Link>
          <span>/</span>
          <Link href="/documents" className="hover:text-foreground">Documents</Link>
          <span>/</span>
          <span className="text-foreground">Retrieval playground</span>
        </div>

        {/* Header and Back Link */}
        <div className="flex items-center justify-between">
          <Link href="/documents">
            <Button variant="outline" size="sm" className="flex items-center gap-2 cursor-pointer">
              <ArrowLeft className="w-4 h-4" />
              <span>Back to Documents</span>
            </Button>
          </Link>
          <Link href={`/chat/${documentId}`}>
            <Button size="sm" className="cursor-pointer">Go to Chat</Button>
          </Link>
        </div>

        {/* Playground description */}
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-foreground font-sans">Similarity Search Playground</h2>
          <p className="text-muted-foreground text-sm mt-1">
            Query the vector space index directly to inspect ranked text chunks and raw similarity scores.
          </p>
        </div>

        {/* Search controls form */}
        <Card>
          <CardContent className="p-6">
            <form onSubmit={handleSearch} className="space-y-4">
              <div className="flex flex-col md:flex-row gap-4">
                {/* Query Input */}
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Enter search query term here..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="pl-9"
                    disabled={isLoading}
                    required
                  />
                </div>

                {/* Top-K Select */}
                <div className="flex items-center gap-2.5 shrink-0">
                  <Sliders className="w-4 h-4 text-muted-foreground" />
                  <span className="text-xs font-semibold text-muted-foreground select-none">Top K:</span>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value) || 5)}
                    className="w-16 h-10 px-2 rounded-lg border border-input bg-card text-foreground focus:outline-none focus:ring-2 focus:ring-ring text-sm text-center"
                    disabled={isLoading}
                  />
                </div>

                <Button type="submit" loading={isLoading} className="shrink-0 cursor-pointer">
                  Search Vectors
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Results Listings */}
        {results.length > 0 && (
          <div className="space-y-4 animate-fadeIn">
            {/* Search stats summary */}
            <div className="flex items-center justify-between text-xs text-muted-foreground px-1 select-none">
              <span className="flex items-center gap-1">
                <Sparkles className="w-3.5 h-3.5 text-primary" />
                Found {totalResults} matching chunks
              </span>
              <span className="flex items-center gap-1">
                <Clock className="w-3.5 h-3.5" />
                Query time: {retrievalTime.toFixed(4)} seconds
              </span>
            </div>

            {/* Chunks Card List */}
            <div className="space-y-4">
              {results.map((item, index) => (
                <Card key={item.chunk_id} className="relative hover:border-primary/45 transition-colors">
                  <CardHeader className="flex flex-row items-center justify-between bg-muted/20 border-b border-border/40 p-4 shrink-0">
                    <div className="flex items-center gap-2.5">
                      <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/15 text-primary font-bold text-xs">
                        {index + 1}
                      </span>
                      <Badge variant="outline" className="flex items-center gap-1">
                        <FileText className="w-3 h-3" />
                        <span>Page {item.page_number}</span>
                      </Badge>
                    </div>
                    
                    <div className="flex items-center gap-3">
                      <div className="text-xs font-semibold text-muted-foreground flex items-center gap-1 select-none">
                        <Info className="w-3.5 h-3.5 text-primary" />
                        Score:{' '}
                        <span className="text-foreground font-mono font-bold">
                          {item.similarity_score.toFixed(4)}
                        </span>
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="p-1 cursor-pointer h-auto"
                        onClick={() => handleCopy(item.text, item.chunk_id)}
                      >
                        {copiedId === item.chunk_id ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent className="p-4 pt-4">
                    <p className="text-sm leading-relaxed text-foreground whitespace-pre-wrap">
                      {item.text}
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}
      </div>
    </MainLayout>
  );
}
