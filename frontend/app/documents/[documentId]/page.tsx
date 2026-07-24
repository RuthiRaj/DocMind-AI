'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useDocuments } from '@/hooks/useDocuments';
import MainLayout from '@/components/layout/MainLayout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import Skeleton from '@/components/ui/Skeleton';
import { formatDate } from '@/lib/formatDate';
import { formatBytes } from '@/lib/formatBytes';
import {
  FileText,
  Activity,
  Layers,
  Database,
  ArrowLeft,
  ChevronRight,
  TrendingUp,
  Sliders,
  Sparkles,
  Info
} from 'lucide-react';


interface DetailsPageProps {
  params: Promise<{ documentId: string }>;
}

type TabType = 'metadata' | 'status' | 'chunking' | 'indexing';

export default function DocumentDetailsPage({ params }: DetailsPageProps) {
  // Unwrap Next 15 params promise
  const { documentId } = React.use(params);

  const { useDetails } = useDocuments();
  const { data: details, isLoading, isError, refetch } = useDetails(documentId);

  const [activeTab, setActiveTab] = useState<TabType>('metadata');

  const tabs: { key: TabType; label: string; icon: any }[] = [
    { key: 'metadata', label: 'File Metadata', icon: FileText },
    { key: 'status', label: 'Pipeline Stage Status', icon: Activity },
    { key: 'chunking', label: 'Smart Chunking Stats', icon: Layers },
    { key: 'indexing', label: 'FAISS Vectors Stats', icon: Database },
  ];

  return (
    <MainLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Navigation Breadcrumb */}
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Link href="/dashboard" className="hover:text-foreground">Dashboard</Link>
          <span>/</span>
          <Link href="/documents" className="hover:text-foreground">Documents</Link>
          <span>/</span>
          <span className="text-foreground">Metadata details</span>
        </div>

        {/* Back Link and Action */}
        <div className="flex items-center justify-between">
          <Link href="/documents">
            <Button variant="outline" size="sm" className="flex items-center gap-2 cursor-pointer">
              <ArrowLeft className="w-4 h-4" />
              <span>Back to Documents</span>
            </Button>
          </Link>

          {details?.status?.indexing_status === 'completed' && (
            <Link href={`/chat/${documentId}`}>
              <Button size="sm" className="flex items-center gap-2 cursor-pointer">
                <span>Start Chat</span>
                <ChevronRight className="w-4 h-4" />
              </Button>
            </Link>
          )}
        </div>

        {/* Main Grid View */}
        {isLoading ? (
          <div className="space-y-6">
            <Skeleton className="w-full h-16" />
            <Skeleton className="w-full h-80" />
          </div>
        ) : isError || !details ? (
          <Card className="border-rose-500/30 bg-rose-500/5">
            <CardContent className="p-12 text-center flex flex-col items-center justify-center space-y-3">
              <FileText className="w-12 h-12 text-rose-500" />
              <h3 className="font-bold text-rose-700 dark:text-rose-400">Failed to load details</h3>
              <p className="text-sm text-rose-500/90 max-w-sm">
                Document folder or metadata files might be corrupted or missing.
              </p>
              <Button size="sm" onClick={() => refetch()}>
                Retry Loading
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-6">
            {/* Header info */}
            <Card>
              <CardContent className="p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="space-y-1">
                  <h3 className="font-bold text-xl text-foreground">
                    {details.metadata?.filename || 'original.pdf'}
                  </h3>
                  <p className="text-xs text-muted-foreground font-mono truncate max-w-[300px]" title={documentId}>
                    ID: {documentId}
                  </p>
                </div>
                <Badge variant={details.status?.indexing_status === 'completed' ? 'success' : 'warning'}>
                  {details.status?.indexing_status === 'completed' ? 'Indexed & Ready' : 'Processing'}
                </Badge>
              </CardContent>
            </Card>

            {/* Tab switch buttons */}
            <div className="flex border-b border-border overflow-x-auto gap-2 shrink-0">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.key;
                return (
                  <button
                    key={tab.key}
                    onClick={() => setActiveTab(tab.key)}
                    className={`flex items-center gap-2 px-4 py-3 text-sm font-semibold border-b-2 transition-colors cursor-pointer select-none whitespace-nowrap ${
                      isActive
                        ? 'border-primary text-primary'
                        : 'border-transparent text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Active Tab Contents */}
            <Card className="min-h-[300px]">
              <CardContent className="p-6">
                {/* 1. Metadata Tab */}
                {activeTab === 'metadata' && (
                  <div className="space-y-6 animate-fadeIn">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 text-sm">
                      <div className="space-y-1">
                        <span className="text-xs text-muted-foreground block font-medium">Filename</span>
                        <span className="text-foreground font-semibold">{details.metadata?.filename || 'N/A'}</span>
                      </div>
                      <div className="space-y-1">
                        <span className="text-xs text-muted-foreground block font-medium">Upload Date</span>
                        <span className="text-foreground font-semibold">{formatDate(details.metadata?.upload_time)}</span>
                      </div>
                      <div className="space-y-1">
                        <span className="text-xs text-muted-foreground block font-medium">File Size</span>
                        <span className="text-foreground font-semibold">{formatBytes(details.metadata?.file_size || 0)}</span>
                      </div>
                      <div className="space-y-1">
                        <span className="text-xs text-muted-foreground block font-medium">Total Pages</span>
                        <span className="text-foreground font-semibold">{details.metadata?.total_pages ?? 'N/A'}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* 2. Pipeline Status Tab */}
                {activeTab === 'status' && (
                  <div className="space-y-6 animate-fadeIn">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                      {[
                        { label: 'Upload stage status', status: details.status?.upload_status },
                        { label: 'PDF parser page processing', status: details.status?.processing_status },
                        { label: 'Text chunking segmenting', status: details.status?.chunking_status },
                        { label: 'Embeddings models encoding', status: details.status?.embedding_status },
                        { label: 'FAISS Index compile mapping', status: details.status?.indexing_status },
                      ].map((item) => (
                        <div key={item.label} className="flex items-center justify-between p-3 rounded-lg bg-muted/30 border border-border/40">
                          <span className="text-muted-foreground font-medium">{item.label}</span>
                          <Badge
                            variant={
                              item.status === 'completed'
                                ? 'success'
                                : item.status === 'failed'
                                ? 'destructive'
                                : 'warning'
                            }
                          >
                            {item.status || 'pending'}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 3. Chunking Tab */}
                {activeTab === 'chunking' && (
                  <div className="space-y-6 animate-fadeIn">
                    {details.chunk_statistics ? (
                      <div className="space-y-6">
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-center">
                          <div className="p-4 bg-muted/30 rounded-xl border border-border/40">
                            <span className="text-2xl font-extrabold text-foreground">
                              {details.chunk_statistics.total_chunks ?? 0}
                            </span>
                            <span className="text-xs text-muted-foreground block mt-1">Total Chunks</span>
                          </div>
                          <div className="p-4 bg-muted/30 rounded-xl border border-border/40">
                            <span className="text-2xl font-extrabold text-foreground">
                              {details.chunk_statistics.average_chunk_size ?? 0}
                            </span>
                            <span className="text-xs text-muted-foreground block mt-1">Avg Character Size</span>
                          </div>
                          <div className="p-4 bg-muted/30 rounded-xl border border-border/40">
                            <span className="text-2xl font-extrabold text-foreground">
                              {details.chunk_statistics.max_chunk_size ?? 0}
                            </span>
                            <span className="text-xs text-muted-foreground block mt-1">Max Character Size</span>
                          </div>
                        </div>

                        <div className="space-y-3 pt-2">
                          <h4 className="font-semibold text-sm text-foreground flex items-center gap-1.5">
                            <Sliders className="w-4 h-4 text-primary" />
                            Chunker Configuration Parameters
                          </h4>
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
                            <div className="p-3 bg-muted/20 border border-border/40 rounded flex justify-between">
                              <span className="text-muted-foreground">Target Chunk Character Limit</span>
                              <span className="font-semibold">{details.chunk_statistics.chunk_size ?? 800} chars</span>
                            </div>
                            <div className="p-3 bg-muted/20 border border-border/40 rounded flex justify-between">
                              <span className="text-muted-foreground">Overlap Character Count</span>
                              <span className="font-semibold">{details.chunk_statistics.chunk_overlap ?? 150} chars</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Info className="w-5 h-5 text-indigo-500" />
                        <span>No smart chunking statistics have been recorded yet.</span>
                      </div>
                    )}
                  </div>
                )}

                {/* 4. Indexing Tab */}
                {activeTab === 'indexing' && (
                  <div className="space-y-6 animate-fadeIn">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 text-sm">
                      <div className="space-y-4">
                        <h4 className="font-semibold text-sm text-foreground flex items-center gap-1.5">
                          <Sparkles className="w-4 h-4 text-primary" />
                          Embedding Engine Definition
                        </h4>
                        {details.embedding_metadata ? (
                          <div className="space-y-3.5">
                            <div className="flex justify-between border-b border-border/60 pb-2">
                              <span className="text-muted-foreground">Model Path</span>
                              <span className="font-semibold">{details.embedding_metadata.model_name || 'BAAI/bge-small-en-v1.5'}</span>
                            </div>
                            <div className="flex justify-between border-b border-border/60 pb-2">
                              <span className="text-muted-foreground">Vector Dimensions</span>
                              <span className="font-semibold">{details.embedding_metadata.dimension || 384}</span>
                            </div>
                          </div>
                        ) : (
                          <p className="text-muted-foreground text-xs">No embedding model metadata available.</p>
                        )}
                      </div>

                      <div className="space-y-4">
                        <h4 className="font-semibold text-sm text-foreground flex items-center gap-1.5">
                          <Database className="w-4 h-4 text-violet-500" />
                          FAISS Retrievers Definition
                        </h4>
                        {details.index_metadata ? (
                          <div className="space-y-3.5">
                            <div className="flex justify-between border-b border-border/60 pb-2">
                              <span className="text-muted-foreground">Indexed Vectors Count</span>
                              <span className="font-semibold">{details.index_metadata.total_indexed_vectors ?? 0}</span>
                            </div>
                            <div className="flex justify-between border-b border-border/60 pb-2">
                              <span className="text-muted-foreground">FAISS Metric Distance</span>
                              <span className="font-semibold">{details.index_metadata.metric || 'METRIC_INNER_PRODUCT (Cosine)'}</span>
                            </div>
                          </div>
                        ) : (
                          <p className="text-muted-foreground text-xs">No vector index metadata available.</p>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </MainLayout>
  );
}
