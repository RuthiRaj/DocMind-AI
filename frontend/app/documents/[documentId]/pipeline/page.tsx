'use client';

import React, { useEffect } from 'react';
import Link from 'next/link';
import { useDocuments } from '@/hooks/useDocuments';
import { usePipeline } from '@/hooks/usePipeline';
import MainLayout from '@/components/layout/MainLayout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import Skeleton from '@/components/ui/Skeleton';
import { PIPELINE_STAGES, STAGE_STYLES } from '@/constants/pipeline';
import {
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  ArrowRight,
  MessageSquare,
  Search,
  Activity,
  Play,
  PlayCircle
} from 'lucide-react';

interface PipelinePageProps {
  params: Promise<{ documentId: string }>;
}

export default function PipelinePage({ params }: PipelinePageProps) {
  const { documentId } = React.use(params);

  const { useStatus } = useDocuments();
  const {
    processDoc,
    chunkDoc,
    embedDoc,
    indexDoc,
    runFullPipeline,
    loadingStage,
    isExecuting,
  } = usePipeline(documentId);
  
  // Auto-refresh polling every 3 seconds if pipeline is executing or not completed/failed
  const { data: statusData, isLoading, isError, refetch } = useStatus(documentId, true);

  const getStageStatus = (stageKey: string): 'waiting' | 'running' | 'completed' | 'failed' => {
    if (!statusData) return 'waiting';

    const upload = statusData.upload_status;
    const processing = statusData.processing_status;
    const chunking = statusData.chunking_status;
    const embedding = statusData.embedding_status;
    const indexing = statusData.indexing_status;

    if (stageKey === 'upload') {
      return upload === 'completed' ? 'completed' : upload === 'failed' ? 'failed' : upload === 'processing' ? 'running' : 'waiting';
    }
    if (stageKey === 'processing') {
      if (upload !== 'completed') return 'waiting';
      return processing === 'completed' ? 'completed' : processing === 'failed' ? 'failed' : processing === 'processing' || (loadingStage === 'processing') ? 'running' : 'waiting';
    }
    if (stageKey === 'chunking') {
      if (processing !== 'completed') return 'waiting';
      return chunking === 'completed' ? 'completed' : chunking === 'failed' ? 'failed' : chunking === 'processing' || (loadingStage === 'chunking') ? 'running' : 'waiting';
    }
    if (stageKey === 'embedding') {
      if (chunking !== 'completed') return 'waiting';
      return embedding === 'completed' ? 'completed' : embedding === 'failed' ? 'failed' : embedding === 'processing' || (loadingStage === 'embedding') ? 'running' : 'waiting';
    }
    if (stageKey === 'indexing') {
      if (embedding !== 'completed') return 'waiting';
      return indexing === 'completed' ? 'completed' : indexing === 'failed' ? 'failed' : indexing === 'processing' || (loadingStage === 'indexing') ? 'running' : 'waiting';
    }

    return 'waiting';
  };

  const getStageIcon = (status: 'waiting' | 'running' | 'completed' | 'failed') => {
    if (status === 'completed') {
      return <CheckCircle2 className="w-6 h-6 text-emerald-500 shrink-0" />;
    }
    if (status === 'failed') {
      return <XCircle className="w-6 h-6 text-rose-500 shrink-0" />;
    }
    if (status === 'running') {
      return <Loader2 className="w-6 h-6 text-blue-500 animate-spin shrink-0" />;
    }
    return <Clock className="w-6 h-6 text-slate-400 shrink-0" />;
  };

  // Determine which action button to show
  const renderStageActionButton = (stageKey: string, status: 'waiting' | 'running' | 'completed' | 'failed') => {
    if (status === 'completed' || status === 'running' || isExecuting) return null;

    // Check if prerequisite is met
    const uploadCompleted = statusData?.upload_status === 'completed';
    const processingCompleted = statusData?.processing_status === 'completed';
    const chunkingCompleted = statusData?.chunking_status === 'completed';
    const embeddingCompleted = statusData?.embedding_status === 'completed';

    if (stageKey === 'processing' && uploadCompleted) {
      return (
        <Button size="sm" variant="outline" className="text-xs h-8 gap-1.5 cursor-pointer" onClick={() => processDoc()}>
          <Play className="w-3 h-3" />
          <span>Run Parse</span>
        </Button>
      );
    }
    if (stageKey === 'chunking' && processingCompleted) {
      return (
        <Button size="sm" variant="outline" className="text-xs h-8 gap-1.5 cursor-pointer" onClick={() => chunkDoc()}>
          <Play className="w-3 h-3" />
          <span>Run Chunk</span>
        </Button>
      );
    }
    if (stageKey === 'embedding' && chunkingCompleted) {
      return (
        <Button size="sm" variant="outline" className="text-xs h-8 gap-1.5 cursor-pointer" onClick={() => embedDoc()}>
          <Play className="w-3 h-3" />
          <span>Run Embed</span>
        </Button>
      );
    }
    if (stageKey === 'indexing' && embeddingCompleted) {
      return (
        <Button size="sm" variant="outline" className="text-xs h-8 gap-1.5 cursor-pointer" onClick={() => indexDoc()}>
          <Play className="w-3 h-3" />
          <span>Compile Index</span>
        </Button>
      );
    }

    return null;
  };

  const isFailed = 
    statusData?.upload_status === 'failed' ||
    statusData?.processing_status === 'failed' ||
    statusData?.chunking_status === 'failed' ||
    statusData?.embedding_status === 'failed' ||
    statusData?.indexing_status === 'failed';

  const isFinished = statusData?.chat_ready === true;

  // Auto-run sequential pipeline if upload completes but remaining are pending
  useEffect(() => {
    if (statusData && !isExecuting && !isFailed && !isFinished) {
      runFullPipeline();
    }
  }, [statusData]);

  return (
    <MainLayout>
      <div className="max-w-3xl mx-auto space-y-6">
        {/* Navigation Breadcrumb */}
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Link href="/dashboard" className="hover:text-foreground">Dashboard</Link>
          <span>/</span>
          <Link href="/documents" className="hover:text-foreground">Documents</Link>
          <span>/</span>
          <span className="text-foreground">Pipeline Status</span>
        </div>

        {/* Title and Run Button */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div>
            <h2 className="text-2xl font-bold tracking-tight text-foreground font-sans">Document Pipeline Timeline</h2>
            <p className="text-muted-foreground text-sm font-mono mt-1 break-all">ID: {documentId}</p>
          </div>
          
          <div className="flex gap-2">
            {!isFinished && !isFailed && (
              <Button
                size="sm"
                loading={isExecuting}
                onClick={() => runFullPipeline()}
                className="flex items-center gap-2 cursor-pointer font-semibold shadow-sm"
              >
                <PlayCircle className="w-4 h-4" />
                <span>{isExecuting ? 'Executing Stage...' : 'Run Pipeline'}</span>
              </Button>
            )}

            {isFinished && (
              <div className="flex items-center gap-1.5 px-3 py-1 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-semibold rounded-full border border-emerald-500/25 select-none">
                <Activity className="w-3.5 h-3.5" />
                <span>Ready for Chat</span>
              </div>
            )}
          </div>
        </div>

        {/* Finished Success Alert Banner */}
        {isFinished && (
          <Card className="border-emerald-500/30 bg-emerald-500/5 fade-in">
            <CardContent className="p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-8 h-8 text-emerald-500 shrink-0" />
                <div>
                  <h3 className="font-bold text-emerald-700 dark:text-emerald-400">Processing Pipeline Successful</h3>
                  <p className="text-sm text-emerald-600/90 dark:text-emerald-500/90 mt-0.5 leading-relaxed">
                    Your PDF file has been parsed, chunked, and indexing is compiled. You can search vectors or chat with the document.
                  </p>
                </div>
              </div>
              <div className="flex gap-2 shrink-0">
                <Link href={`/retrieve/${documentId}`}>
                  <Button variant="outline" size="sm" className="flex items-center gap-1.5 cursor-pointer">
                    <Search className="w-4 h-4" />
                    <span>Search</span>
                  </Button>
                </Link>
                <Link href={`/chat/${documentId}`}>
                  <Button size="sm" className="flex items-center gap-1.5 cursor-pointer">
                    <MessageSquare className="w-4 h-4" />
                    <span>Chat</span>
                    <ArrowRight className="w-4 h-4" />
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Failed Alert Banner */}
        {isFailed && (
          <Card className="border-rose-500/30 bg-rose-500/5 fade-in">
            <CardContent className="p-6 flex flex-col items-center justify-center text-center space-y-3">
              <XCircle className="w-12 h-12 text-rose-500" />
              <h3 className="font-bold text-rose-700 dark:text-rose-400">Pipeline Execution Failed</h3>
              <p className="text-sm text-rose-500/90 max-w-md">
                An error occurred during one of the document processing stages. Click on the action triggers below to retry manually or upload a new file.
              </p>
              <div className="flex gap-2 pt-2">
                <Button size="sm" variant="outline" onClick={() => runFullPipeline()} className="cursor-pointer">
                  Retry Remaining Stages
                </Button>
                <Link href="/upload">
                  <Button variant="destructive" size="sm" className="cursor-pointer">
                    Upload Another File
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Timeline Visual Step Progress */}
        <Card>
          <CardHeader className="shrink-0 border-b border-border/60">
            <CardTitle>Processing Logs Tracker</CardTitle>
            <CardDescription>
              Polling pipeline updates dynamically. Each stage will run sequentially.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-6">
            {isLoading ? (
              <div className="space-y-6">
                <Skeleton className="w-full h-16" />
                <Skeleton className="w-full h-16" />
                <Skeleton className="w-full h-16" />
              </div>
            ) : isError || !statusData ? (
              <div className="flex flex-col items-center justify-center py-12 text-center text-rose-500 text-sm">
                <XCircle className="w-10 h-10 mb-2" />
                <span>Failed to fetch status updates.</span>
                <Button size="sm" variant="outline" className="mt-4" onClick={() => refetch()}>
                  Retry Loading
                </Button>
              </div>
            ) : (
              <div className="space-y-6 relative before:absolute before:inset-0 before:left-8 before:w-0.5 before:bg-border before:z-0">
                {PIPELINE_STAGES.map((stage) => {
                  const stageStatus = getStageStatus(stage.key);
                  const styles = STAGE_STYLES[stageStatus];

                  return (
                    <div
                      key={stage.key}
                      className="flex items-start gap-6 relative z-10 fade-in"
                    >
                      <div className="flex items-center justify-center w-16 h-16 rounded-full bg-card border border-border/80 shadow-sm shrink-0">
                        {getStageIcon(stageStatus)}
                      </div>
                      <div className="flex-1 min-w-0 bg-muted/20 border border-border/50 p-4 rounded-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
                        <div>
                          <h4 className="font-semibold text-sm text-foreground">{stage.label}</h4>
                          <p className="text-xs text-muted-foreground mt-0.5">{stage.desc}</p>
                        </div>
                        <div className="flex items-center gap-3">
                          {renderStageActionButton(stage.key, stageStatus)}
                          <Badge variant={styles.badge as any} className="w-fit shrink-0">
                            {styles.label}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </MainLayout>
  );
}
