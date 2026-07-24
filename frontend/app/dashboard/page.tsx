'use client';

import React from 'react';
import Link from 'next/link';
import { useDocuments } from '@/hooks/useDocuments';
import { useHealth } from '@/hooks/useHealth';
import MainLayout from '@/components/layout/MainLayout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import Skeleton from '@/components/ui/Skeleton';
import { formatBytes } from '@/lib/formatBytes';
import { formatDate } from '@/lib/formatDate';
import {
  FileText,
  Layers,
  HardDrive,
  Activity,
  Upload,
  ArrowRight,
  TrendingUp,
  FileCheck,
  AlertCircle
} from 'lucide-react';

export default function Dashboard() {
  const { documents, stats, listLoading, statsLoading, refetchList, refetchStats } = useDocuments({
    skip: 0,
    limit: 5,
    sort_by: 'upload_time',
    descending: true,
  });

  const { health, isLoading: healthLoading } = useHealth();

  const handleRefreshAll = () => {
    refetchList();
    refetchStats();
  };

  const getHealthBadge = () => {
    if (healthLoading) return <Skeleton className="w-16 h-5" />;
    if (!health) return <Badge variant="destructive">Offline</Badge>;
    return health.status === 'healthy' ? (
      <Badge variant="success">Healthy</Badge>
    ) : (
      <Badge variant="destructive">Issues</Badge>
    );
  };

  return (
    <MainLayout>
      {/* Welcome header banner */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-foreground">Welcome to DocMind Dashboard</h2>
          <p className="text-muted-foreground text-sm">
            Monitor ingestion statistics, system integrity checkpoints, and recent documents.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleRefreshAll}>
            Refresh Stats
          </Button>
          <Link href="/upload">
            <Button size="sm" className="flex items-center gap-2 cursor-pointer">
              <Upload className="w-4 h-4" />
              <span>Ingest PDF</span>
            </Button>
          </Link>
        </div>
      </div>

      {/* Grid of statistics widgets */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Documents Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2 shrink-0">
            <CardTitle className="text-sm font-semibold">Total Documents</CardTitle>
            <FileText className="w-4 h-4 text-primary" />
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Skeleton className="w-16 h-8" />
            ) : (
              <div className="text-3xl font-extrabold">{stats?.total_documents ?? 0}</div>
            )}
            <p className="text-xs text-muted-foreground mt-1">Processed PDF files</p>
          </CardContent>
        </Card>

        {/* Total Chunks Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2 shrink-0">
            <CardTitle className="text-sm font-semibold">Total Chunks</CardTitle>
            <Layers className="w-4 h-4 text-violet-500" />
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Skeleton className="w-16 h-8" />
            ) : (
              <div className="text-3xl font-extrabold">{stats?.total_chunks ?? 0}</div>
            )}
            <p className="text-xs text-muted-foreground mt-1">Segmented sections</p>
          </CardContent>
        </Card>

        {/* Storage Size Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2 shrink-0">
            <CardTitle className="text-sm font-semibold">Disk Storage</CardTitle>
            <HardDrive className="w-4 h-4 text-indigo-500" />
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Skeleton className="w-24 h-8" />
            ) : (
              <div className="text-3xl font-extrabold">
                {formatBytes(stats?.storage_bytes ?? 0)}
              </div>
            )}
            <p className="text-xs text-muted-foreground mt-1">Uploaded folder size</p>
          </CardContent>
        </Card>

        {/* System Health Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2 shrink-0">
            <CardTitle className="text-sm font-semibold">System Health</CardTitle>
            <Activity className="w-4 h-4 text-emerald-500" />
          </CardHeader>
          <CardContent className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-3xl font-extrabold">API</span>
              {getHealthBadge()}
            </div>
            <p className="text-xs text-muted-foreground">
              Uptime: {health ? `${Math.floor(health.uptime_seconds / 60)}m ${Math.floor(health.uptime_seconds % 60)}s` : 'N/A'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* In-depth pipeline states counts */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="flex items-center gap-4 p-5">
          <div className="p-3 bg-emerald-500/10 text-emerald-500 rounded-xl">
            <FileCheck className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold">{stats?.completed_documents ?? 0}</div>
            <div className="text-xs text-muted-foreground">Chat Ready Documents</div>
          </div>
        </Card>

        <Card className="flex items-center gap-4 p-5">
          <div className="p-3 bg-amber-500/10 text-amber-500 rounded-xl">
            <TrendingUp className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold">{stats?.processing_documents ?? 0}</div>
            <div className="text-xs text-muted-foreground">Currently Processing</div>
          </div>
        </Card>

        <Card className="flex items-center gap-4 p-5">
          <div className="p-3 bg-rose-500/10 text-rose-500 rounded-xl">
            <AlertCircle className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold">{stats?.failed_documents ?? 0}</div>
            <div className="text-xs text-muted-foreground">Failed Document Pipelines</div>
          </div>
        </Card>
      </div>

      {/* Row containing Recent documents and Diagnostic Checklist */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Recent Ingested Documents List */}
        <Card className="lg:col-span-2 flex flex-col min-h-[300px]">
          <CardHeader className="flex flex-row justify-between items-center shrink-0">
            <div>
              <CardTitle>Recent Ingestions</CardTitle>
              <CardDescription>Recently uploaded PDF files and their state</CardDescription>
            </div>
            <Link href="/documents" className="text-xs text-primary hover:underline font-semibold cursor-pointer">
              View All
            </Link>
          </CardHeader>
          <CardContent className="flex-1 overflow-x-auto min-w-0">
            {listLoading ? (
              <div className="space-y-3">
                <Skeleton className="w-full h-10" />
                <Skeleton className="w-full h-10" />
                <Skeleton className="w-full h-10" />
              </div>
            ) : documents.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground text-sm">
                <FileText className="w-10 h-10 mb-2 opacity-50" />
                <span>No PDF files uploaded yet.</span>
              </div>
            ) : (
              <div className="w-full min-w-[500px]">
                <table className="w-full border-collapse text-left text-sm">
                  <thead>
                    <tr className="border-b border-border text-muted-foreground">
                      <th className="pb-3 font-medium">Filename</th>
                      <th className="pb-3 font-medium">Upload Date</th>
                      <th className="pb-3 font-medium">Stage</th>
                      <th className="pb-3 font-medium text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/60">
                    {documents.map((doc) => (
                      <tr key={doc.document_id} className="hover:bg-muted/40 transition-colors">
                        <td className="py-3.5 font-medium text-foreground truncate max-w-[200px]" title={doc.filename}>
                          {doc.filename}
                        </td>
                        <td className="py-3.5 text-muted-foreground">{formatDate(doc.upload_time)}</td>
                        <td className="py-3.5">
                          <Badge
                            variant={
                              doc.current_pipeline_stage === 'indexing'
                                ? 'success'
                                : doc.current_pipeline_stage === 'failed'
                                ? 'destructive'
                                : 'warning'
                            }
                          >
                            {doc.current_pipeline_stage}
                          </Badge>
                        </td>
                        <td className="py-3.5 text-right">
                          <Link href={doc.chat_ready ? `/chat/${doc.document_id}` : `/documents/${doc.document_id}/pipeline`}>
                            <Button size="sm" variant={doc.chat_ready ? 'primary' : 'outline'} className="cursor-pointer">
                              <span>{doc.chat_ready ? 'Chat' : 'Track'}</span>
                              <ArrowRight className="w-3.5 h-3.5 ml-1" />
                            </Button>
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Quick Diagnostics Widget */}
        <Card className="flex flex-col">
          <CardHeader className="shrink-0">
            <CardTitle>Diagnostics Summary</CardTitle>
            <CardDescription>Backend environment configurations checks</CardDescription>
          </CardHeader>
          <CardContent className="flex-1 space-y-4">
            {healthLoading ? (
              <div className="space-y-4">
                <Skeleton className="w-full h-8" />
                <Skeleton className="w-full h-8" />
                <Skeleton className="w-full h-8" />
              </div>
            ) : !health ? (
              <div className="flex flex-col items-center justify-center py-8 text-center text-rose-500 text-sm">
                <AlertCircle className="w-8 h-8 mb-2" />
                <span>Backend offline or unreachable.</span>
              </div>
            ) : (
              <div className="space-y-3.5">
                {[
                  { label: 'Upload Path', status: health.uploads_directory.status },
                  { label: 'Write Access', status: health.write_permission.status },
                  { label: 'Embeddings Model', status: health.embedding_model.status },
                  { label: 'FAISS Library', status: health.faiss_library.status },
                  { label: 'Groq Configuration', status: health.groq_service.status },
                ].map((item) => (
                  <div key={item.label} className="flex items-center justify-between p-2 rounded-lg bg-muted/40 border border-border/30">
                    <span className="text-sm text-foreground font-medium">{item.label}</span>
                    <span
                      className={`inline-flex items-center w-2.5 h-2.5 rounded-full ${
                        item.status === 'healthy' ? 'bg-emerald-500' : 'bg-rose-500'
                      }`}
                      title={item.status === 'healthy' ? 'Healthy' : 'Error'}
                    />
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </MainLayout>
  );
}
