'use client';

import React from 'react';
import { useHealth } from '@/hooks/useHealth';
import MainLayout from '@/components/layout/MainLayout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import Skeleton from '@/components/ui/Skeleton';
import { formatBytes } from '@/lib/formatBytes';
import {
  Activity,
  HardDrive,
  FolderOpen,
  CheckCircle,
  AlertCircle,
  Cpu,
  RefreshCw,
  Clock,
  Settings,
  ShieldAlert
} from 'lucide-react';

export default function HealthPage() {
  const { health, isLoading, isError, error, refetch } = useHealth();

  const getStatusIcon = (status: 'healthy' | 'unhealthy') => {
    return status === 'healthy' ? (
      <CheckCircle className="w-5 h-5 text-emerald-500 shrink-0" />
    ) : (
      <AlertCircle className="w-5 h-5 text-rose-500 shrink-0" />
    );
  };

  const getStatusBadge = (status: 'healthy' | 'unhealthy') => {
    return status === 'healthy' ? (
      <Badge variant="success">Healthy</Badge>
    ) : (
      <Badge variant="destructive">Unhealthy</Badge>
    );
  };

  return (
    <MainLayout>
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-foreground">System Health Diagnostics</h2>
          <p className="text-muted-foreground text-sm">
            Inspect core configurations, library imports status, and local storage limits.
          </p>
        </div>
        <Button
          onClick={() => refetch()}
          disabled={isLoading}
          variant="outline"
          size="sm"
          className="flex items-center gap-2 cursor-pointer"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          <span>Refresh Diagnostics</span>
        </Button>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Skeleton className="w-full h-44" />
          <Skeleton className="w-full h-44" />
          <Skeleton className="w-full h-44" />
          <Skeleton className="w-full h-44" />
        </div>
      ) : isError || !health ? (
        <Card className="border-rose-500/30 bg-rose-500/5">
          <CardContent className="flex flex-col items-center justify-center py-12 text-center text-rose-600 space-y-3">
            <ShieldAlert className="w-12 h-12" />
            <h3 className="text-lg font-bold">Diagnostics Offline</h3>
            <p className="text-sm text-rose-500/90 max-w-md">
              {error?.message || 'Unable to communicate with the backend FastAPI services. Verify the server is running locally on port 8000.'}
            </p>
            <Button variant="destructive" onClick={() => refetch()} className="mt-2">
              Retry Connection
            </Button>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Status Bar */}
          <div className={`p-4 rounded-xl flex items-center justify-between border ${
            health.status === 'healthy' 
              ? 'bg-emerald-500/10 border-emerald-500/25 text-emerald-700 dark:text-emerald-400' 
              : 'bg-rose-500/10 border-rose-500/25 text-rose-700 dark:text-rose-400'
          }`}>
            <div className="flex items-center gap-3">
              <Activity className={`w-6 h-6 ${health.status === 'healthy' ? 'animate-pulse' : ''}`} />
              <div>
                <p className="font-semibold text-sm">Overall Status: {health.status.toUpperCase()}</p>
                <p className="text-xs opacity-90">
                  {health.status === 'healthy'
                    ? 'All critical services and environment variables verified.'
                    : 'One or more services or components are unhealthy. Check the details below.'}
                </p>
              </div>
            </div>
            {getStatusBadge(health.status)}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Upload Directory Card */}
            <Card>
              <CardHeader className="flex flex-row items-center gap-3 pb-3 shrink-0">
                <FolderOpen className="w-5 h-5 text-primary" />
                <div>
                  <CardTitle className="text-base font-semibold">Uploads Folder</CardTitle>
                  <CardDescription className="text-xs">Storage path accessibility checks</CardDescription>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start justify-between gap-4 text-sm bg-muted/30 p-3 rounded-lg border border-border/30">
                  <div className="space-y-1">
                    <p className="font-semibold text-xs text-muted-foreground uppercase tracking-wider">Storage Path</p>
                    <p className="font-mono text-xs break-all text-foreground">{health.uploads_directory.path}</p>
                  </div>
                  {getStatusIcon(health.uploads_directory.status)}
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Directory Status</span>
                  <span>{health.uploads_directory.details}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Write Permission</span>
                  <div className="flex items-center gap-2">
                    <span>{health.write_permission.details}</span>
                    {getStatusIcon(health.write_permission.status)}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Disk Usage Card */}
            <Card>
              <CardHeader className="flex flex-row items-center gap-3 pb-3 shrink-0">
                <HardDrive className="w-5 h-5 text-indigo-500" />
                <div>
                  <CardTitle className="text-base font-semibold">Storage Capacity</CardTitle>
                  <CardDescription className="text-xs">Local storage availability bounds</CardDescription>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Progress bar chart */}
                {health.disk_usage.total_bytes && health.disk_usage.used_bytes ? (
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Used: {formatBytes(health.disk_usage.used_bytes)}</span>
                      <span>Total: {formatBytes(health.disk_usage.total_bytes)}</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2.5 overflow-hidden">
                      <div
                        className="bg-indigo-600 h-full rounded-full transition-all"
                        style={{
                          width: `${(health.disk_usage.used_bytes / health.disk_usage.total_bytes) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                ) : null}
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Storage Status</span>
                  <span>{health.disk_usage.details}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Free Disk Space</span>
                  <span>{formatBytes(health.disk_usage.free_bytes ?? 0)}</span>
                </div>
              </CardContent>
            </Card>

            {/* AI Libraries Card */}
            <Card>
              <CardHeader className="flex flex-row items-center gap-3 pb-3 shrink-0">
                <Cpu className="w-5 h-5 text-violet-500" />
                <div>
                  <CardTitle className="text-base font-semibold">AI Models & Search</CardTitle>
                  <CardDescription className="text-xs">SentenceTransformers and FAISS loading checkpoints</CardDescription>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between text-sm p-2 rounded bg-muted/30 border border-border/30">
                  <span className="text-muted-foreground font-medium">SentenceTransformers</span>
                  {getStatusIcon(health.embedding_model.status)}
                </div>
                <p className="text-xs text-muted-foreground pl-1">{health.embedding_model.details}</p>

                <div className="flex items-center justify-between text-sm p-2 rounded bg-muted/30 border border-border/30">
                  <span className="text-muted-foreground font-medium">FAISS Library Check</span>
                  {getStatusIcon(health.faiss_library.status)}
                </div>
                <p className="text-xs text-muted-foreground pl-1">{health.faiss_library.details}</p>
              </CardContent>
            </Card>

            {/* Backend Services Configurations Card */}
            <Card>
              <CardHeader className="flex flex-row items-center gap-3 pb-3 shrink-0">
                <Settings className="w-5 h-5 text-amber-500" />
                <div>
                  <CardTitle className="text-base font-semibold">System Configuration</CardTitle>
                  <CardDescription className="text-xs">Third-party configurations validation checks</CardDescription>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between text-sm p-2 rounded bg-muted/30 border border-border/30">
                  <span className="text-muted-foreground font-medium">Groq completions Service</span>
                  {getStatusIcon(health.groq_service.status)}
                </div>
                <p className="text-xs text-muted-foreground pl-1">{health.groq_service.details}</p>

                <hr className="border-border/60" />

                <div className="grid grid-cols-2 gap-4 text-sm pt-1">
                  <div className="flex flex-col gap-0.5">
                    <span className="text-xs text-muted-foreground flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5" />
                      Uptime
                    </span>
                    <span className="font-semibold text-foreground">
                      {Math.floor(health.uptime_seconds / 3600)}h {Math.floor((health.uptime_seconds % 3600) / 60)}m
                    </span>
                  </div>
                  <div className="flex flex-col gap-0.5">
                    <span className="text-xs text-muted-foreground">Backend version</span>
                    <span className="font-semibold text-foreground font-mono">{health.backend_version}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </MainLayout>
  );
}
