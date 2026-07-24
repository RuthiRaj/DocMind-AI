'use client';

import React, { useState } from 'react';
import { useMaintenance } from '@/hooks/useMaintenance';
import { useDocuments } from '@/hooks/useDocuments';
import MainLayout from '@/components/layout/MainLayout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Dialog from '@/components/ui/Dialog';
import Skeleton from '@/components/ui/Skeleton';
import { formatBytes } from '@/lib/formatBytes';
import {
  Settings,
  HardDrive,
  Trash2,
  CheckCircle2,
  ShieldCheck,
  AlertTriangle,
  Info,
  Clock
} from 'lucide-react';

export default function MaintenancePage() {
  const { cleanup, result, isLoading: isPruning } = useMaintenance();
  const { stats, statsLoading } = useDocuments();

  const [confirmOpen, setConfirmOpen] = useState(false);

  const handlePrune = async () => {
    setConfirmOpen(false);
    await cleanup();
  };

  return (
    <MainLayout>
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-foreground font-sans">System Maintenance Dashboard</h2>
          <p className="text-muted-foreground text-sm">
            Prune garbage temporary file buffers and monitor system-wide storage stats metrics.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Storage stats Card */}
        <Card className="md:col-span-1">
          <CardHeader className="flex flex-row items-center gap-3 pb-3 shrink-0">
            <HardDrive className="w-5 h-5 text-indigo-500" />
            <div>
              <CardTitle className="text-base font-semibold">Storage Metrics</CardTitle>
              <CardDescription className="text-xs">Aggregate storage consumption</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {statsLoading ? (
              <div className="space-y-3">
                <Skeleton className="w-full h-8" />
                <Skeleton className="w-full h-8" />
              </div>
            ) : (
              <div className="space-y-3 text-sm">
                <div className="flex justify-between border-b border-border/60 pb-2">
                  <span className="text-muted-foreground">Total Storage Used</span>
                  <span className="font-bold text-foreground">{formatBytes(stats?.storage_bytes ?? 0)}</span>
                </div>
                <div className="flex justify-between border-b border-border/60 pb-2">
                  <span className="text-muted-foreground">Total Documents</span>
                  <span className="font-semibold text-foreground">{stats?.total_documents ?? 0} files</span>
                </div>
                <div className="flex justify-between border-b border-border/60 pb-2">
                  <span className="text-muted-foreground">Chunks Count</span>
                  <span className="font-semibold text-foreground">{stats?.total_chunks ?? 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">FAISS Indexes</span>
                  <span className="font-semibold text-foreground">{stats?.total_indexes ?? 0} databases</span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Action card */}
        <Card className="md:col-span-2 flex flex-col">
          <CardHeader className="shrink-0">
            <CardTitle>Storage Pruning Utility</CardTitle>
            <CardDescription>
              Cleans swap cache allocations and restores storage resources.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6 flex-1">
            {/* Rules of Pruning */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
              <div className="p-3.5 bg-emerald-500/5 border border-emerald-500/15 rounded-xl space-y-2">
                <h4 className="font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5 uppercase tracking-wider">
                  <ShieldCheck className="w-4 h-4" />
                  Protected Files
                </h4>
                <p className="text-muted-foreground leading-relaxed">
                  Original PDF files, parsed text segments, chunk lists, embeddings, and FAISS indexing databases are protected and will never be pruned.
                </p>
              </div>

              <div className="p-3.5 bg-amber-500/5 border border-amber-500/15 rounded-xl space-y-2">
                <h4 className="font-semibold text-amber-600 dark:text-amber-400 flex items-center gap-1.5 uppercase tracking-wider">
                  <Trash2 className="w-4 h-4" />
                  Cleaned Targets
                </h4>
                <p className="text-muted-foreground leading-relaxed">
                  Temporary *.tmp files, empty document directories, and orphan metadata descriptors inside uploads folder will be deleted.
                </p>
              </div>
            </div>

            {/* Execute Button */}
            <div className="pt-2">
              <Button
                variant="destructive"
                loading={isPruning}
                className="flex items-center gap-2 cursor-pointer w-full sm:w-auto"
                onClick={() => setConfirmOpen(true)}
              >
                <Trash2 className="w-4 h-4" />
                <span>Run Cleanup Maintenance</span>
              </Button>
            </div>

            {/* Result display */}
            {result && (
              <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 text-emerald-700 dark:text-emerald-400 rounded-xl text-sm space-y-2 fade-in">
                <h4 className="font-bold flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 className="w-5 h-5 shrink-0" />
                  Pruning Completed Successfully
                </h4>
                <p className="text-xs">{result.message}</p>
                <div className="grid grid-cols-2 gap-4 text-xs font-semibold pt-2 text-muted-foreground uppercase tracking-wider border-t border-emerald-500/10">
                  <div>
                    Removed Temp Files:{' '}
                    <span className="text-foreground font-bold">{result.removed_temp_files}</span>
                  </div>
                  <div>
                    Pruned Folders:{' '}
                    <span className="text-foreground font-bold">{result.removed_empty_directories}</span>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Confirmation Dialog */}
      <Dialog
        isOpen={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        title="Confirm Storage Cleanup"
      >
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Are you sure you want to run storage maintenance cleanup?
          </p>
          <div className="bg-amber-500/10 text-amber-600 dark:text-amber-400 p-3 rounded-lg border border-amber-500/20 text-xs flex gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>
              This will search the uploads folder directory recursively to delete all temporary and empty folder paths. Valid active files will remain unaffected.
            </span>
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="outline" size="sm" onClick={() => setConfirmOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              size="sm"
              loading={isPruning}
              onClick={handlePrune}
              className="cursor-pointer"
            >
              Confirm Cleanup
            </Button>
          </div>
        </div>
      </Dialog>
    </MainLayout>
  );
}
