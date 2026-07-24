'use client';

import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useUpload } from '@/hooks/useUpload';
import MainLayout from '@/components/layout/MainLayout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import { UploadCloud, FileText, AlertCircle, RefreshCw } from 'lucide-react';
import { formatBytes } from '@/lib/formatBytes';
import { FILE_LIMITS } from '@/constants/limits';

export default function UploadPage() {
  const { upload, isUploading, progress, error, reset } = useUpload();

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        await upload(acceptedFiles[0]);
      }
    },
    [upload]
  );

  const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
    onDrop,
    maxFiles: 1,
    disabled: isUploading,
    accept: {
      'application/pdf': ['.pdf'],
    },
  });

  const selectedFile = acceptedFiles[0];

  return (
    <MainLayout>
      <div className="max-w-2xl mx-auto space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-foreground font-sans">Ingest PDF Document</h2>
          <p className="text-muted-foreground text-sm">
            Upload your document to parse contents, compute chunk segments, and compile vector search indexes.
          </p>
        </div>

        <Card>
          <CardHeader className="shrink-0">
            <CardTitle>File Upload</CardTitle>
            <CardDescription>
              Drag & Drop your file here or click to browse. Max file size: {FILE_LIMITS.MAX_SIZE_BYTES / (1024 * 1024)}MB.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Drag & Drop Zone */}
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-primary bg-primary/5'
                  : 'border-border bg-card hover:bg-muted/30'
              } ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}
            >
              <input {...getInputProps()} />
              <UploadCloud className="w-12 h-12 text-muted-foreground mb-4" />
              <p className="text-sm font-semibold text-foreground text-center">
                {isDragActive ? 'Drop your PDF here...' : 'Drag & Drop PDF document here'}
              </p>
              <p className="text-xs text-muted-foreground mt-1.5">
                or click to browse local files
              </p>
            </div>

            {/* Error alerts */}
            {error && (
              <div className="flex items-start gap-3 p-4 bg-rose-500/10 border border-rose-500/20 text-rose-700 dark:text-rose-400 rounded-xl text-sm fade-in">
                <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="font-semibold">Upload Ingestion Error</p>
                  <p className="text-xs mt-0.5">{error}</p>
                </div>
                <Button size="sm" variant="ghost" className="h-auto p-1" onClick={reset}>
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
            )}

            {/* In-progress tracker bars */}
            {isUploading && (
              <div className="space-y-3 bg-muted/30 p-4 border border-border/60 rounded-xl fade-in">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2 font-medium">
                    <FileText className="w-4 h-4 text-primary animate-pulse" />
                    <span className="truncate max-w-[200px]" title={selectedFile?.name}>
                      {selectedFile?.name}
                    </span>
                  </div>
                  <span className="text-xs text-muted-foreground">{progress}%</span>
                </div>
                
                {/* Progress bar */}
                <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-primary h-full rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-xs text-muted-foreground text-center">
                  Uploading files to DocMind servers. Do not close this window.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </MainLayout>
  );
}
