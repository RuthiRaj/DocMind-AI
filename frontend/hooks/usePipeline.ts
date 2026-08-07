'use client';

import { useState, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { processPdf } from '@/services/processing';
import { chunkDocument } from '@/services/chunking';
import { embedDocument } from '@/services/embedding';
import { indexDocument } from '@/services/indexing';
import { getDocumentStatus } from '@/services/documents';
import { handleApiError } from '@/services/api';
import { useToast } from '@/providers/ToastProvider';

export function usePipeline(documentId: string) {
  const queryClient = useQueryClient();
  const { success: toastSuccess, error: toastError } = useToast();

  const [activeStage, setActiveStage] = useState<string | null>(null);
  const [loadingStage, setLoadingStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPipelineRunning, setIsPipelineRunning] = useState(false);
  const isRunningRef = useRef(false);

  const refreshQueries = () => {
    queryClient.invalidateQueries({ queryKey: ['document-status', documentId] });
    queryClient.invalidateQueries({ queryKey: ['documents'] });
    queryClient.invalidateQueries({ queryKey: ['storage-statistics'] });
  };

  const runStage = async (
    stageName: string,
    actionFn: () => Promise<unknown>,
    successMessage: string
  ): Promise<boolean> => {
    setLoadingStage(stageName);
    setActiveStage(stageName);
    setError(null);

    try {
      await actionFn();
      toastSuccess(`Stage Complete`, successMessage);
      refreshQueries();
      return true;
    } catch (err: unknown) {
      const parsed = handleApiError(err);
      setError(parsed.message);
      toastError(`Stage Failed: ${stageName}`, parsed.message);
      refreshQueries();
      return false;
    } finally {
      setLoadingStage(null);
      setActiveStage(null);
    }
  };

  const processDoc = () =>
    runStage(
      'processing',
      () => processPdf(documentId),
      'Document text and metadata extracted successfully.'
    );

  const chunkDoc = (force: boolean = false) =>
    runStage(
      'chunking',
      () => chunkDocument(documentId, force),
      'Extracted text chunked into semantic paragraphs.'
    );

  const embedDoc = (force: boolean = false) =>
    runStage(
      'embedding',
      () => embedDocument(documentId, force),
      'Dense vector embeddings generated for all text segments.'
    );

  const indexDoc = (force: boolean = false) =>
    runStage(
      'indexing',
      () => indexDocument(documentId, force),
      'FAISS Cosine Similarity Vector search index compiled.'
    );

  // Run all remaining pipeline stages sequentially
  const runFullPipeline = async () => {
    if (isRunningRef.current) return;
    isRunningRef.current = true;
    setIsPipelineRunning(true);
    try {
      // 1. Get current status to see where to resume
      const statusData = await getDocumentStatus(documentId);

      // Upload is assumed completed (otherwise we wouldn't have documentId)
      
      // 2. Process
      if (statusData.processing_status !== 'completed') {
        const ok = await processDoc();
        if (!ok) return;
      }

      // 3. Chunk
      const updatedStatus1 = await getDocumentStatus(documentId);
      if (updatedStatus1.chunking_status !== 'completed') {
        const ok = await chunkDoc();
        if (!ok) return;
      }

      // 4. Embed
      const updatedStatus2 = await getDocumentStatus(documentId);
      if (updatedStatus2.embedding_status !== 'completed') {
        const ok = await embedDoc();
        if (!ok) return;
      }

      // 5. Index
      const updatedStatus3 = await getDocumentStatus(documentId);
      if (updatedStatus3.indexing_status !== 'completed') {
        const ok = await indexDoc();
        if (!ok) return;
      }
    } catch (err: unknown) {
      const parsed = handleApiError(err);
      setError(parsed.message);
      toastError('Pipeline Execution Error', parsed.message);
    } finally {
      isRunningRef.current = false;
      setIsPipelineRunning(false);
    }
  };

  return {
    processDoc,
    chunkDoc,
    embedDoc,
    indexDoc,
    runFullPipeline,
    loadingStage,
    activeStage,
    error,
    isExecuting: isPipelineRunning || loadingStage !== null,
  };
}
