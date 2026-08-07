'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { uploadPdf, UploadSuccessResponse } from '@/services/upload';
import { handleApiError } from '@/services/api';
import { validatePdfFile } from '@/lib/validators';
import { useToast } from '@/providers/ToastProvider';
import { useQueryClient } from '@tanstack/react-query';

export function useUpload() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { success: toastSuccess, error: toastError } = useToast();

  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadSuccessResponse | null>(null);

  const upload = async (file: File): Promise<UploadSuccessResponse | null> => {
    // 1. Client-side Validation
    const validation = validatePdfFile(file);
    if (!validation.isValid) {
      setError(validation.error || 'Invalid file.');
      toastError('Validation Error', validation.error || 'Invalid file type or size.');
      return null;
    }

    setIsUploading(true);
    setProgress(0);
    setError(null);
    setUploadResult(null);

    try {
      const result = await uploadPdf(file, (progressEvent) => {
        if (progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
        }
      });

      setUploadResult(result);
      toastSuccess('Upload Complete', `Successfully uploaded ${file.name}. Initializing pipeline...`);
      
      // Invalidate dashboard/documents list queries
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['storage-statistics'] });

      // Automatically redirect to the pipeline tracking dashboard screen
      router.push(`/documents/${result.document_id}/pipeline`);
      return result;
    } catch (err: unknown) {
      const parsedError = handleApiError(err);
      setError(parsedError.message);
      toastError('Upload Failed', parsedError.message);
      return null;
    } finally {
      setIsUploading(false);
    }
  };

  const reset = () => {
    setIsUploading(false);
    setProgress(0);
    setError(null);
    setUploadResult(null);
  };

  return {
    upload,
    isUploading,
    progress,
    error,
    uploadResult,
    reset,
  };
}
