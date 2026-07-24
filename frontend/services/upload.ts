import apiClient from './api';
import { ENDPOINTS } from '@/constants/api';

export interface UploadSuccessResponse {
  document_id: string;
  filename: string;
  message: string;
}

export async function uploadPdf(
  file: File,
  onUploadProgress?: (progressEvent: any) => void
): Promise<UploadSuccessResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await apiClient.post<UploadSuccessResponse>(ENDPOINTS.UPLOAD, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress,
  });
  return res.data;
}
