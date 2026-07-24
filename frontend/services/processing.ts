import apiClient from './api';

export interface PDFProcessingResponse {
  document_id: string;
  filename: string;
  total_pages: number;
  message: string;
}

export async function processPdf(id: string): Promise<PDFProcessingResponse> {
  const res = await apiClient.post<PDFProcessingResponse>(`/process/${id}`);
  return res.data;
}
