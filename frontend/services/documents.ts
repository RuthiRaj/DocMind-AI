import apiClient from './api';
import { ENDPOINTS } from '@/constants/api';
import { DocumentListResponse, DocumentDetailResponse, DeleteResponse } from '@/types/Document';
import { PipelineStatusResponse } from '@/types/Pipeline';
import { StorageStatistics } from '@/types/Statistics';


export async function listDocuments(params?: {
  skip?: number;
  limit?: number;
  sort_by?: string;
  descending?: boolean;
  status_filter?: string | null;
}): Promise<DocumentListResponse> {
  const res = await apiClient.get<DocumentListResponse>(ENDPOINTS.DOCUMENTS, {
    params,
  });
  return res.data;
}

export async function getDocumentDetails(id: string): Promise<DocumentDetailResponse> {
  const res = await apiClient.get<DocumentDetailResponse>(ENDPOINTS.DOCUMENT_DETAILS(id));
  return res.data;
}

export async function getDocumentStatus(id: string): Promise<PipelineStatusResponse> {
  const res = await apiClient.get<PipelineStatusResponse>(ENDPOINTS.DOCUMENT_STATUS(id));
  return res.data;
}

export async function deleteDocument(id: string): Promise<DeleteResponse> {
  const res = await apiClient.delete<DeleteResponse>(ENDPOINTS.DOCUMENT_DELETE(id));
  return res.data;
}

export async function getStorageStatistics(): Promise<StorageStatistics> {
  const res = await apiClient.get<StorageStatistics>(ENDPOINTS.DOCUMENTS_STATISTICS);
  return res.data;
}
