import apiClient from './api';
import { ENDPOINTS } from '@/constants/api';
import { CleanupResponse } from '@/types/Api';

export async function runMaintenanceCleanup(): Promise<CleanupResponse> {
  const res = await apiClient.post<CleanupResponse>(ENDPOINTS.CLEANUP);
  return res.data;
}
