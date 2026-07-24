import apiClient from './api';
import { ENDPOINTS } from '@/constants/api';
import { SystemHealth } from '@/types/Health';

export async function getSystemHealth(): Promise<SystemHealth> {
  const res = await apiClient.get<SystemHealth>(ENDPOINTS.HEALTH);
  return res.data;
}
