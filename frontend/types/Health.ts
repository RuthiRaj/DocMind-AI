export interface HealthCheckItem {
  status: 'healthy' | 'unhealthy';
  path?: string;
  details: string;
  total_bytes?: number;
  used_bytes?: number;
  free_bytes?: number;
}

export interface SystemHealth {
  status: 'healthy' | 'unhealthy';
  uploads_directory: HealthCheckItem;
  write_permission: HealthCheckItem;
  disk_usage: HealthCheckItem;
  embedding_model: HealthCheckItem;
  faiss_library: HealthCheckItem;
  groq_service: HealthCheckItem;
  backend_version: string;
  uptime_seconds: number;
  total_documents: number;
}
