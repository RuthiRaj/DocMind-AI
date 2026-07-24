export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
}

export interface ApiError {
  message: string;
  statusCode?: number;
  details?: string | Record<string, any>;
}

export interface CleanupResponse {
  success: boolean;
  removed_temp_files: number;
  removed_empty_directories: number;
  message: string;
}
