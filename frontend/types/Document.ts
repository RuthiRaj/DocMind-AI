export interface DocumentListItem {
  document_id: string;
  filename: string;
  upload_time: string | null;
  total_pages: number | null;
  total_chunks: number | null;
  embedding_count: number | null;
  index_status: string | null;
  chat_ready: boolean;
  document_size: number;
  current_pipeline_stage: string;
}

export interface DocumentListResponse {
  success: boolean;
  documents: DocumentListItem[];
  total_count: number;
}

export interface DocumentDetailResponse {
  success: boolean;
  document_id: string;
  metadata: Record<string, any> | null;
  status: Record<string, any> | null;
  chunk_statistics: Record<string, any> | null;
  embedding_metadata: Record<string, any> | null;
  index_metadata: Record<string, any> | null;
}

export interface DeleteResponse {
  success: boolean;
  message: string;
}
