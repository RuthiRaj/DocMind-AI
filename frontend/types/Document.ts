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

export interface DocumentMetadata {
  filename: string;
  upload_time: string;
  file_size: number;
  total_pages: number;
}

export interface DocumentStatus {
  upload_status: string;
  processing_status: string;
  chunking_status: string;
  embedding_status: string;
  indexing_status: string;
  chat_ready: boolean;
}

export interface ChunkStatistics {
  total_chunks: number;
  average_chunk_size: number;
  max_chunk_size: number;
  chunk_size: number;
  chunk_overlap: number;
}

export interface EmbeddingMetadata {
  model_name: string;
  dimension: number;
}

export interface IndexMetadata {
  total_indexed_vectors: number;
  metric: string;
}

export interface DocumentDetailResponse {
  success: boolean;
  document_id: string;
  metadata: DocumentMetadata | null;
  status: DocumentStatus | null;
  chunk_statistics: ChunkStatistics | null;
  embedding_metadata: EmbeddingMetadata | null;
  index_metadata: IndexMetadata | null;
}

export interface DeleteResponse {
  success: boolean;
  message: string;
}
