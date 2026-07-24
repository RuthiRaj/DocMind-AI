export interface StorageStatistics {
  total_documents: number;
  completed_documents: number;
  failed_documents: number;
  processing_documents: number;
  total_pages: number;
  total_chunks: number;
  total_embeddings: number;
  total_indexes: number;
  storage_bytes: number;
  generated_at: string;
}
