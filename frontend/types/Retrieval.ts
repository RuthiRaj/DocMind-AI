export interface RetrievalResultItem {
  chunk_id: string;
  document_id: string;
  page_number: number;
  text: string;
  similarity_score: number;
}

export interface RetrievalResponse {
  query: string;
  results: RetrievalResultItem[];
  total_results: number;
  retrieval_time_seconds: number;
}
