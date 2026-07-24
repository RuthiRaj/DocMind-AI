export type PipelineStageStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'waiting';

export interface PipelineStatus {
  upload_status: PipelineStageStatus;
  processing_status: PipelineStageStatus;
  chunking_status: PipelineStageStatus;
  embedding_status: PipelineStageStatus;
  indexing_status: PipelineStageStatus;
  chat_ready: boolean;
}

export interface PipelineStatusResponse {
  success: boolean;
  document_id: string;
  upload_status: string | null;
  processing_status: string | null;
  chunking_status: string | null;
  embedding_status: string | null;
  indexing_status: string | null;
  chat_ready: boolean;
}
