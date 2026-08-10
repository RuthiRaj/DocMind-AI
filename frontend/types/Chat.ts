export interface ChatCitation {
  chunk_id: string;
  document_id: string;
  page_number: number;
  text: string;
  similarity_score: number;
  chunk_index: number;
}

export interface ChatResponse {
  answer: string;
  citations: ChatCitation[];
  contextTruncated?: boolean;
}

export interface Message {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  citations?: ChatCitation[];
  isError?: boolean;
}
