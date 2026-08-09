export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api';

export const API_TIMEOUT = 30000; // 30 seconds default timeout

export const ENDPOINTS = {
  HEALTH: '/health',
  DOCUMENTS: '/documents',
  DOCUMENTS_STATISTICS: '/documents/statistics',
  RETRIEVAL: (id: string) => `/retrieve/${id}`,
  CHAT: (id: string) => `/chat/${id}`,
  UPLOAD: '/upload',
  CLEANUP: '/maintenance/cleanup',
  DOCUMENT_STATUS: (id: string) => `/documents/${id}/status`,
  DOCUMENT_DETAILS: (id: string) => `/documents/${id}`,
  DOCUMENT_DELETE: (id: string) => `/documents/${id}`,
};
