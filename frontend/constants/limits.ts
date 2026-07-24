export const FILE_LIMITS = {
  MAX_SIZE_BYTES: 25 * 1024 * 1024, // 25 MB max upload
  ALLOWED_EXTENSIONS: ['.pdf'],
  ALLOWED_MIME_TYPES: ['application/pdf'],
};

export const RETRIEVAL_LIMITS = {
  DEFAULT_TOP_K: 5,
  MAX_TOP_K: 20,
  MIN_TOP_K: 1,
};

export const CHAT_LIMITS = {
  MAX_QUERY_CHARACTERS: 1000,
};
