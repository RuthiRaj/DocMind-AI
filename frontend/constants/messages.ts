export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Unable to communicate with the DocMind AI server. Please check your connection.',
  TIMEOUT_ERROR: 'The server request took too long to complete. Please try again.',
  UNKNOWN_ERROR: 'An unexpected system error occurred. Please try again.',
  PDF_ONLY: 'Only PDF documents are supported.',
  FILE_TOO_LARGE: (maxMb: number) => `File size exceeds the limit of ${maxMb}MB.`,
};

export const CHAT_SUGGESTIONS = [
  'What is the primary topic of this document?',
  'Can you summarize the main findings?',
  'What are the key recommendations outlined?',
  'Who is the author or target audience?',
];
