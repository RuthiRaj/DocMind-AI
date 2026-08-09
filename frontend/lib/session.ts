/**
 * Session ID management for chat conversation memory.
 * 
 * Generates and persists a unique session UUID per document in localStorage,
 * allowing the backend to scope conversation history per (document_id, session_id) pair.
 */

const SESSION_KEY_PREFIX = 'docmind_chat_session_';

/**
 * Get the session ID for a specific document. Generates one if it doesn't exist.
 */
export function getSessionId(documentId: string): string {
  if (typeof window === 'undefined') return '';
  
  const key = `${SESSION_KEY_PREFIX}${documentId}`;
  let sessionId = localStorage.getItem(key);
  
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(key, sessionId);
  }
  
  return sessionId;
}

/**
 * Set/update the session ID for a specific document.
 * Called when the server returns a session_id (e.g., on first request when client didn't send one).
 */
export function setSessionId(documentId: string, sessionId: string): void {
  if (typeof window === 'undefined') return;
  
  const key = `${SESSION_KEY_PREFIX}${documentId}`;
  localStorage.setItem(key, sessionId);
}

/**
 * Clear the session ID for a specific document.
 * Used when the user clears chat history to start a fresh conversation.
 */
export function clearSessionId(documentId: string): void {
  if (typeof window === 'undefined') return;
  
  const key = `${SESSION_KEY_PREFIX}${documentId}`;
  localStorage.removeItem(key);
}
