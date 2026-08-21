import logging
import threading
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class ConversationStore:
    """
    A thread-safe in-memory conversation store keyed by (document_id, session_id).
    
    NOTE: In-memory storage — history will not survive server restart and will not 
    be shared correctly across multiple worker processes if the app is ever run 
    with --workers > 1. Flagged as follow-up work.
    """
    
    # Class-level store and lock
    _store: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    _lock: threading.Lock = threading.Lock()

    def add_turn(self, document_id: str, session_id: str, question: str, answer: str) -> None:
        """
        Appends user and assistant messages to the history.

        Args:
            document_id (str): The ID of the document.
            session_id (str): The ID of the session.
            question (str): The user's question.
            answer (str): The assistant's answer.
        """
        key = (document_id, session_id)
        with self._lock:
            if key not in self._store:
                self._store[key] = []
            
            self._store[key].append({"role": "user", "content": question})
            self._store[key].append({"role": "assistant", "content": answer})
            logger.debug(f"Added turn to conversation store for {key}")

    def get_history(self, document_id: str, session_id: str, max_turns: int = 2, max_tokens: int = 350) -> List[Dict[str, str]]:
        """
        Returns the last N conversation turns in chronological order (oldest first),
        truncated by token count (estimate 4 chars = 1 token).

        Args:
            document_id (str): The ID of the document.
            session_id (str): The ID of the session.
            max_turns (int, optional): Maximum number of turns to return. Defaults to 2.
            max_tokens (int, optional): Maximum tokens to include. Defaults to 350.

        Returns:
            List[Dict[str, str]]: Conversation history messages in chronological order.
        """
        key = (document_id, session_id)
        with self._lock:
            history = self._store.get(key, [])
            if not history:
                return []
            
            # Collect turns from most recent backward, respecting token budget
            selected_turns = []
            token_count = 0
            max_message_chars = max(1, (max_tokens * 4) // 2)
            
            # Iterate backward through turn pairs (user + assistant = 2 messages per turn)
            i = len(history) - 2
            while i >= 0 and len(selected_turns) // 2 < max_turns:
                user_msg = {
                    "role": history[i].get("role", "user"),
                    "content": history[i].get("content", "")[-max_message_chars:]
                }
                assistant_msg = {
                    "role": history[i + 1].get("role", "assistant"),
                    "content": history[i + 1].get("content", "")[-max_message_chars:]
                }
                
                # Estimate tokens (4 chars ≈ 1 token)
                turn_chars = len(user_msg.get("content", "")) + len(assistant_msg.get("content", ""))
                turn_tokens = (turn_chars + 3) // 4
                
                if token_count + turn_tokens > max_tokens:
                    break
                    
                token_count += turn_tokens
                # Prepend to maintain chronological order
                selected_turns.insert(0, assistant_msg)
                selected_turns.insert(0, user_msg)
                i -= 2
                
            return selected_turns

    def clear(self, document_id: str, session_id: str) -> None:
        """
        Clears the conversation history for a given document and session.

        Args:
            document_id (str): The ID of the document.
            session_id (str): The ID of the session.
        """
        key = (document_id, session_id)
        with self._lock:
            if key in self._store:
                del self._store[key]
                logger.debug(f"Cleared conversation store for {key}")

# Module-level singleton
conversation_store = ConversationStore()
