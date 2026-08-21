"""
RAG Prompt Construction Builder.

Compiles instructions, grounding constraints, document context, and query variables
into clean prompts to minimize hallucination and enforce contextual bounds.
"""

from app.core.config import settings


class PromptBuilder:
    """
    Builder class for constructing document-grounded system and user prompts.
    """

    @staticmethod
    def get_system_prompt() -> str:
        """
        Retrieves the core grounding instructions system prompt.
        """
        from app.core.prompts import RAG_SYSTEM_PROMPT
        return RAG_SYSTEM_PROMPT

    @staticmethod
    def compile_context(chunks: list) -> str:
        """
        Concatenates chunk texts into a single context string with index labels,
        preserving original reading order, removing repeated sentences, and joining cleanly.
        """
        def _get_val(obj, key, default=None):
            if hasattr(obj, key):
                v = getattr(obj, key)
                return v if v is not None else default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return default

        # Ensure chunks are sorted by start_page and chunk_index to preserve original reading flow
        sorted_chunks = sorted(
            chunks, 
            key=lambda x: (
                _get_val(x, "start_page", 1),
                _get_val(x, "chunk_index", 1)
            )
        )
        
        compiled = []
        for i, chunk in enumerate(sorted_chunks, start=1):
            text = _get_val(chunk, "text", "")
            if not text or not text.strip():
                continue
            
            clean_chunk_text = text.strip()
            
            # Format page numbers and chunk index ranges
            start_page = _get_val(chunk, "start_page", 1)
            end_page = _get_val(chunk, "end_page", 1)
            chunk_index = _get_val(chunk, "chunk_index", 1)
            last_chunk_index = _get_val(chunk, "last_chunk_index", None)
            
            page_str = f"{start_page} to {end_page}" if start_page != end_page else f"{start_page}"
            chunk_str = f"{chunk_index} to {last_chunk_index}" if last_chunk_index is not None else f"{chunk_index}"
            
            segment_info = (
                f"[Document Content (Pages {page_str})]\n"
                f"{clean_chunk_text}"
            )
            compiled.append(segment_info)
                
        return "\n\n".join(compiled)


    @staticmethod
    def get_full_context_system_prompt() -> str:
        """
        Retrieves the full-context grounding instructions system prompt.
        Used when the entire document fits within the LLM context window.
        """
        from app.core.prompts import FULL_CONTEXT_SYSTEM_PROMPT
        return FULL_CONTEXT_SYSTEM_PROMPT

    @staticmethod
    def compile_full_context(full_text: str, pages: list) -> str:
        """
        Inserts [Page N] markers into the full document text at each page boundary,
        using character offset metadata from pages.json.

        Args:
            full_text (str): Complete extracted document text.
            pages (list): List of page metadata dicts with 'page' and 'start_character' keys.

        Returns:
            str: Full document text with [Page N] markers inserted at page boundaries.
        """
        if not pages:
            return full_text

        # Sort pages by start_character ascending to process in order
        sorted_pages = sorted(pages, key=lambda p: p.get("start_character", 0))

        # Build text with page markers inserted at each page boundary
        # Process from end to start to preserve character offsets during insertion
        marked_text = full_text
        for page_info in reversed(sorted_pages):
            page_num = page_info.get("page", 1)
            start_char = page_info.get("start_character", 0)

            # Insert [Page N] marker at the start of each page's content
            marker = f"\n\n[Page {page_num}]\n"
            if start_char == 0:
                # First page — prepend marker at the very beginning
                marked_text = f"[Page {page_num}]\n" + marked_text
            elif start_char <= len(marked_text):
                marked_text = marked_text[:start_char] + marker + marked_text[start_char:]

        return marked_text
