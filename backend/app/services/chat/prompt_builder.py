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
        # Ensure chunks are sorted by start_page and chunk_index to preserve original reading flow
        sorted_chunks = sorted(
            chunks, 
            key=lambda x: (
                getattr(x, "start_page", 1) if hasattr(x, "start_page") else x.get("start_page", 1),
                getattr(x, "chunk_index", 1) if hasattr(x, "chunk_index") else x.get("chunk_index", 1)
            )
        )
        
        import re
        seen_sentences = set()
        compiled = []
        
        for i, chunk in enumerate(sorted_chunks, start=1):
            text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
            
            # Segment text into sentences using simple regex splits on punctuations with whitespace boundaries
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            deduped_sentences = []
            for sentence in sentences:
                s_strip = sentence.strip()
                if not s_strip:
                    continue
                # Normalize sentence for duplicate tracking
                norm_s = " ".join(s_strip.lower().split())
                if norm_s in seen_sentences:
                    continue
                seen_sentences.add(norm_s)
                deduped_sentences.append(s_strip)
                
            if deduped_sentences:
                clean_chunk_text = " ".join(deduped_sentences)
                
                # Format page numbers and chunk index ranges
                start_page = getattr(chunk, "start_page", 1) if hasattr(chunk, "start_page") else chunk.get("start_page", 1)
                end_page = getattr(chunk, "end_page", 1) if hasattr(chunk, "end_page") else chunk.get("end_page", 1)
                chunk_index = getattr(chunk, "chunk_index", 1) if hasattr(chunk, "chunk_index") else chunk.get("chunk_index", 1)
                last_chunk_index = getattr(chunk, "last_chunk_index", None) if hasattr(chunk, "last_chunk_index") else chunk.get("last_chunk_index", None)
                
                page_str = f"{start_page} to {end_page}" if start_page != end_page else f"{start_page}"
                chunk_str = f"{chunk_index} to {last_chunk_index}" if last_chunk_index is not None else f"{chunk_index}"
                
                segment_info = (
                    f"[Document Segment {i}]\n"
                    f"Page: {page_str}\n"
                    f"Chunk: {chunk_str}\n"
                    f"Content:\n"
                    f"{clean_chunk_text}"
                )
                compiled.append(segment_info)
                
        return "\n\n".join(compiled)
