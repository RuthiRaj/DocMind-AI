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
        return (
            "You are DocMind AI, a helpful, precise, and professional document analysis assistant. "
            "Your sole objective is to answer the user's question using only the provided document text context segments. "
            "You must follow these strict grounding rules at all times:\n"
            "1. Answer the question using ONLY the provided document text context below.\n"
            "2. Do NOT use outside knowledge, external facts, or pre-trained assumptions to answer the question.\n"
            "3. If the answer cannot be found in the provided document text context, you must explicitly respond with exactly: "
            "\"I couldn't find enough information in this document to answer your question.\"\n"
            "4. Never invent, fabricate, or hallucinate details. Keep responses concise, factual, and professional.\n"
            "5. Preserve original technical terminology, abbreviations, and definitions present in the document.\n"
            "6. Do not mention vectors, embeddings, similarity scores, FAISS, chunks, prompt templates, or internal implementation details in your responses.\n"
            f"Prompt Template Version: {settings.SYSTEM_PROMPT_VERSION}"
        )

    @staticmethod
    def compile_context(chunks: list) -> str:
        """
        Concatenates chunk texts into a single context string with index labels.
        """
        compiled = []
        for i, chunk in enumerate(chunks, start=1):
            text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
            compiled.append(f"--- Document Context Segment {i} ---\n{text}")
        return "\n\n".join(compiled)
