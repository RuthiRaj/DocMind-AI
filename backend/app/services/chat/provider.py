"""
Abstract LLM Provider Interface.

Defines the base contract for all large language model completion interfaces
ensuring provider inversion.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Abstract interface defining text generation and client health check contract.
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        context: str,
        question: str,
        history: list | None = None,
        context_mode: str = "RAG",
        request_id: str | None = None,
        retrieved_chunks_count: int = 0,
        final_chunks_count: int = 0,
        **kwargs
    ) -> tuple[str, bool]:
        """
        Submits system context and query variables to the LLM backend for completions.

        Args:
            system_prompt (str): Core instructions guiding model responses.
            context (str): Document chunk text context.
            question (str): User query question.
            history (list | None): Optional list of prior conversation turn dicts [{role, content}].

        Returns:
            tuple[str, bool]: Generated completion text and whether the provider
                trimmed history/context to fit the preflight token budget.
        """
        pass

    @abstractmethod
    def model_name(self) -> str:
        """
        Returns active LLM model identifier.

        Returns:
            str: Model name.
        """
        pass

    @abstractmethod
    def provider_name(self) -> str:
        """
        Returns active provider name.

        Returns:
            str: Provider name.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Checks connectivity and API key verification to check if the service is healthy.

        Returns:
            bool: True if service checks pass, False otherwise.
        """
        pass
