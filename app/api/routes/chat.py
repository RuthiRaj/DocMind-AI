"""
PDF AI Chat (RAG) API Route Handler.

Provides the POST /chat/{document_id} endpoint for grounded document question completions.
"""

from fastapi import APIRouter, status
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.upload import HTTPErrorDetail
from app.services.pdf.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()


@router.post(
    "/chat/{document_id}",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Stateless Grounded RAG Chat Completion",
    description="Loads relevant segment context using the semantic retrieval service, constructs modular grounding prompts, queries the abstract LLM provider, and outputs structured source citations.",
    responses={
        200: {
            "model": ChatResponse,
            "description": "Grounded completion query successfully resolved"
        },
        400: {
            "model": HTTPErrorDetail,
            "description": "Bad Request (Empty question, pipeline incomplete, or query boundaries validation failure)"
        },
        404: {
            "model": HTTPErrorDetail,
            "description": "Not Found (Specified document_id does not exist)"
        },
        500: {
            "model": HTTPErrorDetail,
            "description": "Internal Server Error (Provider connection timeouts or server configuration errors)"
        }
    }
)
async def query_chat(
    document_id: str,
    request: ChatRequest
) -> ChatResponse:
    """
    HTTP POST endpoint handler for document-grounded chat queries.

    Args:
        document_id (str): Unique UUID document identifier.
        request (ChatRequest): Query question details.

    Returns:
        ChatResponse: Structured answer summary including sources and execution times.
    """
    return await chat_service.answer_question(document_id, request)
