# -*- coding: utf-8 -*-
"""
Chat API Router Module

Provides OpenAI-compatible chat API endpoints, supporting interaction with Grok models.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from fastapi.responses import StreamingResponse

from app.core.auth import auth_manager
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.services.grok.client import GrokClient
from app.models.openai_schema import OpenAIChatRequest

# Chat Router
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/completions", response_model=None)
async def chat_completions(
    request: OpenAIChatRequest,
    _: Optional[str] = Depends(auth_manager.verify)
):
    """
    Create Chat Completion

    OpenAI-compatible chat API endpoint, supporting streaming and non-streaming responses.

    Args:
        request: OpenAI format chat request
        _: Auth dependency (auto-verified)

    Returns:
        OpenAIChatCompletionResponse: Non-streaming response
        StreamingResponse: Streaming response

    Raises:
        HTTPException: When request processing fails
    """
    try:
        logger.info(f"[Chat] Chat Request - Model: {request.model}")

        # Call Grok client to process request
        result = await GrokClient.openai_to_grok(request.model_dump())
        
        # If streaming response, GrokClient has returned Iterator, wrap as StreamingResponse
        if request.stream:
            return StreamingResponse(
                content=result,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # Return non-streaming response directly
        return result
        
    except GrokApiException as e:
        logger.error(f"[Chat] Grok API Error: {str(e)} - Details: {e.details}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": e.error_code or "grok_api_error",
                    "code": e.error_code or "unknown"
                }
            }
        )
    except Exception as e:
        logger.error(f"[Chat] Chat Request Processing Failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Internal Server Error",
                    "type": "internal_error",
                    "code": "internal_server_error"
                }
            }
        )
