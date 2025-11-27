"""Grok API Response Processor Module"""

import json
import uuid
import time
import asyncio
from typing import AsyncGenerator

from app.core.config import setting
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.models.openai_schema import (
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionMessage,
    OpenAIChatCompletionChunkResponse,
    OpenAIChatCompletionChunkChoice,
    OpenAIChatCompletionChunkMessage
)
from app.services.grok.cache import image_cache_service, video_cache_service
from app.services.cloudinary import cloudinary_client


class StreamTimeoutManager:
    """Stream Response Timeout Manager"""
    
    def __init__(self, chunk_timeout: int = 120, first_response_timeout: int = 30, total_timeout: int = 600):
        """Initialize timeout manager
        
        Args:
            chunk_timeout: Chunk interval timeout (seconds)
            first_response_timeout: First response timeout (seconds)
            total_timeout: Total timeout limit (seconds, 0 means no limit)
        """
        self.chunk_timeout = chunk_timeout
        self.first_response_timeout = first_response_timeout
        self.total_timeout = total_timeout
        
        self.start_time = asyncio.get_event_loop().time()
        self.last_chunk_time = self.start_time
        self.first_chunk_received = False
    
    def check_timeout(self) -> tuple[bool, str]:
        """Check for timeout
        
        Returns:
            (is_timeout, timeout_message): Whether timed out and timeout message
        """
        current_time = asyncio.get_event_loop().time()
        
        # Check first response timeout
        if not self.first_chunk_received:
            if current_time - self.start_time > self.first_response_timeout:
                return True, f"First response timeout ({self.first_response_timeout}s without first chunk)"
        
        # Check total timeout
        if self.total_timeout > 0:
            if current_time - self.start_time > self.total_timeout:
                return True, f"Stream response total timeout ({self.total_timeout}s)"
        
        # Check chunk interval timeout
        if self.first_chunk_received:
            if current_time - self.last_chunk_time > self.chunk_timeout:
                return True, f"Chunk interval timeout ({self.chunk_timeout}s without new data)"
        
        return False, ""
    
    def mark_chunk_received(self):
        """Mark chunk received"""
        self.last_chunk_time = asyncio.get_event_loop().time()
        self.first_chunk_received = True
    
    def get_total_duration(self) -> float:
        """Get total duration (seconds)"""
        return asyncio.get_event_loop().time() - self.start_time


class GrokResponseProcessor:
    """Grok API Response Processor"""

    @staticmethod
    async def process_normal(response, auth_token: str, model: str = None) -> OpenAIChatCompletionResponse:
        """Process non-stream response"""
        response_closed = False
        try:
            for chunk in response.iter_lines():
                if not chunk:
                    continue

                data = json.loads(chunk.decode("utf-8"))

                # Error check
                if error := data.get("error"):
                    raise GrokApiException(
                        f"API Error: {error.get('message', 'Unknown error')}",
                        "API_ERROR",
                        {"code": error.get("code")}
                    )

                # Extract response data
                grok_resp = data.get("result", {}).get("response", {})
                

                
                # Extract video data
                if video_resp := grok_resp.get("streamingVideoGenerationResponse"):
                    
                    if video_url := video_resp.get("videoUrl"):
                        logger.debug(f"[Processor] Video generation detected: {video_url}")
                        full_video_url = f"https://assets.grok.com/{video_url}"
                        
                        # Download and cache video
                        try:
                            cache_path = await video_cache_service.download_video(f"/{video_url}", auth_token)
                            if cache_path:
                                cloudinary_url = await asyncio.to_thread(cloudinary_client.upload_video, str(cache_path))
                                logger.info(f"[Processor] ✅ Video uploaded to Cloudinary: {cloudinary_url}")
                                content = f'<video src="{cloudinary_url}" controls="controls" width="500" height="300"></video>\n'
                            else:
                                logger.warning(f"[Processor] ⚠️  Video caching failed, using direct URL")
                                content = f'<video src="{full_video_url}" controls="controls" width="500" height="300"></video>\n'
                        except Exception as e:
                            logger.error(f"[Processor] ❌ Error caching video: {type(e).__name__}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            content = f'<video src="{full_video_url}" controls="controls" width="500" height="300"></video>\n'
                        
                        logger.info(f"[Processor] Final video content: {content}")
                        
                        # Return video response
                        result = OpenAIChatCompletionResponse(
                            id=f"chatcmpl-{uuid.uuid4()}",
                            object="chat.completion",
                            created=int(time.time()),
                            model=model or "grok-imagine-0.9",
                            choices=[OpenAIChatCompletionChoice(
                                index=0,
                                message=OpenAIChatCompletionMessage(
                                    role="assistant",
                                    content=content
                                ),
                                finish_reason="stop"
                            )],
                            usage=None
                        )
                        response_closed = True
                        response.close()
                        return result
                    else:
                        # Keep iterating - we haven't reached 100% yet
                        continue

                # Extract model response
                model_response = grok_resp.get("modelResponse")
                if not model_response:
                    continue

                # Check error in modelResponse
                if error_msg := model_response.get("error"):
                    raise GrokApiException(
                        f"Model response error: {error_msg}",
                        "MODEL_ERROR"
                    )

                # Build response content
                model_name = model_response.get("model")
                content = model_response.get("message", "")

                # Extract image data
                if images := model_response.get("generatedImageUrls"):
                    # Get image return mode
                    image_mode = setting.global_config.get("image_mode", "url")

                    for img in images:
                        try:
                            if image_mode == "base64":
                                # base64 mode: download and convert to base64
                                base64_str = await image_cache_service.download_base64(f"/{img}", auth_token)
                                if base64_str:
                                    content += f"\n![Generated Image]({base64_str})"
                                else:
                                    content += f"\n![Generated Image](https://assets.grok.com/{img})"
                            else:
                                # url mode: cache and return link
                                cache_path = await image_cache_service.download_image(f"/{img}", auth_token)
                                if cache_path:
                                    img_path = img.replace('/', '-')
                                    base_url = setting.global_config.get("base_url", "")
                                    img_url = f"{base_url}/images/{img_path}" if base_url else f"/images/{img_path}"
                                    content += f"\n![Generated Image]({img_url})"
                                else:
                                    content += f"\n![Generated Image](https://assets.grok.com/{img})"
                        except Exception as e:
                            logger.warning(f"[Processor] Failed to process image: {e}")
                            content += f"\n![Generated Image](https://assets.grok.com/{img})"

                # Return OpenAI response format
                result = OpenAIChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4()}",
                    object="chat.completion",
                    created=int(time.time()),
                    model=model_name,
                    choices=[OpenAIChatCompletionChoice(
                        index=0,
                        message=OpenAIChatCompletionMessage(
                            role="assistant",
                            content=content
                        ),
                        finish_reason="stop"
                    )],
                    usage=None
                )
                response_closed = True
                response.close()
                return result

            raise GrokApiException("No response data", "NO_RESPONSE")

        except json.JSONDecodeError as e:
            logger.error(f"[Processor] JSON decode failed: {e}")
            raise GrokApiException(f"JSON decode failed: {e}", "JSON_ERROR") from e
        except Exception as e:
            logger.error(f"[Processor] Unknown error during response processing: {type(e).__name__}: {e}")
            raise GrokApiException(f"Response processing error: {e}", "PROCESS_ERROR") from e
        finally:
            # Ensure response object is closed to avoid double release
            if not response_closed and hasattr(response, 'close'):
                try:
                    response.close()
                except Exception as e:
                    logger.warning(f"[Processor] Error closing response object: {e}")

    @staticmethod
    async def process_stream(response, auth_token: str) -> AsyncGenerator[str, None]:
        """Process stream response"""
        # Stream generation state
        is_image = False
        is_thinking = False
        thinking_finished = False
        chunk_index = 0
        model = None
        filtered_tags = setting.grok_config.get("filtered_tags", "").split(",")
        video_progress_started = False
        last_video_progress = -1
        response_closed = False

        # Initialize timeout manager
        timeout_manager = StreamTimeoutManager(
            chunk_timeout=setting.grok_config.get("stream_chunk_timeout", 120),
            first_response_timeout=setting.grok_config.get("stream_first_response_timeout", 30),
            total_timeout=setting.grok_config.get("stream_total_timeout", 600)
        )

        def make_chunk(chunk_content: str, finish: str = None):
            """Generate OpenAI format response chunk"""
            chunk_data = OpenAIChatCompletionChunkResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=model or "grok-4-mini-thinking-tahoe",
                choices=[OpenAIChatCompletionChunkChoice(
                    index=chunk_index,
                    delta=OpenAIChatCompletionChunkMessage(
                        role="assistant",
                        content=chunk_content
                    ) if chunk_content else {},
                    finish_reason=finish
                )]
            ).model_dump()
            # Return SSE format
            return f"data: {json.dumps(chunk_data)}\n\n"

        try:
            for chunk in response.iter_lines():
                # Timeout check
                is_timeout, timeout_msg = timeout_manager.check_timeout()
                if is_timeout:
                    logger.warning(f"[Processor] {timeout_msg}")
                    yield make_chunk("", "stop")
                    yield "data: [DONE]\n\n"
                    return

                logger.debug(f"[Processor] Received chunk: {len(chunk)} bytes")
                
                # DEBUG: Log RAW chunk data from Grok before any processing
                if chunk:
                    try:
                        raw_decoded = chunk.decode("utf-8")
                        logger.debug(f"[Processor] 🔴 RAW CHUNK FROM GROK: {raw_decoded}")
                    except Exception as e:
                        logger.debug(f"[Processor] Could not decode chunk: {e}")
                
                if not chunk:
                    continue

                try:
                    data = json.loads(chunk.decode("utf-8"))

                    # Error check
                    if error := data.get("error"):
                        error_msg = error.get('message', 'Unknown error')
                        logger.error(f"[Processor] Grok API returned error: {error_msg}")
                        yield make_chunk(f"Error: {error_msg}", "stop")
                        yield "data: [DONE]\n\n"
                        return

                    # Extract response data
                    grok_resp = data.get("result", {}).get("response", {})
                    logger.debug(f"[Processor] Parsing response data: {len(grok_resp)} fields")
                    
                    if not grok_resp:
                        continue

                    # Update model name
                    if user_resp := grok_resp.get("userResponse"):
                        if m := user_resp.get("model"):
                            model = m

                    # Extract video data
                    if video_resp := grok_resp.get("streamingVideoGenerationResponse"):
                        logger.debug(f"[Processor] 🎬 Stream: Video response chunk: {json.dumps(video_resp, indent=2)}")
                        progress = video_resp.get("progress", 0)
                        
                        if progress > last_video_progress:
                            last_video_progress = progress
                            logger.info(f"[Processor] 📊 Video generation progress: {progress}%")
                            
                            # Add <think> tag
                            if not video_progress_started:
                                content = f"<think>Video generated {progress}%\n"
                                video_progress_started = True
                                logger.info(f"[Processor] Started video progress tracking")
                            elif progress < 100:
                                content = f"Video generated {progress}%\n"
                            else:
                                # Close <think> tag when progress is 100% and process video immediately
                                logger.info(f"[Processor] ✅ Video generation complete (100%)")
                                content = f"Video generated {progress}%</think>\n"
                                
                                # Immediately download and cache video
                                if v_url := video_resp.get("videoUrl"):
                                    logger.info(f"[Processor] 🎥 Video URL found: {v_url}")
                                    full_video_url = f"https://assets.grok.com/{v_url}"
                                    logger.info(f"[Processor] Full video URL: {full_video_url}")
                                    
                                    try:
                                        logger.info(f"[Processor] Attempting to cache video...")
                                        cache_path = await video_cache_service.download_video(f"/{v_url}", auth_token)
                                        if cache_path:
                                            logger.info(f"[Processor] ✅ Video cached at: {cache_path}")
                                            cloudinary_url = await asyncio.to_thread(cloudinary_client.upload_video, str(cache_path))
                                            logger.info(f"[Processor] ✅ Video uploaded to Cloudinary: {cloudinary_url}")
                                            content += f'<video src="{cloudinary_url}" controls="controls"></video>\n'
                                        else:
                                            logger.warning(f"[Processor] ⚠️  Video caching failed, using direct URL")
                                            content += f'<video src="{full_video_url}" controls="controls"></video>\n'
                                    except Exception as e:
                                        logger.error(f"[Processor] ❌ Error caching video: {type(e).__name__}: {e}")
                                        import traceback
                                        logger.error(traceback.format_exc())
                                        content += f'<video src="{full_video_url}" controls="controls"></video>\n'
                                    
                                    logger.info(f"[Processor] Final video content: {content}")
                                else:
                                    logger.error(f"[Processor] ❌ Progress is 100% but no videoUrl found!")

                            yield make_chunk(content)
                            timeout_manager.mark_chunk_received()
                            chunk_index += 1
                        
                        continue

                    # Check generation mode
                    if grok_resp.get("imageAttachmentInfo"):
                        is_image = True

                    # Get token
                    token = grok_resp.get("token", "")

                    # Extract image data
                    if is_image:
                        if model_resp := grok_resp.get("modelResponse"):
                            # Get image return mode
                            image_mode = setting.global_config.get("image_mode", "url")

                            # Initialize content variable
                            content = ""

                            # Generate image link and cache
                            for img in model_resp.get("generatedImageUrls", []):
                                try:
                                    if image_mode == "base64":
                                        # base64 mode: download and convert to base64
                                        base64_str = await image_cache_service.download_base64(f"/{img}", auth_token)
                                        if base64_str:
                                            # Send base64 data in chunks, 8KB per chunk
                                            markdown_prefix = "![Generated Image](data:"
                                            markdown_suffix = ")\n"

                                            # Extract mime and base64 part of data URL
                                            if base64_str.startswith("data:"):
                                                parts = base64_str.split(",", 1)
                                                if len(parts) == 2:
                                                    mime_part = parts[0] + ","
                                                    b64_data = parts[1]

                                                    # Send prefix
                                                    yield make_chunk(markdown_prefix + mime_part)
                                                    timeout_manager.mark_chunk_received()
                                                    chunk_index += 1

                                                    # Send base64 data in chunks
                                                    chunk_size = 8192
                                                    for i in range(0, len(b64_data), chunk_size):
                                                        chunk_data = b64_data[i:i + chunk_size]
                                                        yield make_chunk(chunk_data)
                                                        timeout_manager.mark_chunk_received()
                                                        chunk_index += 1

                                                    # Send suffix
                                                    yield make_chunk(markdown_suffix)
                                                    timeout_manager.mark_chunk_received()
                                                    chunk_index += 1
                                                else:
                                                    yield make_chunk(f"![Generated Image]({base64_str})\n")
                                                    timeout_manager.mark_chunk_received()
                                                    chunk_index += 1
                                            else:
                                                yield make_chunk(f"![Generated Image]({base64_str})\n")
                                                timeout_manager.mark_chunk_received()
                                                chunk_index += 1
                                        else:
                                            yield make_chunk(f"![Generated Image](https://assets.grok.com/{img})\n")
                                            timeout_manager.mark_chunk_received()
                                            chunk_index += 1
                                    else:
                                        # url mode: cache and return link
                                        await image_cache_service.download_image(f"/{img}", auth_token)
                                        # Local image path
                                        img_path = img.replace('/', '-')
                                        base_url = setting.global_config.get("base_url", "")
                                        img_url = f"{base_url}/images/{img_path}" if base_url else f"/images/{img_path}"
                                        content += f"![Generated Image]({img_url})\n"
                                except Exception as e:
                                    logger.warning(f"[Processor] Failed to process image: {e}")
                                    content += f"![Generated Image](https://assets.grok.com/{img})\n"

                            # Send content
                            yield make_chunk(content.strip(), "stop")
                            timeout_manager.mark_chunk_received()
                            return
                        elif token:
                            yield make_chunk(token)
                            timeout_manager.mark_chunk_received()
                            chunk_index += 1

                    # Extract conversation data
                    else:
                        # Filter list format tokens
                        if isinstance(token, list):
                            continue

                        # Filter specific tags
                        if any(tag in token for tag in filtered_tags if token):
                            continue

                        # Get current status
                        current_is_thinking = grok_resp.get("isThinking", False)
                        message_tag = grok_resp.get("messageTag")

                        # Skip subsequent <think> tags
                        if thinking_finished and current_is_thinking:
                            continue

                        # Check toolUsageCardId - ALWAYS stream tool usage to reduce TTFT
                        if tool_usage_id := grok_resp.get("toolUsageCardId"):
                            # Stream tool usage notification IMMEDIATELY to keep connection alive
                            if not is_thinking:
                                # Start thinking block for tool usage
                                tool_notification = f"<think>\n[Tool: {tool_usage_id}]\n"
                                yield make_chunk(tool_notification)
                                timeout_manager.mark_chunk_received()
                                chunk_index += 1
                                is_thinking = True
                            
                            if web_search := grok_resp.get("webSearchResults"):
                                # Stream web search results IMMEDIATELY
                                search_content = ""
                                for result in web_search.get("results", []):
                                    title = result.get("title", "")
                                    url = result.get("url", "")
                                    preview = result.get("preview", "")
                                    preview_clean = preview.replace("\n", "") if isinstance(preview, str) else ""
                                    search_content += f'\n- [{title}]({url} "{preview_clean}")'
                                search_content += "\n"
                                
                                # Stream search results
                                yield make_chunk(search_content)
                                timeout_manager.mark_chunk_received()
                                chunk_index += 1
                                continue
                            else:
                                # Tool usage started but no results yet - still stream notification
                                continue

                        if token:
                            content = token

                            # Header newline after token
                            if message_tag == "header":
                                content = f"\n\n{token}\n\n"

                            # is_thinking status switch
                            if not is_thinking and current_is_thinking:
                                content = f"<think>\n{content}"
                            elif is_thinking and not current_is_thinking:
                                content = f"\n</think>\n{content}"
                                thinking_finished = True

                            yield make_chunk(content)
                            timeout_manager.mark_chunk_received()
                            chunk_index += 1
                            is_thinking = current_is_thinking

                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"[Processor] Failed to parse chunk: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"[Processor] Error processing chunk: {e}")
                    continue

            # Send end chunk
            yield make_chunk("", "stop")

            # Send stream end marker
            yield "data: [DONE]\n\n"
            
            # Log stream response statistics
            logger.info(f"[Processor] Stream response completed, total duration: {timeout_manager.get_total_duration():.2f}s")

        except Exception as e:
            logger.error(f"[Processor] Severe error in stream processing: {e}")
            yield make_chunk(f"Processing error: {e}", "error")
            # Send stream end marker
            yield "data: [DONE]\n\n"
        finally:
            # Ensure response object is closed
            if not response_closed and hasattr(response, 'close'):
                try:
                    response.close()
                    logger.debug("[Processor] Stream response object closed")
                except Exception as e:
                    logger.warning(f"[Processor] Error closing stream response object: {e}")