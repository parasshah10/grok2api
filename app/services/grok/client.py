"""Grok API Client Module"""

import asyncio
import json
from typing import Dict, List, Tuple, Any

from curl_cffi import requests as curl_requests

from app.core.config import setting
from app.core.logger import logger
from app.models.grok_models import Models
from app.services.grok.processer import GrokResponseProcessor
from app.services.grok.statsig import get_dynamic_headers
from app.services.grok.token import token_manager
from app.services.grok.upload import ImageUploadManager
from app.core.exception import GrokApiException

# Constant Definition
GROK_API_ENDPOINT = "https://grok.com/rest/app-chat/conversations/new"
REQUEST_TIMEOUT = 120
IMPERSONATE_BROWSER = "chrome133a"
MAX_RETRY = 3  # Maximum retry attempts


class GrokClient:
    """Grok API Client"""

    @staticmethod
    def _is_data_url(url: str) -> bool:
        """Check if URL is a base64 data URL"""
        return url.strip().startswith("data:image") and ";base64," in url
    
    @staticmethod
    def _decode_data_url(data_url: str) -> Tuple[bytes, str]:
        """Decode base64 data URL to bytes and extract mime type"""
        import re
        
        try:
            # Split header and data
            header, encoded = data_url.split(",", 1)
            
            # Extract mime type from header (e.g., "data:image/png;base64")
            mime_match = re.search(r"data:(image/[a-zA-Z0-9-.+]+);base64", header)
            mime_type = mime_match.group(1) if mime_match else "image/jpeg"
            
            # Decode base64
            import base64
            decoded_bytes = base64.b64decode(encoded)
            
            return decoded_bytes, mime_type
        except Exception as e:
            logger.error(f"[Client] Failed to decode data URL: {e}")
            raise GrokApiException(f"Invalid data URL: {e}", "DATA_URL_DECODE_ERROR")

    @staticmethod
    async def openai_to_grok(openai_request: dict):
        """Convert OpenAI request to Grok request and handle response"""
        model = openai_request["model"]
        messages = openai_request["messages"]
        stream = openai_request.get("stream", False)

        logger.debug(f"[Client] Processing request - Model:{model}, Message Count:{len(messages)}, Streaming:{stream}")

        # Extract message content and image URLs
        content, image_urls = GrokClient._extract_content(messages)
        model_name, model_mode = Models.to_grok(model)
        is_video_model = Models.get_model_info(model).get("is_video_model", False)
        
        # Special handling for video models
        if is_video_model:
            if len(image_urls) > 1:
                logger.warning(f"[Client] Video model allows only one image, currently has {len(image_urls)}, using only the first one")
                image_urls = image_urls[:1]
            
            # If we have an image, use the new video generation flow
            if image_urls:
                from app.services.grok.video import video_generator
                import tempfile
                import os
                import aiohttp
                
                image_url = image_urls[0]
                
                # Log start of video generation without exposing full data URLs
                if GrokClient._is_data_url(image_url):
                    logger.info("[Client] 🎬 Starting video generation flow for image: <data URL>")
                else:
                    logger.info(f"[Client] 🎬 Starting video generation flow for image: {image_url[:100]}...")
                
                # Download or decode image to temp file
                # FIX: Unindented this block so it runs for both Data URLs and HTTP URLs
                try:
                    tmp_path = None
                    
                    # Check if it's a data URL or HTTP URL
                    if GrokClient._is_data_url(image_url):
                        logger.debug("[Client] Detected data URL, decoding base64...")
                        image_bytes, mime_type = GrokClient._decode_data_url(image_url)
                        
                        # Determine file extension from mime type
                        extension_map = {
                            "image/jpeg": ".jpg",
                            "image/png": ".png",
                            "image/gif": ".gif",
                            "image/webp": ".webp",
                            "image/bmp": ".bmp"
                        }
                        suffix = extension_map.get(mime_type, ".jpg")
                        
                        # Write to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(image_bytes)
                            tmp_path = tmp.name
                        logger.debug(f"[Client] ✅ Data URL decoded to: {tmp_path}")
                        
                    else:
                        logger.debug(f"[Client] Detected HTTP URL, downloading from: {image_url[:100]}...")
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                        async with aiohttp.ClientSession() as session:
                            async with session.get(image_url, headers=headers) as resp:
                                if resp.status == 200:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                                        tmp.write(await resp.read())
                                        tmp_path = tmp.name
                                    logger.debug(f"[Client] ✅ HTTP image downloaded to: {tmp_path}")
                                else:
                                    logger.error(f"[Client] Failed to download image: HTTP {resp.status}")
                                    raise GrokApiException(f"Failed to download image: HTTP {resp.status}", "IMAGE_DOWNLOAD_ERROR")
                    
                    # Generate video with user's actual prompt
                    if tmp_path:
                        try:
                            result = await video_generator.generate_video_from_image(
                                image_path=tmp_path,
                                prompt=content,  # Pass the user's actual text prompt!
                                mode="normal",
                                model_name=model_name
                            )
                            
                            if result.get("success"):
                                # Get video URL from result
                                video_url_path = result.get("conversation_result", {}).get("data", {}).get("video_url")
                                
                                if video_url_path:
                                    # Cache and upload to Cloudinary (matching original processor logic)
                                    from app.services.grok.processer import OpenAIChatCompletionResponse, OpenAIChatCompletionChoice, OpenAIChatCompletionMessage
                                    from app.services.grok.cache import video_cache_service
                                    from app.services.cloudinary.client import cloudinary_client
                                    import time
                                    import uuid
                                    
                                    full_video_url = f"https://assets.grok.com/{video_url_path}"
                                    logger.info(f"[Client] ✅ Video URL found: {full_video_url}")
                                    
                                    try:
                                        # Get auth token for downloading
                                        auth_token = token_manager.get_token(model_name)
                                        
                                        # Download and cache video
                                        logger.info(f"[Client] Attempting to cache video...")
                                        cache_path = await video_cache_service.download_video(f"/{video_url_path}", auth_token)
                                        
                                        if cache_path:
                                            logger.info(f"[Client] ✅ Video cached at: {cache_path}")
                                            # Upload to Cloudinary
                                            cloudinary_url = await asyncio.to_thread(cloudinary_client.upload_video, str(cache_path))
                                            logger.info(f"[Client] ✅ Video uploaded to Cloudinary: {cloudinary_url}")
                                            content = f'<video src="{cloudinary_url}" controls="controls" width="500" height="300"></video>\n'
                                        else:
                                            logger.warning(f"[Client] ⚠️  Video caching failed, using direct URL")
                                            content = f'<video src="{full_video_url}" controls="controls" width="500" height="300"></video>\n'
                                    except Exception as cache_error:
                                        logger.error(f"[Client] ❌ Error caching/uploading video: {cache_error}")
                                        content = f'<video src="{full_video_url}" controls="controls" width="500" height="300"></video>\n'
                                    
                                    return OpenAIChatCompletionResponse(
                                        id=f"chatcmpl-{uuid.uuid4()}",
                                        object="chat.completion",
                                        created=int(time.time()),
                                        model=model,
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
                                else:
                                    logger.error(f"[Client] Video generation succeeded but no video URL in response")
                            else:
                                error_msg = result.get("error", "Unknown error")
                                logger.error(f"[Client] Video generation failed: {error_msg}")
                        finally:
                            # Cleanup temp file
                            if tmp_path and os.path.exists(tmp_path):
                                os.remove(tmp_path)
                                logger.debug(f"[Client] Cleaned up temp file: {tmp_path}")
                        
                        # If we got here, video generation failed
                        raise GrokApiException("Video generation failed", "VIDEO_GENERATION_ERROR")
                    else:
                        raise GrokApiException("Failed to prepare image for video generation", "IMAGE_PREPARATION_ERROR")
                        
                except GrokApiException:
                    # Re-raise GrokApiExceptions as-is
                    raise
                except Exception as e:
                    logger.error(f"[Client] Error in video flow: {e}")
                    raise GrokApiException(f"Video generation error: {e}", "VIDEO_ERROR")

        # Retry logic (OLD FLOW - should only be used for non-video or fallback)
        return await GrokClient._try(model, content, image_urls, model_name, model_mode, is_video_model, stream)

    @staticmethod
    async def _try(model: str, content: str, image_urls: List[str], model_name: str, model_mode: str, is_video: bool, stream: bool):
        """Execute request with retries"""
        last_err = None
        
        for i in range(MAX_RETRY):
            try:
                # Get token
                auth_token = token_manager.get_token(model)
                
                # Upload images
                imgs = await GrokClient._upload_imgs(image_urls, auth_token)
                
                # Build and send request
                payload = GrokClient._build_payload(content, model_name, model_mode, imgs, is_video)
                return await GrokClient._send_request(payload, auth_token, model, stream)
                
            except GrokApiException as e:
                last_err = e
                # 401/429 are retryable, other errors raise immediately
                if e.error_code not in ["HTTP_ERROR", "NO_AVAILABLE_TOKEN"]:
                    raise
                
                # Check if status code is retryable
                status = e.context.get("status") if e.context else None
                if status not in [401, 429]:
                    raise
                
                if i < MAX_RETRY - 1:
                    logger.warning(f"[Client] Request failed (Status Code:{status}), Retry {i+1}/{MAX_RETRY}")
                    await asyncio.sleep(0.5)  # Short delay
                else:
                    logger.error(f"[Client] Failed after {MAX_RETRY} retries")
        
        raise last_err if last_err else GrokApiException("Request failed", "REQUEST_ERROR")

    @staticmethod
    def _extract_content(messages: List[Dict]) -> Tuple[str, List[str]]:
        """Extract message content and image URLs"""
        content_parts = []
        image_urls = []

        for msg in messages:
            msg_content = msg.get("content", "")

            # Handle complex message format (containing text and images)
            if isinstance(msg_content, list):
                for item in msg_content:
                    item_type = item.get("type")
                    if item_type == "text":
                        content_parts.append(item.get("text", ""))
                    elif item_type == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            image_urls.append(url)
            # Handle plain text message
            else:
                content_parts.append(msg_content)

        return "".join(content_parts), image_urls

    @staticmethod
    async def _upload_imgs(image_urls: List[str], auth_token: str) -> List[str]:
        """Upload images and return attachment ID list"""
        image_attachments = []
        # Upload all images concurrently
        tasks = [ImageUploadManager.upload(url, auth_token) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, result in zip(image_urls, results):
            if isinstance(result, Exception):
                logger.warning(f"[Client] Image upload failed: {url}, Error: {result}")
            elif result:
                image_attachments.append(result)

        return image_attachments

    @staticmethod
    def _build_payload(content: str, model_name: str, model_mode: str, image_attachments: List[str], is_video_model: bool = False) -> Dict[str, Any]:
        """Build Grok API request payload"""
        payload = {
            "temporary": setting.grok_config.get("temporary", True),
            "modelName": model_name,
            "message": content,
            "fileAttachments": image_attachments,
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "sendFinalMetadata": True,
            "isReasoning": False,
            "webpageUrls": [],
            "disableTextFollowUps": True,
            "responseMetadata": {"requestModelDetails": {"modelId": model_name}},
            "disableMemory": False,
            "forceSideBySide": False,
            "modelMode": model_mode,
            "isAsyncChat": False
        }
        
        # Special configuration for video models
        if is_video_model:
            import uuid
            # Generate a unique parentPostId for the video asset
            parent_post_id = str(uuid.uuid4())
            
            payload["toolOverrides"] = {"videoGen": True}
            # Add video generation config to responseMetadata
            payload["responseMetadata"] = {
                "requestModelDetails": {"modelId": model_name},
                "modelConfigOverride": {
                    "modelMap": {
                        "videoGenModelConfig": {
                            "parentPostId": parent_post_id,  # Critical for completing to 100%
                            "aspectRatio": "16:9",  # Default aspect ratio
                            "videoLength": 5  # Default 5 seconds
                        }
                    }
                }
            }
            logger.info(f"[Client] 🎬 Video generation config: parentPostId={parent_post_id}, aspectRatio=16:9, videoLength=5")
            logger.debug(f"[Client] 📤 Full video payload: {json.dumps(payload, indent=2)}")
        
        return payload

    @staticmethod
    async def _send_request(payload: dict, auth_token: str, model: str, stream: bool):
        """Send HTTP request to Grok API"""
        # Verify auth token
        if not auth_token:
            raise GrokApiException("Missing auth token", "NO_AUTH_TOKEN")

        try:
            # Build headers
            headers = GrokClient._build_headers(auth_token)
            
            # Use service proxy
            proxy_url = setting.get_service_proxy()
            proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
            
            if proxy_url:
                logger.debug(f"[Client] Using service proxy: {proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url}")

            # Build request parameters
            request_kwargs = {
                "headers": headers,
                "data": json.dumps(payload),
                "impersonate": IMPERSONATE_BROWSER,
                "timeout": REQUEST_TIMEOUT,
                "stream": True,
                "proxies": proxies
            }

            # Execute synchronous HTTP request in thread pool to avoid blocking event loop
            response = await asyncio.to_thread(
                curl_requests.post,
                GROK_API_ENDPOINT,
                **request_kwargs
            )

            logger.debug(f"[Client] API Response Status Code: {response.status_code}")

            # Handle non-successful response
            if response.status_code != 200:
                GrokClient._handle_error(response, auth_token)

            # Request successful, reset failure count
            asyncio.create_task(token_manager.reset_failure(auth_token))

            # Process and return response
            return await GrokClient._process_response(response, auth_token, model, stream)

        except curl_requests.RequestsError as e:
            logger.error(f"[Client] Network request error: {e}")
            raise GrokApiException(f"Network error: {e}", "NETWORK_ERROR") from e
        except json.JSONDecodeError as e:
            logger.error(f"[Client] JSON parse error: {e}")
            raise GrokApiException(f"JSON parse error: {e}", "JSON_ERROR") from e
        except Exception as e:
            logger.error(f"[Client] Unknown request error: {type(e).__name__}: {e}")
            raise GrokApiException(f"Request processing error: {e}", "REQUEST_ERROR") from e

    @staticmethod
    def _build_headers(auth_token: str) -> Dict[str, str]:
        """Build request headers"""
        headers = get_dynamic_headers("/rest/app-chat/conversations/new")

        # Build Cookie
        cf_clearance = setting.grok_config.get("cf_clearance", "")
        headers["Cookie"] = f"{auth_token};{cf_clearance}" if cf_clearance else auth_token

        return headers

    @staticmethod
    def _handle_error(response, auth_token: str):
        """Handle error response"""
        try:
            error_data = response.json()
            error_message = str(error_data)
        except Exception as e:
            error_data = response.text
            error_message = error_data[:200] if error_data else e

        # Record Token failure
        asyncio.create_task(token_manager.record_failure(auth_token, response.status_code, error_message))

        raise GrokApiException(
            f"Request failed: {response.status_code} - {error_message}",
            "HTTP_ERROR",
            {"status": response.status_code, "data": error_data}
        )

    @staticmethod
    async def _process_response(response, auth_token: str, model: str, stream: bool):
        """Process API response"""
        if stream:
            result = GrokResponseProcessor.process_stream(response, auth_token)
            asyncio.create_task(GrokClient._update_rate_limits(auth_token, model))
        else:
            result = await GrokResponseProcessor.process_normal(response, auth_token, model)
            asyncio.create_task(GrokClient._update_rate_limits(auth_token, model))

        return result

    @staticmethod
    async def _update_rate_limits(auth_token: str, model: str):
        """Asynchronously update rate limit info"""
        try:
            await token_manager.check_limits(auth_token, model)
        except Exception as e:
            logger.error(f"[Client] Failed to update rate limits: {e}")