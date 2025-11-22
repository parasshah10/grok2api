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
            content = f"{content} --mode=custom"
            logger.debug(f"[Client] Video model text processing: {content}")

        # Retry logic
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
            payload["toolOverrides"] = {"videoGen": True}
            logger.debug("[Client] Video model payload config: toolOverrides.videoGen = True")
        
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