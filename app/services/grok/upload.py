"""Image Upload Manager"""

import base64
import re
from typing import Tuple

from curl_cffi.requests import AsyncSession

from app.core.config import setting
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.services.grok.statsig import get_dynamic_headers

# Constant Definition
UPLOAD_ENDPOINT = "https://grok.com/rest/app-chat/upload-file"
REQUEST_TIMEOUT = 30
IMPERSONATE_BROWSER = "chrome133a"
DEFAULT_MIME_TYPE = "image/jpeg"
DEFAULT_EXTENSION = "jpg"


class ImageUploadManager:
    """
    Grok Image Upload Manager

    Provides image upload functionality, supporting:
    - Base64 format image upload
    - URL image download and upload
    """

    @staticmethod
    async def upload(image_input: str, auth_token: str) -> str:
        """
        Upload image to Grok, automatically detect and handle URL or Base64 Data URI.
        """
        try:
            if ImageUploadManager._is_base64_uri(image_input):
                logger.debug("[Upload] Detected Base64 Data URI, starting processing")
                image_bytes, mime_type = ImageUploadManager._decode_base64_uri(image_input)
            else:
                logger.debug(f"[Upload] Detected URL: {image_input}, starting download")
                image_bytes, mime_type = await ImageUploadManager._download_url(image_input)

            if not image_bytes:
                logger.warning("[Upload] Failed to get image data, upload failed")
                return ""

            return await ImageUploadManager._upload_bytes(image_bytes, mime_type, auth_token)

        except Exception as e:
            logger.error(f"[Upload] Unexpected error during image upload: {e}")
            return ""

    @staticmethod
    def _is_base64_uri(uri: str) -> bool:
        """Check if input is Base64 Data URI"""
        return uri.strip().startswith("data:image") and ";base64," in uri

    @staticmethod
    def _decode_base64_uri(uri: str) -> Tuple[bytes, str]:
        """Decode Base64 Data URI"""
        try:
            header, encoded = uri.split(",", 1)
            mime_match = re.search(r"data:(image/[a-zA-Z0-9-.+]+);base64", header)
            mime_type = mime_match.group(1) if mime_match else DEFAULT_MIME_TYPE

            decoded_bytes = base64.b64decode(encoded)
            return decoded_bytes, mime_type
        except (ValueError, IndexError) as e:
            logger.error(f"[Upload] Failed to decode Base64 URI: {e}")
            return b"", DEFAULT_MIME_TYPE

    @staticmethod
    async def _download_url(url: str) -> Tuple[bytes, str]:
        """Download image from URL"""
        try:
            async with AsyncSession() as session:
                response = await session.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()

                content_type = response.headers.get('content-type', DEFAULT_MIME_TYPE)
                if not content_type.startswith('image/'):
                    logger.warning(f"[Upload] MIME type returned by URL is not image: {content_type}, using default value")
                    content_type = DEFAULT_MIME_TYPE

                return response.content, content_type
        except Exception as e:
            logger.error(f"[Upload] Failed to download image URL: {url}, Error: {e}")
            return b"", DEFAULT_MIME_TYPE

    @staticmethod
    async def _upload_bytes(image_bytes: bytes, mime_type: str, auth_token: str) -> str:
        """Upload image bytes to Grok"""
        if not auth_token:
            raise GrokApiException("Auth token missing or empty", "NO_AUTH_TOKEN")

        try:
            # Prepare request
            base64_content = base64.b64encode(image_bytes).decode('utf-8')
            extension = mime_type.split("/")[-1] if "/" in mime_type else DEFAULT_EXTENSION
            file_name = f"image.{extension}"

            payload = {
                "fileName": file_name,
                "fileMimeType": mime_type,
                "content": base64_content,
            }

            cf_clearance = setting.grok_config.get("cf_clearance", "")
            cookie = f"{auth_token};{cf_clearance}" if cf_clearance else auth_token

            proxy_url = setting.get_service_proxy()
            proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

            if proxy_url:
                logger.debug(f"[Upload] Using proxy: {proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url}")

            async with AsyncSession() as session:
                response = await session.post(
                    UPLOAD_ENDPOINT,
                    headers={
                        **get_dynamic_headers("/rest/app-chat/upload-file"),
                        "Cookie": cookie,
                    },
                    json=payload,
                    impersonate=IMPERSONATE_BROWSER,
                    timeout=REQUEST_TIMEOUT,
                    proxies=proxies,
                )

                response.raise_for_status()
                result = response.json()
                file_id = result.get("fileMetadataId", "")
                logger.debug(f"[Upload] Image upload successful, File ID: {file_id}")
                return file_id

        except Exception as e:
            logger.error(f"[Upload] Failed to upload image bytes: {e}")
            return ""