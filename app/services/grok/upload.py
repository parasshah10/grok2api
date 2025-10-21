"""图片上传管理器"""

import base64
import re
from typing import Tuple

from curl_cffi.requests import AsyncSession

from app.core.config import setting
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.services.grok.statsig import get_dynamic_headers

# 常量定义
UPLOAD_ENDPOINT = "https://grok.com/rest/app-chat/upload-file"
REQUEST_TIMEOUT = 30
IMPERSONATE_BROWSER = "chrome133a"
DEFAULT_MIME_TYPE = "image/jpeg"
DEFAULT_EXTENSION = "jpg"


class ImageUploadManager:
    """
    Grok图片上传管理器

    提供图片上传功能，支持：
    - Base64格式图片上传
    - URL图片下载并上传
    """

    @staticmethod
    async def upload(image_input: str, auth_token: str) -> str:
        """
        上传图片到Grok，自动检测并处理URL或Base64数据URI。
        """
        try:
            if ImageUploadManager._is_base64_uri(image_input):
                logger.debug("[Upload] 检测到Base64数据URI，开始处理")
                image_bytes, mime_type = ImageUploadManager._decode_base64_uri(image_input)
            else:
                logger.debug(f"[Upload] 检测到URL: {image_input}，开始下载")
                image_bytes, mime_type = await ImageUploadManager._download_url(image_input)

            if not image_bytes:
                logger.warning("[Upload] 无法获取图片数据，上传失败")
                return ""

            return await ImageUploadManager._upload_bytes(image_bytes, mime_type, auth_token)

        except Exception as e:
            logger.error(f"[Upload] 上传图片时发生意外错误: {e}")
            return ""

    @staticmethod
    def _is_base64_uri(uri: str) -> bool:
        """检查输入是否为Base64数据URI"""
        return uri.strip().startswith("data:image") and ";base64," in uri

    @staticmethod
    def _decode_base64_uri(uri: str) -> Tuple[bytes, str]:
        """解码Base64数据URI"""
        try:
            header, encoded = uri.split(",", 1)
            mime_match = re.search(r"data:(image/[a-zA-Z0-9-.+]+);base64", header)
            mime_type = mime_match.group(1) if mime_match else DEFAULT_MIME_TYPE

            decoded_bytes = base64.b64decode(encoded)
            return decoded_bytes, mime_type
        except (ValueError, IndexError) as e:
            logger.error(f"[Upload] 解码Base64 URI失败: {e}")
            return b"", DEFAULT_MIME_TYPE

    @staticmethod
    async def _download_url(url: str) -> Tuple[bytes, str]:
        """从URL下载图片"""
        try:
            async with AsyncSession() as session:
                response = await session.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()

                content_type = response.headers.get('content-type', DEFAULT_MIME_TYPE)
                if not content_type.startswith('image/'):
                    logger.warning(f"[Upload] URL返回的MIME类型不是图片: {content_type}，使用默认值")
                    content_type = DEFAULT_MIME_TYPE

                return response.content, content_type
        except Exception as e:
            logger.error(f"[Upload] 下载图片URL失败: {url}, 错误: {e}")
            return b"", DEFAULT_MIME_TYPE

    @staticmethod
    async def _upload_bytes(image_bytes: bytes, mime_type: str, auth_token: str) -> str:
        """将图片字节上传到Grok"""
        if not auth_token:
            raise GrokApiException("认证令牌缺失或为空", "NO_AUTH_TOKEN")

        try:
            # 准备请求
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
                logger.debug(f"[Upload] 使用代理: {proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url}")

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
                logger.debug(f"[Upload] 图片上传成功，文件ID: {file_id}")
                return file_id

        except Exception as e:
            logger.error(f"[Upload] 上传图片字节失败: {e}")
            return ""