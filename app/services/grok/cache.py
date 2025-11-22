"""Cache Service Module"""

import asyncio
import base64
from pathlib import Path
from typing import Optional
from curl_cffi.requests import AsyncSession
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

from app.core.config import setting
from app.core.logger import logger
from app.services.grok.statsig import get_dynamic_headers


class CacheService:
    """Cache Service Base Class"""

    def __init__(self, cache_type: str):
        """Initialize Cache Service"""
        self.cache_type = cache_type
        self.cache_dir = Path(f"data/temp/{cache_type}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, file_path: str) -> Path:
        """Get full path of cache file"""
        filename = file_path.lstrip('/').replace('/', '-')
        return self.cache_dir / filename

    async def download_file(self, file_path: str, auth_token: str, timeout: float = 30.0) -> Optional[Path]:
        """Download and cache file"""
        cache_path = self._cache_path(file_path)
        if cache_path.exists():
            logger.debug(f"[{self.cache_type.upper()}Cache] File cached: {cache_path}")
            return cache_path

        try:
            cf_clearance = setting.grok_config.get("cf_clearance", "")
            headers = {
                **get_dynamic_headers(pathname=file_path),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "same-site",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
                "Referer": "https://grok.com/",
                "Cookie": f"{auth_token};{cf_clearance}" if cf_clearance else auth_token
            }

            # Use cache proxy
            proxy_url = setting.get_cache_proxy()
            proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else {}
            
            if proxy_url:
                logger.debug(f"[{self.cache_type.upper()}Cache] Using cache proxy: {proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url}")

            async with AsyncSession() as session:
                logger.debug(f"[{self.cache_type.upper()}Cache] Start downloading: https://assets.grok.com{file_path}")
                response = await session.get(
                    f"https://assets.grok.com{file_path}",
                    headers=headers,
                    proxies=proxies,
                    timeout=timeout,
                    allow_redirects=True,
                    impersonate="chrome133a"
                )
                response.raise_for_status()
                cache_path.write_bytes(response.content)
                logger.debug(f"[{self.cache_type.upper()}Cache] File cached: {cache_path} ({len(response.content)} bytes)")
                asyncio.create_task(self.cleanup_cache())
                return cache_path
        except Exception as e:
            logger.error(f"[{self.cache_type.upper()}Cache] Failed to download file: {e}")
            return None

    def get_cached(self, file_path: str) -> Optional[Path]:
        """Get cached file path"""
        cache_path = self._cache_path(file_path)
        return cache_path if cache_path.exists() else None

    async def cleanup_cache(self):
        """Clean up cache directory to ensure it doesn't exceed size limit"""
        try:
            max_size_mb = setting.global_config.get(f"{self.cache_type}_cache_max_size_mb", 500)
            max_size_bytes = max_size_mb * 1024 * 1024

            files = [(fp, (stat := fp.stat()).st_size, stat.st_mtime)
                     for fp in self.cache_dir.glob("*") if fp.is_file()]
            total_size = sum(size for _, size, _ in files)

            if total_size <= max_size_bytes:
                logger.debug(f"[{self.cache_type.upper()}Cache] Cache size {total_size / 1024 / 1024:.2f}MB, within limit")
                return

            logger.info(f"[{self.cache_type.upper()}Cache] Cache size {total_size / 1024 / 1024:.2f}MB exceeds limit {max_size_mb}MB, starting cleanup")
            files.sort(key=lambda x: x[2])

            for file_path, size, _ in files:
                if total_size <= max_size_bytes:
                    break
                file_path.unlink()
                total_size -= size
                logger.debug(f"[{self.cache_type.upper()}Cache] Deleted cache file: {file_path}")

            logger.info(f"[{self.cache_type.upper()}Cache] Cleanup complete, current size {total_size / 1024 / 1024:.2f}MB")
        except Exception as e:
            logger.error(f"[{self.cache_type.upper()}Cache] Cleanup failed: {e}")


class ImageCacheService(CacheService):
    """Image Cache Service"""

    def __init__(self):
        super().__init__("image")

    async def download_image(self, image_path: str, auth_token: str) -> Optional[Path]:
        """Download and cache image"""
        cache_path = await self.download_file(image_path, auth_token, timeout=30.0)
        if not cache_path:
            return None

        # If avif format, convert to png
        if cache_path.suffix.lower() == '.avif':
            try:
                with Image.open(cache_path) as img:
                    png_path = cache_path.with_suffix('.png')
                    img.save(png_path, 'PNG')
                    logger.debug(f"[ImageCache] AVIF image converted to PNG: {png_path}")
                    # Delete old avif file
                    cache_path.unlink()
                    return png_path
            except Exception as e:
                logger.error(f"[ImageCache] AVIF to PNG conversion failed: {e}")
                return None  # Or return original path

        return cache_path

    def get_cached(self, image_path: str) -> Optional[Path]:
        """Get cached image path"""
        return super().get_cached(image_path)

    @staticmethod
    def to_base64(image_path: Path) -> Optional[str]:
        """Convert image to base64 encoding"""
        try:
            if not image_path.exists():
                logger.error(f"[ImageCache] Image file does not exist: {image_path}")
                return None

            with open(image_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode('utf-8')

            mime_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                         '.gif': 'image/gif', '.webp': 'image/webp'}.get(image_path.suffix.lower(), 'image/jpeg')

            return f"data:{mime_type};base64,{base64_data}"
        except Exception as e:
            logger.error(f"[ImageCache] Image to base64 failed: {e}")
            return None

    async def download_base64(self, image_path: str, auth_token: str) -> Optional[str]:
        """Download image and convert to base64 encoding (delete cache file immediately after conversion)"""
        try:
            cache_path = await self.download_file(image_path, auth_token, timeout=30.0)
            if not cache_path:
                return None

            base64_str = self.to_base64(cache_path)

            try:
                cache_path.unlink()
                logger.debug(f"[ImageCache] Temporary file deleted: {cache_path}")
            except Exception as e:
                logger.warning(f"[ImageCache] Failed to delete temporary file: {e}")

            return base64_str
        except Exception as e:
            logger.error(f"[ImageCache] Download and convert base64 failed: {e}")
            return None


class VideoCacheService(CacheService):
    """Video Cache Service"""

    def __init__(self):
        super().__init__("video")

    async def download_video(self, video_path: str, auth_token: str) -> Optional[Path]:
        """Download and cache video"""
        return await self.download_file(video_path, auth_token, timeout=60.0)

    def get_cached(self, video_path: str) -> Optional[Path]:
        """Get cached video path"""
        return super().get_cached(video_path)


# Global Instances
image_cache_service = ImageCacheService()
video_cache_service = VideoCacheService()
