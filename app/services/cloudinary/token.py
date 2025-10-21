"""Cloudinary Token Manager"""

import json
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, Any, Optional, List

from app.core.logger import logger
from app.core.exception import GrokApiException

class CloudinaryTokenManager:
    _instance: Optional['CloudinaryTokenManager'] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> 'CloudinaryTokenManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self.token_file = Path(__file__).parents[3] / "data" / "cloudinary.json"
        self._file_lock = asyncio.Lock()
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self._storage = None

        self._load_data()
        self._initialized = True

        logger.debug(f"[CloudinaryToken] Manager initialized, file: {self.token_file}")

    def set_storage(self, storage) -> None:
        self._storage = storage

    def _load_data(self) -> None:
        default_data = {"accounts": []}

        try:
            if self.token_file.exists():
                with open(self.token_file, "r", encoding="utf-8") as f:
                    self.token_data = json.load(f)
            else:
                self.token_data = default_data
                logger.debug("[CloudinaryToken] Creating new token data file")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"[CloudinaryToken] Failed to load token data: {str(e)}")
            self.token_data = default_data

    async def _save_data(self) -> None:
        try:
            if not self._storage:
                async with self._file_lock:
                    async with aiofiles.open(self.token_file, "w", encoding="utf-8") as f:
                        await f.write(json.dumps(self.token_data, indent=2, ensure_ascii=False))
            else:
                await self._storage.save_cloudinary_tokens(self.token_data)
        except IOError as e:
            logger.error(f"[CloudinaryToken] Failed to save token data: {str(e)}")
            raise GrokApiException(
                f"Failed to save token data: {str(e)}",
                "TOKEN_SAVE_ERROR",
                {"file_path": str(self.token_file)}
            )

    async def add_accounts(self, accounts: List[Dict[str, str]]) -> None:
        if not accounts:
            return

        for account in accounts:
            if not any(acc.get('api_key') == account.get('api_key') for acc in self.token_data['accounts']):
                self.token_data['accounts'].append(account)

        await self._save_data()

    async def delete_accounts(self, api_keys: List[str]) -> None:
        if not api_keys:
            return

        self.token_data['accounts'] = [
            acc for acc in self.token_data['accounts'] if acc.get('api_key') not in api_keys
        ]
        await self._save_data()

    def get_accounts(self) -> List[Dict[str, str]]:
        return self.token_data.get('accounts', [])

cloudinary_token_manager = CloudinaryTokenManager()