"""Grok Token Manager Module"""

import json
import time
import asyncio
import aiofiles
from pathlib import Path
from curl_cffi.requests import AsyncSession
from typing import Dict, Any, Optional, Tuple

from app.models.grok_models import TokenType, Models
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.core.config import setting
from app.services.grok.statsig import get_dynamic_headers

# Constant Definition
RATE_LIMIT_ENDPOINT = "https://grok.com/rest/rate-limits"
REQUEST_TIMEOUT = 30
IMPERSONATE_BROWSER = "chrome133a"
MAX_FAILURE_COUNT = 3
TOKEN_INVALID_CODE = 401  # SSO Token invalid
STATSIG_INVALID_CODE = 403  # x-statsig-id invalid


class GrokTokenManager:
    """
    Grok Token Manager
    
    Singleton Token Manager, responsible for:
    - Token file read/write operations
    - Token load balancing
    - Token status management
    - Support for normal Token and Super Token
    """
    
    _instance: Optional['GrokTokenManager'] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> 'GrokTokenManager':
        """Singleton implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Token Manager"""
        if hasattr(self, '_initialized'):
            return

        self.token_file = Path(__file__).parents[3] / "data" / "token.json"
        self._file_lock = asyncio.Lock()
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self._storage = None

        # Sync load initial data
        self._load_data()
        self._initialized = True
        self.video_token_index = 0
 
        logger.debug(f"[Token] Manager initialized, file: {self.token_file}")
 
    def set_storage(self, storage) -> None:
        """Set storage instance"""
        self._storage = storage

    def _load_data(self) -> None:
        """Sync load Token data (for initialization only)"""
        default_data = {
            TokenType.NORMAL.value: {},
            TokenType.SUPER.value: {}
        }

        try:
            if self.token_file.exists():
                with open(self.token_file, "r", encoding="utf-8") as f:
                    self.token_data = json.load(f)
            else:
                self.token_data = default_data
                logger.debug("[Token] Created new Token data file")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"[Token] Failed to load Token data: {str(e)}")
            self.token_data = default_data

    async def _save_data(self) -> None:
        """Async save Token data to storage"""
        try:
            if not self._storage:
                # If no storage set, use legacy file saving (backward compatibility)
                async with self._file_lock:
                    async with aiofiles.open(self.token_file, "w", encoding="utf-8") as f:
                        await f.write(json.dumps(self.token_data, indent=2, ensure_ascii=False))
            else:
                # Use storage abstraction layer
                await self._storage.save_tokens(self.token_data)
        except IOError as e:
            logger.error(f"[Token] Failed to save Token data: {str(e)}")
            raise GrokApiException(
                f"Token data save failed: {str(e)}",
                "TOKEN_SAVE_ERROR",
                {"file_path": str(self.token_file)}
            )

    @staticmethod
    def _extract_sso(auth_token: str) -> Optional[str]:
        """Extract SSO value from auth token"""
        if "sso=" in auth_token:
            return auth_token.split("sso=")[1].split(";")[0]
        logger.warning("[Token] Cannot extract SSO value from auth token")
        return None

    def _find_token(self, sso_value: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Find Token data, return (token_type, token_data)"""
        for token_type in [TokenType.NORMAL.value, TokenType.SUPER.value]:
            if sso_value in self.token_data[token_type]:
                return token_type, self.token_data[token_type][sso_value]
        return None, None

    async def add_token(self, tokens: list[str], token_type: TokenType) -> None:
        """Add Token to manager"""
        if not tokens:
            logger.debug("[Token] Attempting to add empty Token list")
            return

        added_count = 0
        for token in tokens:
            if not token or not token.strip():
                logger.debug("[Token] Skipping empty Token")
                continue

            self.token_data[token_type.value][token] = {
                "createdTime": int(time.time() * 1000),
                "remainingQueries": -1,
                "heavyremainingQueries": -1,
                "status": "active",
                "failedCount": 0,
                "lastFailureTime": None,
                "lastFailureReason": None
            }
            added_count += 1

        await self._save_data()
        logger.info(f"[Token] Successfully added {added_count} {token_type.value} Tokens")

    async def delete_token(self, tokens: list[str], token_type: TokenType) -> None:
        """Delete specified Token"""
        if not tokens:
            logger.debug("[Token] Attempting to delete empty Token list")
            return

        deleted_count = 0
        for token in tokens:
            if token in self.token_data[token_type.value]:
                del self.token_data[token_type.value][token]
                deleted_count += 1
            else:
                logger.debug(f"[Token] Token does not exist: {token[:10]}...")

        await self._save_data()
        logger.info(f"[Token] Successfully deleted {deleted_count} {token_type.value} Tokens")
    
    def get_tokens(self) -> Dict[str, Any]:
        """Get all Token data"""
        return self.token_data.copy()

    def get_token(self, model: str) -> str:
        """Get Token for specified model"""
        jwt_token = self.select_token(model)
        return f"sso-rw={jwt_token};sso={jwt_token}"
    
    def select_token(self, model: str) -> str:
        """Select best Token based on model type and remaining queries"""
        model_info = Models.get_model_info(model)
        if model_info.get("is_video_model"):
            active_tokens = [
                token for token, data in self.token_data[TokenType.NORMAL.value].items()
                if data.get("status") != "expired"
            ]
            
            if not active_tokens:
                # Fallback to Super tokens if no normal ones are active
                active_tokens = [
                    token for token, data in self.token_data[TokenType.SUPER.value].items()
                    if data.get("status") != "expired"
                ]

            if not active_tokens:
                raise GrokApiException("No active tokens available for video generation.", "NO_AVAILABLE_TOKEN")

            # Select token using round-robin index
            selected_token = active_tokens[self.video_token_index % len(active_tokens)]
            
            # Increment index for the next request
            self.video_token_index += 1
            
            logger.debug(f"[Token] Round-robin selection for video model: Chose token at index {self.video_token_index - 1}")
            return selected_token
        def select_best_token(tokens_dict: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
            """Select best token from token dict"""
            unused_tokens = []  # remaining = -1 tokens
            used_tokens = []    # remaining > 0 tokens
 
            for token_key, token_data in tokens_dict.items():
                # Skip expired tokens
                if token_data.get("status") == "expired":
                    continue
 
                remaining = int(token_data.get(remaining_field, -1))
 
                # Skip rate-limited tokens
                if remaining == 0:
                    continue
 
                # Categorize storage
                if remaining == -1:
                    unused_tokens.append(token_key)
                elif remaining > 0:
                    used_tokens.append((token_key, remaining))
 
            # Prefer unused tokens
            if unused_tokens:
                return unused_tokens[0], -1
 
            # Otherwise return token with most remaining queries
            if used_tokens:
                used_tokens.sort(key=lambda x: x[1], reverse=True)
                return used_tokens[0][0], used_tokens[0][1]
 
            return None, None
 
        max_token_key = None
        max_remaining = None
 
        # Deep copy
        token_data_snapshot = {
            TokenType.NORMAL.value: self.token_data[TokenType.NORMAL.value].copy(),
            TokenType.SUPER.value: self.token_data[TokenType.SUPER.value].copy()
        }
 
        if model == "grok-4-heavy":
            # grok-4-heavy can only use Super Token + heavy remaining queries
            remaining_field = "heavyremainingQueries"
            max_token_key, max_remaining = select_best_token(token_data_snapshot[TokenType.SUPER.value])
        else:
            # Other models use remaining Queries
            remaining_field = "remainingQueries"
 
            # Prefer normal Token
            max_token_key, max_remaining = select_best_token(token_data_snapshot[TokenType.NORMAL.value])
 
            # If no normal Token available, try Super Token
            if max_token_key is None:
                max_token_key, max_remaining = select_best_token(token_data_snapshot[TokenType.SUPER.value])
 
        if max_token_key is None:
            raise GrokApiException(
                f"No available Token for model {model}",
                "NO_AVAILABLE_TOKEN",
                {
                    "model": model,
                    "normal_count": len(token_data_snapshot[TokenType.NORMAL.value]),
                    "super_count": len(token_data_snapshot[TokenType.SUPER.value])
                }
            )
 
        status_text = "Unused" if max_remaining == -1 else f"Remaining {max_remaining}"
        logger.debug(f"[Token] Selected Token for model {model} ({status_text})")
        return max_token_key
    
    async def check_limits(self, auth_token: str, model: str) -> Optional[Dict[str, Any]]:
        """Check and update model rate limits"""
        try:
            rate_limit_model_name = Models.to_rate_limit(model)
            logger.debug(f"[Token] Checking rate limits for model {model} (API model: {rate_limit_model_name})")

            # Prepare request
            payload = {"requestKind": "DEFAULT", "modelName": rate_limit_model_name}
            cf_clearance = setting.grok_config.get("cf_clearance", "")
            cookie = f"{auth_token};{cf_clearance}" if cf_clearance else auth_token

            headers = get_dynamic_headers("/rest/rate-limits")
            headers["Cookie"] = cookie

            # Get proxy config
            proxy_url = setting.grok_config.get("proxy_url", "")
            proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
            
            if proxy_url:
                logger.debug(f"[Token] Using proxy: {proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url}")

            # Send async request
            async with AsyncSession() as session:
                response = await session.post(
                    RATE_LIMIT_ENDPOINT,
                    headers=headers,
                    json=payload,
                    impersonate=IMPERSONATE_BROWSER,
                    timeout=REQUEST_TIMEOUT,
                    proxies=proxies
                )

                if response.status_code == 200:
                    rate_limit_data = response.json()
                    logger.debug(f"[Token] Successfully retrieved rate limit info")

                    # Save rate limit info
                    sso_value = self._extract_sso(auth_token)
                    if sso_value:
                        if model == "grok-4-heavy":
                            await self.update_limits(sso_value, normal=None, heavy=rate_limit_data.get("remainingQueries", -1))
                            logger.info(f"[Token] Limits updated: sso={sso_value[:10]}..., heavy={rate_limit_data.get('remainingQueries', -1)}")
                        else:
                            await self.update_limits(sso_value, normal=rate_limit_data.get("remainingTokens", -1), heavy=None)
                            logger.info(f"[Token] Limits updated: sso={sso_value[:10]}..., normal={rate_limit_data.get('remainingTokens', -1)}")

                    return rate_limit_data
                else:
                    logger.warning(f"[Token] Failed to get rate limits, status code: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"[Token] Error checking rate limits: {str(e)}")
            return None

    async def update_limits(self, sso_value: str, normal: Optional[int] = None, heavy: Optional[int] = None) -> None:
        """Update Token limit info"""
        try:
            for token_type in [TokenType.NORMAL.value, TokenType.SUPER.value]:
                if sso_value in self.token_data[token_type]:
                    if normal is not None:
                        self.token_data[token_type][sso_value]["remainingQueries"] = normal
                    if heavy is not None:
                        self.token_data[token_type][sso_value]["heavyremainingQueries"] = heavy

                    await self._save_data()
                    logger.info(f"[Token] Updated limits for Token {sso_value[:10]}...")
                    return

            logger.warning(f"[Token] Token with SSO value {sso_value[:10]}... not found")

        except Exception as e:
            logger.error(f"[Token] Error updating Token limits: {str(e)}")
    
    async def record_failure(self, auth_token: str, status_code: int, error_message: str) -> None:
        """Record Token failure info

        Error code explanation:
        - 401: SSO Token invalid, marks Token as expired
        - 403: Server IP Blocked, does not affect Token status

        Args:
            auth_token: Full auth Token (format: sso-rw=xxx;sso=xxx)
            status_code: HTTP status code
            error_message: Error message
        """
        try:
            # 403 error is Server IP Blocked, not a Token issue
            if status_code == STATSIG_INVALID_CODE:
                logger.warning(
                    f"[Token] Server IP Blocked (403), please 1. Change server IP 2. Use proxy IP "
                    f"3. Log in to Grok.com on server, pass CF check, then get CF value from F12 to fill in admin settings"
                )
                return

            sso_value = self._extract_sso(auth_token)
            if not sso_value:
                return

            _, token_data = self._find_token(sso_value)
            if not token_data:
                logger.warning(f"[Token] Token with SSO value {sso_value[:10]}... not found")
                return

            # Update failure count
            token_data["failedCount"] = token_data.get("failedCount", 0) + 1
            token_data["lastFailureTime"] = int(time.time() * 1000)
            token_data["lastFailureReason"] = f"{status_code}: {error_message}"

            logger.warning(
                f"[Token] Token {sso_value[:10]}... failed (Status: {status_code}), "
                f"Failures: {token_data['failedCount']}/{MAX_FAILURE_COUNT}, "
                f"Reason: {error_message}"
            )

            # Only mark as expired if 401 error (SSO Token invalid) and failure count reaches max
            if status_code == TOKEN_INVALID_CODE and token_data["failedCount"] >= MAX_FAILURE_COUNT:
                token_data["status"] = "expired"
                logger.error(
                    f"[Token] SSO Token {sso_value[:10]}... marked as expired "
                    f"(Consecutive 401 errors: {token_data['failedCount']})"
                )

            await self._save_data()

        except Exception as e:
            logger.error(f"[Token] Error recording Token failure info: {str(e)}")

    async def reset_failure(self, auth_token: str) -> None:
        """Reset Token failure count

        Called when Token successfully completes a request, used to clear failure record.

        Args:
            auth_token: Full auth Token (format: sso-rw=xxx;sso=xxx)
        """
        try:
            sso_value = self._extract_sso(auth_token)
            if not sso_value:
                return

            _, token_data = self._find_token(sso_value)
            if not token_data:
                logger.warning(f"[Token] Token with SSO value {sso_value[:10]}... not found")
                return

            # Reset and save only if there are failure records
            if token_data.get("failedCount", 0) > 0:
                token_data["failedCount"] = 0
                token_data["lastFailureTime"] = None
                token_data["lastFailureReason"] = None

                await self._save_data()
                logger.info(f"[Token] Token {sso_value[:10]}... failure count reset")

        except Exception as e:
            logger.error(f"[Token] Error resetting Token failure count: {str(e)}")


# Global Token Manager Instance
token_manager = GrokTokenManager()
