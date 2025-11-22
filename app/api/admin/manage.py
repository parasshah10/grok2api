"""
Management Interface Module

Provides Token management functions, including login verification, Token create/delete/query operations.
"""

import secrets
import re
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
 
from app.core.config import setting
from app.core.logger import logger
from app.services.grok.token import token_manager
from app.models.grok_models import TokenType
from app.services.cloudinary import cloudinary_token_manager


# Create router
router = APIRouter(tags=["Management"])

# Constant Definition
STATIC_DIR = Path(__file__).parents[2] / "template"
TEMP_DIR = Path(__file__).parents[3] / "data" / "temp"
IMAGE_CACHE_DIR = TEMP_DIR / "image"
VIDEO_CACHE_DIR = TEMP_DIR / "video"
SESSION_EXPIRE_HOURS = 24
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024

# Simple session storage
_sessions: Dict[str, datetime] = {}


# === Request/Response Models ===

class LoginRequest(BaseModel):
    """Login Request"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login Response"""
    success: bool
    token: Optional[str] = None
    message: str


class AddTokensRequest(BaseModel):
    """Batch Add Tokens Request"""
    tokens: List[str]
    token_type: str  # "sso" or "ssoSuper"


class DeleteTokensRequest(BaseModel):
    """Batch Delete Tokens Request"""
    tokens: List[str]
    token_type: str  # "sso" or "ssoSuper"


class TokenInfo(BaseModel):
    """Token Info"""
    token: str
    token_type: str
    created_time: Optional[int] = None
    remaining_queries: int
    heavy_remaining_queries: int
    status: str  # "Unused", "Rate-Limited", "Expired", "Active"


class TokenListResponse(BaseModel):
    """Token List Response"""
    success: bool
    data: List[TokenInfo]
    total: int


class CloudinaryAccount(BaseModel):
    """Cloudinary Account"""
    cloud_name: str
    api_key: str
    api_secret: str


class AddCloudinaryAccountsRequest(BaseModel):
    """Add Cloudinary Accounts Request"""
    urls: List[str]


class DeleteCloudinaryAccountsRequest(BaseModel):
    """Delete Cloudinary Accounts Request"""
    api_keys: List[str]


# === Helper Functions ===

def validate_token_type(token_type_str: str) -> TokenType:
    """Verify and convert token type string to enum"""
    if token_type_str not in ["sso", "ssoSuper"]:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid token type, must be 'sso' or 'ssoSuper'", "code": "INVALID_TYPE"}
        )
    return TokenType.NORMAL if token_type_str == "sso" else TokenType.SUPER


def parse_created_time(created_time) -> Optional[int]:
    """Parse created time, handle different formats uniformly"""
    if isinstance(created_time, str):
        return int(created_time) if created_time else None
    elif isinstance(created_time, int):
        return created_time
    return None


def calculate_token_stats(tokens: Dict[str, Any], token_type: str) -> Dict[str, int]:
    """Calculate Token statistics"""
    total = len(tokens)
    expired = sum(1 for t in tokens.values() if t.get("status") == "expired")

    if token_type == "normal":
        unused = sum(1 for t in tokens.values()
                    if t.get("status") != "expired" and t.get("remainingQueries", -1) == -1)
        limited = sum(1 for t in tokens.values()
                     if t.get("status") != "expired" and t.get("remainingQueries", -1) == 0)
        active = sum(1 for t in tokens.values()
                    if t.get("status") != "expired" and t.get("remainingQueries", -1) > 0)
    else:  # super token
        unused = sum(1 for t in tokens.values()
                    if t.get("status") != "expired" and
                    t.get("remainingQueries", -1) == -1 and t.get("heavyremainingQueries", -1) == -1)
        limited = sum(1 for t in tokens.values()
                     if t.get("status") != "expired" and
                     (t.get("remainingQueries", -1) == 0 or t.get("heavyremainingQueries", -1) == 0))
        active = sum(1 for t in tokens.values()
                    if t.get("status") != "expired" and
                    (t.get("remainingQueries", -1) > 0 or t.get("heavyremainingQueries", -1) > 0))

    return {
        "total": total,
        "unused": unused,
        "limited": limited,
        "expired": expired,
        "active": active
    }


def verify_admin_session(authorization: Optional[str] = Header(None)) -> bool:
    """Verify admin session"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized access", "code": "UNAUTHORIZED"}
        )
    
    token = authorization[7:]  # Remove "Bearer " prefix
    
    # Check if token exists and is not expired
    if token not in _sessions:
        raise HTTPException(
            status_code=401,
            detail={"error": "Session expired or invalid", "code": "SESSION_INVALID"}
        )
    
    # Check if session expired (24 hours)
    if datetime.now() > _sessions[token]:
        del _sessions[token]
        raise HTTPException(
            status_code=401,
            detail={"error": "Session expired", "code": "SESSION_EXPIRED"}
        )
    
    return True


def get_token_status(token_data: Dict[str, Any], token_type: str) -> str:
    """Get Token status"""
    # First check if expired (from status field in token.json)
    if token_data.get("status") == "expired":
        return "Expired"
    
    # Get remaining queries
    remaining_queries = token_data.get("remainingQueries", -1)
    heavy_remaining = token_data.get("heavyremainingQueries", -1)
    
    # Select correct field based on token type
    if token_type == "ssoSuper":
        # Super token may use heavy model
        relevant_remaining = max(remaining_queries, heavy_remaining)
    else:
        # Normal token mainly looks at remaining_queries
        relevant_remaining = remaining_queries
    
    if relevant_remaining == -1:
        return "Unused"
    elif relevant_remaining == 0:
        return "Rate-Limited"
    else:
        return "Active"


# === Page Routes ===

@router.get("/login", response_class=HTMLResponse)
async def login_page():
    """Login Page"""
    login_html = STATIC_DIR / "login.html"
    if login_html.exists():
        return login_html.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="Login page not found")


@router.get("/manage", response_class=HTMLResponse)
async def manage_page():
    """Management Page"""
    admin_html = STATIC_DIR / "admin.html"
    if admin_html.exists():
        return admin_html.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="Management page not found")


# === API Endpoints ===

@router.post("/api/login", response_model=LoginResponse)
async def admin_login(request: LoginRequest) -> LoginResponse:
    """
    Admin Login
    
    Verify username and password, return session token upon success.
    """
    try:
        logger.debug(f"[Admin] Admin login attempt - Username: {request.username}")

        # Verify username and password
        expected_username = setting.global_config.get("admin_username", "")
        expected_password = setting.global_config.get("admin_password", "")

        if request.username != expected_username or request.password != expected_password:
            logger.warning(f"[Admin] Login failed: Username or password incorrect - Username: {request.username}")
            return LoginResponse(
                success=False,
                message="Username or password incorrect"
            )

        # Generate session token
        session_token = secrets.token_urlsafe(32)

        # Set session expiration time
        expire_time = datetime.now() + timedelta(hours=SESSION_EXPIRE_HOURS)
        _sessions[session_token] = expire_time

        logger.debug(f"[Admin] Admin login successful - Username: {request.username}")

        return LoginResponse(
            success=True,
            token=session_token,
            message="Login successful"
        )

    except Exception as e:
        logger.error(f"[Admin] Login processing error - Username: {request.username}, Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Login failed: {str(e)}", "code": "LOGIN_ERROR"}
        )


@router.post("/api/logout")
async def admin_logout(_: bool = Depends(verify_admin_session),
                       authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Admin Logout
    
    Clear session token.
    """
    try:
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
            if token in _sessions:
                del _sessions[token]
                logger.debug("[Admin] Admin logout successful")
                return {"success": True, "message": "Logout successful"}

        logger.warning("[Admin] Logout failed: Invalid or missing session token")
        return {"success": False, "message": "Invalid session"}

    except Exception as e:
        logger.error(f"[Admin] Logout processing error - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Logout failed: {str(e)}", "code": "LOGOUT_ERROR"}
        )


@router.get("/api/tokens", response_model=TokenListResponse)
async def list_tokens(_: bool = Depends(verify_admin_session)) -> TokenListResponse:
    """
    Get Token List
    
    Return all Tokens and their status information in the system.
    """
    try:
        logger.debug("[Admin] Start getting Token list")

        all_tokens_data = token_manager.get_tokens()
        token_list: List[TokenInfo] = []

        # Process Normal Tokens
        normal_tokens = all_tokens_data.get(TokenType.NORMAL.value, {})
        for token, data in normal_tokens.items():
            token_list.append(TokenInfo(
                token=token,
                token_type="sso",
                created_time=parse_created_time(data.get("createdTime")),
                remaining_queries=data.get("remainingQueries", -1),
                heavy_remaining_queries=data.get("heavyremainingQueries", -1),
                status=get_token_status(data, "sso")
            ))

        # Process Super Tokens
        super_tokens = all_tokens_data.get(TokenType.SUPER.value, {})
        for token, data in super_tokens.items():
            token_list.append(TokenInfo(
                token=token,
                token_type="ssoSuper",
                created_time=parse_created_time(data.get("createdTime")),
                remaining_queries=data.get("remainingQueries", -1),
                heavy_remaining_queries=data.get("heavyremainingQueries", -1),
                status=get_token_status(data, "ssoSuper")
            ))

        normal_count = len(normal_tokens)
        super_count = len(super_tokens)
        total_count = len(token_list)

        logger.debug(f"[Admin] Token list retrieved successfully - Normal Tokens: {normal_count}, Super Tokens: {super_count}, Total: {total_count}")

        return TokenListResponse(
            success=True,
            data=token_list,
            total=total_count
        )

    except Exception as e:
        logger.error(f"[Admin] Get Token list error - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get Token list: {str(e)}", "code": "LIST_ERROR"}
        )


@router.post("/api/tokens/add")
async def add_tokens(request: AddTokensRequest,
                    _: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """
    Batch Add Tokens
    
    Support adding Normal Tokens (sso) and Super Tokens (ssoSuper).
    """
    try:
        logger.debug(f"[Admin] Batch add Tokens - Type: {request.token_type}, Count: {len(request.tokens)}")

        # Verify and convert token type
        token_type = validate_token_type(request.token_type)

        # Add Tokens
        await token_manager.add_token(request.tokens, token_type)

        logger.debug(f"[Admin] Tokens added successfully - Type: {request.token_type}, Count: {len(request.tokens)}")

        return {
            "success": True,
            "message": f"Successfully added {len(request.tokens)} Tokens",
            "count": len(request.tokens)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Admin] Token add error - Type: {request.token_type}, Count: {len(request.tokens)}, Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to add Tokens: {str(e)}", "code": "ADD_ERROR"}
        )


@router.post("/api/tokens/delete")
async def delete_tokens(request: DeleteTokensRequest,
                       _: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """
    Batch Delete Tokens
    
    Support deleting Normal Tokens (sso) and Super Tokens (ssoSuper).
    """
    try:
        logger.debug(f"[Admin] Batch delete Tokens - Type: {request.token_type}, Count: {len(request.tokens)}")

        # Verify and convert token type
        token_type = validate_token_type(request.token_type)

        # Delete Tokens
        await token_manager.delete_token(request.tokens, token_type)

        logger.debug(f"[Admin] Tokens deleted successfully - Type: {request.token_type}, Count: {len(request.tokens)}")

        return {
            "success": True,
            "message": f"Successfully deleted {len(request.tokens)} Tokens",
            "count": len(request.tokens)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Admin] Token delete error - Type: {request.token_type}, Count: {len(request.tokens)}, Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to delete Tokens: {str(e)}", "code": "DELETE_ERROR"}
        )


@router.post("/api/refresh-limits")
async def refresh_all_token_limits(_: bool = Depends(verify_admin_session)):
    """
    Manually refresh the rate limits for all active tokens.
    """
    logger.info("[Admin] Starting manual rate limit refresh for all tokens.")
    
    all_tokens = token_manager.get_tokens()
    tasks = []

    for token_type_str in [TokenType.NORMAL.value, TokenType.SUPER.value]:
        for token, data in all_tokens[token_type_str].items():
            if data.get("status") != "expired":
                auth_token = f"sso-rw={token};sso={token}"
                # Check standard quota for all active tokens
                tasks.append(token_manager.check_limits(auth_token, "grok-4-fast"))
                
                # Also check heavy quota for Super tokens
                if token_type_str == TokenType.SUPER.value:
                    tasks.append(token_manager.check_limits(auth_token, "grok-4-heavy"))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    success_count = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    failure_count = len(tasks) - success_count

    message = f"Rate limit refresh complete. Successful updates: {success_count}, Failures: {failure_count}."
    logger.info(f"[Admin] {message}")
    return {"success": True, "message": message}
 
 
@router.get("/api/settings")
async def get_settings(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """Get Global Settings"""
    try:
        logger.debug("[Admin] Get global settings")
        return {
            "success": True,
            "data": {
                "global": setting.global_config,
                "grok": setting.grok_config
            }
        }
    except Exception as e:
        logger.error(f"[Admin] Failed to get settings: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": f"Failed to get settings: {str(e)}", "code": "GET_SETTINGS_ERROR"})


class UpdateSettingsRequest(BaseModel):
    """Update Settings Request"""
    global_config: Optional[Dict[str, Any]] = None
    grok_config: Optional[Dict[str, Any]] = None


class StreamTimeoutSettings(BaseModel):
    """Stream Timeout Settings"""
    stream_chunk_timeout: int = 120
    stream_first_response_timeout: int = 30
    stream_total_timeout: int = 600


@router.post("/api/settings")
async def update_settings(request: UpdateSettingsRequest, _: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """Update Global Settings"""
    try:
        logger.debug("[Admin] Update global settings")

        # Use ConfigManager's save method (supports storage abstraction layer)
        await setting.save(
            global_config=request.global_config,
            grok_config=request.grok_config
        )

        logger.debug("[Admin] Settings updated successfully")
        return {"success": True, "message": "Settings updated successfully"}
    except Exception as e:
        logger.error(f"[Admin] Failed to update settings: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": f"Failed to update settings: {str(e)}", "code": "UPDATE_SETTINGS_ERROR"})


def _calculate_dir_size(directory: Path) -> int:
    """Calculate total size of files in directory (bytes)"""
    total_size = 0
    for file_path in directory.iterdir():
        if file_path.is_file():
            try:
                total_size += file_path.stat().st_size
            except Exception as e:
                logger.warning(f"[Admin] Cannot get file size: {file_path.name}, Error: {str(e)}")
    return total_size


def _format_size(size_bytes: int) -> str:
    """Format bytes size to readable string"""
    size_mb = size_bytes / BYTES_PER_MB
    if size_mb < 1:
        size_kb = size_bytes / BYTES_PER_KB
        return f"{size_kb:.1f} KB"
    return f"{size_mb:.1f} MB"


@router.get("/api/cache/size")
async def get_cache_size(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """Get Cache Size"""
    try:
        logger.debug("[Admin] Start getting cache size")

        # Calculate image cache size
        image_size = 0
        if IMAGE_CACHE_DIR.exists():
            image_size = _calculate_dir_size(IMAGE_CACHE_DIR)
        
        # Calculate video cache size
        video_size = 0
        if VIDEO_CACHE_DIR.exists():
            video_size = _calculate_dir_size(VIDEO_CACHE_DIR)
        
        # Total size
        total_size = image_size + video_size

        logger.debug(f"[Admin] Cache size retrieved - Images: {_format_size(image_size)}, Videos: {_format_size(video_size)}, Total: {_format_size(total_size)}")
        
        return {
            "success": True,
            "data": {
                "image_size": _format_size(image_size),
                "video_size": _format_size(video_size),
                "total_size": _format_size(total_size),
                "image_size_bytes": image_size,
                "video_size_bytes": video_size,
                "total_size_bytes": total_size
            }
        }

    except Exception as e:
        logger.error(f"[Admin] Get cache size exception - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get cache size: {str(e)}", "code": "CACHE_SIZE_ERROR"}
        )


@router.post("/api/cache/clear")
async def clear_cache(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """Clear Cache

    Delete all temporary files"""
    try:
        logger.debug("[Admin] Start clearing cache")

        deleted_count = 0
        image_count = 0
        video_count = 0

        # Clear image cache
        if IMAGE_CACHE_DIR.exists():
            for file_path in IMAGE_CACHE_DIR.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        image_count += 1
                        logger.debug(f"[Admin] Deleted image cache: {file_path.name}")
                    except Exception as e:
                        logger.error(f"[Admin] Failed to delete image cache: {file_path.name}, Error: {str(e)}")

        # Clear video cache
        if VIDEO_CACHE_DIR.exists():
            for file_path in VIDEO_CACHE_DIR.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        video_count += 1
                        logger.debug(f"[Admin] Deleted video cache: {file_path.name}")
                    except Exception as e:
                        logger.error(f"[Admin] Failed to delete video cache: {file_path.name}, Error: {str(e)}")

        deleted_count = image_count + video_count
        logger.debug(f"[Admin] Cache clearing complete - Images: {image_count}, Videos: {video_count}, Total: {deleted_count}")

        return {
            "success": True,
            "message": f"Successfully cleared cache, deleted {image_count} images, {video_count} videos, total {deleted_count} files",
            "data": {
                "deleted_count": deleted_count,
                "image_count": image_count,
                "video_count": video_count
            }
        }

    except Exception as e:
        logger.error(f"[Admin] Cache clear exception - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to clear cache: {str(e)}", "code": "CACHE_CLEAR_ERROR"}
        )


@router.post("/api/cache/clear/images")
async def clear_image_cache(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """Clear Image Cache

    Only delete image cache files"""
    try:
        logger.debug("[Admin] Start clearing image cache")

        deleted_count = 0

        # Clear image cache
        if IMAGE_CACHE_DIR.exists():
            for file_path in IMAGE_CACHE_DIR.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"[Admin] Deleted image cache: {file_path.name}")
                    except Exception as e:
                        logger.error(f"[Admin] Failed to delete image cache: {file_path.name}, Error: {str(e)}")

        logger.debug(f"[Admin] Image cache clearing complete - Deleted {deleted_count} files")

        return {
            "success": True,
            "message": f"Successfully cleared image cache, deleted {deleted_count} files",
            "data": {
                "deleted_count": deleted_count,
                "type": "images"
            }
        }

    except Exception as e:
        logger.error(f"[Admin] Clear image cache exception - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to clear image cache: {str(e)}", "code": "IMAGE_CACHE_CLEAR_ERROR"}
        )


@router.post("/api/cache/clear/videos")
async def clear_video_cache(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """Clear Video Cache

    Only delete video cache files"""
    try:
        logger.debug("[Admin] Start clearing video cache")

        deleted_count = 0

        # Clear video cache
        if VIDEO_CACHE_DIR.exists():
            for file_path in VIDEO_CACHE_DIR.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"[Admin] Deleted video cache: {file_path.name}")
                    except Exception as e:
                        logger.error(f"[Admin] Failed to delete video cache: {file_path.name}, Error: {str(e)}")

        logger.debug(f"[Admin] Video cache clearing complete - Deleted {deleted_count} files")

        return {
            "success": True,
            "message": f"Successfully cleared video cache, deleted {deleted_count} files",
            "data": {
                "deleted_count": deleted_count,
                "type": "videos"
            }
        }

    except Exception as e:
        logger.error(f"[Admin] Clear video cache exception - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to clear video cache: {str(e)}", "code": "VIDEO_CACHE_CLEAR_ERROR"}
        )


@router.get("/api/stats")
async def get_stats(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """
    Get Stats

    Return Token statistics.
    """
    try:
        logger.debug("[Admin] Start getting stats")

        all_tokens_data = token_manager.get_tokens()

        # Stats for Normal Tokens
        normal_tokens = all_tokens_data.get(TokenType.NORMAL.value, {})
        normal_stats = calculate_token_stats(normal_tokens, "normal")

        # Stats for Super Tokens
        super_tokens = all_tokens_data.get(TokenType.SUPER.value, {})
        super_stats = calculate_token_stats(super_tokens, "super")

        total_count = normal_stats["total"] + super_stats["total"]

        stats = {
            "success": True,
            "data": {
                "normal": normal_stats,
                "super": super_stats,
                "total": total_count
            }
        }

        logger.debug(f"[Admin] Stats retrieved successfully - Normal Tokens: {normal_stats['total']}, Super Tokens: {super_stats['total']}, Total: {total_count}")
        return stats

    except Exception as e:
        logger.error(f"[Admin] Get stats exception - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get stats: {str(e)}", "code": "STATS_ERROR"}
        )


@router.get("/api/storage/mode")
async def get_storage_mode(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """
    Get Current Storage Mode

    Returns current storage mode (file/mysql/redis).
    """
    try:
        logger.debug("[Admin] Get storage mode")

        import os
        storage_mode = os.getenv("STORAGE_MODE", "file").upper()

        return {
            "success": True,
            "data": {
                "mode": storage_mode
            }
        }

    except Exception as e:
        logger.error(f"[Admin] Get storage mode exception - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get storage mode: {str(e)}", "code": "STORAGE_MODE_ERROR"}
        )


@router.get("/api/cloudinary/accounts")
async def list_cloudinary_accounts(_: bool = Depends(verify_admin_session)):
    """Get Cloudinary Accounts List"""
    try:
        accounts = cloudinary_token_manager.get_accounts()
        return {"success": True, "data": accounts}
    except Exception as e:
        logger.error(f"[Admin] Get Cloudinary accounts list exception - Error: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": f"Failed to get Cloudinary accounts list: {str(e)}", "code": "LIST_CLOUDINARY_ACCOUNTS_ERROR"})


def parse_cloudinary_url(cloudinary_url):
    """Parses a Cloudinary URL into credentials."""
    match = re.match(r'cloudinary://([^:]+):([^@]+)@(.+)', cloudinary_url)
    if not match:
        raise ValueError(f"Invalid Cloudinary URL format: {cloudinary_url}")
    api_key, api_secret, cloud_name = match.groups()
    return {
        'cloud_name': cloud_name,
        'api_key': api_key,
        'api_secret': api_secret
    }


@router.post("/api/cloudinary/accounts/add")
async def add_cloudinary_accounts(request: AddCloudinaryAccountsRequest, _: bool = Depends(verify_admin_session)):
    """Add Cloudinary Accounts"""
    try:
        accounts = [parse_cloudinary_url(url) for url in request.urls]
        await cloudinary_token_manager.add_accounts(accounts)
        return {"success": True, "message": "Cloudinary accounts added successfully"}
    except ValueError as e:
        logger.error(f"[Admin] Failed to add Cloudinary accounts: Invalid URL format - {str(e)}")
        raise HTTPException(status_code=400, detail={"error": str(e), "code": "INVALID_CLOUDINARY_URL"})
    except Exception as e:
        logger.error(f"[Admin] Add Cloudinary accounts exception - Error: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": f"Failed to add Cloudinary accounts: {str(e)}", "code": "ADD_CLOUDINARY_ACCOUNTS_ERROR"})


@router.post("/api/cloudinary/accounts/delete")
async def delete_cloudinary_accounts(request: DeleteCloudinaryAccountsRequest, _: bool = Depends(verify_admin_session)):
    """Delete Cloudinary Accounts"""
    try:
        await cloudinary_token_manager.delete_accounts(request.api_keys)
        return {"success": True, "message": "Cloudinary accounts deleted successfully"}
    except Exception as e:
        logger.error(f"[Admin] Delete Cloudinary accounts exception - Error: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": f"Failed to delete Cloudinary accounts: {str(e)}", "code": "DELETE_CLOUDINARY_ACCOUNTS_ERROR"})
