"""Grok Video Generation Service"""

import json
import asyncio
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path

from curl_cffi.requests import AsyncSession

from app.core.config import setting
from app.core.logger import logger
from app.services.grok.statsig import get_dynamic_headers
from app.services.grok.token import token_manager

# Constants
CREATE_POST_ENDPOINT = "https://grok.com/rest/media/post/create"
CONVERSATION_ENDPOINT = "https://grok.com/rest/app-chat/conversations/new"
UPLOAD_ENDPOINT = "https://grok.com/rest/app-chat/upload-file"
REQUEST_TIMEOUT = 60
IMPERSONATE_BROWSER = "chrome133a"


class GrokVideoGenerator:
    """Grok Video Generator Service"""

    @staticmethod
    async def generate_video_from_image(
        image_path: str, 
        prompt: str = "",
        file_name: Optional[str] = None, 
        mode: str = "normal",
        model_name: str = "grok-3"
    ) -> Dict[str, Any]:
        """
        Full flow: Upload image -> Create Post -> Generate Video
        """
        try:
            # Step 1: Upload Image
            upload_result = await GrokVideoGenerator._upload_image_file(image_path, file_name)
            
            if not upload_result.get("success"):
                return upload_result
            
            # Get file info
            file_metadata_id = upload_result["data"].get("fileMetadataId")
            file_uri = upload_result["data"].get("fileUri")
            
            if not file_metadata_id or not file_uri:
                return {"error": "Upload successful but missing metadata"}
            
            # Step 2: Create Media Post
            post_result = await GrokVideoGenerator._create_media_post(file_uri, file_metadata_id)
            
            if not post_result.get("success"):
                return {
                    "upload_result": upload_result,
                    "post_result": post_result,
                    "error": "Failed to create media post"
                }
            
            # Step 3: Create Conversation and Generate Video
            # The image URL for the prompt
            image_url = f"https://assets.grok.com/{file_uri}"
            
            conversation_result = await GrokVideoGenerator._create_video_conversation(
                file_metadata_id,
                image_url,
                prompt,  # Pass the user's prompt!
                mode,
                model_name
            )
            
            return {
                "upload_result": upload_result,
                "post_result": post_result,
                "conversation_result": conversation_result,
                "success": conversation_result.get("success", False)
            }
            
        except Exception as e:
            logger.error(f"[Video] Exception in generation flow: {str(e)}")
            return {"error": f"Exception: {str(e)}"}

    @staticmethod
    async def _upload_image_file(image_path: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        """Upload image file to Grok"""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return {"error": f"File not found: {image_path}"}
            
            file_name = file_name or image_path.name
            file_mime_type = GrokVideoGenerator._get_mime_type(image_path.suffix)
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
                content = base64.b64encode(image_data).decode('utf-8')
            
            data = {
                "content": content,
                "fileMimeType": file_mime_type,
                "fileName": file_name,
                "fileSource": "IMAGINE_SELF_UPLOAD_FILE_SOURCE"
            }
            
            auth_token = token_manager.get_token("grok-3")
            headers = GrokVideoGenerator._get_headers(auth_token, "/rest/app-chat/upload-file")
            headers["referer"] = "https://grok.com/imagine"
            
            proxies = GrokVideoGenerator._get_proxies()
            
            async with AsyncSession() as session:
                response = await session.post(
                    UPLOAD_ENDPOINT,
                    headers=headers,
                    json=data,
                    impersonate=IMPERSONATE_BROWSER,
                    timeout=REQUEST_TIMEOUT,
                    proxies=proxies
                )
                
                if response.status_code == 200:
                    return {"success": True, "data": response.json()}
                else:
                    return {"success": False, "error": f"Upload failed: {response.status_code}", "response": response.text}
                    
        except Exception as e:
            logger.error(f"[Video] Upload exception: {str(e)}")
            return {"error": f"Upload exception: {str(e)}"}

    @staticmethod
    async def _create_media_post(file_uri: str, file_metadata_id: str, media_type: str = "MEDIA_POST_TYPE_IMAGE") -> Dict[str, Any]:
        """Create media post from uploaded file"""
        try:
            data = {
                "mediaType": media_type,
                "mediaUrl": f"https://assets.grok.com/{file_uri}"
            }
            
            auth_token = token_manager.get_token("grok-3")
            headers = GrokVideoGenerator._get_headers(auth_token, "/rest/media/post/create")
            headers["referer"] = "https://grok.com/imagine"
            
            proxies = GrokVideoGenerator._get_proxies()
            
            async with AsyncSession() as session:
                response = await session.post(
                    CREATE_POST_ENDPOINT,
                    headers=headers,
                    json=data,
                    impersonate=IMPERSONATE_BROWSER,
                    timeout=REQUEST_TIMEOUT,
                    proxies=proxies
                )
                
                if response.status_code == 200:
                    return {"success": True, "data": response.json()}
                else:
                    return {"success": False, "error": f"Create post failed: {response.status_code}", "response": response.text}
                    
        except Exception as e:
            logger.error(f"[Video] Create post exception: {str(e)}")
            return {"error": f"Create post exception: {str(e)}"}

    @staticmethod
    async def _create_video_conversation(
        file_metadata_id: str, 
        image_url: str,
        prompt: str = "",
        mode: str = "normal", 
        model_name: str = "grok-3"
    ) -> Dict[str, Any]:
        """Create conversation to generate video"""
        try:
            # Construct the message: use the prompt if provided, otherwise just the mode
            if prompt:
                message = f"{prompt} --mode={mode}"
            else:
                message = f"{image_url} --mode={mode}"
            
            # The payload structure based on the working curl and upstream code
            data = {
                "fileAttachments": [file_metadata_id],
                "message": message,
                "modelName": model_name,
                "temporary": True,
                "toolOverrides": {"videoGen": True},
                "videoGen": True,
                 # Adding the responseMetadata that we found was important in the curl
                "responseMetadata": {
                    "modelConfigOverride": {
                        "modelMap": {
                            "videoGenModelConfig": {
                                "parentPostId": file_metadata_id,
                                "aspectRatio": "16:9",
                                "videoLength": 5
                            }
                        }
                    }
                }
            }
            
            auth_token = token_manager.get_token(model_name)
            headers = GrokVideoGenerator._get_headers(auth_token, "/rest/app-chat/conversations/new")
            # CRITICAL: Referer must point to the imagine post
            headers["referer"] = f"https://grok.com/imagine/{file_metadata_id}"
            
            proxies = GrokVideoGenerator._get_proxies()
            
            async with AsyncSession() as session:
                response = await session.post(
                    CONVERSATION_ENDPOINT,
                    headers=headers,
                    json=data,
                    impersonate=IMPERSONATE_BROWSER,
                    timeout=REQUEST_TIMEOUT,
                    proxies=proxies
                )
                
                if response.status_code == 200:
                    # Process streaming response
                    return await GrokVideoGenerator._process_video_stream(response)
                else:
                    return {
                        "success": False,
                        "error": f"Conversation failed: {response.status_code}",
                        "response": response.text
                    }
                    
        except Exception as e:
            logger.error(f"[Video] Conversation exception: {str(e)}")
            return {"error": f"Conversation exception: {str(e)}"}

    @staticmethod
    async def _process_video_stream(response) -> Dict[str, Any]:
        """Process the streaming response to find video URL"""
        response_text = response.text
        responses = []
        
        # Split multiple JSON objects
        json_objects = response_text.strip().split('\n')
        for json_str in json_objects:
            if json_str.strip():
                try:
                    json_obj = json.loads(json_str)
                    responses.append(json_obj)
                except json.JSONDecodeError:
                    continue
        
        video_url = None
        video_id = None
        progress = 0
        
        for resp in responses:
            if 'result' in resp and 'response' in resp['result']:
                response_data = resp['result']['response']
                
                if 'streamingVideoGenerationResponse' in response_data:
                    video_data = response_data['streamingVideoGenerationResponse']
                    video_id = video_data.get('videoId')
                    progress = video_data.get('progress', 0)
                    
                    if progress == 100 and 'videoUrl' in video_data:
                        video_url = video_data['videoUrl']
                        break
        
        return {
            "success": True if video_url else False,
            "data": {
                "responses": responses,
                "video_id": video_id,
                "video_url": video_url,
                "progress": progress,
                "final_video_url": f"https://assets.grok.com/{video_url}" if video_url else None
            }
        }

    @staticmethod
    def _get_headers(auth_token: str, path: str) -> Dict[str, str]:
        headers = get_dynamic_headers(path)
        cf_clearance = setting.grok_config.get("cf_clearance", "")
        headers["Cookie"] = f"{auth_token};{cf_clearance}" if cf_clearance else auth_token
        return headers

    @staticmethod
    def _get_proxies() -> Optional[Dict[str, str]]:
        proxy_url = setting.get_service_proxy()
        return {"http": proxy_url, "https": proxy_url} if proxy_url else None

    @staticmethod
    def _get_mime_type(file_extension: str) -> str:
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }
        return mime_types.get(file_extension.lower(), 'image/jpeg')

# Global instance
video_generator = GrokVideoGenerator()
