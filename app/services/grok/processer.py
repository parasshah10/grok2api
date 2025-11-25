"""Grok API 响应处理器模块"""

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
    """流式响应超时管理器"""
    
    def __init__(self, chunk_timeout: int = 120, first_response_timeout: int = 30, total_timeout: int = 600):
        """初始化超时管理器
        
        Args:
            chunk_timeout: 数据块间隔超时（秒）
            first_response_timeout: 首次响应超时（秒）
            total_timeout: 总超时限制（秒，0表示不限制）
        """
        self.chunk_timeout = chunk_timeout
        self.first_response_timeout = first_response_timeout
        self.total_timeout = total_timeout
        
        self.start_time = asyncio.get_event_loop().time()
        self.last_chunk_time = self.start_time
        self.first_chunk_received = False
    
    def check_timeout(self) -> tuple[bool, str]:
        """检查是否超时
        
        Returns:
            (is_timeout, timeout_message): 是否超时及超时信息
        """
        current_time = asyncio.get_event_loop().time()
        
        # 检查首次响应超时
        if not self.first_chunk_received:
            if current_time - self.start_time > self.first_response_timeout:
                return True, f"首次响应超时 ({self.first_response_timeout}秒未收到首个数据块)"
        
        # 检查总超时
        if self.total_timeout > 0:
            if current_time - self.start_time > self.total_timeout:
                return True, f"流式响应总超时 ({self.total_timeout}秒)"
        
        # 检查数据块间隔超时
        if self.first_chunk_received:
            if current_time - self.last_chunk_time > self.chunk_timeout:
                return True, f"数据块间隔超时 ({self.chunk_timeout}秒无新数据)"
        
        return False, ""
    
    def mark_chunk_received(self):
        """标记收到数据块"""
        self.last_chunk_time = asyncio.get_event_loop().time()
        self.first_chunk_received = True
    
    def get_total_duration(self) -> float:
        """获取总耗时（秒）"""
        return asyncio.get_event_loop().time() - self.start_time


class GrokResponseProcessor:
    """Grok API 响应处理器"""

    @staticmethod
    async def process_normal(response, auth_token: str, model: str = None) -> OpenAIChatCompletionResponse:
        """处理非流式响应"""
        response_closed = False
        try:
            for chunk in response.iter_lines():
                if not chunk:
                    continue

                data = json.loads(chunk.decode("utf-8"))

                # 错误检查
                if error := data.get("error"):
                    raise GrokApiException(
                        f"API错误: {error.get('message', '未知错误')}",
                        "API_ERROR",
                        {"code": error.get("code")}
                    )

                # 提取响应数据
                grok_resp = data.get("result", {}).get("response", {})
                

                
                # 提取视频数据
                if video_resp := grok_resp.get("streamingVideoGenerationResponse"):
                    
                    if video_url := video_resp.get("videoUrl"):
                        logger.debug(f"[Processor] 检测到视频生成: {video_url}")
                        full_video_url = f"https://assets.grok.com/{video_url}"
                        
                        # 下载并缓存视频
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
                        
                        # 返回视频响应
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

                # 提取模型响应
                model_response = grok_resp.get("modelResponse")
                if not model_response:
                    continue

                # 检查 modelResponse 中的错误
                if error_msg := model_response.get("error"):
                    raise GrokApiException(
                        f"模型响应错误: {error_msg}",
                        "MODEL_ERROR"
                    )

                # 构建响应内容
                model_name = model_response.get("model")
                content = model_response.get("message", "")

                # 提取图片数据
                if images := model_response.get("generatedImageUrls"):
                    # 获取图片返回模式
                    image_mode = setting.global_config.get("image_mode", "url")

                    for img in images:
                        try:
                            if image_mode == "base64":
                                # base64 模式：下载并转换为 base64
                                base64_str = await image_cache_service.download_base64(f"/{img}", auth_token)
                                if base64_str:
                                    content += f"\n![Generated Image]({base64_str})"
                                else:
                                    content += f"\n![Generated Image](https://assets.grok.com/{img})"
                            else:
                                # url 模式：缓存并返回链接
                                cache_path = await image_cache_service.download_image(f"/{img}", auth_token)
                                if cache_path:
                                    img_path = img.replace('/', '-')
                                    base_url = setting.global_config.get("base_url", "")
                                    img_url = f"{base_url}/images/{img_path}" if base_url else f"/images/{img_path}"
                                    content += f"\n![Generated Image]({img_url})"
                                else:
                                    content += f"\n![Generated Image](https://assets.grok.com/{img})"
                        except Exception as e:
                            logger.warning(f"[Processor] 处理图片失败: {e}")
                            content += f"\n![Generated Image](https://assets.grok.com/{img})"

                # 返回 OpenAI 响应格式
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

            raise GrokApiException("无响应数据", "NO_RESPONSE")

        except json.JSONDecodeError as e:
            logger.error(f"[Processor] JSON解析失败: {e}")
            raise GrokApiException(f"JSON解析失败: {e}", "JSON_ERROR") from e
        except Exception as e:
            logger.error(f"[Processor] 处理响应时发生未知错误: {type(e).__name__}: {e}")
            raise GrokApiException(f"响应处理错误: {e}", "PROCESS_ERROR") from e
        finally:
            # 确保响应对象被关闭，避免双重释放
            if not response_closed and hasattr(response, 'close'):
                try:
                    response.close()
                except Exception as e:
                    logger.warning(f"[Processor] 关闭响应对象时出错: {e}")

    @staticmethod
    async def process_stream(response, auth_token: str) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        # 流式生成状态
        is_image = False
        is_thinking = False
        thinking_finished = False
        chunk_index = 0
        model = None
        filtered_tags = setting.grok_config.get("filtered_tags", "").split(",")
        video_progress_started = False
        last_video_progress = -1
        response_closed = False

        # 初始化超时管理器
        timeout_manager = StreamTimeoutManager(
            chunk_timeout=setting.grok_config.get("stream_chunk_timeout", 120),
            first_response_timeout=setting.grok_config.get("stream_first_response_timeout", 30),
            total_timeout=setting.grok_config.get("stream_total_timeout", 600)
        )

        def make_chunk(chunk_content: str, finish: str = None):
            """生成OpenAI格式的响应块"""
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
            # SSE 格式返回
            return f"data: {json.dumps(chunk_data)}\n\n"

        try:
            for chunk in response.iter_lines():
                # 超时检查
                is_timeout, timeout_msg = timeout_manager.check_timeout()
                if is_timeout:
                    logger.warning(f"[Processor] {timeout_msg}")
                    yield make_chunk("", "stop")
                    yield "data: [DONE]\n\n"
                    return

                logger.debug(f"[Processor] 接收到数据块: {len(chunk)} bytes")
                
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

                    # 错误检查
                    if error := data.get("error"):
                        error_msg = error.get('message', '未知错误')
                        logger.error(f"[Processor] Grok API返回错误: {error_msg}")
                        yield make_chunk(f"Error: {error_msg}", "stop")
                        yield "data: [DONE]\n\n"
                        return

                    # 提取响应数据
                    grok_resp = data.get("result", {}).get("response", {})
                    logger.debug(f"[Processor] 解析响应数据: {len(grok_resp)} 字段")
                    
                    if not grok_resp:
                        continue

                    # 更新模型名称
                    if user_resp := grok_resp.get("userResponse"):
                        if m := user_resp.get("model"):
                            model = m

                    # 提取视频数据
                    if video_resp := grok_resp.get("streamingVideoGenerationResponse"):
                        logger.debug(f"[Processor] 🎬 Stream: Video response chunk: {json.dumps(video_resp, indent=2)}")
                        progress = video_resp.get("progress", 0)
                        
                        if progress > last_video_progress:
                            last_video_progress = progress
                            logger.info(f"[Processor] 📊 Video generation progress: {progress}%")
                            
                            # 添加 <think> 标签
                            if not video_progress_started:
                                content = f"<think>视频已生成{progress}%\n"
                                video_progress_started = True
                                logger.info(f"[Processor] Started video progress tracking")
                            elif progress < 100:
                                content = f"视频已生成{progress}%\n"
                            else:
                                # 进度100%时关闭 <think> 标签并立即处理视频
                                logger.info(f"[Processor] ✅ Video generation complete (100%)")
                                content = f"视频已生成{progress}%</think>\n"
                                
                                # 立即下载并缓存视频
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

                    # 检查生成模式
                    if grok_resp.get("imageAttachmentInfo"):
                        is_image = True

                    # 获取token
                    token = grok_resp.get("token", "")

                    # 提取图片数据
                    if is_image:
                        if model_resp := grok_resp.get("modelResponse"):
                            # 获取图片返回模式
                            image_mode = setting.global_config.get("image_mode", "url")

                            # 初始化内容变量
                            content = ""

                            # 生成图片链接并缓存
                            for img in model_resp.get("generatedImageUrls", []):
                                try:
                                    if image_mode == "base64":
                                        # base64 模式：下载并转换为 base64
                                        base64_str = await image_cache_service.download_base64(f"/{img}", auth_token)
                                        if base64_str:
                                            # 分块发送 base64 数据，每 8KB 一个 chunk
                                            markdown_prefix = "![Generated Image](data:"
                                            markdown_suffix = ")\n"

                                            # 提取 data URL 的 mime 和 base64 部分
                                            if base64_str.startswith("data:"):
                                                parts = base64_str.split(",", 1)
                                                if len(parts) == 2:
                                                    mime_part = parts[0] + ","
                                                    b64_data = parts[1]

                                                    # 发送前缀
                                                    yield make_chunk(markdown_prefix + mime_part)
                                                    timeout_manager.mark_chunk_received()
                                                    chunk_index += 1

                                                    # 分块发送 base64 数据
                                                    chunk_size = 8192
                                                    for i in range(0, len(b64_data), chunk_size):
                                                        chunk_data = b64_data[i:i + chunk_size]
                                                        yield make_chunk(chunk_data)
                                                        timeout_manager.mark_chunk_received()
                                                        chunk_index += 1

                                                    # 发送后缀
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
                                        # url 模式：缓存并返回链接
                                        await image_cache_service.download_image(f"/{img}", auth_token)
                                        # 本地图片路径
                                        img_path = img.replace('/', '-')
                                        base_url = setting.global_config.get("base_url", "")
                                        img_url = f"{base_url}/images/{img_path}" if base_url else f"/images/{img_path}"
                                        content += f"![Generated Image]({img_url})\n"
                                except Exception as e:
                                    logger.warning(f"[Processor] 处理图片失败: {e}")
                                    content += f"![Generated Image](https://assets.grok.com/{img})\n"

                            # 发送内容
                            yield make_chunk(content.strip(), "stop")
                            timeout_manager.mark_chunk_received()
                            return
                        elif token:
                            yield make_chunk(token)
                            timeout_manager.mark_chunk_received()
                            chunk_index += 1

                    # 提取对话数据
                    else:
                        # 过滤 list 格式的 token
                        if isinstance(token, list):
                            continue

                        # 过滤特定标签
                        if any(tag in token for tag in filtered_tags if token):
                            continue

                        # 获取当前状态
                        current_is_thinking = grok_resp.get("isThinking", False)
                        message_tag = grok_resp.get("messageTag")

                        # 跳过后续的 <think> 标签
                        if thinking_finished and current_is_thinking:
                            continue

                        # 检查 toolUsageCardId - ALWAYS stream tool usage to reduce TTFT
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

                            # header 在 token 后换行
                            if message_tag == "header":
                                content = f"\n\n{token}\n\n"

                            # is_thinking 状态切换
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
                    logger.warning(f"[Processor] 解析chunk失败: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"[Processor] 处理chunk出错: {e}")
                    continue

            # 发送结束块
            yield make_chunk("", "stop")

            # 发送流结束标记
            yield "data: [DONE]\n\n"
            
            # 记录流式响应统计
            logger.info(f"[Processor] 流式响应完成，总耗时: {timeout_manager.get_total_duration():.2f}秒")

        except Exception as e:
            logger.error(f"[Processor] 流式处理严重错误: {e}")
            yield make_chunk(f"处理错误: {e}", "error")
            # 发送流结束标记
            yield "data: [DONE]\n\n"
        finally:
            # 确保响应对象被关闭
            if not response_closed and hasattr(response, 'close'):
                try:
                    response.close()
                    logger.debug("[Processor] 流式响应对象已关闭")
                except Exception as e:
                    logger.warning(f"[Processor] 关闭流式响应对象时出错: {e}")