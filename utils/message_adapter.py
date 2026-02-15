from astrbot.api.event import AstrMessageEvent
import os
import re
import aiohttp
import asyncio
import logging
import base64
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from natsort import natsorted
from asyncio import Queue

logger = logging.getLogger(__name__)


class MessageAdapter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        platform_config = self.config.get("platform", {})
        self.platform_type = platform_config.get("type", "napcat")
        self.http_host = platform_config.get("http_host", "127.0.0.1")
        self.http_port = platform_config.get("http_port", 2333)
        self.api_token = platform_config.get("api_token", "")
        self.use_base64 = platform_config.get("use_base64_upload", False)

    def get_headers(self) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.api_token:
            headers['Authorization'] = f'Bearer {self.api_token}'
        return headers

    async def get_group_root_files(self, group_id: str) -> Dict[str, Any]:
        url = f"http://{self.http_host}:{self.http_port}/get_group_root_files"
        payload = {"group_id": group_id}
        headers = self.get_headers()

        logger.debug(f"发送给消息平台-> {payload}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"获取群文件根目录失败，状态码: {response.status}, 错误信息: {error_text}")

                res = await response.json()
                if res["status"] != "ok":
                    raise Exception(f"获取群文件根目录失败，状态码: {res['status']}\n完整消息: {str(res)}")

                return res["data"]

    async def create_group_file_folder(self, group_id: str, folder_name: str) -> Optional[str]:
        url = f"http://{self.http_host}:{self.http_port}/create_group_file_folder"

        if self.platform_type == 'napcat':
            payload = {
                "group_id": group_id,
                "folder_name": folder_name
            }
        elif self.platform_type == 'llonebot':
            payload = {
                "group_id": group_id,
                "name": folder_name
            }
        elif self.platform_type == 'lagrange':
            payload = {
                "group_id": group_id,
                "name": folder_name,
                "parent_id": "/"
            }
        else:
            raise Exception("消息平台配置有误, 只能是'napcat', 'llonebot'或'lagrange'")

        headers = self.get_headers()
        logger.debug(f"发送给消息平台-> {payload}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"创建群文件夹失败，状态码: {response.status}, 错误信息: {error_text}")

                res = await response.json()
                logger.debug(f"消息平台返回-> {res}")

                if res["status"] != "ok":
                    raise Exception(f"创建群文件夹失败，状态码: {res['status']}\n完整消息: {str(res)}")

                try:
                    return res["data"]["folder_id"]
                except Exception:
                    return None

    async def get_group_folder_id(self, group_id: str, folder_name: str = '/') -> str:
        if folder_name == '/':
            return '/'

        data = await self.get_group_root_files(group_id)
        for folder in data.get('folders', []):
            if folder.get('folder_name') == folder_name:
                return folder.get('folder_id')

        folder_id = await self.create_group_file_folder(group_id, folder_name)
        if folder_id is None:
            data = await self.get_group_root_files(group_id)
            for folder in data.get('folders', []):
                if folder.get('folder_name') == folder_name:
                    return folder.get('folder_id')
            return "/"

        return folder_id

    async def upload_file(self, event: AstrMessageEvent, path: str, name: str, folder_name: str = '/') -> Dict[str, Any]:
        logger.info(f"发送 {name} 中...")

        all_files = os.listdir(path)
        pattern = re.compile(rf"^{re.escape(name)}(?: part \d+)?\.pdf$")
        matching_files = [
            os.path.join(path, f) for f in all_files if pattern.match(f)
        ]

        files = natsorted(matching_files)
        # 转换为绝对路径，防止发送相对路径给机器人端
        files = [os.path.abspath(f) for f in files]
        
        if not files:
            raise FileNotFoundError("未找到符合命名的文件")

        is_private = event.is_private_chat()
        target_id = event.get_sender_id() if is_private else event.get_group_id()
        url_type = "upload_private_file" if is_private else "upload_group_file"
        url = f"http://{self.http_host}:{self.http_port}/{url_type}"

        base_payload = {
            "file": None,
            "name": None,
            "user_id" if is_private else "group_id": target_id
        }

        if not is_private:
            base_payload["folder_id"] = await self.get_group_folder_id(target_id, folder_name)

        queue = Queue()

        async def worker():
            async with aiohttp.ClientSession() as session:
                while not queue.empty():
                    file = await queue.get()
                    payload = base_payload.copy()
                    
                    file_val = file
                    if self.use_base64:
                        try:
                            with open(file, "rb") as f:
                                b64_data = base64.b64encode(f.read()).decode("utf-8")
                                file_val = f"base64://{b64_data}"
                        except Exception as e:
                            logger.error(f"读取文件并进行 Base64 编码失败: {file}, 错误: {e}")
                            # 回退到路径上传，或者你可以选择直接跳过
                    
                    payload.update({
                        "file": file_val,
                        "name": os.path.basename(file)
                    })
                    result = await self._upload_single_file(session, url, self.get_headers(), payload)
                    results.append(result)
                    queue.task_done()

        for file in files:
            await queue.put(file)

        results = []
        workers = [worker() for _ in range(1)]
        await asyncio.gather(*workers)

        process_result = self._process_results(results)
        
        # 向用户反馈结果
        if process_result["failed_count"] == 0:
            logger.info(f"{name} 发送成功")
        else:
            error_msg = f"❌ {name} 发送失败 ({process_result['success_count']}/{process_result['total']} 成功)"
            if process_result["details"]["errors"]:
                first_error = process_result["details"]["errors"][0]
                error_msg += f"\n原因: {first_error}"
            
            await event.send(event.plain_result(error_msg))
            logger.error(f"文件上传失败: {process_result}")

        return process_result

    async def _upload_single_file(self, session: aiohttp.ClientSession, url: str,
                                  headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 增加超时控制
            timeout = aiohttp.ClientTimeout(total=120) # 2分钟上传超时
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
                
                res = await response.json()

                if res.get("status") != "ok":
                    return {"success": False, "error": res.get("message") or res.get("msg") or "未知错误"}

                return {"success": True, "data": res.get("data")}
        except asyncio.TimeoutError:
            return {"success": False, "error": "上传超时，请检查网络连接或 OneBot 适配器状态"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _process_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successes = [r["data"] for r in results if r["success"]]
        errors = [r["error"] for r in results if not r["success"]]

        if errors:
            logger.warning(f"部分文件上传失败: {errors}")

        return {
            "total": len(results),
            "success_count": len(successes),
            "failed_count": len(errors),
            "details": {
                "successes": successes,
                "errors": errors
            }
        }

    async def set_msg_emoji_like(self, message_id: str, emoji_id: str = "76"):
        """对消息添加表情回应 (NapCat/LLOneBot)
        emoji_id: QQ 表情 ID，默认76=赞
        常用: 76=赞, 124=OK, 66=爱心, 4=得意, 277=汪汪
        """
        url = f"http://{self.http_host}:{self.http_port}/set_msg_emoji_like"
        payload = {
            "message_id": message_id,
            "emoji_id": emoji_id
        }
        headers = self.get_headers()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        res = await response.json()
                        if res.get("status") == "ok":
                            logger.info(f"表情回应成功: msg={message_id}, emoji={emoji_id}")
                        else:
                            logger.warning(f"表情回应失败: {res}")
                    else:
                        logger.warning(f"表情回应请求失败: HTTP {response.status}")
        except Exception as e:
            logger.warning(f"表情回应异常: {e}")
