from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import Image, Plain, Nodes, Node
from .utils.downloader import Downloader
from .utils.html_parser import HTMLParser
from .utils.message_adapter import MessageAdapter
from .utils.tag_translator import TagTranslator
from pathlib import Path
import os
import io
import re
import json
import aiohttp
import asyncio
import glob
import logging
import traceback
import tempfile
import base64 # 新增 base64
from typing import List, Optional, Dict, Any, Union
from urllib.parse import urlparse
from PIL import Image as PILImage, ImageDraw, ImageFont
import re # 确保 re 被导入
from bs4 import BeautifulSoup # 导入 BeautifulSoup
try:
    from aiohttp_socks import ProxyConnector
    HAS_SOCKS = True
except ImportError:
    HAS_SOCKS = False

logger = logging.getLogger(__name__)


@register("astrbot_plugin_ehentai_bot", "Doro0721", "适配 AstrBot 的 EHentai画廊 转 PDF 插件", "4.2.7")
class EHentaiBot(Star):
    @staticmethod
    def _parse_proxy_config(proxy_str: str) -> Dict[str, Any]:
        """解析代理配置字符串"""
        if not proxy_str:
            return {}
        
        parsed = urlparse(proxy_str)
        
        if parsed.scheme not in ('http', 'https', 'socks5'):
            raise ValueError("仅支持HTTP/HTTPS/SOCKS5代理协议")
        
        auth = None
        if parsed.username and parsed.password:
            auth = aiohttp.BasicAuth(parsed.username, parsed.password)
        
        if not parsed.hostname:
            logger.warning(f"代理配置 '{proxy_str}' 解析失败：未找到主机名。已忽略代理设置。")
            return {}
            
        proxy_url = f"{parsed.scheme}://{parsed.hostname}"
        if parsed.port:
            proxy_url += f":{parsed.port}"
        
        return {
            'url': proxy_url,
            'auth': auth
        }
    
    @staticmethod
    def _transform_config(config: dict) -> Dict[str, Any]:
        """将扁平配置转换为嵌套字典结构"""
        # 如果已经是嵌套结构，直接返回
        if any(isinstance(v, dict) for v in config.values()):
            return config
        
        # 配置映射表：扁平键 -> 嵌套路径
        json_to_yaml_mapping = {
            "platform_type": ["platform", "type"],
            "platform_http_host": ["platform", "http_host"],
            "platform_http_port": ["platform", "http_port"],
            "platform_api_token": ["platform", "api_token"],
            "platform_use_base64_upload": ["platform", "use_base64_upload"],
            "request_headers_user_agent": ["request", "headers", "User-Agent"],
            "request_website": ["request", "website"],
            "request_cookies_ipb_member_id": ["request", "cookies", "ipb_member_id"],
            "request_cookies_ipb_pass_hash": ["request", "cookies", "ipb_pass_hash"],
            "request_cookies_igneous": ["request", "cookies", "igneous"],
            "request_cookies_sk": ["request", "cookies", "sk"],
            "request_proxies": ["request", "proxies"],
            "request_concurrency": ["request", "concurrency"],
            "request_max_retries": ["request", "max_retries"],
            "request_timeout": ["request", "timeout"],
            "output_image_folder": ["output", "image_folder"],
            "output_pdf_folder": ["output", "pdf_folder"],
            "output_search_cache_folder": ["output", "search_cache_folder"],
            "output_jpeg_quality": ["output", "jpeg_quality"],
            "output_max_pages_per_pdf": ["output", "max_pages_per_pdf"],
            "output_max_filename_length": ["output", "max_filename_length"],
            "features_enable_formatted_message_search": ["features", "enable_formatted_message_search"],
            "features_enable_cover_image_download": ["features", "enable_cover_image_download"],
        }
        
        # 需要类型转换的字段
        int_fields = [
            "platform_http_port",
            "request_concurrency",
            "request_max_retries",
            "request_timeout",
            "output_jpeg_quality",
            "output_max_pages_per_pdf",
            "output_max_filename_length"
        ]
        
        bool_fields = [
            "platform_use_base64_upload",
            "features_enable_formatted_message_search",
            "features_enable_cover_image_download"
        ]
        
        # 处理配置值
        processed_config = {}
        for key, value in config.items():
            if value == "" or value is None:
                continue
            
            if key in int_fields:
                try:
                    processed_config[key] = int(value)
                except (ValueError, TypeError):
                    logger.warning(f"无法将 {key} 的值 '{value}' 转换为整数，已跳过此项")
                    continue
            elif key in bool_fields:
                if isinstance(value, str):
                    processed_config[key] = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    processed_config[key] = bool(value)
            else:
                processed_config[key] = value
        
        # 转换为嵌套结构
        nested_config = {}
        for json_key, value in processed_config.items():
            if json_key in json_to_yaml_mapping:
                path_parts = json_to_yaml_mapping[json_key]
                current = nested_config
                for i, part in enumerate(path_parts[:-1]):
                    current = current.setdefault(part, {})
                current[path_parts[-1]] = value
        
        # 后处理：添加代理配置和验证cookies
        if 'request' in nested_config:
            request = nested_config['request']
            website = request.get('website')
            cookies = request.get('cookies', {})
            
            # 如果设置为exhentai但cookies不完整，切换为e-hentai
            if website == 'exhentai':
                if any(not cookies.get(key, '') for key in ["ipb_member_id", "ipb_pass_hash", "igneous"]):
                    request['website'] = 'e-hentai'
                    logger.warning("网站设置为里站exhentai但cookies不完整，已更换为表站e-hentai")
            
            # 解析代理配置
            proxy_str = request.get('proxies', '')
            request['proxy_str'] = proxy_str # 保留原始字符串
            proxy_config = EHentaiBot._parse_proxy_config(proxy_str)
            request['proxy'] = proxy_config
        
        # 确保关键配置项始终存在默认结构
        if 'output' not in nested_config:
            nested_config['output'] = {}
        if 'request' not in nested_config:
            nested_config['request'] = {}
        if 'features' not in nested_config:
            nested_config['features'] = {}
        if 'platform' not in nested_config:
            nested_config['platform'] = {}
        
        return nested_config
    
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = self._transform_config(config)
        self.parser = HTMLParser()
        self.uploader = MessageAdapter(self.config)
        self.downloader = Downloader(self.config, self.uploader, self.parser)
        self.tag_translator = TagTranslator()

    def add_number_to_image(self, image: PILImage.Image, number: int) -> PILImage.Image:
        """为单张图片添加数字序号"""
        image = image.convert("RGBA")
        txt_layer = PILImage.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)

        try:
            font = ImageFont.truetype("msyh.ttc", size=60)
        except IOError:
            try:
                font = ImageFont.truetype("arial.ttf", size=60)
            except IOError:
                font = ImageFont.load_default()

        text = str(number)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        rect_height = text_height + 20
        rect_pos = (0, image.height - rect_height, image.width, image.height)
        draw.rectangle(rect_pos, fill=(0, 0, 0, 150))

        text_x = (image.width - text_width) / 2
        text_y = image.height - rect_height + 10
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))

        out = PILImage.alpha_composite(image, txt_layer)
        return out.convert("RGB")

    @staticmethod
    def split_text_by_length(text: str, max_length: int = 4000) -> List[str]:
        result = []
        label = '画廊链接'
        start = 0
        last_link_end = -1
        last_newline = -1
        for i, ch in enumerate(text):
            if ch == '\n':
                last_newline = i
            if text.startswith(label, i - len(label) + 1):
                next_newline_pos = text.find('\n', i)
                if next_newline_pos != -1:
                    last_link_end = next_newline_pos + 1
                else:
                    last_link_end = len(text)
            if i - start + 1 >= max_length:
                cut = last_link_end if last_link_end > start else (
                    last_newline + 1 if last_newline >= start else start + max_length)
                result.append(text[start:cut])
                start = cut
                last_link_end = -1
                last_newline = -1
        if start < len(text):
            result.append(text[start:])
        return result

    async def _resolve_url_from_input(self, event: AstrMessageEvent, user_input: str) -> Optional[str]:
        """从用户输入（URL或序号）解析画廊URL"""
        output_config = self.config.get('output', {})
        search_cache_folder = Path(output_config.get('search_cache_folder', 'data/ehentai/searchCache'))
        pattern = re.compile(r'^https://(e-hentai|exhentai)\.org/g/\d{7}/[a-f0-9]{10}/?$')

        if pattern.match(user_input):
            return user_input

        if user_input.isdigit() and int(user_input) > 0:
            cache_file = search_cache_folder / f"{event.get_sender_id()}.json"
            if not cache_file.exists():
                await event.send(event.plain_result("未找到搜索记录，请先使用'搜eh'命令"))
                return None

            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            if user_input in cache_data:
                url = cache_data[user_input]
                await event.send(event.plain_result(f"正在获取画廊链接: {url}"))
                return url
            else:
                await event.send(event.plain_result(f"未找到索引为 {user_input} 的画廊"))
                return None

        await event.send(event.plain_result("输入的画廊链接或序号无效，请重试..."))
        return None

    @staticmethod
    def parse_command(message: str) -> List[str]:
        cleaned_text = re.sub(r'@\S+\s*', '', message).strip()
        return [p for p in cleaned_text.split(' ') if p][1:]

    async def _get_session(self) -> aiohttp.ClientSession:
        """根据配置创建一个带有正确代理设置的 aiohttp.ClientSession"""
        request_config = self.config.get('request', {})
        proxy_str = request_config.get('proxy_str', '')
        
        connector = None
        if proxy_str and proxy_str.startswith('socks5'):
            if HAS_SOCKS:
                connector = ProxyConnector.from_url(proxy_str, ssl=False)
            else:
                logger.error("检测到 SOCKS5 代理配置，但未安装 aiohttp-socks 库。请运行 'pip install aiohttp-socks'")
        
        if connector is None:
            connector = aiohttp.TCPConnector(ssl=False)
            
        return aiohttp.ClientSession(connector=connector)

    async def download_thumbnail(self, url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore):
        """下载封面图片"""
        try:
            # Prefer User-Agent from config, but keep image-specific headers
            headers = {
                'User-Agent': self.config.get('request', {}).get('headers', {}).get('User-Agent',
                                                                                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'),
                'Referer': f"https://{self.config.get('request', {}).get('website', 'e-hentai')}.org/",
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
            }

            request_config = self.config.get('request', {})
            proxy_conf = request_config.get('proxy', {})
            proxy_str = request_config.get('proxy_str', '')
            
            cookies = request_config.get('cookies') if request_config.get('website') == 'exhentai' else None
            timeout = aiohttp.ClientTimeout(total=request_config.get('timeout', 30))
            
            # 只有 http/https 代理使用 aiohttp 原生 proxy 参数
            proxy = None
            proxy_auth = None
            if not proxy_str.startswith('socks5'):
                proxy = proxy_conf.get('url')
                proxy_auth = proxy_conf.get('auth')

            async with semaphore:
                async with session.get(
                        url,
                        headers=headers,
                        cookies=cookies,
                        proxy=proxy,
                        proxy_auth=proxy_auth,
                        timeout=timeout,
                        ssl=False
                ) as response:
                    response.raise_for_status()
                    return PILImage.open(io.BytesIO(await response.read()))
        except Exception as e:
            logger.warning(f"下载封面图片失败: {url} - {e}")
            return None

    async def _download_thumbnail_with_tracking(self, url: str, session: aiohttp.ClientSession,
                                                semaphore: asyncio.Semaphore):
        """包装封面下载任务以进行跟踪"""
        image = await self.download_thumbnail(url, session, semaphore)
        if image:
            return {"success": True, "image": image, "url": url}
        else:
            return {"success": False, "error": "Download failed", "url": url}

    async def _download_covers_with_retry(self, search_results: List[dict]) -> List[PILImage.Image]:
        """带重试机制的封面下载器"""
        if not self.config.get('features', {}).get('enable_cover_image_download', True):
            return []

        concurrency = self.config.get('request', {}).get('concurrency', 5)
        semaphore = asyncio.Semaphore(concurrency)

        urls_to_download = [res['cover_url'] for res in search_results if res.get('cover_url')]
        if not urls_to_download:
            return []

        async with await self._get_session() as session:
            # 首次尝试
            tasks = [self._download_thumbnail_with_tracking(url, session, semaphore) for url in urls_to_download]
            results = await asyncio.gather(*tasks)

            successful_images = [r['image'] for r in results if r.get('success')]
            failed_urls = [r['url'] for r in results if not r.get('success')]

            # 重试逻辑
            if failed_urls:
                logger.info(f"首次封面下载有 {len(failed_urls)} 张失败，正在重试...")
                await asyncio.sleep(1)  # 重试前短暂延迟

                retry_tasks = [self._download_thumbnail_with_tracking(url, session, semaphore) for url in failed_urls]
                retry_results = await asyncio.gather(*retry_tasks)

                successful_images.extend([r['image'] for r in retry_results if r.get('success')])
                final_failed_count = sum(1 for r in retry_results if not r.get('success'))

                if final_failed_count > 0:
                    logger.warning(f"封面下载重试后仍有 {final_failed_count} 张失败。")

        return successful_images

    def create_combined_image(self, images):
        """将多个封面图片拼接成一张图片，按五张一排排列"""
        if not images:
            return None

        valid_images = [img for img in images if img is not None]
        if not valid_images:
            return None

        # 为每张图片添加编号
        numbered_images = [self.add_number_to_image(img, i) for i, img in enumerate(valid_images, 1)]

        target_height = 800
        padding = 10
        images_per_row = 5

        scaled_widths = []
        for img in numbered_images:
            width, height = img.size
            scaled_width = int((width * target_height) / height)
            scaled_widths.append(scaled_width)

        rows = []
        current_row_widths = []
        current_row_total = 0

        for i, scaled_width in enumerate(scaled_widths):
            if len(current_row_widths) < images_per_row:
                current_row_widths.append(scaled_width)
                current_row_total += scaled_width
            else:
                rows.append((current_row_widths, current_row_total))
                current_row_widths = [scaled_width]
                current_row_total = scaled_width

        if current_row_widths:
            rows.append((current_row_widths, current_row_total))

        max_row_width = max(row_total for _, row_total in rows) if rows else 0
        total_width = max_row_width + (images_per_row - 1) * padding

        total_height = len(rows) * target_height + (len(rows) - 1) * padding

        combined_image = PILImage.new('RGB', (total_width, total_height), (255, 255, 255))

        y_offset = 0
        image_index = 0
        for row_widths, row_total in rows:
            row_start_x = (total_width - (row_total + (len(row_widths) - 1) * padding)) // 2
            x_offset = row_start_x

            for scaled_width in row_widths:
                img = numbered_images[image_index]
                img = img.convert('RGB')
                img = img.resize((scaled_width, target_height), PILImage.Resampling.LANCZOS)

                combined_image.paste(img, (x_offset, y_offset))
                x_offset += scaled_width + padding
                image_index += 1

            y_offset += target_height + padding

        self.add_random_blocks(combined_image)
        return combined_image

    def add_random_blocks(self, image):
        """添加随机色块并进行轻微图像变换以规避图片审查"""
        import random
        from PIL import ImageOps, ImageEnhance

        # 1. 随机水平翻转 (极其有效的 Hash 规避)
        if random.random() > 0.5:
            image = ImageOps.mirror(image)

        # 2. 轻微亮度调节 (改变像素值)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.98, 1.02))

        width, height = image.size
        
        # 3. 添加少量极小随机色块
        num_blocks = random.randint(5, 10)
        for _ in range(num_blocks):
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            block_width = random.randint(1, 3)
            block_height = random.randint(1, 3)
            x2 = min(x1 + block_width, width - 1)
            y2 = min(y1 + block_height, height - 1)
            
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            alpha = random.randint(10, 30)
            
            for x in range(x1, x2):
                for y in range(y1, y2):
                    current = image.getpixel((x, y))
                    new_r = int((current[0] * (255 - alpha) + r * alpha) / 255)
                    new_g = int((current[1] * (255 - alpha) + g * alpha) / 255)
                    new_b = int((current[2] * (255 - alpha) + b * alpha) / 255)
                    image.putpixel((x, y), (new_r, new_g, new_b))
                    
        return image


    @filter.command("es")
    async def handle_es(self, event: AstrMessageEvent):
        """
        搜索 EHentai 画廊
        用法: /es <关键词> [页码]
        示例: /es loli
        示例: /es loli 2
        """
        # 解析参数：类似于 nhentai 的解析逻辑
        message = event.message_str.strip()
        parts = message.split(maxsplit=1)
        
        if len(parts) < 2:
            yield event.plain_result(
                "🔍 EHentai 搜索\n"
                "用法: /es <关键词> [页码]\n"
                "示例: /es loli 2"
            )
            return

        query_str = parts[1].strip()
        words = query_str.split()
        
        # 检查最后一个词是否为页码
        target_page = 1
        if len(words) > 1 and words[-1].isdigit():
            target_page = int(words[-1])
            query = " ".join(words[:-1])
        else:
            query = query_str
            
        await self._search_and_reply(event, query, target_page)

    async def _search_and_reply(self, event: AstrMessageEvent, query: str, page: int):
        """执行搜索并回复结果（供 /es 和翻页使用）"""
        # 发送提示
        yield event.plain_result(f"🔍 正在搜索: {query} (第{page}页)...")

        try:
            search_results = await self.downloader.crawl_ehentai(
                query,
                0, # min_rating
                0, # min_pages
                page - 1 # target_page
            )

            if not search_results:
                yield event.plain_result("未找到符合条件的结果")
                return

            # 缓存搜索结果（用于快速下载和翻页）
            user_id = event.get_sender_id()
            cache_data = {
                "results": search_results,
                "time": asyncio.get_event_loop().time(),
                "query": query,
                "page": page
            }
            
            if not hasattr(self, '_search_cache'):
                self._search_cache = {}
            self._search_cache[user_id] = cache_data

            # 构建消息链
            chain = []
            header = f"🔍 搜索结果 (第 {page} 页)\n━━━━━━━━━━━━\n"
            chain.append(Plain(header))

            # 异步下载所有封面
            semaphore = asyncio.Semaphore(5)
            
            # 复用 _download_covers_with_retry
            covers = await self._download_covers_with_retry(search_results)

            for idx, result in enumerate(search_results, 1):
                # 文本部分
                title = result['title']
                
                # 尝试从 gallery_url 提取 gid/token
                g_url = result['gallery_url']
                g_parts = g_url.strip('/').split('/')
                if len(g_parts) >= 2:
                    current_gid = g_parts[-2]
                    current_token = g_parts[-1]
                else:
                    current_gid = "?"
                    current_token = "?"
                
                # 更新 result 以包含 gid (用于快速下载)
                result['_gid'] = current_gid
                result['_token'] = current_token

                info = f"[{idx}] 📖 {title}\n"
                info += f"🔖 ID: {current_gid} | 📄 {result['pages']}页 | ⭐ {result['rating']}\n"
                info += f"✍️ 作者: {result['author']} | 📂 {result['category']}\n"
                info += f"📅 {result['timestamp']}\n"
                
                chain.append(Plain(info))

                # 图片部分
                if idx <= len(covers) and covers[idx-1]:
                    img = covers[idx-1]
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    chain.append(Image.fromBase64(img_b64))
                
                chain.append(Plain("\n━━━━━━━━━━━━\n" if idx < len(search_results) else "\n"))

            footer = "\n💡 30秒内回复:\n• 数字(1-9): 下载对应画廊\n• '下': 下一页 | '上': 上一页"
            chain.append(Plain(footer))
            
            yield event.chain_result(chain)

        except Exception as e:
            logger.exception("搜索处理异常")
            yield event.plain_result(f"搜索出错: {str(e)}")

    @filter.regex(r"^(?:\d+|上|下)$")
    async def handle_quick_interaction(self, event: AstrMessageEvent):
        """处理快速交互：数字下载、翻页"""
        text = event.message_str.strip()
        user_id = event.get_sender_id()
        
        # 检查缓存
        if not hasattr(self, '_search_cache') or user_id not in self._search_cache:
            return 
            
        cache = self._search_cache[user_id]
        # 检查过期 (30秒)
        if asyncio.get_event_loop().time() - cache["time"] > 30:
            del self._search_cache[user_id]
            return 
            
        # 处理翻页
        if text == "上":
            current_page = cache.get("page", 1)
            new_page = current_page - 1
            if new_page < 1:
                yield event.plain_result("🚫 已经是第一页了")
                return
            
            # 更新缓存时间防止过期，虽然 _search_and_reply 会覆盖
            async for result in self._search_and_reply(event, cache["query"], new_page):
                yield result
            return

        elif text == "下":
            current_page = cache.get("page", 1)
            new_page = current_page + 1
            
            async for result in self._search_and_reply(event, cache["query"], new_page):
                yield result
            return

        # 处理下载 (纯数字)
        if not text.isdigit():
            return

        idx = int(text)
        results = cache["results"]
        if idx < 1 or idx > len(results):
            return 
            
        target = results[idx-1]
        gid = target.get('_gid')
        token = target.get('_token')
        
        if not gid or not token:
            yield event.plain_result("无法解析画廊信息，请重新搜索")
            return
            
        # 触发下载流程
        yield event.plain_result(f"🚀 已选择 [{idx}]，开始下载 ID: {gid}...")
        
        # 清除缓存防止重复触发
        del self._search_cache[user_id]
        
        # 调用下载逻辑
        await self.download_gallery(event, gid, token)


    async def send_formatted_search_results(self, event, result_text, search_results, combined_image_obj=None):
        """发送格式化搜索结果（转发消息格式）"""
        text_parts = self.split_text_by_length(result_text)
        sender_name = "图片搜索bot"
        sender_id = event.get_self_id()
        try:
            sender_id = int(sender_id)
        except Exception:
            pass

        nodes_list = []
        temp_file_path = ''
        try:
            if combined_image_obj:
                self.add_random_blocks(combined_image_obj)

                img_byte_arr = io.BytesIO()
                combined_image_obj.save(img_byte_arr, 'JPEG', quality=85)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(img_byte_arr.getvalue())
                    temp_file_path = temp_file.name

                image_node = Node(
                    name=sender_name,
                    uin=sender_id,
                    content=[Image(temp_file_path)]
                )
                nodes_list.append(image_node)

            for i, part in enumerate(text_parts):
                text_node = Node(
                    name=sender_name,
                    uin=sender_id,
                    content=[Plain(f"[  搜索结果 {i + 1} / {len(text_parts)}  ]\n\n{part}")]
                )
                nodes_list.append(text_node)

            if nodes_list:
                nodes = Nodes(nodes_list)
                await event.send(event.chain_result([nodes]))
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    # @filter.command("eh翻页")
    # async def jump_to_page(self, event: AstrMessageEvent):
    #     pass
        

    async def download_gallery(self, event: AstrMessageEvent, gid: str = None, token: str = None):
        """下载画廊（支持直接调用或命令调用）"""
        output_config = self.config.get('output', {})
        image_folder = Path(output_config.get('image_folder', 'data/ehentai/tempImages'))
        image_folder.mkdir(exist_ok=True, parents=True) # 使用绝对路径? main.py 里没有 self.image_folder 存储绝对路径，是在 Downloader 里。
        # 这里只是创建目录，Downloader 会再次处理。
        
        # 修正：移除 main.py 里对 output_config 的路径处理，直接依赖 Downloader
        # 或者为了保险起见，这里不处理目录，只负责解析参数。
        
        try:
            url = ""
            if gid and token:
                website = self.config.get('request', {}).get('website', 'e-hentai')
                url = f"https://{website}.org/g/{gid}/{token}/"
            else:
                args = self.parse_command(event.message_str)
                if len(args) != 1:
                    # 如果不是命令调用，或者是参数不对
                     # 由于移除了 help 命令，这里直接返回提示
                    await event.send(event.plain_result("参数错误"))
                    return

                url = await self._resolve_url_from_input(event, args[0])
            
            if not url:
                return

            # 记录日志而非发送消息
            logger.info(f"开始下载: {url}")

            async with await self.downloader._get_session() as session:
                is_pdf_exist = await self.downloader.process_pagination(event, session, url)

                if not is_pdf_exist:
                    # 使用 downloader 的 stored gallery_title
                    title = self.downloader.gallery_title
                    safe_title = await self.downloader.merge_images_to_pdf(event, title)
                    # output_config 里的 pdf_folder 可能是相对路径，Downloader 里是绝对路径。
                    # upload_file 需要绝对路径。
                    # 从 downloader 获取绝对路径
                    pdf_folder = self.downloader.pdf_folder
                    await self.uploader.upload_file(event, pdf_folder, safe_title)

                    # 发送后自动清理 PDF 文件
                    try:
                        pattern = re.compile(rf"^{re.escape(safe_title)}(?: part \d+)?\.pdf$")
                        for f in os.listdir(pdf_folder):
                            if pattern.match(f):
                                os.remove(os.path.join(pdf_folder, f))
                        logger.info(f"已清理 PDF 文件: {safe_title}")
                    except Exception as e:
                        logger.warning(f"清理 PDF 文件失败: {e}")

        except Exception as e:
            logger.exception("下载失败")
            stack_info = traceback.format_exc()
            await event.send(event.plain_result(f"下载失败：{str(e)}\n{stack_info}"))

    @filter.regex(r"https?://(?:e-hentai|exhentai)\.org/g/\d+/[a-f0-9]+/?")
    async def handle_link_parsing(self, event: AstrMessageEvent, *args):
        """解析 E-Hentai/ExHentai 画廊链接并显示卡片信息"""
        # 兼容性处理：如果 event 不是事件对象（可能是参数偏移），则从参数中寻找
        if not hasattr(event, "message_str"):
            for arg in args:
                if hasattr(arg, "message_str"):
                    event = arg
                    break
        
        if not hasattr(event, "message_str"):
            logger.error(f"无法获取消息内容，event类型: {type(event)}")
            return

        text = event.message_str.strip()
        # 提取链接
        pattern = re.compile(r"https?://(e-hentai|exhentai)\.org/g/(\d+)/([a-f0-9]+)/?")
        match = pattern.search(text)
        if not match:
            return
            
        domain, gid, token = match.groups()
        url = match.group(0)
        
        await event.send(event.plain_result(f"🔍 正在解析画廊: {gid} ..."))
        
        # 保存原始消息ID，用于下载完后表情回应
        original_msg_id = None
        try:
            original_msg_id = event.message_obj.message_id
        except:
            pass
        
        try:
            # 使用同一个 session 完成 HTML 获取和封面下载
            async with await self._get_session() as session:
                # 确保标签翻译数据已加载
                await self.tag_translator.ensure_loaded(session)
                
                html = await self.downloader.fetch_with_retry(session, url)
                
                if not html:
                    await event.send(event.plain_result("无法获取画廊详情"))
                    return
                    
                # 使用 extract_gallery_info 获取标题
                title, _ = self.parser.extract_gallery_info(html)
                
                soup = BeautifulSoup(html, "html.parser")
                
                # 标题
                gn = soup.select_one("#gn")
                gj = soup.select_one("#gj")
                title_en = gn.text.strip() if gn else ""
                title_jp = gj.text.strip() if gj else ""
                
                if title_jp and title_en and title_jp != title_en:
                    display_title = f"{title_jp}\n{title_en}"
                else:
                    display_title = title_jp or title_en or title
                
                # 标签映射表
                tag_mapping = {
                    "language": "语言",
                    "parody": "原作",
                    "character": "角色",
                    "group": "社团",
                    "artist": "艺术家",
                    "female": "女性",
                    "male": "男性",
                    "mixed": "混合",
                    "other": "其他",
                    "misc": "其他"
                }
                
                # 标签解析（使用 EhTagTranslation 翻译标签值）
                tag_rows = soup.select("#taglist tr")
                tags_text = ""
                for row in tag_rows:
                    tds = row.find_all("td")
                    if len(tds) == 2:
                        cat_raw = tds[0].text.strip(":")
                        cat_cn = tag_mapping.get(cat_raw, cat_raw)
                        
                        tag_links = tds[1].find_all("a")
                        tag_names = []
                        for t in tag_links:
                            raw_tag = t.text.strip().split(" | ")[0]
                            cn_tag = self.tag_translator.translate(cat_raw, raw_tag)
                            tag_names.append(f"#{cn_tag}")
                        
                        if tag_names:
                            tags_text += f"{cat_cn}: {' '.join(tag_names)}\n"

                # 构建消息
                chain = []
                
                # 标题 + 标签合为一段
                info_text = f"{display_title}\n"
                if tags_text:
                    info_text += tags_text
                chain.append(Plain(info_text))
                
                # 获取画廊第一张原图作为预览封面
                cover_img_obj = None
                try:
                    subpage_urls = self.parser.extract_subpage_urls(html)
                    if subpage_urls:
                        first_page_html = await self.downloader.fetch_with_retry(session, subpage_urls[0])
                        if first_page_html:
                            sub_soup = BeautifulSoup(first_page_html, "html.parser")
                            first_img_url = sub_soup.select_one("#img")
                            if first_img_url:
                                first_img_url = first_img_url.get("src")
                            
                            if not first_img_url:
                                img_el = sub_soup.select_one("#i3 img")
                                if img_el:
                                    first_img_url = img_el.get("src")
                            
                            if first_img_url:
                                img_bytes = await self.downloader.fetch_bytes_with_retry(session, first_img_url)
                                if img_bytes:
                                    cover_img_obj = PILImage.open(io.BytesIO(img_bytes))
                except Exception as e:
                    logger.warning(f"获取第一张原图失败，回退到缩略图: {e}")
                
                # 回退：如果原图获取失败，使用缩略图
                if not cover_img_obj:
                    cover_url = None
                    cover_img_tag = soup.select_one("#gd1 img")
                    if cover_img_tag:
                        cover_url = cover_img_tag.get("src")
                    if not cover_url:
                        cover_div = soup.select_one("#gd1 div")
                        if cover_div:
                            style = cover_div.get("style", "")
                            m = re.search(r'url\((.+?)\)', style)
                            if m:
                                cover_url = m.group(1).strip("'\"")
                    if not cover_url:
                        og_img = soup.select_one('meta[property="og:image"]')
                        if og_img:
                            cover_url = og_img.get("content")
                    if cover_url:
                        logger.info(f"回退封面 URL: {cover_url}")
                        semaphore = asyncio.Semaphore(1)
                        cover_img_obj = await self.download_thumbnail(cover_url, session, semaphore)
                
                # 构建封面消息
                if cover_img_obj:
                    try:
                        # 降低规格以确保 QQ 发送成功率
                        max_side = 700
                        w, h = cover_img_obj.size
                        if max(w, h) > max_side:
                            ratio = max_side / max(w, h)
                            cover_img_obj = cover_img_obj.resize(
                                (int(w * ratio), int(h * ratio)),
                                PILImage.Resampling.LANCZOS
                            )
                        # 应用反和谐处理
                        cover_img_obj = self.add_random_blocks(cover_img_obj)
                        buffered = io.BytesIO()
                        # 调整 JPEG 质量到 80，兼顾体积和清晰度
                        cover_img_obj.convert("RGB").save(buffered, format="JPEG", quality=80)
                        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        chain.append(Image.fromBase64(img_b64))
                    except Exception as e:
                        logger.error(f"处理封面图失败: {e}")
                else:
                    logger.warning("未能获取封面图")
            
            # 发送详情（如果带图发送失败，回退到纯文字）
            try:
                await event.send(event.chain_result(chain))
            except Exception as e:
                logger.warning(f"带图发送失败（可能被和谐），回退纯文字: {e}")
                text_chain = [item for item in chain if not isinstance(item, Image)]
                if text_chain:
                    try:
                        await event.send(event.chain_result(text_chain))
                    except Exception:
                        pass
            
            # 自动下载
            await self.download_gallery(event, gid, token)
            
            # 下载完成后对原消息添加表情回应
            if original_msg_id:
                try:
                    await self.uploader.set_msg_emoji_like(str(original_msg_id), "66")  # 66=❤️爱心
                except Exception as e:
                    logger.warning(f"表情回应失败: {e}")
            
        except Exception as e:
            logger.error(f"链接解析失败: {e}")
            await event.send(event.plain_result(f"解析失败: {e}"))
            
    # @filter.command("归档eh")
    async def archive_gallery(self, event: AstrMessageEvent):
        output_config = self.config.get('output', {})
        search_cache_folder = Path(output_config.get('search_cache_folder', 'data/ehentai/searchCache'))
        search_cache_folder.mkdir(exist_ok=True, parents=True)

        try:
            args = self.parse_command(event.message_str)
            if len(args) != 1:
                await event.send(event.plain_result("参数错误，归档操作只需要一个参数（画廊链接或搜索结果序号）"))
                return

            url = await self._resolve_url_from_input(event, args[0])
            if not url:
                return

            pattern = re.compile(r'^https://(e-hentai|exhentai)\.org/g/(\d{7})/([a-f0-9]{10})/?$')
            match = pattern.match(url)
            if not match:
                await event.send(event.plain_result("无法解析画廊链接，请重试..."))
                return

            _, gid, token = match.groups()
            
            await event.send(event.plain_result("正在获取归档链接，请稍候..."))
            
            async with await self._get_session() as session:
                download_url = await self.downloader.get_archive_url(session, gid, token)
                
                if download_url:
                    await event.send(event.plain_result(f"归档链接获取成功，请尽快下载（链接仅能访问一次）：\n{download_url}"))
                else:
                    await event.send(event.plain_result("归档链接获取失败，请检查账号权限或重试"))

        except Exception as e:
            logger.exception("归档失败")
            await event.send(event.plain_result(f"归档失败：{str(e)}"))

    @filter.command("eh")
    async def eh_helper(self, event: AstrMessageEvent):
        help_text = """📖 EHentai 插件使用指南 (v4.0.9)
━━━━━━━━━━━━━━━━━━━━━
🔍 搜索与下载
/es <关键词> [页码]
• 搜索画廊，结果中回复数字可快速下载
• 示例: /es loli
• 示例: /es loli 2

🚀 快速下载
• 在搜索结果出现后 30秒内，直接回复序号 (1-9) 即可开始下载

🔗 链接解析
• 发送 E-Hentai/ExHentai 画廊链接，自动解析并提供下载选项

ℹ️ 其他
• /eh <ID> <Token> - 高级下载 (一般由按钮或链接触发)
━━━━━━━━━━━━━━━━━━━━━"""
        await event.send(event.plain_result(help_text))

    @filter.command("重载eh配置")
    async def reload_config(self, event: AstrMessageEvent):
        await event.send(event.plain_result("正在重载配置参数"))
        # 配置由框架管理，无需手动重载
        self.uploader = MessageAdapter(self.config)
        self.downloader = Downloader(self.config, self.uploader, self.parser)
        await event.send(event.plain_result("已重载配置参数"))
    
    # @filter.regex(r"^(?:\[([^\]]+)\]|\(([^\)]+)\))\s*(.*)$")
    # async def search_by_formatted_message(self, event: AstrMessageEvent):
        """
        监听特定格式的消息，自动提取作者名和作品名，并拼接为搜索关键词进行搜索。
        """
        # 检查是否启用格式化消息搜索功能
        if not self.config.get("features", {}).get("enable_formatted_message_search", True):
            return 
            
        match = re.search(r"^(?:\[([^\]]+)\]|\(([^\)]+)\))\s*(.*)$", event.message_str)
        if not match:
            return
            
        author = match.group(1) if match.group(1) else match.group(2)
        title = match.group(3).strip()

        # 移除作品名中可能存在的额外信息，例如[中国翻訳]
        title = re.sub(r'\[[^\]]+\]|\([^\)]+\)', '', title).strip()

        if not author or not title:
            logger.warning(f"未能从消息中提取有效的作者或作品名: {event.message_str}")
            return

        # 将空格替换为+
        search_query = f"{author.replace(' ', '+')}+{title.replace(' ', '+')}"
        
        event.message_str = f"搜eh {search_query}"
        
        await self.search_gallery(event)
        
        return
        
    async def terminate(self):
        pass
