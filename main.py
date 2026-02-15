from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import Image, Plain, Nodes, Node
from .utils.downloader import Downloader
from .utils.html_parser import HTMLParser
from .utils.message_adapter import MessageAdapter
from pathlib import Path
import os
import io
import re
import json
import aiohttp
import asyncio
import glob
import logging
import tempfile
from typing import List, Optional, Dict, Any, Union
from urllib.parse import urlparse
from PIL import Image as PILImage, ImageDraw, ImageFont
try:
    from aiohttp_socks import ProxyConnector
    HAS_SOCKS = True
except ImportError:
    HAS_SOCKS = False

logger = logging.getLogger(__name__)


@register("astrbot_plugin_ehentai_bot", "Doro0721", "适配 AstrBot 的 EHentai画廊 转 PDF 插件", "4.0.4")
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
        """添加随机色块以规避图片审查"""
        import random
        
        width, height = image.size
        
        # 添加10-20个随机色块
        num_blocks = random.randint(10, 20)
        
        for _ in range(num_blocks):
            # 随机位置
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            
            # 随机大小（较小，不影响观看）
            block_width = random.randint(3, 8)
            block_height = random.randint(3, 8)
            
            # 确保色块不超出图片边界
            x2 = min(x1 + block_width, width - 1)
            y2 = min(y1 + block_height, height - 1)
            
            # 随机颜色（半透明）
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            alpha = random.randint(30, 100)  # 透明度
            
            # 获取原始像素
            for x in range(x1, x2):
                for y in range(y1, y2):
                    if 0 <= x < width and 0 <= y < height:
                        # 获取当前像素颜色
                        current = image.getpixel((x, y))
                        
                        # 混合颜色（考虑透明度）
                        new_r = int((current[0] * (255 - alpha) + r * alpha) / 255)
                        new_g = int((current[1] * (255 - alpha) + g * alpha) / 255)
                        new_b = int((current[2] * (255 - alpha) + b * alpha) / 255)
                        
                        # 设置新颜色
                        image.putpixel((x, y), (new_r, new_g, new_b))

    @filter.command("搜eh")
    async def search_gallery(self, event: AstrMessageEvent):
        defaults = {
            "min_rating": 2,
            "min_pages": 1,
            "target_page": 1
        }

        try:
            args = self.parse_command(event.message_str)
            if not args:
                await self.eh_helper(event)
                return

            if len(args) > 4:
                await event.send(event.plain_result("参数过多，最多支持4个参数：标签 评分 页数 页码"))
                return

            raw_tags = args[0]
            tags = re.sub(r'[，,+]+', ' ', args[0])

            params = defaults.copy()
            params["tags"] = raw_tags
            param_names = ["min_rating", "min_pages", "target_page"]

            for i, (name, value) in enumerate(zip(param_names, args[1:]), 1):
                try:
                    params[name] = int(value)
                except ValueError:
                    await event.send(event.plain_result(f"第{i + 1}个参数应为整数: {value}"))
                    return

            await event.send(event.plain_result("正在搜索，请稍候..."))

            search_results = await self.downloader.crawl_ehentai(
                tags,
                params["min_rating"],
                params["min_pages"],
                params["target_page"]
            )

            if not search_results:
                await event.send(event.plain_result("未找到符合条件的结果"))
                return

            cache_data = {"params": params}
            for idx, result in enumerate(search_results, 1):
                cache_data[str(idx)] = result['gallery_url']

            output_config = self.config.get('output', {})
            search_cache_folder = Path(output_config.get('search_cache_folder', 'data/ehentai/searchCache'))
            search_cache_folder.mkdir(exist_ok=True, parents=True)

            cache_file = search_cache_folder / f"{event.get_sender_id()}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            message_components = []
            combined_image_obj = None

            covers = await self._download_covers_with_retry(search_results)
            if covers:
                combined_image_obj = self.create_combined_image(covers)

            output_lines = []
            for idx, result in enumerate(search_results, 1):
                output_lines.append(f"[{idx}] {result['title']}")
                output_lines.append(
                    f" 作者: {result['author']} | 分类: {result['category']} | 页数: {result['pages']} | "
                    f"评分: {result['rating']} | 上传时间: {result['timestamp']}"
                )
                output_lines.append(f" 画廊链接: {result['gallery_url']}")
            output = "\n".join(output_lines)

            if self.config.get('features', {}).get('enable_formatted_message_search', True):
                await self.send_formatted_search_results(event, output, search_results, combined_image_obj)
            else:
                temp_file_path = ''
                try:
                    if combined_image_obj:
                        img_byte_arr = io.BytesIO()
                        combined_image_obj.save(img_byte_arr, format='JPEG', quality=85)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                            temp_file.write(img_byte_arr.getvalue())
                            temp_file_path = temp_file.name
                        message_components.append(Image(temp_file_path))

                    message_components.append(Plain(output))
                    await event.send(MessageEventResult(message_components))
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

        except ValueError as e:
            logger.exception("参数解析失败")
            await event.send(event.plain_result(f"参数错误：{str(e)}"))

        except Exception as e:
            logger.exception("搜索失败")
            await event.send(event.plain_result(f"搜索失败：{str(e)}"))

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

    @filter.command("eh翻页")
    async def jump_to_page(self, event: AstrMessageEvent):
        args = self.parse_command(event.message_str)
        if len(args) != 1:
            await event.send(event.plain_result("参数错误，翻页操作只需要一个参数（页码）"))
            return
    
        page_num = args[0]
        if not page_num.isdigit() or int(page_num) < 1:
            await event.send(event.plain_result("页码应该是大于0的整数"))
            return
    
        output_config = self.config.get('output', {})
        search_cache_folder = Path(output_config.get('search_cache_folder', 'data/ehentai/searchCache'))
        cache_file = search_cache_folder / f"{event.get_sender_id()}.json"
    
        if not cache_file.exists():
            await event.send(event.plain_result("未找到搜索记录，请先使用'搜eh'命令"))
            return
    
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
    
        if 'params' not in cache_data:
            await event.send(event.plain_result("缓存文件中缺少必要参数信息，请使用'搜eh'命令重新搜索"))
            return
    
        params = cache_data['params']
        
        if 'tags' not in params:
            await event.send(event.plain_result("缓存文件中未找到关键词信息，无法跳转到指定页"))
            return
    
        params['target_page'] = int(page_num)
        event.message_str = f"搜eh {params['tags']} {params['min_rating']} {params['min_pages']} {params['target_page']}"
    
        await self.search_gallery(event)
        
    @filter.command("看eh")
    async def download_gallery(self, event: AstrMessageEvent):
        output_config = self.config.get('output', {})
        image_folder = Path(output_config.get('image_folder', 'data/ehentai/tempImages'))
        image_folder.mkdir(exist_ok=True, parents=True)
        pdf_folder = Path(output_config.get('pdf_folder', 'data/ehentai/pdf'))
        pdf_folder.mkdir(exist_ok=True, parents=True)
        search_cache_folder = Path(output_config.get('search_cache_folder', 'data/ehentai/searchCache'))
        search_cache_folder.mkdir(exist_ok=True, parents=True)

        for f in glob.glob(str(image_folder / "*.*")):
            os.remove(f)

        try:
            args = self.parse_command(event.message_str)
            if len(args) != 1:
                await self.eh_helper(event)
                return

            url = await self._resolve_url_from_input(event, args[0])
            if not url:
                return

            async with await self._get_session() as session:
                is_pdf_exist = await self.downloader.process_pagination(event, session, url)

                if not is_pdf_exist:
                    title = self.downloader.gallery_title
                    safe_title = await self.downloader.merge_images_to_pdf(event, title)
                    output_config = self.config.get('output', {})
                    pdf_folder = output_config.get('pdf_folder', 'data/ehentai/pdf')
                    await self.uploader.upload_file(event, pdf_folder, safe_title)

        except Exception as e:
            logger.exception("下载失败")
            await event.send(event.plain_result(f"下载失败：{str(e)}"))

    @filter.command("归档eh")
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
        help_text = """eh指令帮助：
[1] 搜索画廊: 搜eh [关键词] [最低评分（2-5，默认2）] [最少页数（默认1）] [获取第几页的画廊列表（默认1）]
[2] 快速翻页: eh翻页 [获取第几页的画廊列表]
[3] 下载画廊: 看eh [画廊链接/搜索结果序号]
[4] 获取归档链接: 归档eh [画廊链接/搜索结果序号]
[5] 获取指令帮助: eh
[6] 热重载config相关参数: 重载eh配置

可用的搜索方式:
[1] 搜eh [关键词]
[2] 搜eh [关键词] [最低评分]
[3] 搜eh [关键词] [最低评分] [最少页数]
[4] 搜eh [关键词] [最低评分] [最少页数] [获取第几页的画廊列表]
[5] eh翻页 [获取第几页的画廊列表]

可用的下载方式：
[1] 看eh [画廊链接]
[2] 看eh [搜索结果序号]

可用的归档方式：
[1] 归档eh [画廊链接]
[2] 归档eh [搜索结果序号]

注意：
[1] 搜索多关键词时请用以下符号连接`,` `，` `+`，关键词之间不要添加任何空格
[2] 使用"eh翻页 [获取第几页的画廊列表]"、"看eh [搜索结果序号]"和"归档eh [搜索结果序号]"前确保你最近至少使用过一次"搜eh"命令（每个用户的缓存文件是独立的）
[3] 归档链接仅能访问一次，请尽快下载"""
        await event.send(event.plain_result(help_text))

    @filter.command("重载eh配置")
    async def reload_config(self, event: AstrMessageEvent):
        await event.send(event.plain_result("正在重载配置参数"))
        # 配置由框架管理，无需手动重载
        self.uploader = MessageAdapter(self.config)
        self.downloader = Downloader(self.config, self.uploader, self.parser)
        await event.send(event.plain_result("已重载配置参数"))
    
    @filter.regex(r"^(?:\[([^\]]+)\]|\(([^\)]+)\))\s*(.*)$")
    async def search_by_formatted_message(self, event: AstrMessageEvent):
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
