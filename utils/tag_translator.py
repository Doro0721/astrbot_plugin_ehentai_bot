"""
EhTagTranslation 标签翻译模块
从 EhTagTranslation 社区数据库下载标签翻译数据，将英文标签翻译为中文。
"""

import json
import os
import time
import logging
import re
import aiohttp

logger = logging.getLogger(__name__)

# EhTagTranslation 数据库下载地址
DB_URL = "https://raw.githubusercontent.com/EhTagTranslation/DatabaseReleases/master/db.text.json"

# 缓存过期时间（7天）
CACHE_TTL = 7 * 24 * 3600


class TagTranslator:
    """EHentai 标签中文翻译器"""

    def __init__(self, cache_dir: str = "data/ehentai"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "tag_translations.json")
        self._db: dict = {}  # {namespace: {tag: chinese_name}}
        self._loaded = False

    async def ensure_loaded(self, session: aiohttp.ClientSession = None):
        """确保翻译数据已加载（优先从缓存读取，过期则重新下载）"""
        if self._loaded:
            return

        # 尝试从本地缓存加载
        if self._load_from_cache():
            self._loaded = True
            return

        # 缓存不存在或已过期，从网络下载
        await self._download_db(session)
        self._loaded = True

    def _load_from_cache(self) -> bool:
        """从本地缓存文件加载翻译数据"""
        try:
            if not os.path.exists(self.cache_file):
                return False

            # 检查缓存是否过期
            file_mtime = os.path.getmtime(self.cache_file)
            if time.time() - file_mtime > CACHE_TTL:
                logger.info("标签翻译缓存已过期，需要重新下载")
                return False

            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self._db = json.load(f)

            logger.info(f"从缓存加载标签翻译数据，共 {sum(len(v) for v in self._db.values())} 条")
            return True
        except Exception as e:
            logger.warning(f"加载标签翻译缓存失败: {e}")
            return False

    async def _download_db(self, session: aiohttp.ClientSession = None):
        """从 EhTagTranslation 下载翻译数据库"""
        logger.info("正在下载 EhTagTranslation 标签翻译数据库...")

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            async with session.get(DB_URL, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    logger.error(f"下载标签翻译数据库失败，HTTP {resp.status}")
                    return

                raw_data = await resp.json(content_type=None)

            # 解析数据库格式
            self._db = self._parse_db(raw_data)

            # 保存到缓存
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._db, f, ensure_ascii=False)

            logger.info(f"标签翻译数据库下载完成，共 {sum(len(v) for v in self._db.values())} 条翻译")

        except Exception as e:
            logger.error(f"下载标签翻译数据库失败: {e}")
        finally:
            if close_session:
                await session.close()

    @staticmethod
    def _parse_db(raw_data: dict) -> dict:
        """
        解析 EhTagTranslation 的 db.text.json 格式。
        该文件结构为:
        {
            "head": {...},
            "data": [
                {
                    "namespace": "female",
                    "data": {
                        "tag_name": {"name": "中文翻译", "intro": "...", "links": "..."},
                        ...
                    }
                },
                ...
            ]
        }
        """
        result = {}
        data_list = raw_data.get("data", [])

        for ns_entry in data_list:
            namespace = ns_entry.get("namespace", "")
            if not namespace:
                continue

            ns_data = ns_entry.get("data", {})
            translated = {}

            for tag_key, tag_info in ns_data.items():
                if isinstance(tag_info, dict):
                    cn_name = tag_info.get("name", "")
                    if cn_name:
                        # 清理 markdown 格式标记（如 ![img](...) 等）
                        cn_name = re.sub(r'!\[.*?\]\(.*?\)', '', cn_name)
                        cn_name = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cn_name)
                        cn_name = cn_name.strip()
                        if cn_name:
                            translated[tag_key] = cn_name

            if translated:
                result[namespace] = translated

        return result

    def translate(self, namespace: str, tag: str) -> str:
        """
        翻译单个标签。

        Args:
            namespace: 标签命名空间（如 "female", "male", "artist" 等）
            tag: 英文标签名（如 "big breasts"）

        Returns:
            中文翻译，如果未找到则返回原始标签
        """
        if not self._db:
            return tag

        # 标准化查找：将空格替换为空格（EhTag 数据库中用空格）
        tag_key = tag.strip().replace(" ", " ")

        ns_data = self._db.get(namespace, {})
        cn = ns_data.get(tag_key)
        if cn:
            return cn

        # 尝试不区分大小写匹配
        tag_lower = tag_key.lower()
        for k, v in ns_data.items():
            if k.lower() == tag_lower:
                return v

        return tag

    def translate_category(self, category: str) -> str:
        """
        翻译画廊分类名称（如 "Doujinshi" → "同人志"）

        Args:
            category: 英文分类名

        Returns:
            中文分类名
        """
        # 先从 reclass 命名空间查找
        reclass_data = self._db.get("reclass", {})
        cat_lower = category.strip().lower()

        for k, v in reclass_data.items():
            if k.lower() == cat_lower:
                return v

        return category
