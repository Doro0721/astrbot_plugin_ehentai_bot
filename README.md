# 适配 AstrBot 的 EHentai画廊 转 PDF 插件

## 安装方法

1. **通过 插件市场 安装**  
- 打开 "AstrBot WebUI" -> "插件市场" -> "右上角 Search"  
- 搜索任何与本项目相关的关键词，找到插件后点击安装
- 推荐通过唯一标识符搜索：```astrbot_plugin_ehentai_bot```

2. **通过 Github仓库链接 安装**  
- 打开 "AstrBot WebUI" -> "插件市场" -> "右下角 '+' 按钮"  
- 输入以下地址并点击安装：
```
https://github.com/drdon1234/astrbot_plugin_ehentai_bot
```

---

## 使用说明

### 指令帮助

- **搜索画廊**  
`搜eh [关键词] [最低评分（2-5，默认2）] [最少页数（默认1）] [获取第几页的画廊列表（默认1）]`

- **快速翻页**   
`eh翻页 [获取第几页的画廊列表]`

- **下载画廊**  
`看eh [画廊链接/搜索结果序号]`

- **获取指令帮助**  
`eh`

- **热重载config配置**  
`重载eh配置`

### 可用的搜索方式

1. 基础搜索：  
`搜eh [关键词]`

2. 高级搜索：  
`搜eh [关键词] [最低评分]`
 
    `搜eh [关键词] [最低评分] [最少页数]`
   
    `搜eh [关键词] [最低评分] [最少页数] [获取第几页的画廊列表]`

3. 快速翻页：  
`eh翻页 [获取第几页的画廊列表]`

### 可用的下载方式

1. 通过画廊链接下载：  
`看eh [画廊链接]`

2. 通过画廊索引下载：  
`看eh [搜索结果序号]`

**注意：**  
- 搜索多关键词时请用以下符号连接 `,` `，` `+` ，关键词之间不要添加任何空格
- 使用 `eh翻页 [获取第几页的画廊列表]` 和 `看eh [搜索结果序号]` 前确保你最近至少使用过一次 `搜eh` 命令（每个用户的缓存文件是独立的）

---

## 配置文件修改（重要！）

使用前请先修改配置文件：

**通过 WebUI 的插件管理面板设置**
- 打开 "AstrBot WebUI" -> "插件管理" -> 找到本插件 -> "操作" -> "插件配置":
![image](https://github.com/user-attachments/assets/3f6487f6-27c6-4624-8cb7-9a8179538298)

所有配置项均通过WebUI的插件管理面板进行设置。配置项说明如下：

- **平台设置**
  - `platform_type`: 消息平台类型，兼容 napcat, llonebot, lagrange，默认值：napcat
  - `platform_http_host`: HTTP 服务器 IP，非 docker 部署一般为 127.0.0.1，docker 部署一般为宿主机局域网 ip，默认值：127.0.0.1
  - `platform_http_port`: HTTP 服务器端口，默认值：2333
  - `platform_api_token`: HTTP 服务器 Token，没有则不填

- **请求设置**
  - `request_headers_user_agent`: HTTP请求使用的User-Agent，默认值：Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36
  - `request_website`: 网站类型，表站: e-hentai | 里站: exhentai，缺少有效cookie时请不要将website设置为exhentai，默认值：e-hentai
  - `request_cookies_ipb_member_id`: Cookie中的ipb_member_id值
  - `request_cookies_ipb_pass_hash`: Cookie中的ipb_pass_hash值
  - `request_cookies_igneous`: Cookie中的igneous值
  - `request_cookies_sk`: Cookie中的sk值
  - `request_proxies`: 代理设置（墙内用户必填项），代理软件位于宿主机时，非docker部署一般为http://127.0.0.1:port，docker部署一般为http://{宿主机局域网ip}:port
  - `request_concurrency`: 并发数量限制，默认值：10
  - `request_max_retries`: 请求重试次数，如果你的代理不稳定或带宽不够建议适量增加次数，默认值：3
  - `request_timeout`: 请求超时时间（秒），默认值：5

- **输出设置**
  - `output_image_folder`: 缓存画廊图片的路径，默认值：/app/sharedFolder/ehentai/tempImages
  - `output_pdf_folder`: 存放PDF文件的路径，默认值：/app/sharedFolder/ehentai/pdf
  - `output_search_cache_folder`: 缓存搜索结果的路径，默认值：/app/sharedFolder/ehentai/searchCache
  - `output_jpeg_quality`: 图片质量，100为不压缩，85左右可以达到文件大小和图片质量的最佳平衡，默认值：85
  - `output_max_pages_per_pdf`: 单个PDF文件最大页数，超过此页数将分割为多个PDF文件，默认值：200
  - `output_max_filename_length`: 文件名最大长度限制，超出部分将被截取，默认值：200

- **功能特性**
  - `features_enable_formatted_message_search`: 是否启用格式化消息搜索功能，默认值：true
  - `features_enable_cover_image_download`: 是否下载拼接封面图片，默认值：true

---

## 依赖库安装（重要！）

使用前请先安装以下依赖库：
- aiofiles
- aiohttp
- natsort
- glob2
- python-magic
- beautifulsoup4
- img2pdf
- Pillow

在您的终端输入以下命令并回车：
```
pip install <module>
```
*使用具体模块名替换 &lt;module&gt;*

---

## Docker 部署注意事项

如果您是 Docker 部署，请务必为消息平台容器挂载 PDF 文件所在的文件夹，否则消息平台将无法解析文件路径

示例挂载方式(NapCat)：
- 对 AstrBot：`/vol3/1000/dockerSharedFolder -> /app/sharedFolder`
- 对 NapCat：`/vol3/1000/dockerSharedFolder -> /app/sharedFolder`

---

## 已知 BUG

---

## 开发中的功能

- 随机画廊

---

## 使用示例
- 搜索  

![搜索示例](https://github.com/user-attachments/assets/68f7c828-5891-4b2e-abc3-f17e3b57eb37)

- 下载  

![下载示例](https://github.com/user-attachments/assets/f5f6085a-078c-4235-9bff-51e635bba3d6)

---

## 鸣谢

用户指令清洗和消息适配器参考了[exneverbur](https://github.com/exneverbur)的[ShowMeJM](https://github.com/exneverbur/ShowMeJM)项目，感谢

---
