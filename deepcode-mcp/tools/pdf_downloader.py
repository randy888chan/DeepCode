#!/usr/bin/env python3
"""
Smart File Downloader MCP Tool using FastMCP
能够理解自然语言指令，识别URL和目标路径，并执行下载的MCP工具
"""

import os
import re
import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from urllib.parse import urlparse, unquote
from datetime import datetime
import mimetypes
import sys
import io

from mcp.server import FastMCP

# Docling imports for document conversion
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: docling package not available. Document conversion will be disabled.")

# Fallback PDF text extraction
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Warning: PyPDF2 package not available. Fallback PDF extraction will be disabled.")

# 设置标准输出编码为UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        else:
            sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
    except Exception as e:
        print(f"Warning: Could not set UTF-8 encoding: {e}")

# 创建 FastMCP 实例
mcp = FastMCP("smart-file-downloader")

class URLExtractor:
    """URL提取器"""
    
    # URL正则模式
    URL_PATTERNS = [
        # 标准HTTP/HTTPS URL
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/(?:[-\w._~!$&\'()*+,;=:@]|%[\da-fA-F]{2})*)*(?:\?(?:[-\w._~!$&\'()*+,;=:@/?]|%[\da-fA-F]{2})*)?(?:#(?:[-\w._~!$&\'()*+,;=:@/?]|%[\da-fA-F]{2})*)?',
        # FTP URL
        r'ftp://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/(?:[-\w._~!$&\'()*+,;=:@]|%[\da-fA-F]{2})*)*',
        # 文件路径形式的URL（如 www.example.com/file.pdf）
        r'(?<!\S)(?:www\.)?[-\w]+(?:\.[-\w]+)+/(?:[-\w._~!$&\'()*+,;=:@/]|%[\da-fA-F]{2})+',
    ]
    
    @classmethod
    def extract_urls(cls, text: str) -> List[str]:
        """从文本中提取URL"""
        urls = []
        
        # 首先处理特殊情况：@开头的URL
        at_url_pattern = r'@(https?://[^\s]+)'
        at_matches = re.findall(at_url_pattern, text, re.IGNORECASE)
        for match in at_matches:
            urls.append(match.rstrip('/'))
        
        # 然后使用原有的正则模式
        for pattern in cls.URL_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # 处理可能缺少协议的URL
                if not match.startswith(('http://', 'https://', 'ftp://')):
                    # 检查是否是 www 开头
                    if match.startswith('www.'):
                        match = 'https://' + match
                    else:
                        # 其他情况也添加 https
                        match = 'https://' + match
                
                # 清理URL
                url = match.rstrip('/')
                urls.append(url)
        
        # 去重并保持顺序
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return unique_urls
    
    @staticmethod
    def infer_filename_from_url(url: str) -> str:
        """从URL推断文件名"""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        
        # 从路径中提取文件名
        filename = os.path.basename(path)
        
        # 特殊处理：arxiv PDF链接
        if 'arxiv.org' in parsed.netloc and '/pdf/' in path:
            if filename:
                # 检查是否已经有合适的文件扩展名
                if not filename.lower().endswith(('.pdf', '.doc', '.docx', '.txt')):
                    filename = f"{filename}.pdf"
            else:
                path_parts = [p for p in path.split('/') if p]
                if path_parts and path_parts[-1]:
                    filename = f"{path_parts[-1]}.pdf"
                else:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"arxiv_paper_{timestamp}.pdf"
        
        # 如果没有文件名或没有扩展名，生成一个
        elif not filename or '.' not in filename:
            # 尝试从URL生成有意义的文件名
            domain = parsed.netloc.replace('www.', '').replace('.', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 尝试根据路径推断文件类型
            if not path or path == '/':
                filename = f"{domain}_{timestamp}.html"
            else:
                # 使用路径的最后一部分
                path_parts = [p for p in path.split('/') if p]
                if path_parts:
                    filename = f"{path_parts[-1]}_{timestamp}"
                else:
                    filename = f"{domain}_{timestamp}"
                
                # 如果还是没有扩展名，根据路径推断
                if '.' not in filename:
                    # 根据路径中的关键词推断文件类型
                    if '/pdf/' in path.lower() or path.lower().endswith('pdf'):
                        filename += '.pdf'
                    elif any(ext in path.lower() for ext in ['/doc/', '/word/', '.docx']):
                        filename += '.docx'
                    elif any(ext in path.lower() for ext in ['/ppt/', '/powerpoint/', '.pptx']):
                        filename += '.pptx'
                    elif any(ext in path.lower() for ext in ['/csv/', '.csv']):
                        filename += '.csv'
                    elif any(ext in path.lower() for ext in ['/zip/', '.zip']):
                        filename += '.zip'
                    else:
                        filename += '.html'
        
        return filename

class PathExtractor:
    """路径提取器"""
    
    @staticmethod
    def extract_target_path(text: str) -> Optional[str]:
        """从文本中提取目标路径"""
        # 路径指示词模式
        patterns = [
            # 英文指示词
            r'(?:save|download|store|put|place|write|copy)\s+(?:to|into|in|at)\s+["\']?([^\s"\']+)["\']?',
            r'(?:to|into|in|at)\s+(?:folder|directory|dir|path|location)\s*["\']?([^\s"\']+)["\']?',
            r'(?:destination|target|output)\s*(?:is|:)?\s*["\']?([^\s"\']+)["\']?',
            # 中文指示词
            r'(?:保存|下载|存储|放到|写入|复制)(?:到|至|去)\s*["\']?([^\s"\']+)["\']?',
            r'(?:到|在|至)\s*["\']?([^\s"\']+)["\']?\s*(?:文件夹|目录|路径|位置)',
        ]
        
        # 需要过滤的通用词
        filter_words = {
            'here', 'there', 'current', 'local', 'this', 'that',
            '这里', '那里', '当前', '本地', '这个', '那个'
        }
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                path = match.group(1).strip('。，,.、')
                
                # 过滤通用词
                if path and path.lower() not in filter_words:
                    return path
        
        return None

class SimplePdfConverter:
    """简单的PDF转换器，使用PyPDF2提取文本"""
    
    def convert_pdf_to_markdown(self, input_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        使用PyPDF2将PDF转换为Markdown格式
        
        Args:
            input_file: 输入PDF文件路径
            output_file: 输出Markdown文件路径（可选）
            
        Returns:
            转换结果字典
        """
        if not PYPDF2_AVAILABLE:
            return {
                "success": False,
                "error": "PyPDF2 package is not available"
            }
        
        try:
            # 检查输入文件是否存在
            if not os.path.exists(input_file):
                return {
                    "success": False,
                    "error": f"Input file not found: {input_file}"
                }
            
            # 如果没有指定输出文件，自动生成
            if not output_file:
                base_name = os.path.splitext(input_file)[0]
                output_file = f"{base_name}.md"
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 执行转换
            start_time = datetime.now()
            
            # 读取PDF文件
            with open(input_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                
                # 提取每页文本
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(f"## Page {page_num}\n\n{text.strip()}\n\n")
            
            # 生成Markdown内容
            markdown_content = f"# Extracted from {os.path.basename(input_file)}\n\n"
            markdown_content += f"*Total pages: {len(pdf_reader.pages)}*\n\n"
            markdown_content += "---\n\n"
            markdown_content += "".join(text_content)
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            # 计算转换时间
            duration = (datetime.now() - start_time).total_seconds()
            
            # 获取文件大小
            input_size = os.path.getsize(input_file)
            output_size = os.path.getsize(output_file)
            
            return {
                "success": True,
                "input_file": input_file,
                "output_file": output_file,
                "input_size": input_size,
                "output_size": output_size,
                "duration": duration,
                "markdown_content": markdown_content,
                "pages_extracted": len(pdf_reader.pages)
            }
            
        except Exception as e:
            return {
                "success": False,
                "input_file": input_file,
                "error": f"Conversion failed: {str(e)}"
            }

class DoclingConverter:
    """文档转换器，使用docling将文档转换为Markdown格式"""
    
    def __init__(self):
        if not DOCLING_AVAILABLE:
            raise ImportError("docling package is not available. Please install it first.")
        
        # 配置PDF处理选项
        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = False  # 暂时禁用OCR以避免认证问题
        pdf_pipeline_options.do_table_structure = False  # 暂时禁用表格结构识别
        
        # 创建文档转换器（使用基础模式）
        try:
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
                }
            )
        except Exception as e:
            # 如果失败，尝试更简单的配置
            self.converter = DocumentConverter()
    
    def is_supported_format(self, file_path: str) -> bool:
        """检查文件格式是否支持转换"""
        if not DOCLING_AVAILABLE:
            return False
            
        supported_extensions = {'.pdf', '.docx', '.pptx', '.html', '.md', '.txt'}
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in supported_extensions
    
    def convert_to_markdown(self, input_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        将文档转换为Markdown格式
        
        Args:
            input_file: 输入文件路径
            output_file: 输出Markdown文件路径（可选）
            
        Returns:
            转换结果字典
        """
        if not DOCLING_AVAILABLE:
            return {
                "success": False,
                "error": "docling package is not available"
            }
        
        try:
            # 检查输入文件是否存在
            if not os.path.exists(input_file):
                return {
                    "success": False,
                    "error": f"Input file not found: {input_file}"
                }
            
            # 检查文件格式是否支持
            if not self.is_supported_format(input_file):
                return {
                    "success": False,
                    "error": f"Unsupported file format: {os.path.splitext(input_file)[1]}"
                }
            
            # 如果没有指定输出文件，自动生成
            if not output_file:
                base_name = os.path.splitext(input_file)[0]
                output_file = f"{base_name}.md"
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 执行转换
            start_time = datetime.now()
            result = self.converter.convert(input_file)
            
            # 获取Markdown内容
            markdown_content = result.document.export_to_markdown()
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            # 计算转换时间
            duration = (datetime.now() - start_time).total_seconds()
            
            # 获取文件大小
            input_size = os.path.getsize(input_file)
            output_size = os.path.getsize(output_file)
            
            return {
                "success": True,
                "input_file": input_file,
                "output_file": output_file,
                "input_size": input_size,
                "output_size": output_size,
                "duration": duration,
                "markdown_content": markdown_content
            }
            
        except Exception as e:
            return {
                "success": False,
                "input_file": input_file,
                "error": f"Conversion failed: {str(e)}"
            }

async def check_url_accessible(url: str) -> Dict[str, Any]:
    """检查URL是否可访问"""
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.head(url, allow_redirects=True) as response:
                return {
                    "accessible": response.status < 400,
                    "status": response.status,
                    "content_type": response.headers.get('Content-Type', ''),
                    "content_length": response.headers.get('Content-Length', 0)
                }
    except:
        return {
            "accessible": False,
            "status": 0,
            "content_type": "",
            "content_length": 0
        }

async def download_file(url: str, destination: str) -> Dict[str, Any]:
    """下载单个文件"""
    start_time = datetime.now()
    chunk_size = 8192
    
    try:
        timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                # 检查响应状态
                response.raise_for_status()
                
                # 获取文件信息
                total_size = int(response.headers.get('Content-Length', 0))
                content_type = response.headers.get('Content-Type', 'application/octet-stream')
                
                # 确保目标目录存在
                parent_dir = os.path.dirname(destination)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                
                # 下载文件
                downloaded = 0
                async with aiofiles.open(destination, 'wb') as file:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        await file.write(chunk)
                        downloaded += len(chunk)
                
                # 计算下载时间
                duration = (datetime.now() - start_time).total_seconds()
                
                return {
                    "success": True,
                    "url": url,
                    "destination": destination,
                    "size": downloaded,
                    "content_type": content_type,
                    "duration": duration,
                    "speed": downloaded / duration if duration > 0 else 0
                }
                
    except aiohttp.ClientError as e:
        return {
            "success": False,
            "url": url,
            "destination": destination,
            "error": f"Network error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "destination": destination,
            "error": f"Download error: {str(e)}"
        }

@mcp.tool()
async def download_files(instruction: str) -> str:
    """
    Download files from URLs mentioned in natural language instructions.
    
    Args:
        instruction: Natural language instruction containing URLs and optional destination paths
        
    Returns:
        Status message about the download operations
        
    Examples:
        - "Download https://example.com/file.pdf to documents folder"
        - "Please get https://raw.githubusercontent.com/user/repo/main/data.csv and save it to ~/downloads"
        - "下载 https://example.com/image.jpg 到 /tmp/images/"
        - "Download www.example.com/report.xlsx"
    """
    # 提取URLs
    urls = URLExtractor.extract_urls(instruction)
    if not urls:
        return "[ERROR] No downloadable URLs found in the instruction"
    
    # 提取目标路径
    target_path = PathExtractor.extract_target_path(instruction)
    
    # 下载文件
    results = []
    for url in urls:
        try:
            # 推断文件名
            filename = URLExtractor.infer_filename_from_url(url)
            
            # 构建完整的目标路径
            if target_path:
                # 处理路径
                if target_path.startswith('~'):
                    target_path = os.path.expanduser(target_path)
                
                # 确保使用相对路径（如果不是绝对路径）
                if not os.path.isabs(target_path):
                    target_path = os.path.normpath(target_path)
                
                # 判断是文件路径还是目录路径
                if os.path.splitext(target_path)[1]:  # 有扩展名，是文件
                    destination = target_path
                else:  # 是目录
                    destination = os.path.join(target_path, filename)
            else:
                # 默认下载到当前目录
                destination = filename
            
            # 检查文件是否已存在
            if os.path.exists(destination):
                results.append(f"[WARNING] Skipped {url}: File already exists at {destination}")
                continue
            
            # 先检查URL是否可访问
            check_result = await check_url_accessible(url)
            if not check_result["accessible"]:
                results.append(f"[ERROR] Failed to access {url}: HTTP {check_result['status'] or 'Connection failed'}")
                continue
            
            # 执行下载
            result = await download_file(url, destination)
            
            if result["success"]:
                size_mb = result["size"] / (1024 * 1024)
                speed_mb = result["speed"] / (1024 * 1024)
                msg = f"[SUCCESS] Successfully downloaded: {url}\n"
                msg += f"   File: {destination}\n"
                msg += f"   Size: {size_mb:.2f} MB\n"
                msg += f"   Time: {result['duration']:.2f} seconds\n"
                msg += f"   Speed: {speed_mb:.2f} MB/s"
                
                # 尝试转换为Markdown
                conversion_success = False
                
                # 首先尝试使用简单的PDF转换器（对于PDF文件）
                if destination.lower().endswith('.pdf') and PYPDF2_AVAILABLE:
                    try:
                        simple_converter = SimplePdfConverter()
                        conversion_result = simple_converter.convert_pdf_to_markdown(destination)
                        if conversion_result["success"]:
                            msg += f"\n   [INFO] PDF converted to Markdown (PyPDF2)"
                            msg += f"\n   Markdown file: {conversion_result['output_file']}"
                            msg += f"\n   Conversion time: {conversion_result['duration']:.2f} seconds"
                            msg += f"\n   Pages extracted: {conversion_result['pages_extracted']}"
                            conversion_success = True
                        else:
                            msg += f"\n   [WARNING] PDF conversion failed: {conversion_result['error']}"
                    except Exception as conv_error:
                        msg += f"\n   [WARNING] PDF conversion error: {str(conv_error)}"
                
                # 如果简单转换失败，尝试使用docling
                if not conversion_success and DOCLING_AVAILABLE:
                    try:
                        converter = DoclingConverter()
                        if converter.is_supported_format(destination):
                            conversion_result = converter.convert_to_markdown(destination)
                            if conversion_result["success"]:
                                msg += f"\n   [INFO] Document converted to Markdown (docling)"
                                msg += f"\n   Markdown file: {conversion_result['output_file']}"
                                msg += f"\n   Conversion time: {conversion_result['duration']:.2f} seconds"
                            else:
                                msg += f"\n   [WARNING] Docling conversion failed: {conversion_result['error']}"
                    except Exception as conv_error:
                        msg += f"\n   [WARNING] Docling conversion error: {str(conv_error)}"
            else:
                msg = f"[ERROR] Failed to download: {url}\n"
                msg += f"   Error: {result.get('error', 'Unknown error')}"
            
        except Exception as e:
            msg = f"[ERROR] Failed to download: {url}\n"
            msg += f"   Error: {str(e)}"
        
        results.append(msg)
    
    return "\n\n".join(results)

@mcp.tool()
async def parse_download_urls(text: str) -> str:
    """
    Extract URLs and target paths from text without downloading.
    
    Args:
        text: Text containing URLs and optional download paths
        
    Returns:
        Parsed URLs and target path information
    """
    # 提取URLs
    urls = URLExtractor.extract_urls(text)
    
    # 提取路径
    target_path = PathExtractor.extract_target_path(text)
    
    content = "[INFO] Parsed download information:\n\n"
    
    if urls:
        content += f"URLs found ({len(urls)}):\n"
        for i, url in enumerate(urls, 1):
            filename = URLExtractor.infer_filename_from_url(url)
            content += f"  {i}. {url}\n"
            content += f"     -> Filename: {filename}\n"
    else:
        content += "No URLs found\n"
    
    if target_path:
        content += f"\nTarget path: {target_path}"
        if target_path.startswith('~'):
            content += f"\n  (Expanded: {os.path.expanduser(target_path)})"
    else:
        content += "\nTarget path: Not specified (will use current directory)"
    
    return content

@mcp.tool()
async def download_file_to(url: str, destination: Optional[str] = None, filename: Optional[str] = None) -> str:
    """
    Download a specific file with detailed options.
    
    Args:
        url: URL to download from
        destination: Target directory or full file path (optional)
        filename: Specific filename to use (optional, ignored if destination is a full file path)
        
    Returns:
        Status message about the download operation
    """
    # 确定文件名
    if not filename:
        filename = URLExtractor.infer_filename_from_url(url)
    
    # 确定完整路径
    if destination:
        # 展开用户目录
        if destination.startswith('~'):
            destination = os.path.expanduser(destination)
        
        # 检查是否是完整文件路径
        if os.path.splitext(destination)[1]:  # 有扩展名
            target_path = destination
        else:  # 是目录
            target_path = os.path.join(destination, filename)
    else:
        target_path = filename
    
    # 确保使用相对路径（如果不是绝对路径）
    if not os.path.isabs(target_path):
        target_path = os.path.normpath(target_path)
    
    # 检查文件是否已存在
    if os.path.exists(target_path):
        return f"[ERROR] Error: File already exists at {target_path}"
    
    # 先检查URL
    check_result = await check_url_accessible(url)
    if not check_result["accessible"]:
        return f"[ERROR] Error: Cannot access URL {url} (HTTP {check_result['status'] or 'Connection failed'})"
    
    # 显示下载信息
    size_mb = int(check_result["content_length"]) / (1024 * 1024) if check_result["content_length"] else 0
    msg = f"[INFO] Downloading file:\n"
    msg += f"   URL: {url}\n"
    msg += f"   Target: {target_path}\n"
    if size_mb > 0:
        msg += f"   Expected size: {size_mb:.2f} MB\n"
    msg += "\n"
    
    # 执行下载
    result = await download_file(url, target_path)
    
    if result["success"]:
        actual_size_mb = result["size"] / (1024 * 1024)
        speed_mb = result["speed"] / (1024 * 1024)
        msg += f"[SUCCESS] Download completed!\n"
        msg += f"   Saved to: {target_path}\n"
        msg += f"   Size: {actual_size_mb:.2f} MB\n"
        msg += f"   Duration: {result['duration']:.2f} seconds\n"
        msg += f"   Speed: {speed_mb:.2f} MB/s\n"
        msg += f"   Type: {result['content_type']}"
        
        # 尝试转换为Markdown
        conversion_success = False
        
        # 首先尝试使用简单的PDF转换器（对于PDF文件）
        if target_path.lower().endswith('.pdf') and PYPDF2_AVAILABLE:
            try:
                simple_converter = SimplePdfConverter()
                conversion_result = simple_converter.convert_pdf_to_markdown(target_path)
                if conversion_result["success"]:
                    msg += f"\n\n[INFO] PDF converted to Markdown (PyPDF2)"
                    msg += f"\n   Markdown file: {conversion_result['output_file']}"
                    msg += f"\n   Conversion time: {conversion_result['duration']:.2f} seconds"
                    msg += f"\n   Original size: {conversion_result['input_size'] / 1024:.1f} KB"
                    msg += f"\n   Markdown size: {conversion_result['output_size'] / 1024:.1f} KB"
                    msg += f"\n   Pages extracted: {conversion_result['pages_extracted']}"
                    conversion_success = True
                else:
                    msg += f"\n\n[WARNING] PDF conversion failed: {conversion_result['error']}"
            except Exception as conv_error:
                msg += f"\n\n[WARNING] PDF conversion error: {str(conv_error)}"
        
        # 如果简单转换失败，尝试使用docling
        if not conversion_success and DOCLING_AVAILABLE:
            try:
                converter = DoclingConverter()
                if converter.is_supported_format(target_path):
                    conversion_result = converter.convert_to_markdown(target_path)
                    if conversion_result["success"]:
                        msg += f"\n\n[INFO] Document converted to Markdown (docling)"
                        msg += f"\n   Markdown file: {conversion_result['output_file']}"
                        msg += f"\n   Conversion time: {conversion_result['duration']:.2f} seconds"
                        msg += f"\n   Original size: {conversion_result['input_size'] / 1024:.1f} KB"
                        msg += f"\n   Markdown size: {conversion_result['output_size'] / 1024:.1f} KB"
                    else:
                        msg += f"\n\n[WARNING] Docling conversion failed: {conversion_result['error']}"
            except Exception as conv_error:
                msg += f"\n\n[WARNING] Docling conversion error: {str(conv_error)}"
    else:
        msg += f"[ERROR] Download failed!\n"
        msg += f"   Error: {result['error']}"
    
    return msg

@mcp.tool()
async def convert_document_to_markdown(file_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert a document to Markdown format using docling.
    
    Args:
        file_path: Path to the input document file
        output_path: Path for the output Markdown file (optional)
        
    Returns:
        Status message about the conversion operation
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return f"[ERROR] Input file not found: {file_path}"
    
    # 检查是否是PDF文件，优先使用简单转换器
    if file_path.lower().endswith('.pdf') and PYPDF2_AVAILABLE:
        try:
            simple_converter = SimplePdfConverter()
            result = simple_converter.convert_pdf_to_markdown(file_path, output_path)
        except Exception as e:
            return f"[ERROR] PDF conversion error: {str(e)}"
    elif DOCLING_AVAILABLE:
        try:
            converter = DoclingConverter()
            
            # 检查文件格式是否支持
            if not converter.is_supported_format(file_path):
                supported_formats = ['.pdf', '.docx', '.pptx', '.html', '.md', '.txt']
                return f"[ERROR] Unsupported file format. Supported formats: {', '.join(supported_formats)}"
            
            # 执行转换
            result = converter.convert_to_markdown(file_path, output_path)
        except Exception as e:
            return f"[ERROR] Docling conversion error: {str(e)}"
    else:
        return "[ERROR] No conversion tools available. Please install docling or PyPDF2."
    
    if result["success"]:
        msg = f"[SUCCESS] Document converted successfully!\n"
        msg += f"   Input file: {result['input_file']}\n"
        msg += f"   Output file: {result['output_file']}\n"
        msg += f"   Conversion time: {result['duration']:.2f} seconds\n"
        msg += f"   Original size: {result['input_size'] / 1024:.1f} KB\n"
        msg += f"   Markdown size: {result['output_size'] / 1024:.1f} KB\n"
        
        # 显示Markdown内容的前几行作为预览
        content_lines = result['markdown_content'].split('\n')
        preview_lines = content_lines[:5]
        if len(content_lines) > 5:
            preview_lines.append('...')
        
        msg += f"\n[PREVIEW] First few lines of converted Markdown:\n"
        for line in preview_lines:
            msg += f"   {line}\n"
    else:
        msg = f"[ERROR] Conversion failed!\n"
        msg += f"   Error: {result['error']}"
    
    return msg

@mcp.tool()
async def download_and_convert(instruction: str, auto_convert: bool = True) -> str:
    """
    Download files and optionally convert them to Markdown format.
    
    Args:
        instruction: Natural language instruction containing URLs and optional destination paths
        auto_convert: Whether to automatically convert supported documents to Markdown
        
    Returns:
        Status message about the download and conversion operations
    """
    # 首先执行下载
    download_result = await download_files(instruction)
    
    # 如果禁用自动转换或docling不可用，直接返回下载结果
    if not auto_convert or not DOCLING_AVAILABLE:
        return download_result
    
    # 如果启用自动转换，下载函数已经自动处理了转换
    # 这里只是提供一个明确的接口
    return download_result

# 主程序入口
if __name__ == "__main__":
    print("Smart File Downloader MCP Tool")
    print("Natural language file downloading with intelligent parsing")
    
    if DOCLING_AVAILABLE:
        print("Document conversion to Markdown is ENABLED (docling available)")
    else:
        print("Document conversion to Markdown is DISABLED (docling not available)")
        print("Install docling to enable: pip install docling")
    
    print("\nExamples:")
    print('  • "Download https://example.com/file.pdf to documents"')
    print('  • "Get https://raw.githubusercontent.com/user/repo/main/data.csv and save to ~/downloads"')
    print('  • "下载 https://example.com/image.jpg 到 /tmp/images/"')
    print('  • "Please download www.example.com/data.csv"')
    print("\nAvailable tools:")
    print("  • download_files - Download files from natural language instructions (auto-converts to MD)")
    print("  • download_and_convert - Download files with explicit conversion control")
    print("  • convert_document_to_markdown - Convert existing documents to Markdown")
    print("  • parse_download_urls - Extract URLs and paths without downloading")
    print("  • download_file_to - Download a specific file with options")
    
    if DOCLING_AVAILABLE:
        print("\nSupported formats for Markdown conversion:")
        print("  • PDF (.pdf)")
        print("  • Word documents (.docx)")
        print("  • PowerPoint (.pptx)")
        print("  • HTML (.html)")
        print("  • Text files (.txt, .md)")
    
    print("")
    
    # 运行服务器
    mcp.run()