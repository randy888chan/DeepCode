"""
工具函数
"""
import os
import aiofiles
from typing import Optional, Dict, Any
from pathlib import Path
import chromadb.utils.embedding_functions as embedding_functions

def detect_language(file_path: str) -> Optional[str]:
    """检测文件的编程语言"""
    from .analyzer import CodeAnalyzer
    
    ext = os.path.splitext(file_path)[1].lower()
    return CodeAnalyzer.SUPPORTED_EXTENSIONS.get(ext)

async def read_file_safe(file_path: str, encoding: str = 'utf-8') -> Optional[str]:
    """安全地读取文件内容"""
    try:
        async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
            return await f.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
                return await f.read()
        except:
            return None
    except Exception:
        return None

def get_embedding_function():
    """获取嵌入函数"""
    # 优先使用OpenAI，如果没有则使用默认的
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
    else:
        # 使用默认的句子转换器
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

def format_file_tree(structure: Dict[str, Any], 
                    indent: str = "", 
                    is_last: bool = True,
                    current_depth: int = 0,
                    max_depth: int = 3) -> str:
    """格式化文件树结构"""
    if current_depth >= max_depth:
        return ""
    
    output = ""
    
    # 获取文件和目录
    files = structure.get('files', [])
    directories = structure.get('directories', {})
    
    # 排序
    files.sort()
    dir_items = sorted(directories.items())
    
    # 显示文件
    for i, file in enumerate(files):
        is_last_item = (i == len(files) - 1) and not directories
        prefix = "└── " if is_last_item else "├── "
        output += f"{indent}{prefix}📄 {file}\n"
    
    # 显示目录
    for i, (dir_name, dir_content) in enumerate(dir_items):
        is_last_item = (i == len(dir_items) - 1)
        prefix = "└── " if is_last_item else "├── "
        output += f"{indent}{prefix}📁 {dir_name}/\n"
        
        # 递归显示子目录
        new_indent = indent + ("    " if is_last_item else "│   ")
        output += format_file_tree(
            dir_content, 
            new_indent, 
            is_last_item,
            current_depth + 1,
            max_depth
        )
    
    return output

def truncate_content(content: str, max_lines: int = 20) -> str:
    """截断内容到指定行数"""
    lines = content.split('\n')
    if len(lines) <= max_lines:
        return content
    
    truncated = '\n'.join(lines[:max_lines])
    truncated += f"\n... ({len(lines) - max_lines} more lines)"
    return truncated

def calculate_metrics(repo_info: Dict[str, Any]) -> Dict[str, Any]:
    """计算仓库指标"""
    metrics = {
        'total_files': repo_info.get('total_files', 0),
        'total_lines': repo_info.get('total_lines', 0),
        'languages': repo_info.get('languages', {}),
        'primary_language': None,
        'language_diversity': 0
    }
    
    if metrics['languages']:
        # 主要语言
        metrics['primary_language'] = max(
            metrics['languages'].items(), 
            key=lambda x: x[1]
        )[0]
        
        # 语言多样性（使用香农熵）
        total_lines = metrics['total_lines']
        if total_lines > 0:
            import math
            entropy = 0
            for lines in metrics['languages'].values():
                p = lines / total_lines
                if p > 0:
                    entropy -= p * math.log2(p)
            metrics['language_diversity'] = entropy
    
    return metrics