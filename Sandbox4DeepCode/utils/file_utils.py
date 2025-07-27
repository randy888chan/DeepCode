#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件操作工具
功能：处理文件和目录操作
作者：AI Assistant
创建时间：2024-01-01
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional


def setup_directories(directories: List[str]) -> None:
    """
    创建必要的目录结构
    
    Args:
        directories: 目录路径列表
    """
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logging.debug(f"创建目录: {directory}")
        except Exception as e:
            logging.error(f"创建目录失败 {directory}: {str(e)}")


def copy_directory(src: str, dst: str, exclude_patterns: Optional[List[str]] = None) -> bool:
    """
    复制目录
    
    Args:
        src: 源目录路径
        dst: 目标目录路径
        exclude_patterns: 排除的文件模式列表
        
    Returns:
        是否复制成功
    """
    try:
        if exclude_patterns is None:
            exclude_patterns = ['.git', '__pycache__', '.pytest_cache', '.DS_Store']
        
        def should_exclude(path: str) -> bool:
            for pattern in exclude_patterns:
                if pattern in path:
                    return True
            return False
        
        if os.path.exists(dst):
            shutil.rmtree(dst)
        
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*exclude_patterns))
        logging.info(f"目录复制完成: {src} -> {dst}")
        return True
        
    except Exception as e:
        logging.error(f"目录复制失败: {str(e)}")
        return False


def get_file_info(file_path: str) -> dict:
    """
    获取文件信息
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件信息字典
    """
    try:
        stat = os.stat(file_path)
        return {
            'path': file_path,
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'is_file': os.path.isfile(file_path),
            'is_dir': os.path.isdir(file_path)
        }
    except Exception as e:
        logging.error(f"获取文件信息失败 {file_path}: {str(e)}")
        return {}


def find_files_by_extension(directory: str, extensions: List[str]) -> List[str]:
    """
    根据扩展名查找文件
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表
        
    Returns:
        文件路径列表
    """
    files = []
    try:
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, filename)
                    files.append(file_path)
    except Exception as e:
        logging.error(f"查找文件失败: {str(e)}")
    
    return files


def read_file_content(file_path: str, encoding: str = 'utf-8') -> Optional[str]:
    """
    读取文件内容
    
    Args:
        file_path: 文件路径
        encoding: 文件编码
        
    Returns:
        文件内容
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logging.error(f"读取文件失败 {file_path}: {str(e)}")
        return None


def write_file_content(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
    """
    写入文件内容
    
    Args:
        file_path: 文件路径
        content: 文件内容
        encoding: 文件编码
        
    Returns:
        是否写入成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        logging.debug(f"文件写入成功: {file_path}")
        return True
        
    except Exception as e:
        logging.error(f"文件写入失败 {file_path}: {str(e)}")
        return False


def cleanup_temp_files(temp_dir: str) -> None:
    """
    清理临时文件
    
    Args:
        temp_dir: 临时目录路径
    """
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"临时目录已清理: {temp_dir}")
    except Exception as e:
        logging.error(f"清理临时文件失败: {str(e)}")


def get_project_structure(directory: str, max_depth: int = 3) -> dict:
    """
    获取项目结构
    
    Args:
        directory: 项目目录
        max_depth: 最大深度
        
    Returns:
        项目结构字典
    """
    # 需要跳过的目录和文件
    skip_items = {
        '.git', '.vscode', '.idea', '__pycache__', '.pytest_cache',
        'venv', 'env', '.env', 'node_modules', '.DS_Store',
        '.gitignore', '*.pyc', '*.pyo', '*.pyd', '.Python'
    }
    
    def build_tree(path: str, depth: int = 0) -> dict:
        if depth > max_depth:
            return {}
        
        result = {}
        try:
            for item in os.listdir(path):
                # 跳过不需要的目录和文件
                if item in skip_items or item.startswith('.'):
                    continue
                    
                item_path = os.path.join(path, item)
                
                # 安全检查：跳过符号链接和无法访问的文件
                if os.path.islink(item_path) or not os.access(item_path, os.R_OK):
                    continue
                
                if os.path.isdir(item_path):
                    result[item] = {
                        'type': 'directory',
                        'children': build_tree(item_path, depth + 1)
                    }
                else:
                    try:
                        result[item] = {
                            'type': 'file',
                            'size': os.path.getsize(item_path)
                        }
                    except (OSError, IOError):
                        # 跳过无法访问的文件
                        continue
        except Exception as e:
            logging.error(f"获取目录结构失败 {path}: {str(e)}")
        
        return result
    
    return build_tree(directory) 