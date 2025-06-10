#!/usr/bin/env python3
"""
简化的文件创建工具
"""

import os
from pathlib import Path
from typing import List

def create_files_from_list(base_directory: str, file_list: List[str]) -> str:
    """
    根据文件列表在指定目录创建文件
    
    Args:
        base_directory: 基础目录路径
        file_list: 文件路径列表
        
    Returns:
        创建结果的描述
    """
    try:
        base_path = Path(base_directory)
        base_path.mkdir(parents=True, exist_ok=True)
        
        created_items = []
        
        for file_path_str in file_list:
            file_path = base_path / file_path_str
            
            try:
                # 创建父目录
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 如果路径以/结尾，创建目录，否则创建文件
                if file_path_str.endswith('/'):
                    file_path.mkdir(exist_ok=True)
                    created_items.append(f"目录: {file_path}")
                else:
                    file_path.touch()
                    created_items.append(f"文件: {file_path}")
                    
            except Exception as e:
                created_items.append(f"错误: 创建 {file_path} 失败 - {str(e)}")
        
        return f"成功创建文件在 {base_directory}:\n" + "\n".join(created_items)
        
    except Exception as e:
        return f"创建文件失败: {str(e)}"

def create_file_tree(base_directory: str, file_structure: str) -> str:
    """
    根据文件结构描述在指定目录创建文件树
    
    Args:
        base_directory: 基础目录路径
        file_structure: 文件结构描述，支持类似树状结构的文本
        
    Returns:
        创建结果的描述
    """
    try:
        # 确保基础目录存在
        base_path = Path(base_directory)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # 解析文件结构并创建文件和目录
        lines = file_structure.strip().split('\n')
        created_items = []
        
        # 用于跟踪目录层级
        dir_stack = []
        
        for line in lines:
            if not line.strip():
                continue
                
            # 计算缩进级别
            original_line = line
            indent_level = (len(line) - len(line.lstrip())) // 4
            
            # 提取文件/目录名称
            clean_line = line.replace('├──', '').replace('└──', '').replace('│', '').strip()
            
            # 移除注释部分 - 只保留文件名部分
            if '#' in clean_line:
                clean_line = clean_line.split('#')[0].strip()
            
            # 移除其他无关内容
            if not clean_line or clean_line.startswith('project') or len(clean_line) == 0:
                continue
            
            # 确保clean_line只包含文件名，不包含注释
            clean_line = clean_line.split()[0] if clean_line else ""
            if not clean_line:
                continue
            
            # 更新目录堆栈
            if indent_level <= len(dir_stack):
                dir_stack = dir_stack[:indent_level]
            
            # 构建完整路径
            if dir_stack:
                full_path = '/'.join(dir_stack + [clean_line])
            else:
                full_path = clean_line
                
            file_path = base_path / full_path
                
            try:
                if clean_line.endswith('/') or '.' not in clean_line or clean_line in ['src', 'core', 'utils', 'tests', 'docs', 'experiments', 'configs', 'notebooks', 'models']:
                    # 目录
                    dir_name = clean_line.rstrip('/')
                    if dir_name:
                        dir_stack.append(dir_name)
                        file_path = base_path / '/'.join(dir_stack)
                        file_path.mkdir(parents=True, exist_ok=True)
                        created_items.append(f"目录: {file_path}")
                else:
                    # 文件
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.touch()
                    created_items.append(f"文件: {file_path}")
            except Exception as e:
                created_items.append(f"错误: 创建 {file_path} 失败 - {str(e)}")
        
        return f"成功创建文件树在 {base_directory}:\n" + "\n".join(created_items)
        
    except Exception as e:
        return f"创建文件树失败: {str(e)}"

def parse_file_tree_to_list(file_structure: str) -> List[str]:
    """
    将文件树结构解析为文件路径列表
    
    Args:
        file_structure: 文件树结构文本
        
    Returns:
        文件路径列表
    """
    lines = file_structure.strip().split('\n')
    file_list = []
    
    for line in lines:
        if not line.strip():
            continue
            
        # 提取文件/目录名称
        clean_line = line.replace('├──', '').replace('└──', '').replace('│', '').strip()
        if not clean_line or clean_line.startswith('project'):
            continue
            
        file_list.append(clean_line)
    
    return file_list

if __name__ == "__main__":
    # 测试
    test_structure = """
project/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── gcn.py
│   └── utils/
│       ├── __init__.py
│       └── data.py
├── tests/
│   └── test_gcn.py
└── requirements.txt
"""
    
    result = create_file_tree("test_output/generate_code", test_structure)
    print(result) 