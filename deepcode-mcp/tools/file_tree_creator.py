import os
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from mcp.server import FastMCP

app = FastMCP('file-tree-creator')

@app.tool()
async def create_file_tree(base_directory: str, file_structure: str) -> str:
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
        
        for line in lines:
            if not line.strip() or '├──' not in line and '│' not in line and '└──' not in line:
                continue
                
            # 提取文件/目录名称
            clean_line = line.replace('├──', '').replace('└──', '').replace('│', '').strip()
            if not clean_line:
                continue
                
            # 计算缩进级别来确定目录层级
            indent_level = (len(line) - len(line.lstrip())) // 4
            
            # 构建相对路径
            if '/' in clean_line:
                parts = clean_line.split('/')
                file_path = base_path / '/'.join(parts)
            else:
                file_path = base_path / clean_line
                
            try:
                if clean_line.endswith('/'):
                    # 目录
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

@app.tool()
async def create_files_from_list(base_directory: str, file_list: List[str]) -> str:
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

@app.tool()
async def execute_shell_commands(commands: str, working_directory: str) -> str:
    """
    执行多个shell命令来创建文件树
    
    Args:
        commands: 要执行的命令列表（每行一个命令）
        working_directory: 工作目录
        
    Returns:
        命令执行结果
    """
    try:
        # 确保工作目录存在
        work_path = Path(working_directory)
        work_path.mkdir(parents=True, exist_ok=True)
        
        # 解析命令
        command_lines = [line.strip() for line in commands.strip().split('\n') if line.strip()]
        
        results = []
        created_items = []
        
        for command in command_lines:
            if not command or command.startswith('#'):
                continue
                
            try:
                # 在工作目录中执行命令
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=working_directory,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    # 记录创建的项目
                    if command.startswith('mkdir'):
                        path = command.replace('mkdir -p ', '').strip()
                        created_items.append(f"目录: {path}")
                    elif command.startswith('touch'):
                        path = command.replace('touch ', '').strip()
                        created_items.append(f"文件: {path}")
                    
                    results.append(f"✅ {command}")
                else:
                    results.append(f"❌ {command} - 错误: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                results.append(f"❌ {command} - 超时")
            except Exception as e:
                results.append(f"❌ {command} - 异常: {str(e)}")
        
        summary = f"执行了 {len(command_lines)} 个命令，创建了 {len(created_items)} 个项目"
        
        return f"命令执行完成:\n\n{summary}\n\n执行结果:\n" + "\n".join(results) + f"\n\n创建的项目:\n" + "\n".join(created_items)
        
    except Exception as e:
        return f"命令执行失败: {str(e)}"

@app.tool()
async def execute_shell_command(command: str, working_directory: str = None) -> str:
    """
    执行单个shell命令
    
    Args:
        command: 要执行的命令
        working_directory: 工作目录，可选
        
    Returns:
        命令执行结果
    """
    try:
        if working_directory:
            # 切换到指定目录执行命令
            full_command = f"cd {working_directory} && {command}"
        else:
            full_command = command
            
        # 使用subprocess执行命令
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30  # 30秒超时
        )
        
        output = []
        if result.stdout:
            output.append(f"标准输出:\n{result.stdout}")
        if result.stderr:
            output.append(f"标准错误:\n{result.stderr}")
        if result.returncode != 0:
            output.append(f"返回码: {result.returncode}")
            
        return "\n".join(output) if output else "命令执行成功，无输出"
        
    except subprocess.TimeoutExpired:
        return "命令执行超时"
    except Exception as e:
        return f"命令执行失败: {str(e)}"

if __name__ == "__main__":
    app.run() 