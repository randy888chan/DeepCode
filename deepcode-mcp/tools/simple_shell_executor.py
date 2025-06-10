#!/usr/bin/env python3
"""
简化的shell命令执行器
"""

import subprocess
import os
from pathlib import Path
from typing import List
import platform

def convert_to_windows_command(command: str) -> str:
    """
    将Unix命令转换为Windows兼容命令
    """
    if platform.system() == "Windows":
        if command.startswith('mkdir -p '):
            # mkdir -p path -> mkdir path (Windows会自动创建父目录)
            path = command.replace('mkdir -p ', '').strip()
            return f'mkdir "{path}" 2>nul || echo Directory exists'
        elif command.startswith('touch '):
            # touch file -> type nul > file
            path = command.replace('touch ', '').strip()
            return f'echo. > "{path}"'
    
    return command

def execute_shell_commands(commands: str, working_directory: str) -> str:
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
        success_count = 0
        
        for command in command_lines:
            if not command or command.startswith('#'):
                continue
                
            try:
                # 转换为Windows兼容命令
                win_command = convert_to_windows_command(command)
                
                # 在工作目录中执行命令
                result = subprocess.run(
                    win_command,
                    shell=True,
                    cwd=working_directory,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    success_count += 1
                    # 记录创建的项目
                    if command.startswith('mkdir'):
                        path = command.replace('mkdir -p ', '').strip()
                        created_items.append(f"✅ 目录: {path}")
                    elif command.startswith('touch'):
                        path = command.replace('touch ', '').strip()
                        created_items.append(f"✅ 文件: {path}")
                    
                    results.append(f"✅ {command}")
                else:
                    results.append(f"❌ {command} - 错误: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                results.append(f"❌ {command} - 超时")
            except Exception as e:
                results.append(f"❌ {command} - 异常: {str(e)}")
        
        summary = f"执行了 {len(command_lines)} 个命令，成功 {success_count} 个"
        
        return f"命令执行完成:\n\n{summary}\n\n创建的项目:\n" + "\n".join(created_items[:10])
        
    except Exception as e:
        return f"命令执行失败: {str(e)}"

if __name__ == "__main__":
    # 测试
    test_commands = """
mkdir -p test_project/src/core
mkdir -p test_project/tests
touch test_project/src/core/__init__.py
touch test_project/src/core/main.py
touch test_project/tests/test_main.py
touch test_project/requirements.txt
"""
    
    result = execute_shell_commands(test_commands, "test_output")
    print(result) 