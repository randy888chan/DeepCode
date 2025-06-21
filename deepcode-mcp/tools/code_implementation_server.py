#!/usr/bin/env python3
"""
Code Implementation MCP Server - 代码实现MCP服务器

这个MCP服务器提供了论文代码复现所需的核心功能：
1. 文件读写操作 (File read/write operations)
2. 代码执行和测试 (Code execution and testing)
3. 代码搜索和分析 (Code search and analysis)
4. 迭代式改进支持 (Iterative improvement support)

This MCP server provides core functions needed for paper code reproduction:
1. File read/write operations
2. Code execution and testing
3. Code search and analysis
4. Iterative improvement support

使用方法 / Usage:
python tools/code_implementation_server.py
"""

import os
import subprocess
import json
import sys
import io
from pathlib import Path
import re
from typing import Dict, Any
import tempfile
import shutil
import logging
from datetime import datetime

# 设置标准输出编码为UTF-8
if sys.stdout.encoding != "utf-8":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        else:
            sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")
            sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8")
    except Exception as e:
        print(f"Warning: Could not set UTF-8 encoding: {e}")

# 导入MCP相关模块
from mcp.server.fastmcp import FastMCP

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastMCP服务器实例
mcp = FastMCP("code-implementation-server")

# 全局变量：工作空间目录和操作历史
WORKSPACE_DIR = None
OPERATION_HISTORY = []
CURRENT_FILES = {}


def initialize_workspace(workspace_dir: str = None):
    """初始化工作空间"""
    global WORKSPACE_DIR
    if workspace_dir is None:
        # 默认使用当前目录下的generate_code目录
        WORKSPACE_DIR = Path.cwd() / "generate_code"
    else:
        WORKSPACE_DIR = Path(workspace_dir).resolve()

    # 确保工作空间目录存在
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"工作空间初始化: {WORKSPACE_DIR}")


def validate_path(path: str) -> Path:
    """验证路径是否在工作空间内"""
    if WORKSPACE_DIR is None:
        initialize_workspace()

    full_path = (WORKSPACE_DIR / path).resolve()
    if not str(full_path).startswith(str(WORKSPACE_DIR)):
        raise ValueError(f"路径 {path} 超出工作空间范围")
    return full_path


def log_operation(action: str, details: Dict[str, Any]):
    """记录操作历史"""
    OPERATION_HISTORY.append(
        {"timestamp": datetime.now().isoformat(), "action": action, "details": details}
    )


# ==================== 文件操作工具 ====================


@mcp.tool()
async def read_file(
    file_path: str, start_line: int = None, end_line: int = None
) -> str:
    """
    读取文件内容，支持指定行号范围

    Args:
        file_path: 文件路径，相对于工作空间
        start_line: 起始行号（从1开始，可选）
        end_line: 结束行号（从1开始，可选）

    Returns:
        文件内容或错误信息的JSON字符串
    """
    try:
        full_path = validate_path(file_path)

        if not full_path.exists():
            result = {"status": "error", "message": f"文件不存在: {file_path}"}
            log_operation(
                "read_file_error", {"file_path": file_path, "error": "file_not_found"}
            )
            return json.dumps(result, ensure_ascii=False, indent=2)

        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 处理行号范围
        if start_line is not None or end_line is not None:
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else len(lines)
            lines = lines[start_idx:end_idx]

        content = "".join(lines)

        result = {
            "status": "success",
            "content": content,
            "file_path": file_path,
            "total_lines": len(lines),
            "size_bytes": len(content.encode("utf-8")),
        }

        log_operation(
            "read_file",
            {
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "lines_read": len(lines),
            },
        )

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        result = {
            "status": "error",
            "message": f"读取文件失败: {str(e)}",
            "file_path": file_path,
        }
        log_operation("read_file_error", {"file_path": file_path, "error": str(e)})
        return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def write_file(
    file_path: str, content: str, create_dirs: bool = True, create_backup: bool = False
) -> str:
    """
    写入内容到文件

    Args:
        file_path: 文件路径，相对于工作空间
        content: 要写入的文件内容
        create_dirs: 如果目录不存在是否创建
        create_backup: 如果文件已存在是否创建备份文件

    Returns:
        操作结果的JSON字符串
    """
    try:
        full_path = validate_path(file_path)

        # 创建目录（如果需要）
        if create_dirs:
            full_path.parent.mkdir(parents=True, exist_ok=True)

        # 备份现有文件（仅在明确要求时）
        backup_created = False
        if full_path.exists() and create_backup:
            backup_path = full_path.with_suffix(full_path.suffix + ".backup")
            shutil.copy2(full_path, backup_path)
            backup_created = True

        # 写入文件
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        # 更新当前文件记录
        CURRENT_FILES[file_path] = {
            "last_modified": datetime.now().isoformat(),
            "size_bytes": len(content.encode("utf-8")),
            "lines": len(content.split("\n")),
        }

        result = {
            "status": "success",
            "message": f"文件写入成功: {file_path}",
            "file_path": file_path,
            "size_bytes": len(content.encode("utf-8")),
            "lines_written": len(content.split("\n")),
            "backup_created": backup_created,
        }

        log_operation(
            "write_file",
            {
                "file_path": file_path,
                "size_bytes": len(content.encode("utf-8")),
                "lines": len(content.split("\n")),
                "backup_created": backup_created,
            },
        )

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        result = {
            "status": "error",
            "message": f"写入文件失败: {str(e)}",
            "file_path": file_path,
        }
        log_operation("write_file_error", {"file_path": file_path, "error": str(e)})
        return json.dumps(result, ensure_ascii=False, indent=2)


# ==================== 代码执行工具 ====================


@mcp.tool()
async def execute_python(code: str, timeout: int = 30) -> str:
    """
    执行Python代码并返回输出

    Args:
        code: 要执行的Python代码
        timeout: 超时时间（秒）

    Returns:
        执行结果的JSON字符串
    """
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            # 执行Python代码
            result = subprocess.run(
                [sys.executable, temp_file],
                cwd=WORKSPACE_DIR,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
            )

            execution_result = {
                "status": "success" if result.returncode == 0 else "error",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timeout": timeout,
            }

            if result.returncode != 0:
                execution_result["message"] = "Python代码执行失败"
            else:
                execution_result["message"] = "Python代码执行成功"

            log_operation(
                "execute_python",
                {
                    "return_code": result.returncode,
                    "stdout_length": len(result.stdout),
                    "stderr_length": len(result.stderr),
                },
            )

            return json.dumps(execution_result, ensure_ascii=False, indent=2)

        finally:
            # 清理临时文件
            os.unlink(temp_file)

    except subprocess.TimeoutExpired:
        result = {
            "status": "error",
            "message": f"Python代码执行超时 ({timeout}秒)",
            "timeout": timeout,
        }
        log_operation("execute_python_timeout", {"timeout": timeout})
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        result = {"status": "error", "message": f"Python代码执行失败: {str(e)}"}
        log_operation("execute_python_error", {"error": str(e)})
        return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def execute_bash(command: str, timeout: int = 30) -> str:
    """
    执行bash命令

    Args:
        command: 要执行的bash命令
        timeout: 超时时间（秒）

    Returns:
        执行结果的JSON字符串
    """
    try:
        # 安全检查：禁止危险命令
        dangerous_commands = ["rm -rf", "sudo", "chmod 777", "mkfs", "dd if="]
        if any(dangerous in command.lower() for dangerous in dangerous_commands):
            result = {"status": "error", "message": f"禁止执行危险命令: {command}"}
            log_operation(
                "execute_bash_blocked",
                {"command": command, "reason": "dangerous_command"},
            )
            return json.dumps(result, ensure_ascii=False, indent=2)

        # 执行命令
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
        )

        execution_result = {
            "status": "success" if result.returncode == 0 else "error",
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": command,
            "timeout": timeout,
        }

        if result.returncode != 0:
            execution_result["message"] = "Bash命令执行失败"
        else:
            execution_result["message"] = "Bash命令执行成功"

        log_operation(
            "execute_bash",
            {
                "command": command,
                "return_code": result.returncode,
                "stdout_length": len(result.stdout),
                "stderr_length": len(result.stderr),
            },
        )

        return json.dumps(execution_result, ensure_ascii=False, indent=2)

    except subprocess.TimeoutExpired:
        result = {
            "status": "error",
            "message": f"Bash命令执行超时 ({timeout}秒)",
            "command": command,
            "timeout": timeout,
        }
        log_operation("execute_bash_timeout", {"command": command, "timeout": timeout})
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        result = {
            "status": "error",
            "message": f"Bash命令执行失败: {str(e)}",
            "command": command,
        }
        log_operation("execute_bash_error", {"command": command, "error": str(e)})
        return json.dumps(result, ensure_ascii=False, indent=2)


# ==================== 代码搜索工具 ====================


@mcp.tool()
async def search_code(
    pattern: str, file_pattern: str = "*.py", use_regex: bool = False
) -> str:
    """
    在代码文件中搜索模式

    Args:
        pattern: 搜索模式
        file_pattern: 文件模式（如 '*.py'）
        use_regex: 是否使用正则表达式

    Returns:
        搜索结果的JSON字符串
    """
    try:
        if WORKSPACE_DIR is None:
            initialize_workspace()

        import glob

        # 获取匹配的文件
        file_paths = glob.glob(str(WORKSPACE_DIR / "**" / file_pattern), recursive=True)

        matches = []
        total_files_searched = 0

        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                total_files_searched += 1
                relative_path = os.path.relpath(file_path, WORKSPACE_DIR)

                for line_num, line in enumerate(lines, 1):
                    if use_regex:
                        if re.search(pattern, line):
                            matches.append(
                                {
                                    "file": relative_path,
                                    "line_number": line_num,
                                    "line_content": line.strip(),
                                    "match_type": "regex",
                                }
                            )
                    else:
                        if pattern.lower() in line.lower():
                            matches.append(
                                {
                                    "file": relative_path,
                                    "line_number": line_num,
                                    "line_content": line.strip(),
                                    "match_type": "substring",
                                }
                            )

            except Exception as e:
                logger.warning(f"搜索文件时出错 {file_path}: {e}")
                continue

        result = {
            "status": "success",
            "pattern": pattern,
            "file_pattern": file_pattern,
            "use_regex": use_regex,
            "total_matches": len(matches),
            "total_files_searched": total_files_searched,
            "matches": matches[:50],  # 限制返回前50个匹配
        }

        if len(matches) > 50:
            result["note"] = f"显示前50个匹配，总共找到{len(matches)}个匹配"

        log_operation(
            "search_code",
            {
                "pattern": pattern,
                "file_pattern": file_pattern,
                "use_regex": use_regex,
                "total_matches": len(matches),
                "files_searched": total_files_searched,
            },
        )

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        result = {
            "status": "error",
            "message": f"代码搜索失败: {str(e)}",
            "pattern": pattern,
        }
        log_operation("search_code_error", {"pattern": pattern, "error": str(e)})
        return json.dumps(result, ensure_ascii=False, indent=2)


# ==================== 文件结构工具 ====================


@mcp.tool()
async def get_file_structure(directory: str = ".", max_depth: int = 5) -> str:
    """
    获取目录的文件结构

    Args:
        directory: 目录路径，相对于工作空间
        max_depth: 最大遍历深度

    Returns:
        文件结构的JSON字符串
    """
    try:
        if WORKSPACE_DIR is None:
            initialize_workspace()

        if directory == ".":
            target_dir = WORKSPACE_DIR
        else:
            target_dir = validate_path(directory)

        if not target_dir.exists():
            result = {"status": "error", "message": f"目录不存在: {directory}"}
            return json.dumps(result, ensure_ascii=False, indent=2)

        def scan_directory(path: Path, current_depth: int = 0) -> Dict[str, Any]:
            """递归扫描目录"""
            if current_depth >= max_depth:
                return {"type": "directory", "name": path.name, "truncated": True}

            items = []
            try:
                for item in sorted(path.iterdir()):
                    relative_path = os.path.relpath(item, WORKSPACE_DIR)

                    if item.is_file():
                        file_info = {
                            "type": "file",
                            "name": item.name,
                            "path": relative_path,
                            "size_bytes": item.stat().st_size,
                            "extension": item.suffix,
                        }
                        items.append(file_info)
                    elif item.is_dir() and not item.name.startswith("."):
                        dir_info = scan_directory(item, current_depth + 1)
                        dir_info["path"] = relative_path
                        items.append(dir_info)
            except PermissionError:
                pass

            return {
                "type": "directory",
                "name": path.name,
                "items": items,
                "item_count": len(items),
            }

        structure = scan_directory(target_dir)

        # 统计信息
        def count_items(node):
            if node["type"] == "file":
                return {"files": 1, "directories": 0}
            else:
                counts = {"files": 0, "directories": 1}
                for item in node.get("items", []):
                    item_counts = count_items(item)
                    counts["files"] += item_counts["files"]
                    counts["directories"] += item_counts["directories"]
                return counts

        counts = count_items(structure)

        result = {
            "status": "success",
            "directory": directory,
            "max_depth": max_depth,
            "structure": structure,
            "summary": {
                "total_files": counts["files"],
                "total_directories": counts["directories"] - 1,  # 减去根目录
            },
        }

        log_operation(
            "get_file_structure",
            {
                "directory": directory,
                "max_depth": max_depth,
                "total_files": counts["files"],
                "total_directories": counts["directories"] - 1,
            },
        )

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        result = {
            "status": "error",
            "message": f"获取文件结构失败: {str(e)}",
            "directory": directory,
        }
        log_operation(
            "get_file_structure_error", {"directory": directory, "error": str(e)}
        )
        return json.dumps(result, ensure_ascii=False, indent=2)


# ==================== 工作空间管理工具 ====================


@mcp.tool()
async def set_workspace(workspace_path: str) -> str:
    """
    设置工作空间目录

    Args:
        workspace_path: 工作空间路径

    Returns:
        操作结果的JSON字符串
    """
    try:
        global WORKSPACE_DIR
        new_workspace = Path(workspace_path).resolve()

        # 创建目录（如果不存在）
        new_workspace.mkdir(parents=True, exist_ok=True)

        old_workspace = WORKSPACE_DIR
        WORKSPACE_DIR = new_workspace

        result = {
            "status": "success",
            "message": f"工作空间设置成功: {workspace_path}",
            "old_workspace": str(old_workspace) if old_workspace else None,
            "new_workspace": str(WORKSPACE_DIR),
        }

        log_operation(
            "set_workspace",
            {
                "old_workspace": str(old_workspace) if old_workspace else None,
                "new_workspace": str(WORKSPACE_DIR),
            },
        )

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        result = {
            "status": "error",
            "message": f"设置工作空间失败: {str(e)}",
            "workspace_path": workspace_path,
        }
        log_operation(
            "set_workspace_error", {"workspace_path": workspace_path, "error": str(e)}
        )
        return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_operation_history(last_n: int = 10) -> str:
    """
    获取操作历史

    Args:
        last_n: 返回最近的N个操作

    Returns:
        操作历史的JSON字符串
    """
    try:
        recent_history = (
            OPERATION_HISTORY[-last_n:] if last_n > 0 else OPERATION_HISTORY
        )

        result = {
            "status": "success",
            "total_operations": len(OPERATION_HISTORY),
            "returned_operations": len(recent_history),
            "workspace": str(WORKSPACE_DIR) if WORKSPACE_DIR else None,
            "history": recent_history,
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        result = {"status": "error", "message": f"获取操作历史失败: {str(e)}"}
        return json.dumps(result, ensure_ascii=False, indent=2)


# ==================== 服务器初始化 ====================


def main():
    """启动MCP服务器"""
    print("🚀 Code Implementation MCP Server")
    print("📝 论文代码复现工具服务器 / Paper Code Implementation Tool Server")
    print("")
    print("Available tools / 可用工具:")
    print("  • read_file           - 读取文件内容 / Read file contents")
    print("  • write_file          - 写入文件内容 / Write file contents")
    print("  • execute_python      - 执行Python代码 / Execute Python code")
    print("  • execute_bash        - 执行bash命令 / Execute bash commands")
    print("  • search_code         - 搜索代码模式 / Search code patterns")
    print("  • get_file_structure  - 获取文件结构 / Get file structure")
    print("  • set_workspace       - 设置工作空间 / Set workspace")
    print("  • get_operation_history - 获取操作历史 / Get operation history")
    print("")
    print("🔧 Server starting...")

    # 初始化默认工作空间
    initialize_workspace()

    # 启动服务器
    mcp.run()


if __name__ == "__main__":
    main()
