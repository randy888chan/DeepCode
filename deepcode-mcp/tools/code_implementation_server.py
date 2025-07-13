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
    """
    初始化工作空间
    
    默认情况下，工作空间将通过 set_workspace 工具由工作流设置为:
    {plan_file_parent}/generate_code
    
    Args:
        workspace_dir: 可选的工作空间目录路径
    """
    global WORKSPACE_DIR
    if workspace_dir is None:
        # 默认使用当前目录下的generate_code目录，但不立即创建
        # 这个默认值将被工作流通过 set_workspace 工具覆盖
        WORKSPACE_DIR = Path.cwd() / "generate_code"
        # logger.info(f"工作空间初始化 (默认值，将被工作流覆盖): {WORKSPACE_DIR}")
        # logger.info("注意: 实际工作空间将由工作流通过 set_workspace 工具设置为 {plan_file_parent}/generate_code")
    else:
        WORKSPACE_DIR = Path(workspace_dir).resolve()
        # 只有明确指定目录时才创建
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"工作空间初始化: {WORKSPACE_DIR}")


def ensure_workspace_exists():
    """确保工作空间目录存在"""
    global WORKSPACE_DIR
    if WORKSPACE_DIR is None:
        initialize_workspace()
    
    # 创建工作空间目录（如果不存在）
    if not WORKSPACE_DIR.exists():
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"工作空间目录已创建: {WORKSPACE_DIR}")


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
            # 确保工作空间目录存在
            ensure_workspace_exists()
            
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

        # 确保工作空间目录存在
        ensure_workspace_exists()
        
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


@mcp.tool()
async def read_code_mem(file_path: str) -> str:
    """
    Check if file summary exists in implement_code_summary.md
    
    Args:
        file_path: File path to check for summary information in implement_code_summary.md
        
    Returns:
        Summary information if available
    """
    try:
        if not file_path:
            result = {
                "status": "error",
                "message": "file_path parameter is required"
            }
            log_operation("read_code_mem_error", {"error": "missing_file_path"})
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        # Ensure workspace exists
        ensure_workspace_exists()
        
        # Look for implement_code_summary.md in the workspace
        current_path = Path(WORKSPACE_DIR)
        summary_file_path = current_path.parent / "implement_code_summary.md"
        
        if not summary_file_path.exists():
            result = {
                "status": "no_summary",
                "file_path": file_path,
                "message": f"No summary file found.",
                # "recommendation": f"read_file(file_path='{file_path}')"
            }
            log_operation("read_code_mem", {"file_path": file_path, "status": "no_summary_file"})
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        # Read the summary file
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_content = f.read()
        
        if not summary_content.strip():
            result = {
                "status": "no_summary",
                "file_path": file_path,
                "message": f"Summary file is empty.",
                # "recommendation": f"read_file(file_path='{file_path}')"
            }
            log_operation("read_code_mem", {"file_path": file_path, "status": "empty_summary"})
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        # Extract file-specific section from summary
        file_section = _extract_file_section_from_summary(summary_content, file_path)
        
        if file_section:
            result = {
                "status": "summary_found",
                "file_path": file_path,
                "summary_content": file_section,
                "message": f"Summary information found for {file_path} in implement_code_summary.md"
            }
            log_operation("read_code_mem", {"file_path": file_path, "status": "summary_found", "section_length": len(file_section)})
            return json.dumps(result, ensure_ascii=False, indent=2)
        else:
            result = {
                "status": "no_summary",
                "file_path": file_path,
                "message": f"No summary found for {file_path} in implement_code_summary.md",
                # "recommendation": f"Use read_file tool to read the actual file: read_file(file_path='{file_path}')"
            }
            log_operation("read_code_mem", {"file_path": file_path, "status": "no_match"})
            return json.dumps(result, ensure_ascii=False, indent=2)
            
    except Exception as e:
        result = {
            "status": "error",
            "message": f"Failed to check code memory: {str(e)}",
            "file_path": file_path,
            # "recommendation": "Use read_file tool instead"
        }
        log_operation("read_code_mem_error", {"file_path": file_path, "error": str(e)})
        return json.dumps(result, ensure_ascii=False, indent=2)


def _extract_file_section_from_summary(summary_content: str, target_file_path: str) -> str:
    """
    Extract the specific section for a file from the summary content
    
    Args:
        summary_content: Full summary content
        target_file_path: Path of the target file
        
    Returns:
        File-specific section or None if not found
    """
    import re
    
    # Normalize the target path for comparison
    normalized_target = _normalize_file_path(target_file_path)
    
    # Pattern to match implementation sections with separator lines
    section_pattern = r'={80}\s*\n## IMPLEMENTATION File ([^;]+); ROUND \d+\s*\n={80}(.*?)(?=\n={80}|\Z)'
    
    matches = re.findall(section_pattern, summary_content, re.DOTALL)
    
    for file_path_in_summary, section_content in matches:
        file_path_in_summary = file_path_in_summary.strip()
        section_content = section_content.strip()
        
        # Normalize the path from summary for comparison
        normalized_summary_path = _normalize_file_path(file_path_in_summary)
        
        # Check if paths match using multiple strategies
        if _paths_match(normalized_target, normalized_summary_path, target_file_path, file_path_in_summary):
            # Return the complete section with proper formatting
            file_section = f"""================================================================================
## IMPLEMENTATION File {file_path_in_summary}; ROUND [X]
================================================================================

{section_content}

---
*Extracted from implement_code_summary.md*"""
            return file_section
    
    # If no section-based match, try alternative parsing method
    return _extract_file_section_alternative(summary_content, target_file_path)


def _normalize_file_path(file_path: str) -> str:
    """Normalize file path for comparison"""
    # Remove leading/trailing slashes and convert to lowercase
    normalized = file_path.strip('/').lower()
    # Replace backslashes with forward slashes
    normalized = normalized.replace('\\', '/')
    
    # Remove common prefixes to make matching more flexible
    common_prefixes = ['rice/', 'src/', './rice/', './src/', './']
    for prefix in common_prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    
    return normalized


def _paths_match(normalized_target: str, normalized_summary: str, original_target: str, original_summary: str) -> bool:
    """Check if two file paths match using multiple strategies"""
    
    # Strategy 1: Exact normalized match
    if normalized_target == normalized_summary:
        return True
    
    # Strategy 2: Basename match (filename only)
    target_basename = os.path.basename(original_target)
    summary_basename = os.path.basename(original_summary)
    if target_basename == summary_basename and len(target_basename) > 4:
        return True
    
    # Strategy 3: Suffix match (remove common prefixes and compare)
    target_suffix = _remove_common_prefixes(normalized_target)
    summary_suffix = _remove_common_prefixes(normalized_summary)
    if target_suffix == summary_suffix:
        return True
    
    # Strategy 4: Ends with match
    if normalized_target.endswith(normalized_summary) or normalized_summary.endswith(normalized_target):
        return True
    
    # Strategy 5: Contains match for longer paths
    if len(normalized_target) > 10 and normalized_target in normalized_summary:
        return True
    if len(normalized_summary) > 10 and normalized_summary in normalized_target:
        return True
    
    return False


def _remove_common_prefixes(file_path: str) -> str:
    """Remove common prefixes from file path"""
    prefixes_to_remove = ['rice/', 'src/', 'core/', './']
    path = file_path
    
    for prefix in prefixes_to_remove:
        if path.startswith(prefix):
            path = path[len(prefix):]
    
    return path


def _extract_file_section_alternative(summary_content: str, target_file_path: str) -> str:
    """Alternative method to extract file section using simpler pattern matching"""
    
    # Get the basename for fallback matching
    target_basename = os.path.basename(target_file_path)
    
    # Split by separator lines to get individual sections
    sections = summary_content.split('=' * 80)
    
    for i, section in enumerate(sections):
        if '## IMPLEMENTATION File' in section:
            # Extract the file path from the header
            lines = section.strip().split('\n')
            for line in lines:
                if '## IMPLEMENTATION File' in line:
                    # Extract file path between "File " and "; ROUND"
                    try:
                        file_part = line.split('File ')[1].split('; ROUND')[0].strip()
                        
                        # Check if this matches our target
                        if (_normalize_file_path(target_file_path) == _normalize_file_path(file_part) or
                            target_basename == os.path.basename(file_part) or
                            target_file_path in file_part or
                            file_part.endswith(target_file_path)):
                            
                            # Get the next section which contains the content
                            if i + 1 < len(sections):
                                content_section = sections[i + 1].strip()
                                return f"""================================================================================
## IMPLEMENTATION File {file_part}
================================================================================

{content_section}

---
*Extracted from implement_code_summary.md using alternative method*"""
                    except (IndexError, AttributeError):
                        continue
    
    return None


# ==================== 代码搜索工具 ====================


@mcp.tool()
async def search_code(
    pattern: str, 
    file_pattern: str = "*.json", 
    use_regex: bool = False,
    search_directory: str = None
) -> str:
    """
    在代码文件中搜索模式

    Args:
        pattern: 搜索模式
        file_pattern: 文件模式（如 '*.py'）
        use_regex: 是否使用正则表达式
        search_directory: 指定搜索目录（可选，如果不指定则使用WORKSPACE_DIR）

    Returns:
        搜索结果的JSON字符串
    """
    try:
        # 确定搜索目录
        if search_directory:
            # 如果指定了搜索目录，使用指定的目录
            if os.path.isabs(search_directory):
                search_path = Path(search_directory)
            else:
                # 相对路径，相对于当前工作目录
                search_path = Path.cwd() / search_directory
        else:
            # 如果没有指定搜索目录，使用默认的WORKSPACE_DIR
            ensure_workspace_exists()
            search_path = WORKSPACE_DIR
        
        # 检查搜索目录是否存在
        if not search_path.exists():
            result = {
                "status": "error",
                "message": f"搜索目录不存在: {search_path}",
                "pattern": pattern,
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

        import glob

        # 获取匹配的文件
        file_paths = glob.glob(str(search_path / "**" / file_pattern), recursive=True)

        matches = []
        total_files_searched = 0

        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                total_files_searched += 1
                relative_path = os.path.relpath(file_path, search_path)

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
            "search_directory": str(search_path),
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
                "search_directory": str(search_path),
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
        ensure_workspace_exists()

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
    
    由工作流调用以将工作空间设置为: {plan_file_parent}/generate_code
    这确保所有文件操作都相对于正确的项目目录执行

    Args:
        workspace_path: 工作空间路径 (通常是 {plan_file_parent}/generate_code)

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

        logger.info(f"New Workspace: {WORKSPACE_DIR}")

        result = {
            "status": "success",
            "message": f"Workspace setup successful: {workspace_path}",
            "new_workspace": str(WORKSPACE_DIR),
        }

        log_operation(
            "set_workspace",
            {
                "old_workspace": str(old_workspace) if old_workspace else None,
                "new_workspace": str(WORKSPACE_DIR),
                "workspace_alignment": "plan_file_parent/generate_code",
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
    # print("  • read_file           - 读取文件内容 / Read file contents")
    print("  • read_code_mem       - 读取代码摘要 / Read code summary from implement_code_summary.md")
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
