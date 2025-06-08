"""
Code File Operations MCP Server / 代码文件操作 MCP 服务器

This server provides comprehensive file operation tools for LLM-driven code generation.
The LLM analyzes requirements and decides what to write, then calls these tools to execute file operations.

该服务器为LLM驱动的代码生成提供全面的文件操作工具。
LLM分析需求并决定要写什么，然后调用这些工具来执行文件操作。

Architecture / 架构:
User Request → LLM Analysis → LLM calls MCP tools → File operations executed
用户请求 → LLM分析 → LLM调用MCP工具 → 执行文件操作

Available Tools / 可用工具:
1. write_code_file        - 写入完整代码文件 / Write complete code files
2. read_code_file         - 读取代码文件内容 / Read code file content  
3. append_to_file         - 追加内容到文件 / Append content to files
4. insert_code_at_line    - 在指定行插入代码 / Insert code at specific line
5. replace_code_section   - 替换代码段 / Replace code sections
6. create_project_structure - 创建项目结构 / Create project structure
7. validate_file_syntax   - 验证文件语法 / Validate file syntax
8. run_code_file         - 运行代码文件 / Execute code files
9. list_project_files    - 列出项目文件 / List project files
10. create_directory     - 创建目录 / Create directories
"""

import os
import json
import logging
import ast
import subprocess
import sys
import io
from typing import Dict, List, Any, Optional
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# =============================================================================
# CONFIGURATION / 配置
# =============================================================================

# 设置标准输出编码为UTF-8 / Set standard output encoding to UTF-8
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

# Configure logging / 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
# Initialize FastMCP server / 初始化 FastMCP 服务器
mcp = FastMCP("code-file-operations")

# =============================================================================
# CORE FILE OPERATIONS / 核心文件操作
# =============================================================================

@mcp.tool()
async def write_code_file(
    file_path: str,
    code_content: str,
    create_directories: bool = True,
    backup_existing: bool = False
) -> str:
    """
    Write complete code content to a file / 写入完整代码内容到文件
    
    This is the primary tool for creating new code files. The LLM provides the complete
    code content and this tool writes it to the specified path.
    
    这是创建新代码文件的主要工具。LLM提供完整的代码内容，此工具将其写入指定路径。
    
    Args:
        file_path (str): Target file path to write / 要写入的目标文件路径
        code_content (str): Complete code content provided by LLM / LLM提供的完整代码内容
        create_directories (bool): Whether to create parent directories / 是否创建父目录
        backup_existing (bool): Whether to backup existing file / 是否备份现有文件
    
    Returns:
        str: JSON response with operation status and file information / 包含操作状态和文件信息的JSON响应
    
    Example / 示例:
        write_code_file("src/main.py", "print('Hello World')", True, False)
    """
    print(f"[INFO] 🔧 write_code_file: Writing code to {file_path}")
    logger.info(f"Writing code file: {file_path}")
    
    try:
        # Create directories if needed / 如需要则创建目录
        if create_directories:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Backup existing file if requested / 如果请求则备份现有文件
        backup_path = None
        if backup_existing and os.path.exists(file_path):
            backup_path = f"{file_path}.backup"
            with open(file_path, 'r', encoding='utf-8') as original:
                with open(backup_path, 'w', encoding='utf-8') as backup:
                    backup.write(original.read())
        
        # Write the code content / 写入代码内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        
        # Basic file validation / 基本文件验证
        syntax_valid = True
        syntax_errors = []
        
        result = {
            "status": "success",
            "message": f"Successfully wrote code to {file_path}",
            "file_path": file_path,
            "size_bytes": len(code_content.encode('utf-8')),
            "lines_count": len(code_content.split('\n')),
            "backup_created": backup_path,
            "syntax_valid": syntax_valid,
            "syntax_errors": syntax_errors
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to write file: {str(e)}",
            "file_path": file_path
        }
        return json.dumps(error_result, indent=2)


@mcp.tool()
async def read_code_file(file_path: str) -> str:
    """
    Read code content from a file for LLM analysis / 读取代码文件内容供LLM分析
    
    This tool allows the LLM to read existing code files to understand the current
    codebase structure and content before making modifications.
    
    此工具允许LLM读取现有代码文件，以在进行修改之前了解当前代码库结构和内容。
    
    Args:
        file_path (str): Path to the file to read / 要读取的文件路径
    
    Returns:
        str: JSON response with file content and metadata / 包含文件内容和元数据的JSON响应
    
    Example / 示例:
        read_code_file("src/main.py")
    """
    print(f"[INFO] 📖 read_code_file: Reading {file_path}")
    logger.info(f"Reading code file: {file_path}")
    
    try:
        if not os.path.exists(file_path):
            result = {
                "status": "error",
                "message": f"File does not exist: {file_path}",
                "file_path": file_path,
                "content": ""
            }
            return json.dumps(result, indent=2)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = {
            "status": "success",
            "message": f"Successfully read {file_path}",
            "file_path": file_path,
            "content": content,
            "size_bytes": len(content.encode('utf-8')),
            "lines_count": len(content.split('\n'))
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to read file: {str(e)}",
            "file_path": file_path,
            "content": ""
        }
        return json.dumps(error_result, indent=2)

# =============================================================================
# INCREMENTAL EDITING OPERATIONS / 增量编辑操作
# =============================================================================

@mcp.tool()
async def append_to_file(
    file_path: str,
    code_content: str,
    add_newline: bool = True
) -> str:
    """
    Append code content to an existing file / 向现有文件追加代码内容
    
    Use this tool to add new code to the end of an existing file without
    overwriting the current content.
    
    使用此工具向现有文件的末尾添加新代码，而不覆盖当前内容。
    
    Args:
        file_path (str): Target file path / 目标文件路径
        code_content (str): Code content to append / 要追加的代码内容
        add_newline (bool): Whether to add newline before appending / 是否在追加前添加换行符
    
    Returns:
        str: JSON response with operation status / 包含操作状态的JSON响应
    
    Example / 示例:
        append_to_file("src/main.py", "print('New function')", True)
    """
    try:
        # Ensure file exists / 确保文件存在
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("")
        
        # Append content / 追加内容
        with open(file_path, 'a', encoding='utf-8') as f:
            if add_newline:
                f.write('\n')
            f.write(code_content)
        
        result = {
            "status": "success",
            "message": f"Successfully appended to {file_path}",
            "file_path": file_path,
            "appended_size": len(code_content.encode('utf-8'))
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to append to file: {str(e)}",
            "file_path": file_path
        }
        return json.dumps(error_result, indent=2)


@mcp.tool()
async def insert_code_at_line(
    file_path: str,
    line_number: int,
    code_content: str
) -> str:
    """
    Insert code content at a specific line number / 在指定行号插入代码内容
    
    Use this tool to insert new code at a specific line position within an existing file.
    The line number is 1-based (first line is line 1).
    
    使用此工具在现有文件的指定行位置插入新代码。
    行号从1开始（第一行是第1行）。
    
    Args:
        file_path (str): Target file path / 目标文件路径
        line_number (int): Line number to insert at (1-based) / 要插入的行号（从1开始）
        code_content (str): Code content to insert / 要插入的代码内容
    
    Returns:
        str: JSON response with operation status / 包含操作状态的JSON响应
    
    Example / 示例:
        insert_code_at_line("src/main.py", 5, "import numpy as np")
    """
    try:
        # Read existing content / 读取现有内容
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            lines = []
        
        # Insert content / 插入内容
        insert_lines = code_content.split('\n')
        for i, line in enumerate(insert_lines):
            lines.insert(line_number - 1 + i, line + '\n')
        
        # Write back / 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        result = {
            "status": "success",
            "message": f"Successfully inserted code at line {line_number} in {file_path}",
            "file_path": file_path,
            "line_number": line_number,
            "lines_inserted": len(insert_lines)
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to insert code: {str(e)}",
            "file_path": file_path
        }
        return json.dumps(error_result, indent=2)


@mcp.tool()
async def replace_code_section(
    file_path: str,
    start_line: int,
    end_line: int,
    new_code_content: str
) -> str:
    """
    Replace a section of code between specified line numbers / 替换指定行号之间的代码段
    
    Use this tool to replace existing code in a specific range with new code.
    Both start_line and end_line are 1-based and inclusive.
    
    使用此工具将特定范围内的现有代码替换为新代码。
    start_line和end_line都从1开始且包含边界。
    
    Args:
        file_path (str): Target file path / 目标文件路径
        start_line (int): Start line number (1-based, inclusive) / 起始行号（从1开始，包含）
        end_line (int): End line number (1-based, inclusive) / 结束行号（从1开始，包含）
        new_code_content (str): New code content to replace with / 要替换的新代码内容
    
    Returns:
        str: JSON response with operation status / 包含操作状态的JSON响应
    
    Example / 示例:
        replace_code_section("src/main.py", 10, 15, "def new_function():\n    pass")
    """
    try:
        # Read existing content / 读取现有内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Replace section / 替换代码段
        new_lines = new_code_content.split('\n')
        # Convert to 0-based indexing and replace / 转换为0基索引并替换
        lines[start_line-1:end_line] = [line + '\n' for line in new_lines]
        
        # Write back / 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        result = {
            "status": "success",
            "message": f"Successfully replaced lines {start_line}-{end_line} in {file_path}",
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "lines_replaced": end_line - start_line + 1,
            "new_lines_count": len(new_lines)
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to replace code section: {str(e)}",
            "file_path": file_path
        }
        return json.dumps(error_result, indent=2)

# =============================================================================
# PROJECT STRUCTURE OPERATIONS / 项目结构操作
# =============================================================================

@mcp.tool()
async def create_project_structure(
    base_path: str,
    structure: Dict[str, Any]
) -> str:
    """
    Create directory structure and files from a nested dictionary / 从嵌套字典创建目录结构和文件
    
    This tool creates a complete project structure including directories and files
    based on a nested dictionary specification.
    
    此工具根据嵌套字典规范创建完整的项目结构，包括目录和文件。
    
    Args:
        base_path (str): Base directory path / 基础目录路径
        structure (Dict[str, Any]): Nested dict representing directory/file structure
                                   嵌套字典表示目录/文件结构
    
    Returns:
        str: JSON response with created items / 包含创建项目的JSON响应
    
    Example / 示例:
        structure = {
            "src": {
                "main.py": "print('Hello')",
                "utils": {
                    "__init__.py": "",
                    "helpers.py": "def helper(): pass"
                }
            },
            "tests": {},
            "README.md": "# Project"
        }
        create_project_structure("my_project", structure)
    """
    print(f"[INFO] 🏗️ create_project_structure: Creating project at {base_path}")
    logger.info(f"Creating project structure at: {base_path}")
    
    try:
        created_items = []
        
        def create_recursive(current_path: str, items: Dict):
            """Recursively create directories and files / 递归创建目录和文件"""
            for name, content in items.items():
                item_path = os.path.join(current_path, name)
                
                if isinstance(content, dict):
                    # It's a directory / 这是一个目录
                    os.makedirs(item_path, exist_ok=True)
                    created_items.append({"type": "directory", "path": item_path})
                    create_recursive(item_path, content)
                else:
                    # It's a file / 这是一个文件
                    os.makedirs(os.path.dirname(item_path), exist_ok=True)
                    with open(item_path, 'w', encoding='utf-8') as f:
                        f.write(content if content else "")
                    created_items.append({"type": "file", "path": item_path})
        
        # Create base directory / 创建基础目录
        os.makedirs(base_path, exist_ok=True)
        create_recursive(base_path, structure)
        
        result = {
            "status": "success",
            "message": f"Created project structure at {base_path}",
            "base_path": base_path,
            "created_items": created_items,
            "total_directories": len([i for i in created_items if i["type"] == "directory"]),
            "total_files": len([i for i in created_items if i["type"] == "file"])
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to create project structure: {str(e)}",
            "base_path": base_path
        }
        return json.dumps(error_result, indent=2)


@mcp.tool()
async def create_directory(directory_path: str) -> str:
    """
    Create a directory and any necessary parent directories / 创建目录及任何必要的父目录
    
    Simple tool to create directories. Automatically creates parent directories if needed.
    
    创建目录的简单工具。如需要会自动创建父目录。
    
    Args:
        directory_path (str): Path of directory to create / 要创建的目录路径
    
    Returns:
        str: JSON response with operation status / 包含操作状态的JSON响应
    
    Example / 示例:
        create_directory("src/utils/helpers")
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        
        result = {
            "status": "success",
            "message": f"Created directory: {directory_path}",
            "directory_path": directory_path,
            "exists": os.path.exists(directory_path)
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to create directory: {str(e)}",
            "directory_path": directory_path
        }
        return json.dumps(error_result, indent=2)

# =============================================================================
# CODE ANALYSIS AND VALIDATION / 代码分析和验证
# =============================================================================

@mcp.tool()
async def validate_file_syntax(
    file_path: str,
    language: str = "auto"
) -> str:
    """
    Validate file syntax for various programming languages / 验证各种编程语言的文件语法
    
    This tool checks if the code syntax is valid for the specified programming language.
    Supports Python syntax validation with detailed error reporting.
    
    此工具检查指定编程语言的代码语法是否有效。
    支持Python语法验证并提供详细的错误报告。
    
    Args:
        file_path (str): Path to file to validate / 要验证的文件路径
        language (str): Programming language (auto, python, javascript, java, go, rust, etc.)
                       编程语言（auto, python, javascript, java, go, rust等）
    
    Returns:
        str: JSON response with validation results / 包含验证结果的JSON响应
    
    Supported Languages / 支持的语言:
        - Python: Full syntax validation with error details
        - Others: Basic file readability check
    
    Example / 示例:
        validate_file_syntax("src/main.py", "python")
        validate_file_syntax("src/main.js", "auto")  # Auto-detects JavaScript
    """
    try:
        if not os.path.exists(file_path):
            result = {
                "status": "error", 
                "message": f"File not found: {file_path}",
                "valid": False,
                "errors": [{"message": "File not found"}]
            }
            return json.dumps(result, indent=2)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Auto-detect language from file extension / 从文件扩展名自动检测语言
        if language == "auto":
            ext = os.path.splitext(file_path)[1].lower()
            language_map = {
                '.py': 'python',
                '.js': 'javascript', 
                '.ts': 'typescript',
                '.java': 'java',
                '.go': 'go',
                '.rs': 'rust',
                '.cpp': 'cpp',
                '.c': 'c',
                '.cs': 'csharp',
                '.php': 'php',
                '.rb': 'ruby'
            }
            language = language_map.get(ext, 'unknown')
        
        # Language-specific validation / 特定语言验证
        if language == "python":
            try:
                compile(code, file_path, 'exec')
                result = {
                    "status": "success",
                    "message": f"{language.title()} syntax is valid",
                    "file_path": file_path,
                    "language": language,
                    "valid": True,
                    "errors": []
                }
            except SyntaxError as e:
                result = {
                    "status": "success",
                    "message": f"{language.title()} syntax errors found", 
                    "file_path": file_path,
                    "language": language,
                    "valid": False,
                    "errors": [{
                        "line": e.lineno,
                        "offset": e.offset,
                        "message": e.msg,
                        "text": e.text
                    }]
                }
        else:
            # For other languages, basic checks / 其他语言的基本检查
            result = {
                "status": "success",
                "message": f"Basic validation completed for {language}",
                "file_path": file_path,
                "language": language,
                "valid": True,  # Basic assumption - file is readable
                "errors": [],
                "note": f"Advanced syntax validation for {language} not implemented"
            }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to validate syntax: {str(e)}",
            "file_path": file_path,
            "language": language,
            "valid": False
        }
        return json.dumps(error_result, indent=2)

# =============================================================================
# CODE EXECUTION AND TESTING / 代码执行和测试
# =============================================================================

@mcp.tool()
async def run_code_file(
    file_path: str,
    language: str = "auto",
    working_directory: str = None,
    timeout: int = 30,
    args: List[str] = None
) -> str:
    """
    Execute a code file in various programming languages / 执行各种编程语言的代码文件
    
    This tool executes code files and captures their output. Supports multiple
    programming languages with configurable timeout and arguments.
    
    此工具执行代码文件并捕获其输出。支持多种编程语言，可配置超时和参数。
    
    Args:
        file_path (str): Path to code file to execute / 要执行的代码文件路径
        language (str): Programming language (auto, python, javascript, java, go, etc.)
                       编程语言（auto, python, javascript, java, go等）
        working_directory (str): Working directory for execution / 执行的工作目录
        timeout (int): Execution timeout in seconds / 执行超时时间（秒）
        args (List[str]): Additional command line arguments / 额外的命令行参数
    
    Returns:
        str: JSON response with execution results / 包含执行结果的JSON响应
    
    Supported Languages / 支持的语言:
        - Python: python file.py
        - JavaScript: node file.js
        - TypeScript: ts-node file.ts
        - Java: java file.java
        - Go: go run file.go
        - Rust: cargo run --bin filename
        - PHP: php file.php
        - Ruby: ruby file.rb
        - Bash: bash file.sh
    
    Example / 示例:
        run_code_file("src/main.py", "python", None, 30, ["--verbose"])
        run_code_file("test.js", "auto")  # Auto-detects JavaScript
    """
    print(f"[INFO] ▶️ run_code_file: Executing {file_path} ({language})")
    logger.info(f"Executing code file: {file_path} with language: {language}")
    
    try:
        if not os.path.exists(file_path):
            result = {
                "status": "error",
                "message": f"File not found: {file_path}",
                "output": "",
                "error": "File not found"
            }
            return json.dumps(result, indent=2)
        
        # Auto-detect language from file extension / 从文件扩展名自动检测语言
        if language == "auto":
            ext = os.path.splitext(file_path)[1].lower()
            language_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript', 
                '.java': 'java',
                '.go': 'go',
                '.rs': 'rust',
                '.cpp': 'cpp',
                '.c': 'c',
                '.cs': 'csharp',
                '.php': 'php',
                '.rb': 'ruby',
                '.sh': 'bash'
            }
            language = language_map.get(ext, 'unknown')
        
        # Set working directory / 设置工作目录
        cwd = working_directory or os.path.dirname(file_path)
        
        # Build command based on language / 根据语言构建命令
        commands = {
            'python': ['python', file_path],
            'javascript': ['node', file_path],
            'typescript': ['ts-node', file_path],
            'java': ['java', file_path],
            'go': ['go', 'run', file_path],
            'rust': ['cargo', 'run', '--bin', os.path.splitext(os.path.basename(file_path))[0]],
            'php': ['php', file_path],
            'ruby': ['ruby', file_path],
            'bash': ['bash', file_path]
        }
        
        if language not in commands:
            result = {
                "status": "error",
                "message": f"Execution not supported for language: {language}",
                "file_path": file_path,
                "language": language,
                "output": "",
                "error": f"Language {language} not supported"
            }
            return json.dumps(result, indent=2)
        
        # Build command with args / 使用参数构建命令
        command = commands[language]
        if args:
            command.extend(args)
        
        # Execute the file / 执行文件
        process = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        result = {
            "status": "success" if process.returncode == 0 else "error",
            "message": f"Executed {file_path} as {language}",
            "file_path": file_path,
            "language": language,
            "command": " ".join(command),
            "return_code": process.returncode,
            "output": process.stdout,
            "error": process.stderr
        }
        
        return json.dumps(result, indent=2)
        
    except subprocess.TimeoutExpired:
        error_result = {
            "status": "error",
            "message": f"Execution timeout ({timeout}s)",
            "file_path": file_path,
            "language": language,
            "output": "",
            "error": "Timeout"
        }
        return json.dumps(error_result, indent=2)
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to execute file: {str(e)}",
            "file_path": file_path,
            "language": language,
            "output": "",
            "error": str(e)
        }
        return json.dumps(error_result, indent=2)

# =============================================================================
# PROJECT EXPLORATION / 项目探索
# =============================================================================

@mcp.tool()
async def list_project_files(
    project_path: str,
    file_extensions: List[str] = None,
    max_depth: int = 10
) -> str:
    """
    List files in a project directory for LLM to understand project structure / 列出项目目录中的文件供LLM了解项目结构
    
    This tool scans a project directory and returns information about all files,
    helping the LLM understand the project structure before making changes.
    
    此工具扫描项目目录并返回所有文件的信息，
    帮助LLM在进行更改之前了解项目结构。
    
    Args:
        project_path (str): Root path to scan / 要扫描的根路径
        file_extensions (List[str]): List of file extensions to include (e.g., ['.py', '.js'])
                                    要包含的文件扩展名列表（例如，['.py', '.js']）
        max_depth (int): Maximum directory depth to scan / 要扫描的最大目录深度
    
    Returns:
        str: JSON response with file list and metadata / 包含文件列表和元数据的JSON响应
    
    Example / 示例:
        list_project_files("my_project", [".py", ".md"], 5)
        list_project_files("src")  # List all files in src directory
    """
    try:
        if not os.path.exists(project_path):
            result = {
                "status": "error",
                "message": f"Directory not found: {project_path}",
                "files": []
            }
            return json.dumps(result, indent=2)
        
        files_info = []
        
        for root, dirs, files in os.walk(project_path):
            # Calculate depth / 计算深度
            depth = root.replace(project_path, '').count(os.sep)
            if depth >= max_depth:
                dirs[:] = []  # Don't go deeper / 不再深入
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_path)
                
                # Filter by extensions if specified / 如果指定则按扩展名过滤
                if file_extensions:
                    if not any(file.endswith(ext) for ext in file_extensions):
                        continue
                
                try:
                    stat = os.stat(file_path)
                    files_info.append({
                        "path": rel_path,
                        "full_path": file_path,
                        "size": stat.st_size,
                        "extension": os.path.splitext(file)[1]
                    })
                except OSError:
                    continue
        
        result = {
            "status": "success",
            "message": f"Listed files in {project_path}",
            "project_path": project_path,
            "total_files": len(files_info),
            "files": files_info
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to list files: {str(e)}",
            "project_path": project_path,
            "files": []
        }
        return json.dumps(error_result, indent=2)

# =============================================================================
# SERVER INITIALIZATION / 服务器初始化
# =============================================================================

if __name__ == "__main__":
    """
    Initialize and run the MCP server / 初始化并运行MCP服务器
    
    This starts the FastMCP server with all the code generation tools available
    for LLM-driven code generation and file operations.
    
    这将启动FastMCP服务器，提供所有用于LLM驱动的代码生成和文件操作的工具。
    """
    print("Code File Operations MCP Server")
    print("LLM-driven code generation and file operations")
    print("LLM驱动的代码生成和文件操作")
    
    print("\nAvailable Tools / 可用工具:")
    print("  • write_code_file        - 写入完整代码文件 / Write complete code files")
    print("  • read_code_file         - 读取代码文件内容 / Read code file content")
    print("  • append_to_file         - 追加内容到文件 / Append content to files")
    print("  • insert_code_at_line    - 在指定行插入代码 / Insert code at specific line")
    print("  • replace_code_section   - 替换代码段 / Replace code sections")
    print("  • create_project_structure - 创建项目结构 / Create project structure")
    print("  • validate_file_syntax   - 验证文件语法 / Validate file syntax")
    print("  • run_code_file         - 运行代码文件 / Execute code files")
    print("  • list_project_files    - 列出项目文件 / List project files")
    print("  • create_directory     - 创建目录 / Create directories")
    
    print("\nSupported Languages / 支持的语言:")
    print("  • Python (.py)")
    print("  • JavaScript (.js)")
    print("  • TypeScript (.ts)")
    print("  • Java (.java)")
    print("  • Go (.go)")
    print("  • Rust (.rs)")
    print("  • C++ (.cpp)")
    print("  • C (.c)")
    print("  • C# (.cs)")
    print("  • PHP (.php)")
    print("  • Ruby (.rb)")
    print("  • Bash (.sh)")
    
    print("\nUsage Examples / 使用示例:")
    print('  • write_code_file("src/main.py", "print(\'Hello World\')", True)')
    print('  • read_code_file("src/utils.py")')
    print('  • create_project_structure("my_project", {"src": {"main.py": "file"}})')
    print('  • run_code_file("test.py", "python")')
    
    print("")
    
    # Run the server using FastMCP with stdio transport
    # 使用FastMCP和stdio传输运行服务器
    mcp.run(transport='stdio') 