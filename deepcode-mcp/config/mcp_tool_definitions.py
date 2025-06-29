"""
MCP工具定义配置模块
MCP Tool Definitions Configuration Module

将工具定义从主程序逻辑中分离，提供标准化的工具定义格式
Separate tool definitions from main program logic, providing standardized tool definition format

支持的工具类型：
- 文件操作工具 (File Operations)
- 代码执行工具 (Code Execution)
- 搜索工具 (Search Tools)
- 项目结构工具 (Project Structure Tools)
"""

from typing import Dict, List, Any


class MCPToolDefinitions:
    """MCP工具定义管理器"""

    @staticmethod
    def get_code_implementation_tools() -> List[Dict[str, Any]]:
        """
        获取代码实现相关的工具定义
        Get tool definitions for code implementation
        """
        return [
            MCPToolDefinitions._get_read_file_tool(),
            MCPToolDefinitions._get_write_file_tool(),
            MCPToolDefinitions._get_execute_python_tool(),
            MCPToolDefinitions._get_execute_bash_tool(),
            MCPToolDefinitions._get_search_code_tool(),
            MCPToolDefinitions._get_file_structure_tool(),
            MCPToolDefinitions._get_search_reference_code_tool(),
        ]

    @staticmethod
    def _get_read_file_tool() -> Dict[str, Any]:
        """读取文件工具定义"""
        return {
            "name": "read_file",
            "description": "Read file content, supports specifying line number range",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path, relative to workspace",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Start line number (starting from 1, optional)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "End line number (starting from 1, optional)",
                    },
                },
                "required": ["file_path"],
            },
        }

    @staticmethod
    def _get_write_file_tool() -> Dict[str, Any]:
        """写入文件工具定义"""
        return {
            "name": "write_file",
            "description": "Write content to file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path, relative to workspace",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to file",
                    },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Whether to create directories if they don't exist",
                        "default": True,
                    },
                    "create_backup": {
                        "type": "boolean",
                        "description": "Whether to create backup file if file already exists",
                        "default": False,
                    },
                },
                "required": ["file_path", "content"],
            },
        }

    @staticmethod
    def _get_execute_python_tool() -> Dict[str, Any]:
        """Python执行工具定义"""
        return {
            "name": "execute_python",
            "description": "Execute Python code and return output",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                        "default": 30,
                    },
                },
                "required": ["code"],
            },
        }

    @staticmethod
    def _get_execute_bash_tool() -> Dict[str, Any]:
        """Bash执行工具定义"""
        return {
            "name": "execute_bash",
            "description": "Execute bash command",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                        "default": 30,
                    },
                },
                "required": ["command"],
            },
        }

    @staticmethod
    def _get_search_code_tool() -> Dict[str, Any]:
        """代码搜索工具定义"""
        return {
            "name": "search_code",
            "description": "Search for patterns in code files",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern"},
                    "file_pattern": {
                        "type": "string",
                        "description": "File pattern (e.g., '*.py')",
                        "default": "*.py",
                    },
                    "use_regex": {
                        "type": "boolean",
                        "description": "Whether to use regular expressions",
                        "default": False,
                    },
                },
                "required": ["pattern"],
            },
        }

    @staticmethod
    def _get_file_structure_tool() -> Dict[str, Any]:
        """文件结构获取工具定义"""
        return {
            "name": "get_file_structure",
            "description": "Get directory file structure",
            "input_schema": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path, relative to workspace",
                        "default": ".",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum traversal depth",
                        "default": 5,
                    },
                },
            },
        }

    @staticmethod
    def _get_search_reference_code_tool() -> Dict[str, Any]:
        """代码参考搜索工具定义"""
        return {
            "name": "search_reference_code",
            "description": "Search relevant reference code from indexes folder for implementation guidance",
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_file": {
                        "type": "string",
                        "description": "Target file path to be implemented"
                    },
                    "keywords": {
                        "type": "string",
                        "description": "Search keywords, comma-separated",
                        "default": ""
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    }
                },
                "required": ["target_file"]
            }
        }
    
    @staticmethod
    def get_available_tool_sets() -> Dict[str, str]:
        """
        获取可用的工具集合
        Get available tool sets
        """
        return {
            "code_implementation": "代码实现相关工具集 / Code implementation tool set",
            # 可以在这里添加更多工具集
            # "data_analysis": "数据分析工具集 / Data analysis tool set",
            # "web_scraping": "网页爬取工具集 / Web scraping tool set",
        }

    @staticmethod
    def get_tool_set(tool_set_name: str) -> List[Dict[str, Any]]:
        """
        根据名称获取特定的工具集
        Get specific tool set by name
        """
        tool_sets = {
            "code_implementation": MCPToolDefinitions.get_code_implementation_tools(),
        }

        return tool_sets.get(tool_set_name, [])

    @staticmethod
    def get_all_tools() -> List[Dict[str, Any]]:
        """
        获取所有可用工具
        Get all available tools
        """
        all_tools = []
        for tool_set_name in MCPToolDefinitions.get_available_tool_sets().keys():
            all_tools.extend(MCPToolDefinitions.get_tool_set(tool_set_name))
        return all_tools


# 便捷访问函数
def get_mcp_tools(tool_set: str = "code_implementation") -> List[Dict[str, Any]]:
    """
    便捷函数：获取MCP工具定义
    Convenience function: Get MCP tool definitions

    Args:
        tool_set: 工具集名称 (默认: "code_implementation")

    Returns:
        工具定义列表
    """
    return MCPToolDefinitions.get_tool_set(tool_set)
