"""
MCP工具定义配置模块 - 代码评估专用
MCP Tool Definitions Configuration Module - Code Evaluation Specific

为代码评估智能体提供专门的工具定义
Provides specialized tool definitions for code evaluation agents

支持的评估工具类型：
- 仓库结构分析工具 (Repository Structure Analysis)
- 依赖检测工具 (Dependency Detection)
- 代码质量评估工具 (Code Quality Assessment)
- 文档评估工具 (Documentation Evaluation)
- Docker环境管理工具 (Docker Environment Management)
"""

from typing import Dict, List, Any


class MCPEvaluationToolDefinitions:
    """MCP代码评估工具定义管理器"""

    @staticmethod
    def get_code_evaluation_tools() -> List[Dict[str, Any]]:
        """
        获取代码评估相关的工具定义
        Get tool definitions for code evaluation
        """
        return [
            MCPEvaluationToolDefinitions._get_analyze_repo_structure_tool(),
            MCPEvaluationToolDefinitions._get_detect_dependencies_tool(),
            MCPEvaluationToolDefinitions._get_assess_code_quality_tool(),
            MCPEvaluationToolDefinitions._get_evaluate_documentation_tool(),
            MCPEvaluationToolDefinitions._get_check_reproduction_readiness_tool(),
            MCPEvaluationToolDefinitions._get_generate_evaluation_summary_tool(),
        ]

    @staticmethod
    def get_docker_management_tools() -> List[Dict[str, Any]]:
        """
        获取Docker管理相关的工具定义
        Get tool definitions for Docker management
        """
        return [
            MCPEvaluationToolDefinitions._get_create_evaluation_container_tool(),
            MCPEvaluationToolDefinitions._get_setup_container_workspace_tool(),
            MCPEvaluationToolDefinitions._get_setup_conda_environment_tool(),
            MCPEvaluationToolDefinitions._get_install_dependencies_tool(),
            MCPEvaluationToolDefinitions._get_execute_in_container_tool(),
            MCPEvaluationToolDefinitions._get_monitor_container_resources_tool(),
            MCPEvaluationToolDefinitions._get_cleanup_container_tool(),
            MCPEvaluationToolDefinitions._get_list_evaluation_containers_tool(),
            MCPEvaluationToolDefinitions._get_read_file_in_container_tool(),
            MCPEvaluationToolDefinitions._get_write_file_in_container_tool(),
            MCPEvaluationToolDefinitions._get_list_files_in_container_tool(),
            MCPEvaluationToolDefinitions._get_analyze_repo_structure_in_container_tool(),
        ]

    # ==================== 代码评估工具定义 ====================

    @staticmethod
    def _get_analyze_repo_structure_tool() -> Dict[str, Any]:
        """分析仓库结构工具定义"""
        return {
            "name": "analyze_repo_structure",
            "description": "Analyze repository structure, file types, organization, and detect programming languages and frameworks",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the repository to analyze",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum directory depth to analyze",
                        "default": 10,
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Whether to include hidden files and directories",
                        "default": False,
                    },
                },
                "required": ["repo_path"],
            },
        }

    @staticmethod
    def _get_detect_dependencies_tool() -> Dict[str, Any]:
        """检测依赖工具定义"""
        return {
            "name": "detect_dependencies",
            "description": "Detect and analyze project dependencies across different languages (Python, JavaScript, etc.)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the repository to analyze",
                    },
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific languages to focus on (optional, auto-detected if not provided)",
                        "default": [],
                    },
                },
                "required": ["repo_path"],
            },
        }

    @staticmethod
    def _get_assess_code_quality_tool() -> Dict[str, Any]:
        """评估代码质量工具定义"""
        return {
            "name": "assess_code_quality",
            "description": "Assess code quality including complexity, maintainability, potential issues, and test coverage",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the repository to analyze",
                    },
                    "include_tests": {
                        "type": "boolean",
                        "description": "Whether to include test files in analysis",
                        "default": True,
                    },
                    "complexity_threshold": {
                        "type": "integer",
                        "description": "Threshold for reporting high complexity functions",
                        "default": 10,
                    },
                },
                "required": ["repo_path"],
            },
        }

    @staticmethod
    def _get_evaluate_documentation_tool() -> Dict[str, Any]:
        """评估文档工具定义"""
        return {
            "name": "evaluate_documentation",
            "description": "Evaluate documentation completeness, quality, and reproduction readiness",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the repository to analyze",
                    },
                    "docs_path": {
                        "type": "string",
                        "description": "Path to specific documentation file (optional)",
                        "default": None,
                    },
                    "check_api_docs": {
                        "type": "boolean",
                        "description": "Whether to check for API documentation",
                        "default": True,
                    },
                },
                "required": ["repo_path"],
            },
        }

    @staticmethod
    def _get_check_reproduction_readiness_tool() -> Dict[str, Any]:
        """检查复现准备度工具定义"""
        return {
            "name": "check_reproduction_readiness",
            "description": "Check if repository is ready for reproduction based on documentation, dependencies, and setup instructions",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the repository to analyze",
                    },
                    "docs_path": {
                        "type": "string",
                        "description": "Path to reproduction documentation (optional)",
                        "default": None,
                    },
                    "check_environment": {
                        "type": "boolean",
                        "description": "Whether to check environment setup requirements",
                        "default": True,
                    },
                },
                "required": ["repo_path"],
            },
        }

    @staticmethod
    def _get_generate_evaluation_summary_tool() -> Dict[str, Any]:
        """生成评估摘要工具定义"""
        return {
            "name": "generate_evaluation_summary",
            "description": "Generate comprehensive evaluation summary combining all analysis results",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the repository to analyze",
                    },
                    "docs_path": {
                        "type": "string",
                        "description": "Path to reproduction documentation (optional)",
                        "default": None,
                    },
                    "include_recommendations": {
                        "type": "boolean",
                        "description": "Whether to include improvement recommendations",
                        "default": True,
                    },
                },
                "required": ["repo_path"],
            },
        }

    # ==================== Docker管理工具定义 ====================

    @staticmethod
    def _get_create_evaluation_container_tool() -> Dict[str, Any]:
        """创建评估容器工具定义"""
        return {
            "name": "create_evaluation_container",
            "description": "Create a Docker container for safe code evaluation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "base_image": {
                        "type": "string",
                        "description": "Base Docker image to use",
                        "default": "python:3.9-slim",
                    },
                    "container_name": {
                        "type": "string",
                        "description": "Name for the container (optional, auto-generated if not provided)",
                        "default": None,
                    },
                    "memory_limit": {
                        "type": "string",
                        "description": "Memory limit for container (e.g., '512m', '1g')",
                        "default": "1g",
                    },
                },
                "required": [],
            },
        }

    @staticmethod
    def _get_setup_container_workspace_tool() -> Dict[str, Any]:
        """设置容器工作空间工具定义"""
        return {
            "name": "setup_container_workspace",
            "description": "Mount repository into container and setup workspace",
            "input_schema": {
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Container ID or name",
                    },
                    "repo_path": {
                        "type": "string",
                        "description": "Local repository path to mount",
                    },
                    "workspace_path": {
                        "type": "string",
                        "description": "Path inside container where repo will be mounted",
                        "default": "/workspace",
                    },
                },
                "required": ["container_id", "repo_path"],
            },
        }

    @staticmethod
    def _get_setup_conda_environment_tool() -> Dict[str, Any]:
        """Setup conda environment tool definition"""
        return {
            "name": "setup_conda_environment",
            "description": "Setup conda environment in container based on grader.Dockerfile approach",
            "input_schema": {
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Docker container ID",
                    },
                    "python_version": {
                        "type": "string",
                        "description": "Python version to install in conda environment",
                        "default": "3.12",
                    },
                    "env_name": {
                        "type": "string",
                        "description": "Name of conda environment to create",
                        "default": "grader",
                    },
                },
                "required": ["container_id"],
            },
        }

    @staticmethod
    def _get_install_dependencies_tool() -> Dict[str, Any]:
        """安装依赖工具定义"""
        return {
            "name": "install_dependencies",
            "description": "Install project dependencies inside container",
            "input_schema": {
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Container ID or name",
                    },
                    "requirements_file": {
                        "type": "string",
                        "description": "Requirements file path (e.g., requirements.txt, package.json)",
                        "default": "requirements.txt",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (python, nodejs, etc.)",
                        "default": "python",
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Working directory inside container for dependency installation",
                        "default": "/root/workbase",
                    },
                },
                "required": ["container_id"],
            },
        }

    @staticmethod
    def _get_execute_in_container_tool() -> Dict[str, Any]:
        """容器内执行工具定义"""
        return {
            "name": "execute_in_container",
            "description": "Execute commands inside container",
            "input_schema": {
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Container ID or name",
                    },
                    "command": {
                        "type": "string",
                        "description": "Command to execute",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory inside container",
                        "default": "/workspace",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                        "default": 60,
                    },
                },
                "required": ["container_id", "command"],
            },
        }

    @staticmethod
    def _get_monitor_container_resources_tool() -> Dict[str, Any]:
        """监控容器资源工具定义"""
        return {
            "name": "monitor_container_resources",
            "description": "Monitor container resource usage (CPU, memory, etc.)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Container ID or name",
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Monitoring duration in seconds",
                        "default": 10,
                    },
                },
                "required": ["container_id"],
            },
        }

    @staticmethod
    def _get_cleanup_container_tool() -> Dict[str, Any]:
        """清理容器工具定义"""
        return {
            "name": "cleanup_container",
            "description": "Stop and remove container",
            "input_schema": {
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Container ID or name",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force removal even if container is running",
                        "default": True,
                    },
                },
                "required": ["container_id"],
            },
        }

    @staticmethod
    def _get_list_evaluation_containers_tool() -> Dict[str, Any]:
        """列出容器工具定义"""
        return {
            "name": "list_evaluation_containers",
            "description": "List Docker containers and available images",
            "input_schema": {
                "type": "object",
                "properties": {
                    "show_all": {
                        "type": "boolean",
                        "description": "Show all containers including stopped ones",
                        "default": True,
                    },
                    "show_images": {
                        "type": "boolean",
                        "description": "Also show available Docker images",
                        "default": True,
                    },
                },
                "required": [],
            },
        }

    @staticmethod
    def _get_read_file_in_container_tool() -> Dict[str, Any]:
        """容器内文件读取工具定义"""
        return {
            "name": "read_file_in_container",
            "description": "Read a file from inside the Docker container",
            "input_schema": {
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Container ID or name",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to file inside container (relative to working_dir)",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory in container",
                        "default": "/workspace/repo",
                    },
                },
                "required": ["container_id", "file_path"],
            },
        }

    @staticmethod
    def _get_write_file_in_container_tool() -> Dict[str, Any]:
        """容器内文件写入工具定义"""
        return {
            "name": "write_file_in_container",
            "description": "Write content to a file inside the Docker container",
            "input_schema": {
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Container ID or name",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to file inside container (relative to working_dir)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory in container",
                        "default": "/workspace/repo",
                    },
                    "create_backup": {
                        "type": "boolean",
                        "description": "Whether to create a backup before writing",
                        "default": True,
                    },
                },
                "required": ["container_id", "file_path", "content"],
            },
        }

    @staticmethod
    def _get_list_files_in_container_tool() -> Dict[str, Any]:
        """容器内文件列表工具定义"""
        return {
            "name": "list_files_in_container",
            "description": "List files in a directory inside the Docker container",
            "input_schema": {
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Container ID or name",
                    },
                    "directory_path": {
                        "type": "string",
                        "description": "Directory to list",
                        "default": "/workspace/repo",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list recursively",
                        "default": False,
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by file extensions (e.g., [\".py\", \".js\"])",
                        "default": None,
                    },
                },
                "required": ["container_id"],
            },
        }

    @staticmethod
    def _get_analyze_repo_structure_in_container_tool() -> Dict[str, Any]:
        """容器内代码仓库结构分析工具定义"""
        return {
            "name": "analyze_repo_structure_in_container",
            "description": "Analyze repository structure inside the container to understand how to run the code",
            "input_schema": {
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Container ID or name",
                    },
                    "repo_path": {
                        "type": "string",
                        "description": "Path to repository in container",
                        "default": "/workspace/repo",
                    },
                },
                "required": ["container_id"],
            },
        }

    @staticmethod
    def get_available_tool_sets() -> Dict[str, str]:
        """
        获取可用的工具集合
        Get available tool sets
        """
        return {
            "code_evaluation": "代码评估工具集 / Code evaluation tool set",
            "docker_management": "Docker管理工具集 / Docker management tool set",
        }

    @staticmethod
    def get_tool_set(tool_set_name: str) -> List[Dict[str, Any]]:
        """
        根据名称获取特定的工具集
        Get specific tool set by name
        """
        tool_sets = {
            "code_evaluation": MCPEvaluationToolDefinitions.get_code_evaluation_tools(),
            "docker_management": MCPEvaluationToolDefinitions.get_docker_management_tools(),
        }

        return tool_sets.get(tool_set_name, [])

    @staticmethod
    def get_all_evaluation_tools() -> List[Dict[str, Any]]:
        """
        获取所有评估相关工具
        Get all evaluation related tools
        """
        all_tools = []
        for tool_set_name in MCPEvaluationToolDefinitions.get_available_tool_sets().keys():
            all_tools.extend(MCPEvaluationToolDefinitions.get_tool_set(tool_set_name))
        return all_tools


# 便捷访问函数
def get_evaluation_mcp_tools(tool_set: str = "code_evaluation") -> List[Dict[str, Any]]:
    """
    便捷函数：获取评估相关的MCP工具定义
    Convenience function: Get evaluation MCP tool definitions

    Args:
        tool_set: 工具集名称 (默认: "code_evaluation")

    Returns:
        工具定义列表
    """
    return MCPEvaluationToolDefinitions.get_tool_set(tool_set)


def get_all_evaluation_tools() -> List[Dict[str, Any]]:
    """
    便捷函数：获取所有评估工具定义
    Convenience function: Get all evaluation tool definitions

    Returns:
        所有评估工具定义列表
    """
    return MCPEvaluationToolDefinitions.get_all_evaluation_tools()
