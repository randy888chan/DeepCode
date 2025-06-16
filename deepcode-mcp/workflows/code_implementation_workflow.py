"""
论文代码复现工作流 - 基于MCP标准的迭代式开发
Paper Code Implementation Workflow - MCP-compliant Iterative Development

实现论文代码复现的完整工作流：
1. 文件树创建 (File Tree Creation)
2. 代码实现 (Code Implementation) - 基于aisi-basic-agent的迭代式开发

使用标准MCP架构：
- MCP服务器：tools/code_implementation_server.py
- MCP客户端：通过mcp_agent框架调用
- 配置文件：mcp_agent.config.yaml
"""

import asyncio
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import json
import time

# 导入MCP代理相关模块
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM

# 导入提示词
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.code_prompts import STRUCTURE_GENERATOR_PROMPT
from prompts.iterative_code_prompts import (
    ITERATIVE_CODE_SYSTEM_PROMPT, 
    CONTINUE_CODE_MESSAGE,
    INITIAL_ANALYSIS_PROMPT,
    COMPLETION_CHECK_PROMPT,
    ERROR_HANDLING_PROMPT,
    TOOL_USAGE_EXAMPLES
)


class CodeImplementationWorkflow:
    """
    论文代码复现工作流管理器
    
    使用标准MCP架构：
    1. 通过MCP客户端连接到code-implementation服务器
    2. 使用MCP协议进行工具调用
    3. 支持工作空间管理和操作历史追踪
    """
    
    def __init__(self, config_path: str = "mcp_agent.secrets.yaml"):
        self.config_path = config_path
        self.api_config = self._load_api_config()
        self.logger = self._create_logger()
        self.mcp_agent = None
    
    def _load_api_config(self) -> Dict[str, Any]:
        """加载API配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"无法加载API配置文件: {e}")

    def _create_logger(self) -> logging.Logger:
        """创建日志记录器"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _read_plan_file(self, plan_file_path: str) -> str:
        """读取计划文件"""
        plan_path = Path(plan_file_path)
        if not plan_path.exists():
            raise FileNotFoundError(f"实现计划文件不存在: {plan_file_path}")
        
        with open(plan_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _check_file_tree_exists(self, target_directory: str) -> bool:
        """检查文件树是否已存在"""
        code_directory = os.path.join(target_directory, "generate_code")
        return os.path.exists(code_directory) and len(os.listdir(code_directory)) > 0

    async def _initialize_mcp_agent(self, code_directory: str):
        """初始化MCP代理，连接到code-implementation服务器"""
        try:
            # 创建连接到code-implementation服务器的代理
            self.mcp_agent = Agent(
                name="CodeImplementationAgent",
                instruction="You are a code implementation assistant, using MCP tools to implement paper code replication.",
                server_names=["code-implementation"],  # 连接到我们的MCP服务器
            )
            
            # 启动代理连接（不使用上下文管理器，手动管理生命周期）
            await self.mcp_agent.__aenter__()
            
            # 初始化LLM
            llm = await self.mcp_agent.attach_llm(AnthropicAugmentedLLM)
            
            # 设置工作空间
            workspace_result = await self.mcp_agent.call_tool(
                "set_workspace", 
                {"workspace_path": code_directory}
            )
            self.logger.info(f"工作空间设置结果: {workspace_result}")
            
            return llm
                
        except Exception as e:
            self.logger.error(f"初始化MCP代理失败: {e}")
            # 如果初始化失败，确保清理资源
            if self.mcp_agent:
                try:
                    await self.mcp_agent.__aexit__(None, None, None)
                except:
                    pass
                self.mcp_agent = None
            raise

    async def _cleanup_mcp_agent(self):
        """清理MCP代理资源"""
        if self.mcp_agent:
            try:
                await self.mcp_agent.__aexit__(None, None, None)
                self.logger.info("MCP代理连接已关闭")
            except Exception as e:
                self.logger.warning(f"关闭MCP代理连接时出错: {e}")
            finally:
                self.mcp_agent = None

    # ==================== 文件树创建流程 ====================
    
    async def create_file_structure(self, plan_content: str, target_directory: str) -> str:
        """创建文件树结构"""
        self.logger.info("开始创建文件树结构...")
        
        # 创建文件结构生成代理
        structure_agent = Agent(
            name="StructureGeneratorAgent",
            instruction=STRUCTURE_GENERATOR_PROMPT,
            server_names=["command-executor"],
        )
        
        async with structure_agent:
            creator = await structure_agent.attach_llm(AnthropicAugmentedLLM)
            
            message = f"""Analyze the following implementation plan and generate shell commands to create the file tree structure.

Target Directory: {target_directory}/generate_code

Implementation Plan:
{plan_content}

Tasks:
1. Find the file tree structure in the implementation plan
2. Generate shell commands (mkdir -p, touch) to create that structure
3. Use the execute_commands tool to run the commands and create the file structure

Requirements:
- Use mkdir -p to create directories
- Use touch to create files
- Include __init__.py file for Python packages
- Use relative paths to the target directory
- Execute commands to actually create the file structure"""
            
            result = await creator.generate_str(message=message)
            self.logger.info("文件树结构创建完成")
            return result

    # ==================== 代码实现流程 ====================
    
    async def implement_code(self, plan_content: str, target_directory: str) -> str:
        """迭代式代码实现 - 使用MCP服务器"""
        self.logger.info("开始迭代式代码实现...")
        
        code_directory = os.path.join(target_directory, "generate_code")
        if not os.path.exists(code_directory):
            raise FileNotFoundError("文件树结构不存在，请先运行文件树创建")
        
        try:
            # 初始化LLM客户端
            client, client_type = await self._initialize_llm_client()
            
            # 初始化MCP代理
            mcp_llm = await self._initialize_mcp_agent(code_directory)
            
            # 准备工具定义 (MCP标准格式)
            tools = self._prepare_mcp_tool_definitions()
            
            # 初始化对话
            system_message = ITERATIVE_CODE_SYSTEM_PROMPT + "\n\n" + TOOL_USAGE_EXAMPLES
            messages = []
            
            # 初始分析消息
            initial_message = f"""Working Directory: {code_directory}

Implementation Plan:
{plan_content}

{INITIAL_ANALYSIS_PROMPT}

Note: Use the get_file_structure tool to explore the current project structure and understand what files already exist."""
            
            messages.append({"role": "user", "content": initial_message})
            
            # 迭代开发循环
            result = await self._iterative_development_loop(
                client, client_type, system_message, messages, tools
            )
            
            return result
            
        finally:
            # 确保清理MCP代理资源
            await self._cleanup_mcp_agent()

    async def _get_file_structure_overview(self) -> str:
        """获取文件结构概览（轻量级，仅显示主要目录和文件数量）"""
        try:
            if not self.mcp_agent:
                return "MCP agent not initialized"
            
            # 获取浅层文件结构（深度限制为2）
            result = await self.mcp_agent.call_tool("get_file_structure", {
                "directory": ".", 
                "max_depth": 2
            })
            
            # 解析结果并生成概览
            import json
            try:
                data = json.loads(result) if isinstance(result, str) else result
                if data.get("status") == "success":
                    summary = data.get("summary", {})
                    return f"""File Structure Overview:
- Total files: {summary.get('total_files', 0)}
- Total directories: {summary.get('total_directories', 0)}
- Scan depth: 2 levels (overview mode)

💡 Tip: Use the get_file_structure tool to get complete real-time file structure"""
                else:
                    return f"Failed to get file structure overview: {data.get('message', 'unknown error')}"
            except json.JSONDecodeError:
                return f"File structure data: {result}"
                
        except Exception as e:
            self.logger.error(f"获取文件结构概览失败: {e}")
            return f"Error getting file structure overview: {str(e)}"

    async def _get_file_structure_via_mcp(self) -> str:
        """通过MCP获取文件结构（保留原方法以兼容性）"""
        try:
            if self.mcp_agent:
                result = await self.mcp_agent.call_tool("get_file_structure", {"directory": ".", "max_depth": 5})
                return f"File Structure:\n{result}"
            else:
                return "MCP agent not initialized"
        except Exception as e:
            self.logger.error(f"获取文件结构失败: {e}")
            return f"Error getting file structure: {str(e)}"

    async def _initialize_llm_client(self):
        """初始化LLM客户端"""
        # 尝试Anthropic API
        try:
            anthropic_key = self.api_config.get('anthropic', {}).get('api_key')
            if anthropic_key:
                from anthropic import AsyncAnthropic
                client = AsyncAnthropic(api_key=anthropic_key)
                # 测试连接
                await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "test"}]
                )
                self.logger.info("使用Anthropic API")
                return client, "anthropic"
        except Exception as e:
            self.logger.warning(f"Anthropic API不可用: {e}")
        
        # 尝试OpenAI API
        try:
            openai_key = self.api_config.get('openai', {}).get('api_key')
            if openai_key:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=openai_key)
                # 测试连接
                await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "test"}]
                )
                self.logger.info("使用OpenAI API")
                return client, "openai"
        except Exception as e:
            self.logger.warning(f"OpenAI API不可用: {e}")
        
        raise ValueError("没有可用的LLM API")

    async def _iterative_development_loop(self, client, client_type, system_message, messages, tools):
        """迭代开发循环 - 使用MCP工具调用"""
        max_iterations = 50
        iteration = 0
        start_time = time.time()
        max_time = 3600  # 1小时
        
        while iteration < max_iterations:
            iteration += 1
            elapsed_time = time.time() - start_time
            
            if elapsed_time > max_time:
                self.logger.warning(f"达到时间限制: {elapsed_time:.2f}s")
                break
            
            if iteration % 5 == 0:
                progress_msg = f"\n[Progress Update] Iteration {iteration}, Time elapsed: {elapsed_time:.2f}s / {max_time}s"
                messages.append({"role": "user", "content": progress_msg})
            
            self.logger.info(f"迭代 {iteration}: 生成响应")
            
            # 调用LLM
            response = await self._call_llm_with_tools(
                client, client_type, system_message, messages, tools
            )
            
            messages.append({"role": "assistant", "content": response["content"]})
            
            # 处理工具调用 - 使用MCP
            if response.get("tool_calls"):
                tool_results = await self._execute_mcp_tool_calls(response["tool_calls"])
                
                for tool_result in tool_results:
                    messages.append({
                        "role": "user",
                        "content": f"Tool Result {tool_result['tool_name']}:\n{tool_result['result']}"
                    })
                
                if any("error" in result['result'] for result in tool_results):
                    messages.append({"role": "user", "content": ERROR_HANDLING_PROMPT})
            else:
                messages.append({"role": "user", "content": CONTINUE_CODE_MESSAGE})
            
            # 检查完成
            if "implementation is complete" in response["content"].lower():
                self.logger.info("代码实现声明完成")
                messages.append({"role": "user", "content": COMPLETION_CHECK_PROMPT})
                final_response = await self._call_llm_with_tools(
                    client, client_type, system_message, messages, tools
                )
                if "complete" in final_response["content"].lower():
                    break
            
            # 防止消息历史过长
            if len(messages) > 100:
                messages = messages[:1] + messages[-50:]
                self.logger.info("裁剪消息历史")
        
        return await self._generate_final_report_via_mcp(iteration, time.time() - start_time)
    
    def _prepare_mcp_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        准备Anthropic API标准格式的工具定义
        """
        return [
            {
                "name": "read_file",
                "description": "Read file content, supports specifying line number range",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string", 
                            "description": "File path, relative to workspace"
                        },
                        "start_line": {
                            "type": "integer", 
                            "description": "Start line number (starting from 1, optional)"
                        },
                        "end_line": {
                            "type": "integer", 
                            "description": "End line number (starting from 1, optional)"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "write_file",
                "description": "Write content to file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string", 
                            "description": "File path, relative to workspace"
                        },
                        "content": {
                            "type": "string", 
                            "description": "Content to write to file"
                        },
                        "create_dirs": {
                            "type": "boolean", 
                            "description": "Whether to create directories if they don't exist",
                            "default": True
                        },
                        "create_backup": {
                            "type": "boolean", 
                            "description": "Whether to create backup file if file already exists",
                            "default": False
                        }
                    },
                    "required": ["file_path", "content"]
                }
            },
            {
                "name": "execute_python",
                "description": "Execute Python code and return output",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string", 
                            "description": "Python code to execute"
                        },
                        "timeout": {
                            "type": "integer", 
                            "description": "Timeout in seconds",
                            "default": 30
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "execute_bash",
                "description": "Execute bash command",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string", 
                            "description": "Bash command to execute"
                        },
                        "timeout": {
                            "type": "integer", 
                            "description": "Timeout in seconds",
                            "default": 30
                        }
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "search_code",
                "description": "Search for patterns in code files",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string", 
                            "description": "Search pattern"
                        },
                        "file_pattern": {
                            "type": "string", 
                            "description": "File pattern (e.g., '*.py')",
                            "default": "*.py"
                        },
                        "use_regex": {
                            "type": "boolean", 
                            "description": "Whether to use regular expressions",
                            "default": False
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "get_file_structure",
                "description": "Get directory file structure",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string", 
                            "description": "Directory path, relative to workspace",
                            "default": "."
                        },
                        "max_depth": {
                            "type": "integer", 
                            "description": "Maximum traversal depth",
                            "default": 5
                        }
                    }
                }
            }
        ]
    
    async def _call_llm_with_tools(self, client, client_type, system_message, messages, tools, max_tokens=4096):
        """调用LLM"""
        try:
            if client_type == "anthropic":
                return await self._call_anthropic_with_tools(client, system_message, messages, tools, max_tokens)
            elif client_type == "openai":
                return await self._call_openai_with_tools(client, system_message, messages, tools, max_tokens)
            else:
                raise ValueError(f"不支持的客户端类型: {client_type}")
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            raise
    
    async def _call_anthropic_with_tools(self, client, system_message, messages, tools, max_tokens):
        """调用Anthropic API"""
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            system=system_message,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=0.2
        )
        
        content = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        
        return {"content": content, "tool_calls": tool_calls}
    
    async def _call_openai_with_tools(self, client, system_message, messages, tools, max_tokens):
        """调用OpenAI API"""
        # 转换MCP工具格式为OpenAI格式
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            })
        
        openai_messages = [{"role": "system", "content": system_message}]
        openai_messages.extend(messages)
        
        response = await client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=openai_messages,
            tools=openai_tools if openai_tools else None,
            max_tokens=max_tokens,
            temperature=0.2
        )
        
        message = response.choices[0].message
        content = message.content or ""
        
        tool_calls = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": json.loads(tool_call.function.arguments)
                })
        
        return {"content": content, "tool_calls": tool_calls}
    
    async def _execute_mcp_tool_calls(self, tool_calls):
        """
        通过MCP协议执行工具调用
        
        这是标准的MCP实现方式，通过MCP代理调用服务器工具
        """
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]
            
            self.logger.info(f"执行MCP工具: {tool_name}")
            
            try:
                if self.mcp_agent:
                    # 通过MCP协议调用工具
                    result = await self.mcp_agent.call_tool(tool_name, tool_input)
                    
                    results.append({
                        "tool_id": tool_call["id"],
                        "tool_name": tool_name,
                        "result": result
                    })
                else:
                    results.append({
                        "tool_id": tool_call["id"],
                        "tool_name": tool_name,
                        "result": json.dumps({
                            "status": "error",
                            "message": "MCP agent not initialized"
                        }, ensure_ascii=False)
                    })
                
            except Exception as e:
                self.logger.error(f"MCP工具执行失败: {e}")
                results.append({
                    "tool_id": tool_call["id"],
                    "tool_name": tool_name,
                    "result": json.dumps({
                        "status": "error",
                        "message": str(e)
                    }, ensure_ascii=False)
                })
        
        return results
    
    async def _generate_final_report_via_mcp(self, iterations: int, elapsed_time: float):
        """通过MCP生成最终报告"""
        try:
            # 获取操作历史
            if self.mcp_agent:
                history_result = await self.mcp_agent.call_tool("get_operation_history", {"last_n": 20})
                history_data = json.loads(history_result) if isinstance(history_result, str) else history_result
            else:
                history_data = {"total_operations": 0, "history": []}
            
            # 统计操作
            operation_counts = {}
            if "history" in history_data:
                for item in history_data["history"]:
                    action = item.get("action", "unknown")
                    operation_counts[action] = operation_counts.get(action, 0) + 1
            
            report = f"""
# 代码实现完成报告 (MCP版本)

## 执行摘要
- 总迭代次数: {iterations}
- 总耗时: {elapsed_time:.2f} 秒
- 总操作数: {history_data.get('total_operations', 0)}

## 操作统计
"""
            for action, count in operation_counts.items():
                report += f"- {action}: {count} 次\n"
            
            report += """
## 实施方法
使用了基于aisi-basic-agent的迭代式开发方法：
1. 分析实现计划和文件结构
2. 识别核心组件并确定实现顺序  
3. 迭代式实现每个组件
4. 测试和验证代码
5. 修复问题并优化

## MCP架构说明
✅ 使用标准MCP客户端/服务器架构
✅ 通过MCP协议进行工具调用
✅ 支持工作空间管理和操作历史追踪
✅ 完全符合MCP规范
"""
            return report
            
        except Exception as e:
            self.logger.error(f"生成最终报告失败: {e}")
            return f"生成最终报告失败: {str(e)}"

    # ==================== 主工作流 ====================
    
    async def run_workflow(self, plan_file_path: str, target_directory: Optional[str] = None):
        """运行完整工作流"""
        try:
            # 读取实现计划
            plan_content = self._read_plan_file(plan_file_path)
            
            # 确定目标目录
            if target_directory is None:
                target_directory = str(Path(plan_file_path).parent)
            
            self.logger.info(f"开始工作流: {plan_file_path}")
            self.logger.info(f"目标目录: {target_directory}")
            
            results = {}
            
            # 检查文件树是否已存在
            if self._check_file_tree_exists(target_directory):
                self.logger.info("文件树已存在，跳过创建步骤")
                results["file_tree"] = "已存在，跳过创建"
            else:
                self.logger.info("创建文件树...")
                results["file_tree"] = await self.create_file_structure(plan_content, target_directory)
            
            # 代码实现
            self.logger.info("开始代码实现...")
            results["code_implementation"] = await self.implement_code(plan_content, target_directory)
            
            self.logger.info("工作流执行成功")
            
            return {
                "status": "success",
                "plan_file": plan_file_path,
                "target_directory": target_directory,
                "code_directory": os.path.join(target_directory, "generate_code"),
                "results": results,
                "mcp_architecture": "standard"
            }
            
        except Exception as e:
            self.logger.error(f"工作流执行失败: {e}")
            return {"status": "error", "message": str(e), "plan_file": plan_file_path}
        finally:
            # 确保清理所有MCP资源
            await self._cleanup_mcp_agent()


# ==================== 主函数 ====================

async def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # 示例用法
    plan_file = "agent_folders/papers/1/initial_plan.txt"
    
    workflow = CodeImplementationWorkflow()
    
    # 运行工作流
    result = await workflow.run_workflow(plan_file)
    
    # 显示结果
    print("=" * 60)
    print("工作流执行结果:")
    print(f"状态: {result['status']}")
    
    if result['status'] == 'success':
        print(f"代码目录: {result['code_directory']}")
        print(f"MCP架构: {result.get('mcp_architecture', 'unknown')}")
        print("执行完成!")
    else:
        print(f"错误信息: {result['message']}")
    
    print("=" * 60)
    print("\n✅ 使用标准MCP架构")
    print("🔧 MCP服务器: tools/code_implementation_server.py")
    print("📋 配置文件: mcp_agent.config.yaml")


if __name__ == "__main__":
    asyncio.run(main())
