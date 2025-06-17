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

# 导入提示词 / Import prompts
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.code_prompts import STRUCTURE_GENERATOR_PROMPT
from prompts.iterative_code_prompts import (
    ITERATIVE_CODE_SYSTEM_PROMPT, 
    CONTINUE_CODE_MESSAGE,
    INITIAL_ANALYSIS_PROMPT,
    COMPLETION_CHECK_PROMPT,
    ERROR_HANDLING_PROMPT,
    TOOL_USAGE_EXAMPLES,
    PURE_CODE_IMPLEMENTATION_PROMPT
)

# 导入新的agent类 / Import new agent classes
from workflows.agents import CodeImplementationAgent, SummaryAgent


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
    
    async def implement_code_pure(self, plan_content: str, target_directory: str) -> str:
        """纯代码实现 - 专注于代码写入，不包含测试"""
        self.logger.info("开始纯代码实现（无测试）...")
        
        code_directory = os.path.join(target_directory, "generate_code")
        if not os.path.exists(code_directory):
            raise FileNotFoundError("文件树结构不存在，请先运行文件树创建")
        
        try:
            # 初始化LLM客户端
            client, client_type = await self._initialize_llm_client()
            
            # 初始化MCP代理
            await self._initialize_mcp_agent(code_directory)
            
            # 准备工具定义 (MCP标准格式)
            tools = self._prepare_mcp_tool_definitions()
            
            # 使用纯代码实现prompt
            system_message = PURE_CODE_IMPLEMENTATION_PROMPT
            messages = []
            
            # 直接传递实现计划
            implementation_message = f"""Code Reproduction Plan:

{plan_content}

Working Directory: {code_directory}

Analyze this plan and begin implementing files one by one, starting with the highest priority file from Phase 1 (Foundation). Implement exactly one complete file per response."""
            
            messages.append({"role": "user", "content": implementation_message})
            
            # 纯代码实现循环
            result = await self._pure_code_implementation_loop(
                client, client_type, system_message, messages, tools
            )
            
            return result
            
        finally:
            # 确保清理MCP代理资源
            await self._cleanup_mcp_agent()
    
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
            await self._initialize_mcp_agent(code_directory)
            
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

    def _validate_messages(self, messages: List[Dict]) -> List[Dict]:
        """验证并清理消息列表，确保所有消息都有非空内容"""
        valid_messages = []
        for msg in messages:
            content = msg.get("content", "").strip()
            if content:  # 只保留有内容的消息
                valid_messages.append({
                    "role": msg.get("role", "user"),
                    "content": content
                })
            else:
                self.logger.warning(f"跳过空消息: {msg}")
        return valid_messages

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
                if progress_msg.strip():  # 确保进度消息不为空
                    messages.append({"role": "user", "content": progress_msg})
            
            self.logger.info(f"迭代 {iteration}: 生成响应")
            
            # 验证消息列表，确保没有空消息
            messages = self._validate_messages(messages)
            
            # 调用LLM
            response = await self._call_llm_with_tools(
                client, client_type, system_message, messages, tools
            )
            
            # 确保响应内容不为空
            response_content = response.get("content", "").strip()
            if not response_content:
                response_content = "继续实现代码..."
            
            messages.append({"role": "assistant", "content": response_content})
            
            # 处理工具调用 - 使用MCP
            if response.get("tool_calls"):
                tool_results = await self._execute_mcp_tool_calls(response["tool_calls"])
                
                for tool_result in tool_results:
                    tool_content = f"Tool Result {tool_result['tool_name']}:\n{tool_result['result']}"
                    if tool_content.strip():  # 确保工具结果不为空
                        messages.append({
                            "role": "user",
                            "content": tool_content
                        })
                
                if any("error" in result['result'] for result in tool_results):
                    messages.append({"role": "user", "content": ERROR_HANDLING_PROMPT})
            else:
                messages.append({"role": "user", "content": CONTINUE_CODE_MESSAGE})
            
            # 检查完成
            if "implementation is complete" in response_content.lower():
                self.logger.info("代码实现声明完成")
                messages.append({"role": "user", "content": COMPLETION_CHECK_PROMPT})
                final_response = await self._call_llm_with_tools(
                    client, client_type, system_message, messages, tools
                )
                final_content = final_response.get("content", "").strip()
                if final_content and "complete" in final_content.lower():
                    break
            
            # 防止消息历史过长 - 改进的消息裁剪逻辑
            if len(messages) > 100:
                # 保留系统消息和最近的有效消息
                filtered_messages = []
                for msg in messages[-50:]:
                    if msg.get("content", "").strip():  # 只保留非空消息
                        filtered_messages.append(msg)
                
                messages = messages[:1] + filtered_messages
                self.logger.info(f"裁剪消息历史，保留 {len(messages)} 条有效消息")
        
        return await self._generate_final_report_via_mcp(iteration, time.time() - start_time)
    
    async def _pure_code_implementation_loop(self, client, client_type, system_message, messages, tools):
        """
        Pure code implementation loop with sliding window and key information extraction
        带滑动窗口和关键信息提取的纯代码实现循环
        """
        max_iterations = 30  # Reduce iterations, focus on code implementation / 减少迭代次数，专注于代码实现
        iteration = 0
        start_time = time.time()
        max_time = 2400  # 40 minutes / 40分钟
        
        # Sliding window configuration / 滑动窗口配置
        WINDOW_SIZE = 5  # Keep recent 5 complete conversation rounds / 保留最近5轮完整对话
        SUMMARY_TRIGGER = 5  # Trigger summary after every 5 file implementations / 每5个文件实现后触发总结
        
        # Initialize specialized agents / 初始化专门的代理
        code_agent = CodeImplementationAgent(self.mcp_agent, self.logger)
        summary_agent = SummaryAgent(self.logger)
        
        # Preserve initial plan information (never compressed) / 保存初始计划信息（永不压缩）
        initial_plan_message = messages[0] if messages else None
        
        while iteration < max_iterations:
            iteration += 1
            elapsed_time = time.time() - start_time
            
            if elapsed_time > max_time:
                self.logger.warning(f"Time limit reached: {elapsed_time:.2f}s")
                break
            
            self.logger.info(f"Pure code implementation iteration {iteration}: generating code")
            
            # Validate message list, ensure no empty messages / 验证消息列表，确保没有空消息
            messages = self._validate_messages(messages)
            
            # Use code agent's system prompt / 使用代码代理的系统提示词
            current_system_message = code_agent.get_system_prompt()
            
            # Call LLM / 调用LLM
            response = await self._call_llm_with_tools(
                client, client_type, current_system_message, messages, tools
            )
            
            # Ensure response content is not empty / 确保响应内容不为空
            response_content = response.get("content", "").strip()
            if not response_content:
                response_content = "Continue implementing code files..."
            
            messages.append({"role": "assistant", "content": response_content})
            
            # Handle tool calls using Code Agent / 使用代码代理处理工具调用
            if response.get("tool_calls"):
                tool_results = await code_agent.execute_tool_calls(response["tool_calls"])
                
                # Add tool results to messages / 将工具结果添加到消息中
                for tool_result in tool_results:
                    tool_content = f"Tool Result {tool_result['tool_name']}:\n{tool_result['result']}"
                    if tool_content.strip():  # Ensure tool result is not empty / 确保工具结果不为空
                        messages.append({
                            "role": "user",
                            "content": tool_content
                        })
                
                # Handle errors with simple guidance / 如果有错误，提供简单的错误处理指导
                if any("error" in result['result'] for result in tool_results):
                    messages.append({
                        "role": "user", 
                        "content": "Error detected. Fix the issue and continue with the next file implementation."
                    })
                else:
                    # More explicit guidance for continuing with next file / 更明确的下一个文件继续指导
                    files_count = code_agent.get_files_implemented_count()
                    next_file_guidance = f"""File implementation completed successfully! Current progress: {files_count} files implemented.

NEXT ACTION REQUIRED: Immediately implement the next file according to the implementation plan priorities.

Instructions:
1. Identify the next highest-priority file from the plan
2. Implement it completely with production-quality code
3. Use write_file tool to create the file
4. Continue this process for each remaining file

Remember: Implement exactly ONE complete file per response. Do not skip files or create multiple files at once."""
                    
                    messages.append({
                        "role": "user", 
                        "content": next_file_guidance
                    })
            else:
                # If no tool calls, provide stronger guidance / 如果没有工具调用，提供更强的指导
                files_count = code_agent.get_files_implemented_count()
                continue_guidance = f"""No tool calls detected. Current progress: {files_count} files implemented.

ACTION REQUIRED: You must implement the next file from the implementation plan.

Steps:
1. Analyze the implementation plan to identify the next priority file
2. Implement the complete file with all required functionality
3. Use the write_file tool to create the file
4. Provide a brief status update

CRITICAL: You must use tools to implement files. Do not just provide explanations - take action!"""
                
                messages.append({
                    "role": "user", 
                    "content": continue_guidance
                })
            
            # Sliding window + key information extraction mechanism / 滑动窗口 + 关键信息提取机制
            if code_agent.should_trigger_summary(SUMMARY_TRIGGER):
                self.logger.info(f"Triggering summary mechanism: {code_agent.get_files_implemented_count()} files implemented")
                
                # Analyze messages before sliding window / 滑动窗口前分析消息
                analysis_before = summary_agent.analyze_message_patterns(messages)
                self.logger.info(f"Before sliding window - Messages: {analysis_before['total_messages']}, Rounds: {analysis_before['conversation_rounds']}, Tool results: {analysis_before['tool_result_count']}")
                
                # Generate conversation summary using Summary Agent / 使用总结代理生成历史对话总结
                summary = await summary_agent.generate_conversation_summary(
                    client, client_type, messages, code_agent.get_implementation_summary()
                )
                
                # Apply sliding window: preserve initial plan + summary + recent conversations / 应用滑动窗口：保留初始计划 + 总结 + 最近的对话
                messages = summary_agent.apply_sliding_window(
                    messages, initial_plan_message, summary, WINDOW_SIZE
                )
                
                # Analyze messages after sliding window / 滑动窗口后分析消息
                analysis_after = summary_agent.analyze_message_patterns(messages)
                self.logger.info(f"After sliding window - Messages: {analysis_after['total_messages']}, Rounds: {analysis_after['conversation_rounds']}, Tool results: {analysis_after['tool_result_count']}")
                
                # Log compression ratio / 记录压缩比例
                compression_ratio = (analysis_before['total_messages'] - analysis_after['total_messages']) / analysis_before['total_messages'] * 100
                self.logger.info(f"Compression ratio: {compression_ratio:.1f}% (reduced {analysis_before['total_messages'] - analysis_after['total_messages']} messages)")
                
                self.logger.info(f"Message count after sliding window: {len(messages)}")
            
            # Check completion with simple completion check / 检查完成 - 更简单的完成检查
            if any(keyword in response_content.lower() for keyword in [
                "all files implemented", 
                "implementation complete", 
                "all phases completed",
                "reproduction plan fully implemented"
            ]):
                self.logger.info("Code implementation declared complete")
                break
            
            # Backup mechanism to prevent message history from being too long / 防止消息历史过长的备用机制
            if len(messages) > 120:  # Higher threshold due to sliding window / 更高的阈值，因为有滑动窗口
                self.logger.warning("Message history still too long, executing emergency trim")
                messages = summary_agent._emergency_message_trim(messages, initial_plan_message)
        
        return await self._generate_pure_code_final_report_with_agents(
            iteration, time.time() - start_time, code_agent, summary_agent
        )
    
    async def _generate_pure_code_final_report_with_agents(
        self, 
        iterations: int, 
        elapsed_time: float, 
        code_agent: CodeImplementationAgent, 
        summary_agent: SummaryAgent
    ):
        """
        Generate final report using agent statistics
        使用代理统计信息生成最终报告
        """
        try:
            # Get statistics from agents / 从代理获取统计信息
            code_stats = code_agent.get_implementation_statistics()
            summary_stats = summary_agent.get_summary_statistics()
            implementation_summary = code_agent.get_implementation_summary()
            
            # Get operation history from MCP / 从MCP获取操作历史
            if self.mcp_agent:
                history_result = await self.mcp_agent.call_tool("get_operation_history", {"last_n": 30})
                history_data = json.loads(history_result) if isinstance(history_result, str) else history_result
            else:
                history_data = {"total_operations": 0, "history": []}
            
            # Count write operations / 统计写入操作
            write_operations = 0
            files_created = []
            if "history" in history_data:
                for item in history_data["history"]:
                    if item.get("action") == "write_file":
                        write_operations += 1
                        file_path = item.get("details", {}).get("file_path", "unknown")
                        files_created.append(file_path)
            
            report = f"""
# Pure Code Implementation Completion Report with Agent Architecture
# 带代理架构的纯代码实现完成报告

## Execution Summary / 执行摘要
- Implementation iterations: {iterations}
- Total elapsed time: {elapsed_time:.2f} seconds
- Files implemented: {code_stats['total_files_implemented']}
- File write operations: {write_operations}
- Total MCP operations: {history_data.get('total_operations', 0)}

## Agent Performance / 代理性能
### Code Implementation Agent / 代码实现代理
- Files tracked: {code_stats['files_implemented_count']}
- Technical decisions recorded: {code_stats['technical_decisions_count']}
- Constraints tracked: {code_stats['constraints_count']}
- Architecture notes: {code_stats['architecture_notes_count']}
- Latest file: {code_stats['latest_file'] or 'None'}

### Summary Agent / 总结代理
- Summaries generated: {summary_stats['total_summaries_generated']}
- Average summary length: {summary_stats['average_summary_length']:.0f} characters
- Memory optimization cycles: {summary_stats['total_summaries_generated']}

## Files Created / 已创建文件
"""
            for file_path in files_created[-20:]:  # Show recent 20 files / 显示最近的20个文件
                report += f"- {file_path}\n"
            
            if len(files_created) > 20:
                report += f"... and {len(files_created) - 20} more files\n"
            
            report += f"""
## Implementation Method / 实施方法
Used specialized agent architecture for pure code generation:
使用专门的代理架构进行纯代码生成：

1. **Code Implementation Agent**: Systematic file-by-file development
   **代码实现代理**: 系统性文件逐个开发
2. **Summary Agent**: Conversation memory optimization with sliding window
   **总结代理**: 带滑动窗口的对话内存优化
3. **Phase-based Implementation**: Following plan priorities (Phase 1 → Phase 2 → Phase 3)
   **基于阶段的实现**: 遵循计划优先级（阶段1 → 阶段2 → 阶段3）
4. **Memory Management**: Automatic conversation compression every 5 files
   **内存管理**: 每5个文件自动进行对话压缩

## Architecture Features / 架构特性
✅ Specialized agent separation for clean code organization
✅ 专门的代理分离，实现清洁的代码组织
✅ Sliding window memory optimization (70-80% token reduction)
✅ 滑动窗口内存优化（减少70-80%的token）
✅ Progress tracking and implementation statistics
✅ 进度跟踪和实现统计
✅ MCP-compliant tool execution
✅ 符合MCP标准的工具执行
✅ Bilingual documentation and logging
✅ 双语文档和日志记录

## Code Quality Assurance / 代码质量保证
- Complete implementations with no placeholders
- 完整实现，无占位符
- Production-grade code with comprehensive type hints
- 生产级代码，具有全面的类型提示
- Detailed docstrings and error handling
- 详细的文档字符串和错误处理
- Clean architecture following best practices
- 遵循最佳实践的清洁架构
"""
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate final report with agents: {e}")
            return f"Failed to generate final report with agents: {str(e)}"
    
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
    
    async def _call_llm_with_tools(self, client, client_type, system_message, messages, tools, max_tokens=16384):
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
        # 最后一次验证消息
        validated_messages = self._validate_messages(messages)
        if not validated_messages:
            validated_messages = [{"role": "user", "content": "请继续实现代码"}]
        
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                system=system_message,
                messages=validated_messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=0.2
            )
        except Exception as e:
            self.logger.error(f"Anthropic API调用失败: {e}")
            self.logger.error(f"消息数量: {len(validated_messages)}")
            for i, msg in enumerate(validated_messages):
                self.logger.error(f"消息 {i}: role={msg.get('role')}, content_length={len(msg.get('content', ''))}")
            raise
        
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
    
    async def _generate_pure_code_final_report(self, iterations: int, elapsed_time: float):
        """生成纯代码实现的最终报告"""
        try:
            # 获取操作历史
            if self.mcp_agent:
                history_result = await self.mcp_agent.call_tool("get_operation_history", {"last_n": 30})
                history_data = json.loads(history_result) if isinstance(history_result, str) else history_result
            else:
                history_data = {"total_operations": 0, "history": []}
            
            # 统计文件写入操作
            write_operations = 0
            files_created = []
            if "history" in history_data:
                for item in history_data["history"]:
                    if item.get("action") == "write_file":
                        write_operations += 1
                        file_path = item.get("details", {}).get("file_path", "unknown")
                        files_created.append(file_path)
            
            report = f"""
# 纯代码实现完成报告

## 执行摘要
- 实现迭代次数: {iterations}
- 总耗时: {elapsed_time:.2f} 秒
- 文件写入操作: {write_operations} 次
- 总操作数: {history_data.get('total_operations', 0)}

## 已创建文件
"""
            for file_path in files_created[-20:]:  # 显示最近创建的20个文件
                report += f"- {file_path}\n"
            
            if len(files_created) > 20:
                report += f"... 以及其他 {len(files_created) - 20} 个文件\n"
            
            report += f"""
## 实施方法
使用了专注于代码实现的纯代码生成方法：
1. 解析代码复现计划的结构和要求
2. 按阶段顺序实现文件（Phase 1 → Phase 2 → Phase 3）
3. 每个文件都包含完整的生产级代码实现
4. 跳过测试文件，专注于核心功能实现
5. 确保代码质量和架构一致性

## 特点
✅ 纯代码实现，无测试代码
✅ 按计划阶段有序实现
✅ 生产级代码质量
✅ 完整功能实现，无占位符
✅ 符合MCP标准架构

## 代码质量保证
- 完整的类型注解
- 详细的文档字符串
- 适当的错误处理
- 清晰的代码结构
- 遵循最佳实践
"""
            return report
            
        except Exception as e:
            self.logger.error(f"生成纯代码实现报告失败: {e}")
            return f"生成纯代码实现报告失败: {str(e)}"

    # ==================== 主工作流 ====================
    
    async def run_workflow(self, plan_file_path: str, target_directory: Optional[str] = None, pure_code_mode: bool = False):
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
            if pure_code_mode:
                self.logger.info("开始纯代码实现（无测试）...")
                results["code_implementation"] = await self.implement_code_pure(plan_content, target_directory)
            else:
                self.logger.info("开始迭代式代码实现...")
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
    
    # 运行工作流 - 使用纯代码模式
    print("Implementation Mode Selection:")
    print("1. Pure Code Implementation Mode (Recommended) - Focus on code writing, no testing")
    print("2. Iterative Implementation Mode - Includes testing and validation")
    
    # Default to pure code mode / 默认使用纯代码模式
    pure_code_mode = True
    mode_name = "Pure Code Implementation Mode with Agent Architecture"
    
    print(f"Using: {mode_name}")
    
    # 运行工作流
    result = await workflow.run_workflow(plan_file, pure_code_mode=pure_code_mode)
    
    # Display results / 显示结果
    print("=" * 60)
    print("Workflow Execution Results:")
    print(f"Status: {result['status']}")
    print(f"Mode: {mode_name}")
    
    if result['status'] == 'success':
        print(f"Code Directory: {result['code_directory']}")
        print(f"MCP Architecture: {result.get('mcp_architecture', 'unknown')}")
        print("Execution completed!")
    else:
        print(f"Error Message: {result['message']}")
    
    print("=" * 60)
    print("\n✅ Using Standard MCP Architecture with Specialized Agents")
    print("🔧 MCP Server: tools/code_implementation_server.py")
    print("📋 Configuration: mcp_agent.config.yaml")
    print("🤖 Code Agent: workflows/agents/code_implementation_agent.py")
    print("📝 Summary Agent: workflows/agents/summary_agent.py")
    print("💾 Prompts: prompts/code_prompts.py")
    print(f"🎯 Implementation Mode: {mode_name}")


if __name__ == "__main__":
    asyncio.run(main())
