"""
迭代式代码实现模块 - 从主工作流中分离
Iterative Code Implementation Module - Separated from Main Workflow

负责迭代式代码实现的具体逻辑，包括：
1. 迭代开发循环
2. 最终报告生成
3. 工具调用和LLM交互

使用标准MCP架构：
- MCP服务器：tools/code_implementation_server.py
- MCP客户端：通过mcp_agent框架调用
"""

import asyncio
import yaml
import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# 导入提示词
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.iterative_code_prompts import (
    ITERATIVE_CODE_SYSTEM_PROMPT,
    CONTINUE_CODE_MESSAGE,
    INITIAL_ANALYSIS_PROMPT,
    COMPLETION_CHECK_PROMPT,
    ERROR_HANDLING_PROMPT,
    TOOL_USAGE_EXAMPLES,
)

# 导入工具定义
from config.mcp_tool_definitions import get_mcp_tools


class IterativeCodeImplementation:
    """
    迭代式代码实现类
    
    负责处理迭代式代码实现的具体逻辑
    """
    
    def __init__(self, logger: logging.Logger, mcp_agent=None):
        self.logger = logger
        self.mcp_agent = mcp_agent
    
    async def implement_code_from_workflow(self, plan_content: str, target_directory: str, 
                                         workflow_instance) -> str:
        """迭代式代码实现 - 使用MCP服务器（从主工作流移动的方法）"""
        try:
            # 初始化LLM客户端
            client, client_type = await workflow_instance._initialize_llm_client()
            
            # 初始化MCP代理
            code_directory = os.path.join(target_directory, "generate_code")
            await workflow_instance._initialize_mcp_agent(code_directory)
            
            # 更新本地mcp_agent引用
            self.mcp_agent = workflow_instance.mcp_agent
            
            # 委托给已有的implement_code方法
            result = await self.implement_code(
                plan_content, target_directory, client, client_type, workflow_instance.mcp_agent,
                workflow_instance._initialize_llm_client, workflow_instance._prepare_mcp_tool_definitions,
                workflow_instance._call_llm_with_tools, workflow_instance._execute_mcp_tool_calls
            )
            
            return result
            
        finally:
            # 确保清理MCP代理资源
            if workflow_instance and hasattr(workflow_instance, '_cleanup_mcp_agent'):
                await workflow_instance._cleanup_mcp_agent()
                
    async def implement_code_standalone(self, plan_content: str, target_directory: str, 
                                      workflow_instance) -> str:
        """迭代式代码实现 - 独立方法（从主工作流移动过来的完整函数）"""
        try:
            # 初始化LLM客户端
            client, client_type = await workflow_instance._initialize_llm_client()
            
            # 初始化MCP代理
            code_directory = os.path.join(target_directory, "generate_code")
            await workflow_instance._initialize_mcp_agent(code_directory)
            
            # 创建迭代式代码实现实例
            # 注意：这里我们用已有的self，不需要再创建新实例
            self.mcp_agent = workflow_instance.mcp_agent
            
            # 委托给专门的实现模块
            result = await self.implement_code(
                plan_content, target_directory, client, client_type, workflow_instance.mcp_agent,
                workflow_instance._initialize_llm_client, workflow_instance._prepare_mcp_tool_definitions,
                workflow_instance._call_llm_with_tools, workflow_instance._execute_mcp_tool_calls
            )
            
            return result
            
        finally:
            # 确保清理MCP代理资源
            await workflow_instance._cleanup_mcp_agent()

    async def implement_code(self, plan_content: str, target_directory: str, 
                           client, client_type, mcp_agent, 
                           initialize_llm_client_func, prepare_tools_func,
                           call_llm_with_tools_func, execute_mcp_tool_calls_func) -> str:
        """迭代式代码实现 - 使用MCP服务器"""
        self.logger.info("开始迭代式代码实现...")
        
        code_directory = os.path.join(target_directory, "generate_code")
        if not os.path.exists(code_directory):
            raise FileNotFoundError("文件树结构不存在，请先运行文件树创建")
        
        self.mcp_agent = mcp_agent
        
        # 准备工具定义 (MCP标准格式)
        tools = prepare_tools_func()
        
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
            client, client_type, system_message, messages, tools,
            call_llm_with_tools_func, execute_mcp_tool_calls_func
        )
        
        return result
    
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

    async def _iterative_development_loop(self, client, client_type, system_message, messages, tools,
                                        call_llm_with_tools_func, execute_mcp_tool_calls_func):
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
            response = await call_llm_with_tools_func(
                client, client_type, system_message, messages, tools
            )
            
            # 确保响应内容不为空
            response_content = response.get("content", "").strip()
            if not response_content:
                response_content = "继续实现代码..."
            
            messages.append({"role": "assistant", "content": response_content})
            
            # 处理工具调用 - 使用MCP
            if response.get("tool_calls"):
                tool_results = await execute_mcp_tool_calls_func(response["tool_calls"])
                
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
                final_response = await call_llm_with_tools_func(
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