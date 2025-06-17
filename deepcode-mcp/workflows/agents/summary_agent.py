"""
Summary Agent for Conversation Management
对话管理的总结代理

Handles conversation summarization and sliding window memory optimization
for long-running code implementation sessions.
处理长时间代码实现会话的对话总结和滑动窗口内存优化。
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional

# Import prompts from code_prompts
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompts.code_prompts import CONVERSATION_SUMMARY_PROMPT


class SummaryAgent:
    """
    Summary Agent for conversation management and memory optimization
    用于对话管理和内存优化的总结代理
    
    Responsibilities / 职责:
    - Generate conversation summaries / 生成对话总结
    - Apply sliding window mechanism / 应用滑动窗口机制
    - Preserve critical implementation context / 保留关键实现上下文
    - Optimize token usage / 优化token使用
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize Summary Agent
        初始化总结代理
        
        Args:
            logger: Logger instance for tracking operations
        """
        self.logger = logger or self._create_default_logger()
        self.summary_history = []  # Store generated summaries / 存储生成的总结
        
    def _create_default_logger(self) -> logging.Logger:
        """Create default logger if none provided / 如果未提供则创建默认日志记录器"""
        logger = logging.getLogger(f"{__name__}.SummaryAgent")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def generate_conversation_summary(
        self, 
        client, 
        client_type: str, 
        messages: List[Dict], 
        implementation_summary: Dict[str, Any]
    ) -> str:
        """
        Generate conversation summary using LLM
        使用LLM生成对话总结
        
        Args:
            client: LLM client instance
            client_type: Type of LLM client ('anthropic' or 'openai')
            messages: Conversation messages to summarize
            implementation_summary: Current implementation progress data
            
        Returns:
            Generated summary string
        """
        try:
            self.logger.info("Generating conversation summary using Summary Agent")
            
            # Prepare summary request / 准备总结请求
            recent_messages = messages[-20:] if len(messages) > 20 else messages
            summary_messages = [
                {"role": "user", "content": CONVERSATION_SUMMARY_PROMPT},
                {"role": "user", "content": f"Conversation to summarize:\n{json.dumps(recent_messages, ensure_ascii=False, indent=2)}"}
            ]
            
            # Call LLM for summary generation / 调用LLM生成总结
            summary_response = await self._call_llm_for_summary(
                client, client_type, summary_messages
            )
            
            summary_content = summary_response.get("content", "").strip()
            
            # Update implementation summary / 更新实现总结
            self._update_implementation_summary(
                implementation_summary, summary_content, len(recent_messages)
            )
            
            # Store in summary history / 存储到总结历史
            self.summary_history.append({
                "timestamp": time.time(),
                "summary": summary_content,
                "message_count": len(recent_messages)
            })
            
            self.logger.info(f"Summary generated successfully, length: {len(summary_content)} characters")
            return summary_content
            
        except Exception as e:
            self.logger.error(f"Failed to generate conversation summary: {e}")
            # Return fallback summary / 返回备用总结
            return self._generate_fallback_summary(implementation_summary)
    
    async def _call_llm_for_summary(
        self, 
        client, 
        client_type: str, 
        summary_messages: List[Dict]
    ) -> Dict[str, Any]:
        """
        Call LLM for summary generation
        调用LLM生成总结
        """
        if client_type == "anthropic":
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                system="You are a helpful conversation summarization assistant.",
                messages=summary_messages,
                max_tokens=1000,
                temperature=0.2
            )
            
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text
            
            return {"content": content}
            
        elif client_type == "openai":
            openai_messages = [{"role": "system", "content": "You are a helpful conversation summarization assistant."}]
            openai_messages.extend(summary_messages)
            
            response = await client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=openai_messages,
                max_tokens=1000,
                temperature=0.2
            )
            
            return {"content": response.choices[0].message.content or ""}
        
        else:
            raise ValueError(f"Unsupported client type: {client_type}")
    
    def _update_implementation_summary(
        self, 
        implementation_summary: Dict[str, Any], 
        summary_content: str, 
        message_count: int
    ):
        """
        Update implementation summary with new information
        使用新信息更新实现总结
        """
        implementation_summary["technical_decisions"].append({
            "iteration_range": f"recent-{message_count}",
            "summary": summary_content,
            "timestamp": time.time()
        })
    
    def _generate_fallback_summary(self, implementation_summary: Dict[str, Any]) -> str:
        """
        Generate fallback summary when LLM call fails
        当LLM调用失败时生成备用总结
        """
        completed_files = [f["file"] for f in implementation_summary["completed_files"]]
        return f"""Implementation Progress Summary:
- Completed files: {', '.join(completed_files[-10:])}
- Total files implemented: {len(completed_files)}
- Continue with next file implementation according to plan priorities."""
    
    def apply_sliding_window(
        self, 
        messages: List[Dict], 
        initial_plan_message: Optional[Dict], 
        summary: str, 
        window_size: int = 5
    ) -> List[Dict]:
        """
        Apply sliding window mechanism to optimize message history
        应用滑动窗口机制优化消息历史
        
        Args:
            messages: Current message list
            initial_plan_message: Initial plan message (never compressed)
            summary: Generated summary for historical context
            window_size: Number of recent conversation rounds to keep
            
        Returns:
            Optimized message list
        """
        try:
            self.logger.info(f"Applying sliding window mechanism, window_size: {window_size}")
            self.logger.info(f"Input messages count: {len(messages)}")
            
            # Debug: Log message types and roles / 调试：记录消息类型和角色
            for i, msg in enumerate(messages[-10:]):  # Show last 10 messages for debugging
                role = msg.get("role", "unknown")
                content_preview = msg.get("content", "")[:50] + "..." if len(msg.get("content", "")) > 50 else msg.get("content", "")
                self.logger.debug(f"Message {len(messages)-10+i}: role={role}, content_preview='{content_preview}'")
            
            # Build new message list / 构建新的消息列表
            new_messages = []
            
            # 1. Preserve initial plan (never compressed) / 保留初始计划（永不压缩）
            if initial_plan_message:
                new_messages.append(initial_plan_message)
                self.logger.info("Initial plan message preserved")
            
            # 2. Add summary information / 添加总结信息
            summary_message = {
                "role": "user",
                "content": f"[CONVERSATION SUMMARY - Previous Implementation Progress]\n{summary}\n\n[CONTINUE IMPLEMENTATION]"
            }
            new_messages.append(summary_message)
            self.logger.info(f"Summary message added, summary length: {len(summary)} characters")
            
            # 3. Keep recent complete conversation rounds / 保留最近的完整对话轮次
            messages_to_keep = window_size * 3  # Each round ~3 messages / 每轮约3条消息
            self.logger.info(f"Calculated messages_to_keep: {messages_to_keep} (window_size={window_size} * 3)")
            
            # Extract recent messages, excluding the initial plan message / 提取最近的消息，排除初始计划消息
            if len(messages) > messages_to_keep:
                # Take the last N messages / 取最后N条消息
                recent_messages = messages[-messages_to_keep:]
            else:
                # If total messages are less than window, take all except initial plan / 如果总消息少于窗口，取除初始计划外的所有消息
                start_idx = 1 if initial_plan_message and len(messages) > 1 else 0
                recent_messages = messages[start_idx:]
            
            self.logger.info(f"Recent messages extracted: {len(recent_messages)} messages (from total {len(messages)})")
            
            # Additional validation: remove any duplicate initial plan messages / 额外验证：移除任何重复的初始计划消息
            if initial_plan_message:
                recent_messages = [msg for msg in recent_messages if msg != initial_plan_message]
                self.logger.info(f"After removing duplicate initial plan: {len(recent_messages)} recent messages")
            
            # Ensure message integrity (avoid truncating conversation rounds) / 确保消息完整性（避免截断对话轮次）
            if recent_messages:
                # Find recent assistant message as starting point / 找到最近的assistant消息作为起点
                start_idx = 0
                for i, msg in enumerate(recent_messages):
                    if msg.get("role") == "assistant":
                        start_idx = i
                        self.logger.debug(f"Found assistant message at index {i} in recent_messages")
                        break
                
                self.logger.info(f"Starting from index {start_idx} in recent_messages")
                final_recent_messages = recent_messages[start_idx:]
                new_messages.extend(final_recent_messages)
                
                self.logger.info(f"Final recent messages added: {len(final_recent_messages)} messages")
            
            self.logger.info(f"Sliding window applied: {len(messages)} -> {len(new_messages)} messages")
            
            # Debug: Log new message structure / 调试：记录新消息结构
            self.logger.info("Final message structure:")
            for i, msg in enumerate(new_messages):
                role = msg.get("role", "unknown")
                content_type = "INITIAL_PLAN" if msg == initial_plan_message else "SUMMARY" if "[CONVERSATION SUMMARY" in msg.get("content", "") else "RECENT"
                self.logger.info(f"  {i}: {role} - {content_type}")
            
            return new_messages
            
        except Exception as e:
            self.logger.error(f"Failed to apply sliding window: {e}")
            # Return emergency trimmed messages / 返回紧急裁剪的消息
            return self._emergency_message_trim(messages, initial_plan_message)
    
    def _emergency_message_trim(
        self, 
        messages: List[Dict], 
        initial_plan_message: Optional[Dict]
    ) -> List[Dict]:
        """
        Emergency message trimming mechanism
        紧急消息裁剪机制
        """
        try:
            new_messages = []
            
            # Preserve initial plan / 保留初始计划
            if initial_plan_message:
                new_messages.append(initial_plan_message)
            
            # Add emergency status explanation / 添加紧急状态说明
            emergency_message = {
                "role": "user",
                "content": "[EMERGENCY TRIM] Previous conversation history has been compressed due to length. Continue implementing files according to the original plan."
            }
            new_messages.append(emergency_message)
            
            # Keep recent 20 valid messages / 保留最近的20条有效消息
            recent_valid_messages = []
            for msg in messages[-20:]:
                if msg.get("content", "").strip() and msg != initial_plan_message:
                    recent_valid_messages.append(msg)
            
            new_messages.extend(recent_valid_messages)
            
            self.logger.warning(f"Emergency trim applied: {len(messages)} -> {len(new_messages)} messages")
            return new_messages
            
        except Exception as e:
            self.logger.error(f"Emergency message trim failed: {e}")
            # Last resort protection / 最后的保险措施
            return [initial_plan_message] if initial_plan_message else messages[-10:]
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for monitoring
        获取用于监控的总结统计信息
        """
        return {
            "total_summaries_generated": len(self.summary_history),
            "latest_summary_time": self.summary_history[-1]["timestamp"] if self.summary_history else None,
            "average_summary_length": sum(len(s["summary"]) for s in self.summary_history) / len(self.summary_history) if self.summary_history else 0
        }
    
    def analyze_message_patterns(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Analyze message patterns for debugging sliding window
        分析消息模式以调试滑动窗口
        """
        try:
            analysis = {
                "total_messages": len(messages),
                "role_distribution": {},
                "conversation_rounds": 0,
                "message_lengths": [],
                "tool_result_count": 0
            }
            
            # Analyze role distribution / 分析角色分布
            for msg in messages:
                role = msg.get("role", "unknown")
                analysis["role_distribution"][role] = analysis["role_distribution"].get(role, 0) + 1
                
                # Count message length / 计算消息长度
                content_length = len(msg.get("content", ""))
                analysis["message_lengths"].append(content_length)
                
                # Count tool results / 计算工具结果
                if "Tool Result" in msg.get("content", ""):
                    analysis["tool_result_count"] += 1
            
            # Estimate conversation rounds / 估计对话轮次
            assistant_messages = analysis["role_distribution"].get("assistant", 0)
            analysis["conversation_rounds"] = assistant_messages
            
            # Calculate averages / 计算平均值
            if analysis["message_lengths"]:
                analysis["average_message_length"] = sum(analysis["message_lengths"]) / len(analysis["message_lengths"])
                analysis["max_message_length"] = max(analysis["message_lengths"])
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze message patterns: {e}")
            return {"error": str(e)} 