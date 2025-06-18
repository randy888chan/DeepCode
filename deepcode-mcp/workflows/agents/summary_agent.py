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
        Generate role-aware conversation summary using LLM
        使用LLM生成角色感知的对话总结
        
        Args:
            client: LLM client instance
            client_type: Type of LLM client ('anthropic' or 'openai')
            messages: Conversation messages to summarize
            implementation_summary: Current implementation progress data
            
        Returns:
            Generated role-aware summary string
        """
        try:
            self.logger.info("Generating role-aware conversation summary using Summary Agent")
            
            # Prepare role-aware summary request / 准备角色感知的总结请求
            recent_messages = messages[-25:] if len(messages) > 25 else messages  # 稍微增加上下文
            
            # Analyze and clean messages for role clarity / 分析并清理消息以确保角色清晰
            role_analysis = self._analyze_role_distribution(recent_messages)
            cleaned_messages = self._prepare_messages_for_summary(recent_messages)
            
            self.logger.info(f"Role analysis before summary: {role_analysis}")
            
            # Create enhanced summary request with role context / 创建带角色上下文的增强总结请求
            summary_messages = [
                {"role": "user", "content": CONVERSATION_SUMMARY_PROMPT},
                {"role": "user", "content": f"""ROLE DISTRIBUTION ANALYSIS:
{json.dumps(role_analysis, ensure_ascii=False, indent=2)}

CONVERSATION TO SUMMARIZE (Role-Cleaned):
{json.dumps(cleaned_messages, ensure_ascii=False, indent=2)}

IMPLEMENTATION CONTEXT:
- Files completed: {len(implementation_summary.get('completed_files', []))}
- Technical decisions: {len(implementation_summary.get('technical_decisions', []))}
- Constraints tracked: {len(implementation_summary.get('important_constraints', []))}

Please provide a role-aware summary that maintains clear distinction between user guidance and assistant responses."""}
            ]
            
            # Call LLM for summary generation / 调用LLM生成总结
            summary_response = await self._call_llm_for_summary(
                client, client_type, summary_messages
            )
            
            summary_content = summary_response.get("content", "").strip()
            
            # Validate summary contains role context / 验证总结包含角色上下文
            if not self._validate_role_aware_summary(summary_content):
                self.logger.warning("Generated summary lacks role clarity, enhancing...")
                summary_content = self._enhance_summary_with_role_context(
                    summary_content, role_analysis, implementation_summary
                )
            
            # Update implementation summary / 更新实现总结
            self._update_implementation_summary(
                implementation_summary, summary_content, len(recent_messages)
            )
            
            # Store in summary history / 存储到总结历史
            self.summary_history.append({
                "timestamp": time.time(),
                "summary": summary_content,
                "message_count": len(recent_messages),
                "role_analysis": role_analysis
            })
            
            self.logger.info(f"Role-aware summary generated successfully, length: {len(summary_content)} characters")
            return summary_content
            
        except Exception as e:
            self.logger.error(f"Failed to generate role-aware conversation summary: {e}")
            # Return fallback summary with role context / 返回带角色上下文的备用总结
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
    
    def _prepare_messages_for_summary(self, messages: List[Dict]) -> List[Dict]:
        """
        Prepare and clean messages for role-aware summarization
        为角色感知总结准备和清理消息
        """
        cleaned_messages = []
        
        for i, msg in enumerate(messages):
            cleaned_msg = msg.copy()
            
            # Ensure role is properly set / 确保角色正确设置
            role = msg.get("role", "unknown")
            if role == "unknown":
                # Infer role based on content patterns / 根据内容模式推断角色
                content = msg.get("content", "")
                if any(pattern in content for pattern in ["Tool Result", "[", "Error detected", "继续"]):
                    cleaned_msg["role"] = "user"
                else:
                    cleaned_msg["role"] = "assistant"
                
                self.logger.debug(f"Inferred role for message {i}: {cleaned_msg['role']}")
            
            # Truncate very long content for summary / 为总结截断过长的内容
            content = cleaned_msg.get("content", "")
            if len(content) > 800:  # Truncate very long messages
                cleaned_msg["content"] = content[:400] + "\n...[content truncated for summary]...\n" + content[-400:]
                self.logger.debug(f"Truncated long message {i} from {len(content)} to {len(cleaned_msg['content'])} chars")
            
            cleaned_messages.append(cleaned_msg)
        
        return cleaned_messages
    
    def _validate_role_aware_summary(self, summary: str) -> bool:
        """
        Validate if summary contains proper role awareness
        验证总结是否包含适当的角色感知
        """
        # Check for role-aware indicators / 检查角色感知指标
        role_indicators = [
            "user guidance", "assistant", "user instructions", 
            "technical decisions made", "user feedback", 
            "assistant responses", "implementation decisions"
        ]
        
        summary_lower = summary.lower()
        found_indicators = sum(1 for indicator in role_indicators if indicator in summary_lower)
        
        # Also check for structured sections / 同时检查结构化部分
        structured_sections = [
            "implementation progress", "technical context", 
            "conversation context", "continuation context"
        ]
        
        found_sections = sum(1 for section in structured_sections if section.lower() in summary_lower)
        
        is_valid = found_indicators >= 2 and found_sections >= 2
        self.logger.debug(f"Summary validation: {found_indicators} role indicators, {found_sections} sections, valid: {is_valid}")
        
        return is_valid
    
    def _enhance_summary_with_role_context(
        self, 
        original_summary: str, 
        role_analysis: Dict[str, Any], 
        implementation_summary: Dict[str, Any]
    ) -> str:
        """
        Enhance summary with explicit role context
        用明确的角色上下文增强总结
        """
        completed_files = [f.get("file", "unknown") for f in implementation_summary.get("completed_files", [])]
        
        enhanced_summary = f"""**IMPLEMENTATION PROGRESS:**
- Files completed: {', '.join(completed_files[-8:]) if completed_files else 'None yet'}
- Total files implemented: {len(completed_files)}
- Role distribution: {role_analysis.get('role_counts', {})}

**TECHNICAL CONTEXT:**
- Assistant implemented files following structured approach
- User provided tool results and implementation feedback
- Technical decisions made systematically by assistant

**CONVERSATION CONTEXT:**
- User messages: {role_analysis.get('role_counts', {}).get('user', 0)} (guidance, tool results, feedback)
- Assistant messages: {role_analysis.get('role_counts', {}).get('assistant', 0)} (implementations, analysis)
- Recent role pattern: {' → '.join(role_analysis.get('last_5_roles', []))}

**CONTINUATION CONTEXT:**
- Next implementation targets: Continue with remaining files according to plan
- Role clarity: Assistant continues systematic file-by-file implementation
- Context preserved: Implementation progress and technical decisions maintained

**ORIGINAL SUMMARY CONTENT:**
{original_summary}"""
        
        self.logger.info("Enhanced summary with explicit role context")
        return enhanced_summary
    
    def _generate_fallback_summary(self, implementation_summary: Dict[str, Any]) -> str:
        """
        Generate fallback summary when LLM call fails
        当LLM调用失败时生成备用总结
        """
        completed_files = [f.get("file", "unknown") for f in implementation_summary.get("completed_files", [])]
        return f"""**IMPLEMENTATION PROGRESS:**
- Completed files: {', '.join(completed_files[-10:]) if completed_files else 'None yet'}
- Total files implemented: {len(completed_files)}

**ROLE CONTEXT:**
- Assistant: Systematic file-by-file implementation
- User: Provided guidance and tool result feedback
- Continuation: Assistant continues implementation role

**Next Steps:** Continue with next file implementation according to plan priorities."""
    
    def apply_sliding_window(
        self, 
        messages: List[Dict], 
        initial_plan_message: Optional[Dict], 
        summary: str, 
        window_size: int = 5
    ) -> List[Dict]:
        """
        Apply sliding window mechanism to optimize message history with role-aware conversation preservation
        应用滑动窗口机制优化消息历史，具有角色感知的对话保留功能
        
        Args:
            messages: Current message list
            initial_plan_message: Initial plan message (never compressed)
            summary: Generated summary for historical context
            window_size: Number of recent conversation rounds to keep
            
        Returns:
            Optimized message list with clear role separation
        """
        try:
            self.logger.info(f"Applying sliding window mechanism with role-aware processing, window_size: {window_size}")
            self.logger.info(f"Input messages count: {len(messages)}")
            
            # Analyze role distribution before processing / 处理前分析角色分布
            role_analysis_before = self._analyze_role_distribution(messages)
            self.logger.info(f"Before sliding window - Role distribution: {role_analysis_before['role_counts']}")
            self.logger.info(f"Before sliding window - Last 10 roles: {[msg.get('role') for msg in messages[-10:]]}")
            
            # Build new message list / 构建新的消息列表
            new_messages = []
            
            # 1. Preserve initial plan (never compressed) / 保留初始计划（永不压缩）
            if initial_plan_message:
                # Ensure initial plan has proper role
                if initial_plan_message.get("role") != "user":
                    self.logger.warning(f"Initial plan message has unexpected role: {initial_plan_message.get('role')}, forcing to 'user'")
                    initial_plan_message = initial_plan_message.copy()
                    initial_plan_message["role"] = "user"
                
                new_messages.append(initial_plan_message)
                self.logger.info("Initial plan message preserved with role validation")
            
            # 2. Add role-aware summary information / 添加角色感知的总结信息
            summary_message = {
                "role": "assistant",  # 改为assistant角色，避免连续user消息
                "content": f"""I understand the previous implementation progress. Based on the conversation history summary:

{summary}

I will now continue implementing the remaining files according to the original plan. Please provide next guidance or tool results."""
            }
            new_messages.append(summary_message)
            self.logger.info(f"Role-aware summary message added as 'assistant' role, summary length: {len(summary)} characters")
            
            # 3. 关键修复：只保留最后几条最新的对话，避免重复保存已总结的内容
            # Critical fix: Only keep the last few newest messages to avoid duplicating summarized content
            recent_cutoff = max(1, min(3, window_size // 2))  # 只保留1-3条最新消息
            if len(messages) > recent_cutoff:
                # 只取最后few条消息，这些是在生成总结时可能还没完全处理的最新对话
                # Only take the last few messages that might not be fully processed when generating summary
                very_recent_messages = messages[-recent_cutoff:]
                
                # 确保这些消息不包括initial_plan_message（避免重复）
                # Ensure these messages don't include initial_plan_message (avoid duplication)
                filtered_recent = []
                for msg in very_recent_messages:
                    if initial_plan_message is None or msg != initial_plan_message:
                        filtered_recent.append(msg)
                
                if filtered_recent:
                    # 验证并纠正角色序列
                    validated_recent_messages = self._validate_role_sequences(filtered_recent)
                    new_messages.extend(validated_recent_messages)
                    
                    self.logger.info(f"Added only {len(validated_recent_messages)} most recent messages (cutoff={recent_cutoff}) to avoid duplicating summarized content")
                else:
                    self.logger.info("No recent messages to add after filtering out initial plan")
            else:
                self.logger.info(f"Message count ({len(messages)}) <= cutoff ({recent_cutoff}), no additional recent messages added")
            
            # 4. 确保对话能够继续：检查是否需要添加继续指令
            # Ensure conversation can continue: check if we need to add continuation instruction
            if new_messages:
                last_role = new_messages[-1].get("role")
                if last_role != "user":
                    # 添加一个简洁的继续指令，让对话能正常进行
                    # Add a concise continuation instruction to keep conversation flowing
                    continuation_msg = {
                        "role": "user",
                        "content": "Continue implementing the next file according to the plan."
                    }
                    new_messages.append(continuation_msg)
                    self.logger.info("Added continuation message to ensure proper conversation flow")
            
            # Log final role structure for verification / 记录最终角色结构进行验证
            self._log_final_role_structure(new_messages)
            
            # Analyze role distribution after processing / 处理后分析角色分布
            role_analysis_after = self._analyze_role_distribution(new_messages)
            self.logger.info(f"After sliding window - Role distribution: {role_analysis_after['role_counts']}")
            self.logger.info(f"After sliding window - Final message roles: {[msg.get('role') for msg in new_messages]}")
            
            # Verify conversation continuity / 验证对话连续性
            if role_analysis_after['role_counts'].get('user', 0) == 0 or role_analysis_after['role_counts'].get('assistant', 0) == 0:
                self.logger.warning("⚠️  Role imbalance detected after sliding window - missing user or assistant messages!")
            
            self.logger.info(f"Sliding window applied with role awareness: {len(messages)} -> {len(new_messages)} messages")
            
            return new_messages
            
        except Exception as e:
            self.logger.error(f"Failed to apply sliding window with role awareness: {e}")
            # Return emergency trimmed messages with role validation / 返回带角色验证的紧急裁剪消息
            return self._emergency_message_trim(messages, initial_plan_message)
    
    def _analyze_role_distribution(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Analyze role distribution in messages for debugging
        分析消息中的角色分布进行调试
        """
        role_counts = {}
        role_sequence = []
        
        for msg in messages:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
            role_sequence.append(role)
        
        return {
            "total_messages": len(messages),
            "role_counts": role_counts,
            "last_5_roles": role_sequence[-5:] if len(role_sequence) >= 5 else role_sequence,
            "role_alternation_issues": self._detect_role_issues(role_sequence)
        }
    
    def _detect_role_issues(self, role_sequence: List[str]) -> List[str]:
        """
        Detect potential role sequence issues
        检测潜在的角色序列问题
        """
        issues = []
        
        # Check for consecutive same roles (potential issue)
        for i in range(len(role_sequence) - 1):
            if role_sequence[i] == role_sequence[i + 1] and role_sequence[i] in ["user", "assistant"]:
                issues.append(f"Consecutive {role_sequence[i]} roles at positions {i}-{i+1}")
        
        # Check for unknown roles
        if "unknown" in role_sequence:
            issues.append("Unknown role detected")
        
        return issues
    
    def _extract_recent_complete_rounds(
        self, 
        messages: List[Dict], 
        initial_plan_message: Optional[Dict], 
        window_size: int
    ) -> List[Dict]:
        """
        Extract recent complete conversation rounds ensuring proper user->assistant alternation
        提取最近的完整对话轮次，确保正确的user->assistant交替模式
        """
        # Filter out initial plan message from candidates / 从候选消息中过滤掉初始计划消息
        candidate_messages = []
        for msg in messages:
            if initial_plan_message is None or msg != initial_plan_message:
                candidate_messages.append(msg)
        
        if not candidate_messages:
            self.logger.warning("No candidate messages for recent rounds extraction")
            return []
        
        # Extract complete conversation rounds (user -> assistant pairs)
        # 提取完整的对话轮次（user -> assistant 对）
        rounds = []
        current_round = []
        
        for i, msg in enumerate(candidate_messages):
            role = msg.get("role", "unknown")
            current_round.append(msg)
            
            # A complete round is: user message -> assistant response
            # 完整的轮次是：user消息 -> assistant回应
            if role == "assistant" and len(current_round) >= 2:
                # Check if the previous message was from user
                # 检查前一条消息是否来自user
                prev_role = current_round[-2].get("role", "unknown")
                if prev_role == "user":
                    # We have a complete user->assistant round
                    # 我们有一个完整的user->assistant轮次
                    rounds.append(current_round[:])
                    current_round = []  # Start fresh for next round
                elif len(current_round) > 2:
                    # Look for earlier user message in current round
                    # 在当前轮次中寻找更早的user消息
                    user_assistant_pairs = []
                    temp_round = []
                    
                    for round_msg in current_round:
                        temp_round.append(round_msg)
                        if round_msg.get("role") == "assistant" and len(temp_round) >= 2:
                            # Check if we have a user message before this assistant message
                            for j in range(len(temp_round) - 2, -1, -1):
                                if temp_round[j].get("role") == "user":
                                    # Found a user->assistant pair
                                    user_assistant_pairs.extend(temp_round[j:])
                                    temp_round = []
                                    break
                    
                    if user_assistant_pairs:
                        rounds.append(user_assistant_pairs)
                        current_round = temp_round  # Keep remaining messages for next round
        
        # Handle remaining messages - try to form a valid round or keep as incomplete
        # 处理剩余消息 - 尝试形成有效轮次或保留为不完整轮次
        if current_round:
            # Check if we have at least a user message to start a round
            # 检查是否至少有一个user消息来开始一轮
            user_msg_found = False
            valid_remaining = []
            
            for msg in current_round:
                if msg.get("role") == "user":
                    user_msg_found = True
                
                if user_msg_found:  # Only include messages after we find a user message
                    valid_remaining.append(msg)
            
            if valid_remaining:
                rounds.append(valid_remaining)
                self.logger.info(f"Added incomplete round with {len(valid_remaining)} messages")
        
        # Keep the most recent complete rounds / 保留最近的完整轮次
        recent_rounds = rounds[-window_size:] if len(rounds) > window_size else rounds
        
        # Flatten rounds back to message list and ensure proper role alternation
        # 将轮次扁平化回消息列表并确保正确的角色交替
        recent_messages = []
        for round_msgs in recent_rounds:
            recent_messages.extend(round_msgs)
        
        # Post-process to ensure proper role alternation
        # 后处理以确保正确的角色交替
        recent_messages = self._ensure_role_alternation(recent_messages)
        
        self.logger.info(f"Extracted {len(recent_rounds)} complete rounds containing {len(recent_messages)} messages")
        self.logger.info(f"Final role sequence: {[msg.get('role') for msg in recent_messages[-10:]]}")
        
        return recent_messages
    
    def _ensure_role_alternation(self, messages: List[Dict]) -> List[Dict]:
        """
        Ensure proper role alternation in message sequence
        确保消息序列中的正确角色交替
        """
        if not messages:
            return messages
        
        # Clean and validate message sequence
        # 清理和验证消息序列
        clean_messages = []
        last_role = None
        
        for i, msg in enumerate(messages):
            current_role = msg.get("role", "unknown")
            content = msg.get("content", "").strip()
            
            # Skip empty messages
            # 跳过空消息
            if not content:
                self.logger.debug(f"Skipping empty message at position {i}")
                continue
            
            # Skip consecutive messages with same role (keep the more informative one)
            # 跳过连续的相同角色消息（保留信息量更大的）
            if current_role == last_role and last_role in ["user", "assistant"]:
                # Choose which message to keep based on content quality
                # 根据内容质量选择保留哪个消息
                if len(content) > len(clean_messages[-1].get("content", "")):
                    # Replace the last message with current one if it's more informative
                    # 如果当前消息信息量更大，则替换上一个消息
                    clean_messages[-1] = msg.copy()
                    self.logger.debug(f"Replaced duplicate {current_role} role at position {i} with more informative content")
                else:
                    self.logger.debug(f"Skipping duplicate {current_role} role at position {i}")
                continue
            
            # Add the message
            # 添加消息
            clean_messages.append(msg.copy())
            last_role = current_role
        
        # Ensure the sequence ends with a user message for conversation continuity
        # 确保序列以user消息结束，以便对话继续
        if clean_messages:
            final_role = clean_messages[-1].get("role")
            if final_role != "user":
                continuation_message = {
                    "role": "user",
                    "content": "Please continue implementing the next file according to the plan priorities."
                }
                clean_messages.append(continuation_message)
                self.logger.info("Added continuation message to ensure proper role ending")
        
        # Final validation: ensure we have a balanced conversation
        # 最终验证：确保对话平衡
        role_counts = {}
        for msg in clean_messages:
            role = msg.get("role")
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # If we only have one type of role, add a minimal counterpart
        # 如果只有一种角色，添加最小的对应角色
        if "user" in role_counts and "assistant" not in role_counts:
            assistant_msg = {
                "role": "assistant", 
                "content": "I will start implementing the code file."
            }
            clean_messages.insert(-1, assistant_msg)  # Insert before the final user message
            self.logger.info("Added assistant message for role balance")
        elif "assistant" in role_counts and "user" not in role_counts:
            user_msg = {
                "role": "user", 
                "content": "Please continue implementing the next file."
            }
            clean_messages.append(user_msg)
            self.logger.info("Added user message for role balance")
        
        return clean_messages
    
    def _validate_role_sequences(self, messages: List[Dict]) -> List[Dict]:
        """
        Validate and correct role sequences in messages
        验证并纠正消息中的角色序列
        """
        validated_messages = []
        
        for i, msg in enumerate(messages):
            validated_msg = msg.copy()
            original_role = msg.get("role", "unknown")
            
            # Ensure role is properly set / 确保角色正确设置
            if original_role == "unknown" or original_role not in ["user", "assistant"]:
                # Try to infer role based on content / 尝试根据内容推断角色
                content = msg.get("content", "")
                if "Tool Result" in content or content.startswith("[") or "CONTINUE" in content.upper():
                    validated_msg["role"] = "user"
                    self.logger.warning(f"Message {i}: role corrected from '{original_role}' to 'user' based on content")
                else:
                    # Default to alternating pattern / 默认使用交替模式
                    expected_role = "assistant" if len(validated_messages) % 2 == 0 else "user"
                    validated_msg["role"] = expected_role
                    self.logger.warning(f"Message {i}: role corrected from '{original_role}' to '{expected_role}' based on position")
            
            # Ensure content is not empty / 确保内容不为空
            if not validated_msg.get("content", "").strip():
                self.logger.warning(f"Message {i}: empty content detected, adding placeholder")
                if validated_msg["role"] == "assistant":
                    validated_msg["content"] = "Continue implementing code..."
                else:
                    validated_msg["content"] = "Continue with the next step implementation"
            
            validated_messages.append(validated_msg)
        
        return validated_messages
    
    def _log_final_role_structure(self, messages: List[Dict]):
        """
        Log final role structure for verification
        记录最终角色结构进行验证
        """
        self.logger.info("Final message structure with roles:")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content_type = self._classify_message_type(msg)
            content_preview = msg.get("content", "")[:60] + "..." if len(msg.get("content", "")) > 60 else msg.get("content", "")
            self.logger.info(f"  [{i:2d}] {role:9s} | {content_type:12s} | {content_preview}")
    
    def _classify_message_type(self, msg: Dict) -> str:
        """
        Classify message type for logging
        分类消息类型用于日志记录
        """
        content = msg.get("content", "")
        
        if "Implementation Plan:" in content:
            return "INITIAL_PLAN"
        elif "[CONVERSATION SUMMARY" in content:
            return "SUMMARY"
        elif "Tool Result" in content:
            return "TOOL_RESULT"
        elif content.startswith("[") and content.endswith("]"):
            return "SYSTEM_MSG"
        else:
            return "CONVERSATION"
    
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