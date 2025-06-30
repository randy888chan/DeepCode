"""
Code Implementation Agent for File-by-File Development
文件逐个开发的代码实现代理

Handles systematic code implementation with progress tracking and
memory optimization for long-running development sessions.
处理系统性代码实现，具有进度跟踪和长时间开发会话的内存优化。
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional

# Import tiktoken for token calculation
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Import prompts from code_prompts
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from prompts.code_prompts import PURE_CODE_IMPLEMENTATION_SYSTEM_PROMPT


class CodeImplementationAgent:
    """
    Code Implementation Agent for systematic file-by-file development
    用于系统性文件逐个开发的代码实现代理

    Responsibilities / 职责:
    - Track file implementation progress / 跟踪文件实现进度
    - Execute MCP tool calls for code generation / 执行MCP工具调用进行代码生成
    - Monitor implementation status / 监控实现状态
    - Coordinate with Summary Agent for memory optimization / 与总结代理协调进行内存优化
    - Calculate token usage for context management / 计算token使用量用于上下文管理
    """

    def __init__(self, mcp_agent, logger: Optional[logging.Logger] = None):
        """
        Initialize Code Implementation Agent
        初始化代码实现代理

        Args:
            mcp_agent: MCP agent instance for tool calls
            logger: Logger instance for tracking operations
        """
        self.mcp_agent = mcp_agent
        self.logger = logger or self._create_default_logger()
        self.implementation_summary = {
            "completed_files": [],
            "technical_decisions": [],
            "important_constraints": [],
            "architecture_notes": [],
            "dependency_analysis": [],  # Track dependency analysis and file reads
        }
        self.files_implemented_count = 0
        self.implemented_files_set = set()  # Track unique file paths to avoid duplicate counting / 跟踪唯一文件路径以避免重复计数
        self.files_read_for_dependencies = (
            set()
        )  # Track files read for dependency analysis / 跟踪为依赖分析而读取的文件
        self.last_summary_file_count = 0  # Track the file count when last summary was triggered / 跟踪上次触发总结时的文件数
        
        # Token calculation settings / Token计算设置
        self.max_context_tokens = 200000  # Default max context tokens for Claude-3.5-Sonnet / Claude-3.5-Sonnet的默认最大上下文tokens
        self.token_buffer = 10000  # Safety buffer before reaching max / 达到最大值前的安全缓冲区
        self.summary_trigger_tokens = self.max_context_tokens - self.token_buffer  # Trigger summary when approaching limit / 接近限制时触发总结
        self.last_summary_token_count = 0  # Track token count when last summary was triggered / 跟踪上次触发总结时的token数
        
        # Initialize tokenizer / 初始化tokenizer
        if TIKTOKEN_AVAILABLE:
            try:
                # Use Claude-3 tokenizer (approximation with OpenAI's o200k_base) / 使用Claude-3 tokenizer（用OpenAI的o200k_base近似）
                self.tokenizer = tiktoken.get_encoding("o200k_base")
                self.logger.info("Token calculation enabled with o200k_base encoding")
            except Exception as e:
                self.tokenizer = None
                self.logger.warning(f"Failed to initialize tokenizer: {e}")
        else:
            self.tokenizer = None
            self.logger.warning("tiktoken not available, token-based summary triggering disabled")

    def _create_default_logger(self) -> logging.Logger:
        """Create default logger if none provided / 如果未提供则创建默认日志记录器"""
        logger = logging.getLogger(f"{__name__}.CodeImplementationAgent")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for code implementation
        获取代码实现的系统提示词
        """
        return PURE_CODE_IMPLEMENTATION_SYSTEM_PROMPT

    async def execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """
        Execute MCP tool calls and track implementation progress
        执行MCP工具调用并跟踪实现进度

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool execution results
        """
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]

            self.logger.info(f"Executing MCP tool: {tool_name}")

            try:
                if self.mcp_agent:
                    # Execute tool call through MCP protocol / 通过MCP协议执行工具调用
                    result = await self.mcp_agent.call_tool(tool_name, tool_input)

                    # Track file implementation progress / 跟踪文件实现进度
                    if tool_name == "write_file":
                        self._track_file_implementation(tool_call, result)
                    elif tool_name == "read_file":
                        self._track_dependency_analysis(tool_call, result)

                    results.append(
                        {
                            "tool_id": tool_call["id"],
                            "tool_name": tool_name,
                            "result": result,
                        }
                    )
                else:
                    results.append(
                        {
                            "tool_id": tool_call["id"],
                            "tool_name": tool_name,
                            "result": json.dumps(
                                {
                                    "status": "error",
                                    "message": "MCP agent not initialized",
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )

            except Exception as e:
                self.logger.error(f"MCP tool execution failed: {e}")
                results.append(
                    {
                        "tool_id": tool_call["id"],
                        "tool_name": tool_name,
                        "result": json.dumps(
                            {"status": "error", "message": str(e)}, ensure_ascii=False
                        ),
                    }
                )

        return results

    def _track_file_implementation(self, tool_call: Dict, result: Any):
        """
        Track file implementation progress
        跟踪文件实现进度
        """
        try:
            # Handle different result types from MCP / 处理MCP的不同结果类型
            result_data = None

            # Check if result is a CallToolResult object / 检查结果是否为CallToolResult对象
            if hasattr(result, "content"):
                # Extract content from CallToolResult / 从CallToolResult提取内容
                if hasattr(result.content, "text"):
                    result_content = result.content.text
                else:
                    result_content = str(result.content)

                # Try to parse as JSON / 尝试解析为JSON
                try:
                    result_data = json.loads(result_content)
                except json.JSONDecodeError:
                    # If not JSON, create a structure / 如果不是JSON，创建一个结构
                    result_data = {
                        "status": "success",
                        "file_path": tool_call["input"].get("file_path", "unknown"),
                    }
            elif isinstance(result, str):
                # Try to parse string result / 尝试解析字符串结果
                try:
                    result_data = json.loads(result)
                except json.JSONDecodeError:
                    result_data = {
                        "status": "success",
                        "file_path": tool_call["input"].get("file_path", "unknown"),
                    }
            elif isinstance(result, dict):
                # Direct dictionary result / 直接字典结果
                result_data = result
            else:
                # Fallback: assume success and extract file path from input / 后备方案：假设成功并从输入中提取文件路径
                result_data = {
                    "status": "success",
                    "file_path": tool_call["input"].get("file_path", "unknown"),
                }

            # Extract file path for tracking / 提取文件路径用于跟踪
            file_path = None
            if result_data and result_data.get("status") == "success":
                file_path = result_data.get(
                    "file_path", tool_call["input"].get("file_path", "unknown")
                )
            else:
                file_path = tool_call["input"].get("file_path")

            # Only count unique files, not repeated tool calls on same file / 只计数唯一文件，不重复计数同一文件的工具调用
            if file_path and file_path not in self.implemented_files_set:
                # This is a new file implementation / 这是一个新的文件实现
                self.implemented_files_set.add(file_path)
                self.files_implemented_count += 1

                # Add to completed files list / 添加到已完成文件列表
                self.implementation_summary["completed_files"].append(
                    {
                        "file": file_path,
                        "iteration": self.files_implemented_count,
                        "timestamp": time.time(),
                        "size": result_data.get("size", 0) if result_data else 0,
                    }
                )

                self.logger.info(
                    f"New file implementation tracked: count={self.files_implemented_count}, file={file_path}"
                )
            elif file_path and file_path in self.implemented_files_set:
                # This file was already implemented (duplicate tool call) / 这个文件已经被实现过了（重复工具调用）
                self.logger.debug(
                    f"File already tracked, skipping duplicate count: {file_path}"
                )
            else:
                # No valid file path found / 没有找到有效的文件路径
                self.logger.warning("No valid file path found for tracking")

        except Exception as e:
            self.logger.warning(f"Failed to track file implementation: {e}")
            # Even if tracking fails, try to count based on tool input (but check for duplicates) / 即使跟踪失败，也尝试根据工具输入计数（但检查重复）
            try:
                file_path = tool_call["input"].get("file_path")
                if file_path and file_path not in self.implemented_files_set:
                    self.implemented_files_set.add(file_path)
                    self.files_implemented_count += 1
                    self.logger.info(
                        f"File implementation counted (emergency fallback): count={self.files_implemented_count}, file={file_path}"
                    )
            except:
                pass

    def _track_dependency_analysis(self, tool_call: Dict, result: Any):
        """
        Track dependency analysis through read_file calls
        跟踪通过read_file调用进行的依赖分析
        """
        try:
            file_path = tool_call["input"].get("file_path")
            if file_path:
                # Track unique files read for dependency analysis / 跟踪为依赖分析而读取的唯一文件
                if file_path not in self.files_read_for_dependencies:
                    self.files_read_for_dependencies.add(file_path)

                    # Add to dependency analysis summary / 添加到依赖分析总结
                    self.implementation_summary["dependency_analysis"].append(
                        {
                            "file_read": file_path,
                            "timestamp": time.time(),
                            "purpose": "dependency_analysis",
                        }
                    )

                    self.logger.info(
                        f"Dependency analysis tracked: file_read={file_path}"
                    )

        except Exception as e:
            self.logger.warning(f"Failed to track dependency analysis: {e}")

    def calculate_messages_token_count(self, messages: List[Dict]) -> int:
        """
        Calculate total token count for a list of messages
        计算消息列表的总token数

        Args:
            messages: List of chat messages with 'role' and 'content' keys

        Returns:
            Total token count
        """
        if not self.tokenizer:
            # Fallback: rough estimation based on character count / 回退：基于字符数的粗略估计
            total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
            # Rough approximation: 1 token ≈ 4 characters / 粗略近似：1个token ≈ 4个字符
            return total_chars // 4
        
        try:
            total_tokens = 0
            for message in messages:
                content = str(message.get("content", ""))
                role = message.get("role", "")
                
                # Count tokens for content / 计算内容的token数
                if content:
                    content_tokens = len(self.tokenizer.encode(content, disallowed_special=()))
                    total_tokens += content_tokens
                
                # Add tokens for role and message structure / 为角色和消息结构添加token
                role_tokens = len(self.tokenizer.encode(role, disallowed_special=()))
                total_tokens += role_tokens + 4  # Extra tokens for message formatting / 消息格式化的额外token
            
            return total_tokens
            
        except Exception as e:
            self.logger.warning(f"Token calculation failed: {e}")
            # Fallback estimation / 回退估计
            total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
            return total_chars // 4

    def should_trigger_summary_by_tokens(self, messages: List[Dict]) -> bool:
        """
        Check if summary should be triggered based on token count
        根据token数检查是否应触发总结

        Args:
            messages: Current conversation messages

        Returns:
            True if summary should be triggered based on token count
        """
        if not messages:
            return False
        
        # Calculate current token count / 计算当前token数
        current_token_count = self.calculate_messages_token_count(messages)
        
        # Check if we should trigger summary / 检查是否应触发总结
        should_trigger = (
            current_token_count > self.summary_trigger_tokens and
            current_token_count > self.last_summary_token_count + 10000  # Minimum 10k tokens between summaries / 总结间最少10k tokens
        )
        
        if should_trigger:
            self.logger.info(
                f"Token-based summary trigger: current={current_token_count:,}, "
                f"threshold={self.summary_trigger_tokens:,}, "
                f"last_summary={self.last_summary_token_count:,}"
            )
        
        return should_trigger

    def should_trigger_summary(self, summary_trigger: int = 5, messages: List[Dict] = None) -> bool:
        """
        Check if summary should be triggered based on token count (preferred) or file count (fallback)
        根据token数（首选）或文件数（回退）检查是否应触发总结

        Args:
            summary_trigger: Number of files after which to trigger summary (fallback)
            messages: Current conversation messages for token calculation

        Returns:
            True if summary should be triggered
        """
        # Primary: Token-based triggering / 主要：基于token的触发
        if messages and self.tokenizer:
            return self.should_trigger_summary_by_tokens(messages)
        
        # Fallback: File-based triggering (original logic) / 回退：基于文件的触发（原始逻辑）
        self.logger.info("Using fallback file-based summary triggering")
        should_trigger = (
            self.files_implemented_count > 0
            and self.files_implemented_count % summary_trigger == 0
            and self.files_implemented_count > self.last_summary_file_count
        )

        return should_trigger

    def mark_summary_triggered(self, messages: List[Dict] = None):
        """
        Mark that summary has been triggered for current state
        标记当前状态的总结已被触发

        Args:
            messages: Current conversation messages for token tracking
        """
        # Update file-based tracking / 更新基于文件的跟踪
        self.last_summary_file_count = self.files_implemented_count
        
        # Update token-based tracking / 更新基于token的跟踪
        if messages and self.tokenizer:
            self.last_summary_token_count = self.calculate_messages_token_count(messages)
            self.logger.info(
                f"Summary marked as triggered - file_count: {self.files_implemented_count}, "
                f"token_count: {self.last_summary_token_count:,}"
            )
        else:
            self.logger.info(
                f"Summary marked as triggered for file count: {self.files_implemented_count}"
            )

    def get_implementation_summary(self) -> Dict[str, Any]:
        """
        Get current implementation summary
        获取当前实现总结
        """
        return self.implementation_summary.copy()

    def get_files_implemented_count(self) -> int:
        """
        Get the number of files implemented so far
        获取到目前为止实现的文件数量
        """
        return self.files_implemented_count

    def add_technical_decision(self, decision: str, context: str = ""):
        """
        Add a technical decision to the implementation summary
        向实现总结添加技术决策

        Args:
            decision: Description of the technical decision
            context: Additional context for the decision
        """
        self.implementation_summary["technical_decisions"].append(
            {"decision": decision, "context": context, "timestamp": time.time()}
        )
        self.logger.info(f"Technical decision recorded: {decision}")

    def add_constraint(self, constraint: str, impact: str = ""):
        """
        Add an important constraint to the implementation summary
        向实现总结添加重要约束

        Args:
            constraint: Description of the constraint
            impact: Impact of the constraint on implementation
        """
        self.implementation_summary["important_constraints"].append(
            {"constraint": constraint, "impact": impact, "timestamp": time.time()}
        )
        self.logger.info(f"Constraint recorded: {constraint}")

    def add_architecture_note(self, note: str, component: str = ""):
        """
        Add an architecture note to the implementation summary
        向实现总结添加架构注释

        Args:
            note: Architecture note description
            component: Related component or module
        """
        self.implementation_summary["architecture_notes"].append(
            {"note": note, "component": component, "timestamp": time.time()}
        )
        self.logger.info(f"Architecture note recorded: {note}")

    def get_implementation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive implementation statistics
        获取全面的实现统计信息
        """
        return {
            "total_files_implemented": self.files_implemented_count,
            "files_implemented_count": self.files_implemented_count,
            "technical_decisions_count": len(
                self.implementation_summary["technical_decisions"]
            ),
            "constraints_count": len(
                self.implementation_summary["important_constraints"]
            ),
            "architecture_notes_count": len(
                self.implementation_summary["architecture_notes"]
            ),
            "dependency_analysis_count": len(
                self.implementation_summary["dependency_analysis"]
            ),
            "files_read_for_dependencies": len(self.files_read_for_dependencies),
            "unique_files_implemented": len(self.implemented_files_set),
            "completed_files_list": [
                f["file"] for f in self.implementation_summary["completed_files"]
            ],
            "dependency_files_read": list(self.files_read_for_dependencies),
            "last_summary_file_count": self.last_summary_file_count,
        }

    def reset_implementation_tracking(self):
        """
        Reset implementation tracking (useful for new sessions)
        重置实现跟踪（对新会话有用）
        """
        self.implementation_summary = {
            "completed_files": [],
            "technical_decisions": [],
            "important_constraints": [],
            "architecture_notes": [],
            "dependency_analysis": [],  # Reset dependency analysis and file reads
        }
        self.files_implemented_count = 0
        self.implemented_files_set = (
            set()
        )  # Reset the unique files set / 重置唯一文件集合
        self.files_read_for_dependencies = (
            set()
        )  # Reset files read for dependency analysis / 重置为依赖分析而读取的文件
        self.last_summary_file_count = 0  # Reset the file count when last summary was triggered / 重置上次触发总结时的文件数
        self.last_summary_token_count = 0  # Reset token count when last summary was triggered / 重置上次触发总结时的token数
        self.logger.info("Implementation tracking reset")
