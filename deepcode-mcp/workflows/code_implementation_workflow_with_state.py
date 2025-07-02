"""
Paper Code Implementation Workflow with State Management - MCP-compliant with Submission Support

Features:
1. State-based message management (aligned with _basic_agent_plus.py approach)
2. MCP Architecture with submission and scoring capabilities
3. Code Implementation with evaluation support
4. Compatible with paper2code benchmark evaluation framework

Architecture:
- TaskState-like management for messages, tools, and completion status
- MCP Server integration: tools/code_implementation_server.py
- Submission mechanism for evaluation alignment
- Configuration: mcp_agent.config.yaml
"""

import asyncio
import json
import logging
import os
import sys
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# MCP Agent imports
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.code_prompts import STRUCTURE_GENERATOR_PROMPT
from prompts.iterative_code_prompts import PURE_CODE_IMPLEMENTATION_PROMPT
from workflows.agents import CodeImplementationAgent, SummaryAgent
from workflows.iterative_code_implementation import IterativeCodeImplementation
from config.mcp_tool_definitions import get_mcp_tools


@dataclass
class CodeImplementationState:
    """
    State management for code implementation workflow
    Inspired by TaskState from inspect_ai but adapted for MCP architecture
    """
    # Core message management
    messages: List[Dict[str, str]] = field(default_factory=list)
    
    # Tools and MCP agent
    mcp_agent: Optional[Agent] = None
    available_tools: List[Dict[str, Any]] = field(default_factory=list)
    
    # Completion and limits
    completed: bool = False
    message_limit: Optional[int] = None
    token_limit: Optional[int] = None
    real_time_limit: Optional[int] = None
    
    # Implementation tracking
    files_implemented: int = 0
    current_directory: str = ""
    plan_content: str = ""
    
    # Submission and evaluation
    submission_content: Optional[str] = None
    submission_attempts: int = 0
    max_attempts: int = 3
    
    # Performance tracking
    start_time: float = field(default_factory=time.time)
    iteration_count: int = 0
    
    # Agents
    code_agent: Optional[CodeImplementationAgent] = None
    summary_agent: Optional[SummaryAgent] = None
    
    def is_over_limits(self) -> bool:
        """Check if state exceeds configured limits"""
        if self.message_limit and len(self.messages) >= self.message_limit:
            return True
        if self.real_time_limit and (time.time() - self.start_time) >= self.real_time_limit:
            return True
        return False
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation"""
        if content.strip():
            self.messages.append({"role": role, "content": content})
    
    def should_complete(self) -> bool:
        """Determine if the workflow should complete"""
        return (
            self.completed or 
            self.is_over_limits() or 
            self.submission_attempts >= self.max_attempts
        )


class CodeImplementationWorkflowWithState:
    """
    Enhanced Code Implementation Workflow with State Management
    
    Combines:
    1. MCP architecture from original workflow
    2. State management approach from _basic_agent_plus
    3. Submission and evaluation support
    4. Compatible with paper2code benchmark framework
    """
    
    # ==================== 1. Class Initialization and Configuration ====================
    
    def __init__(self, config_path: str = "mcp_agent.secrets.yaml"):
        """Initialize workflow with configuration"""
        self.config_path = config_path
        self.api_config = self._load_api_config()
        self.logger = self._create_logger()

    def _load_api_config(self) -> Dict[str, Any]:
        """Load API configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Failed to load API config: {e}")

    def _create_logger(self) -> logging.Logger:
        """Create and configure logger"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _read_plan_file(self, plan_file_path: str) -> str:
        """Read implementation plan file"""
        plan_path = Path(plan_file_path)
        if not plan_path.exists():
            raise FileNotFoundError(f"Implementation plan file not found: {plan_file_path}")
        
        with open(plan_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _check_file_tree_exists(self, target_directory: str) -> bool:
        """Check if file tree structure already exists"""
        code_directory = os.path.join(target_directory, "generate_code")
        return os.path.exists(code_directory) and len(os.listdir(code_directory)) > 0

    # ==================== 2. State Management and Initialization ====================

    async def _initialize_state(
        self, 
        plan_content: str, 
        target_directory: str,
        message_limit: Optional[int] = 100,
        real_time_limit: Optional[int] = 2400  # 40 minutes
    ) -> CodeImplementationState:
        """Initialize the state for code implementation"""
        state = CodeImplementationState(
            plan_content=plan_content,
            current_directory=target_directory,
            message_limit=message_limit,
            real_time_limit=real_time_limit
        )
        
        # Initialize MCP agent
        code_directory = os.path.join(target_directory, "generate_code")
        state.mcp_agent = await self._initialize_mcp_agent(code_directory)
        
        # Initialize specialized agents
        state.code_agent = CodeImplementationAgent(state.mcp_agent, self.logger)
        state.summary_agent = SummaryAgent(self.logger)
        
        # Prepare tool definitions
        state.available_tools = self._prepare_mcp_tool_definitions()
        
        # Add initial system message
        system_message = self._create_system_message()
        state.add_message("system", system_message)
        
        # Add initial task message
        initial_message = self._create_initial_task_message(plan_content, code_directory)
        state.add_message("user", initial_message)
        
        return state

    def _create_system_message(self) -> str:
        """Create system message for code implementation"""
        return """You are a code implementation assistant that helps reproduce research papers through systematic code development.

**Core Objectives:**
1. Implement complete, working code that reproduces paper results
2. Create all necessary files with proper dependencies
3. Ensure code can be executed to generate all paper metrics/figures
4. Use systematic file-by-file implementation approach

**Available Tools:**
- write_file: Create/update code files
- read_file: Read existing files for context
- search_reference_code: Find relevant reference implementations
- execute_commands: Run shell commands for testing
- get_operation_history: Track implementation progress

**Implementation Strategy:**
1. Analyze the reproduction plan thoroughly
2. Implement foundation files first (core classes, utilities)
3. Build dependent files systematically
4. Test implementations incrementally
5. Create main execution scripts and reproduce.sh

**Submission Criteria:**
- Complete codebase that reproduces paper results
- All metrics, figures, and tables can be generated
- reproduce.sh script works correctly
- Code is well-structured and documented

**When finished:** Call end_task with comprehensive completion summary.

Focus on systematic, dependency-aware implementation. Read related files before implementing dependent components."""

    def _create_initial_task_message(self, plan_content: str, code_directory: str) -> str:
        """Create initial task message"""
        return f"""**Task: Implement complete code reproduction based on the following plan**

**Code Reproduction Plan:**
{plan_content}

**Working Directory:** {code_directory}

**Current Objective:** Begin systematic implementation by:
1. Analyzing the plan structure and dependencies
2. Examining current project layout
3. Implementing foundation files according to dependency order
4. Building towards complete paper reproduction

**Important:** This is a complete reproduction task. Implement ALL components needed to reproduce paper results, not just examples or partial implementations."""

    async def _initialize_mcp_agent(self, code_directory: str) -> Agent:
        """Initialize MCP agent and connect to code-implementation server"""
        try:
            mcp_agent = Agent(
                name="CodeImplementationAgent",
                instruction="You are a code implementation assistant using MCP tools for paper code replication.",
                server_names=["code-implementation", "code-reference-indexer"],
            )
            
            await mcp_agent.__aenter__()
            await mcp_agent.attach_llm(AnthropicAugmentedLLM)
            
            return mcp_agent
                
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP agent: {e}")
            raise

    # ==================== 3. Main Workflow Execution ====================

    async def run_workflow_with_state(
        self, 
        plan_file_path: str, 
        target_directory: Optional[str] = None,
        pure_code_mode: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Run complete workflow with state management - Main public interface"""
        try:
            plan_content = self._read_plan_file(plan_file_path)
            
            if target_directory is None:
                target_directory = str(Path(plan_file_path).parent)
            
            self.logger.info(f"Starting state-managed workflow: {plan_file_path}")
            self.logger.info(f"Target directory: {target_directory}")
            
            results = {}
            
            # Check if file tree exists
            if self._check_file_tree_exists(target_directory):
                self.logger.info("File tree exists, skipping creation")
                results["file_tree"] = "Already exists, skipped creation"
            else:
                self.logger.info("Creating file tree...")
                results["file_tree"] = await self.create_file_structure(plan_content, target_directory)
            
            # Initialize state
            state = await self._initialize_state(plan_content, target_directory, **kwargs)
            
            # Code implementation with state management
            if pure_code_mode:
                self.logger.info("Starting state-managed pure code implementation...")
                results["code_implementation"] = await self._run_pure_code_implementation_with_state(state)
            else:
                self.logger.info("Starting state-managed iterative implementation...")
                results["code_implementation"] = await self._run_iterative_implementation_with_state(state)
            
            self.logger.info("State-managed workflow execution successful")
            
            return {
                "status": "success",
                "plan_file": plan_file_path,
                "target_directory": target_directory,
                "code_directory": os.path.join(target_directory, "generate_code"),
                "results": results,
                "final_state": {
                    "messages_count": len(state.messages),
                    "files_implemented": state.files_implemented,
                    "iterations": state.iteration_count,
                    "submission_attempts": state.submission_attempts,
                    "elapsed_time": time.time() - state.start_time,
                    "submission_content": state.submission_content
                },
                "architecture": "state_managed_mcp"
            }
            
        except Exception as e:
            self.logger.error(f"State-managed workflow execution failed: {e}")
            return {"status": "error", "message": str(e), "plan_file": plan_file_path}
        finally:
            if 'state' in locals() and state.mcp_agent:
                await self._cleanup_mcp_agent(state.mcp_agent)

    async def create_file_structure(self, plan_content: str, target_directory: str) -> str:
        """Create file tree structure based on implementation plan"""
        self.logger.info("Starting file tree creation...")
        
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
            self.logger.info("File tree structure creation completed")
            return result

    # ==================== 4. State-Managed Implementation Loops ====================

    async def _run_pure_code_implementation_with_state(self, state: CodeImplementationState) -> str:
        """Run pure code implementation with state management"""
        self.logger.info("Starting state-managed pure code implementation loop...")
        
        # Initialize LLM client
        client, client_type = await self._initialize_llm_client()
        
        # Main implementation loop with state management
        while not state.should_complete():
            state.iteration_count += 1
            elapsed_time = time.time() - state.start_time
            
            self.logger.info(f"State-managed iteration {state.iteration_count}: {state.files_implemented} files implemented, {elapsed_time:.2f}s elapsed")
            
            # Check time limit
            if state.real_time_limit and elapsed_time >= state.real_time_limit:
                self.logger.warning(f"Time limit reached: {elapsed_time:.2f}s")
                state.completed = True
                break
            
            # Validate and prepare messages
            validated_messages = self._validate_state_messages(state)
            if not validated_messages:
                state.add_message("user", "Please continue implementing code files...")
                validated_messages = self._validate_state_messages(state)
            
            # Call LLM with tools
            response = await self._call_llm_with_tools(
                client, client_type, "", validated_messages, state.available_tools
            )
            
            response_content = response.get("content", "").strip()
            if not response_content:
                response_content = "Continue implementing code files..."
            
            state.add_message("assistant", response_content)
            
            # Handle tool calls
            if response.get("tool_calls"):
                tool_results = await state.code_agent.execute_tool_calls(response["tool_calls"])
                
                # Check for submission (end_task call)
                submission = self._extract_submission_from_tools(response["tool_calls"])
                if submission:
                    state.submission_content = submission
                    state.submission_attempts += 1
                    
                    # Evaluate submission
                    if await self._evaluate_submission(state):
                        self.logger.info("Submission accepted - workflow completed")
                        state.completed = True
                        break
                    else:
                        if state.submission_attempts >= state.max_attempts:
                            self.logger.warning("Max submission attempts reached")
                            state.completed = True
                            break
                        else:
                            feedback = "Your submission was not complete. Please continue implementing missing components."
                            state.add_message("user", feedback)
                            continue
                
                # Process regular tool results
                has_error = self._check_tool_results_for_errors(tool_results)
                state.files_implemented = state.code_agent.get_files_implemented_count()
                
                if has_error:
                    guidance = self._generate_error_guidance()
                else:
                    guidance = self._generate_success_guidance(state.files_implemented)
                
                compiled_response = self._compile_user_response(tool_results, guidance)
                state.add_message("user", compiled_response)
                
            else:
                # No tool calls - urge to continue
                no_tools_guidance = self._generate_no_tools_guidance(state.files_implemented)
                state.add_message("user", no_tools_guidance)
            
            # Memory management with sliding window
            if self._should_trigger_summary(state):
                await self._apply_sliding_window_to_state(state)
            
            # Check completion conditions
            if self._check_completion_keywords(response_content):
                self.logger.info("Implementation completion declared by model")
                # Don't complete immediately - allow for final submission
                state.add_message("user", "Please call end_task to submit your complete implementation.")
        
        return await self._generate_state_final_report(state)

    async def _run_iterative_implementation_with_state(self, state: CodeImplementationState) -> str:
        """Run iterative implementation with state management"""
        # This would be similar to the pure code version but with testing cycles
        # For now, delegate to the pure code version
        return await self._run_pure_code_implementation_with_state(state)

    # ==================== 5. Submission and Evaluation ====================

    def _extract_submission_from_tools(self, tool_calls: List[Dict]) -> Optional[str]:
        """Extract submission content from tool calls"""
        for tool_call in tool_calls:
            if tool_call.get("name") == "end_task":
                return tool_call.get("input", {}).get("end_message", "")
        return None

    async def _evaluate_submission(self, state: CodeImplementationState) -> bool:
        """Evaluate if submission is complete and correct"""
        if not state.submission_content:
            return False
        
        # Basic evaluation criteria
        evaluation_score = 0.0
        
        # Check if reproduce.sh exists
        reproduce_script = os.path.join(state.current_directory, "generate_code", "reproduce.sh")
        if os.path.exists(reproduce_script):
            evaluation_score += 0.3
            self.logger.info("✅ reproduce.sh exists")
        else:
            self.logger.warning("❌ reproduce.sh not found")
        
        # Check number of files implemented
        if state.files_implemented >= 5:
            evaluation_score += 0.3
            self.logger.info(f"✅ {state.files_implemented} files implemented")
        else:
            self.logger.warning(f"❌ Only {state.files_implemented} files implemented")
        
        # Check submission content quality
        submission_lower = state.submission_content.lower()
        quality_keywords = [
            "reproduce", "complete", "implemented", "metrics", "figures", 
            "results", "paper", "all", "finished"
        ]
        
        keyword_matches = sum(1 for keyword in quality_keywords if keyword in submission_lower)
        if keyword_matches >= 3:
            evaluation_score += 0.4
            self.logger.info(f"✅ Submission content quality: {keyword_matches}/10 keywords")
        else:
            self.logger.warning(f"❌ Low submission content quality: {keyword_matches}/10 keywords")
        
        # Final evaluation
        is_complete = evaluation_score >= 0.7
        self.logger.info(f"Submission evaluation score: {evaluation_score:.2f}, Complete: {is_complete}")
        
        return is_complete

    # ==================== 6. LLM Communication ====================

    async def _initialize_llm_client(self):
        """Initialize LLM client (Anthropic or OpenAI)"""
        # Try Anthropic API first
        try:
            anthropic_key = self.api_config.get('anthropic', {}).get('api_key')
            if anthropic_key:
                from anthropic import AsyncAnthropic
                client = AsyncAnthropic(api_key=anthropic_key)
                # Test connection
                await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "test"}]
                )
                self.logger.info("Using Anthropic API")
                return client, "anthropic"
        except Exception as e:
            self.logger.warning(f"Anthropic API unavailable: {e}")
        
        # Try OpenAI API
        try:
            openai_key = self.api_config.get('openai', {}).get('api_key')
            if openai_key:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=openai_key)
                # Test connection
                await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "test"}]
                )
                self.logger.info("Using OpenAI API")
                return client, "openai"
        except Exception as e:
            self.logger.warning(f"OpenAI API unavailable: {e}")
        
        raise ValueError("No available LLM API")

    async def _call_llm_with_tools(self, client, client_type, system_message, messages, tools, max_tokens=16384):
        """Call LLM with tools, adding end_task tool automatically"""
        # Add end_task tool to the available tools
        enhanced_tools = tools + [self._get_end_task_tool_definition()]
        
        try:
            if client_type == "anthropic":
                return await self._call_anthropic_with_tools(client, system_message, messages, enhanced_tools, max_tokens)
            elif client_type == "openai":
                return await self._call_openai_with_tools(client, system_message, messages, enhanced_tools, max_tokens)
            else:
                raise ValueError(f"Unsupported client type: {client_type}")
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise

    def _get_end_task_tool_definition(self) -> Dict[str, Any]:
        """Get end_task tool definition for submission"""
        return {
            "name": "end_task",
            "description": "Signal that you are completely finished with the code implementation and ready to submit",
            "input_schema": {
                "type": "object",
                "properties": {
                    "end_message": {
                        "type": "string",
                        "description": "Final summary of what was implemented and how to reproduce paper results"
                    }
                },
                "required": ["end_message"]
            }
        }

    async def _call_anthropic_with_tools(self, client, system_message, messages, tools, max_tokens):
        """Call Anthropic API"""
        validated_messages = [msg for msg in messages if msg.get("role") != "system"]
        if not validated_messages:
            validated_messages = [{"role": "user", "content": "Please continue implementing code"}]
        
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                system=system_message or "You are a helpful code implementation assistant.",
                messages=validated_messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=0.2
            )
        except Exception as e:
            self.logger.error(f"Anthropic API call failed: {e}")
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
        """Call OpenAI API"""
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
        
        openai_messages = []
        if system_message:
            openai_messages.append({"role": "system", "content": system_message})
        
        for msg in messages:
            if msg.get("role") != "system":
                openai_messages.append(msg)
        
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

    # ==================== 7. Utility Methods ====================

    def _validate_state_messages(self, state: CodeImplementationState) -> List[Dict]:
        """Validate and clean message list from state"""
        valid_messages = []
        for msg in state.messages:
            content = msg.get("content", "").strip()
            role = msg.get("role", "user")
            if content and role != "system":  # Skip system messages for API calls
                valid_messages.append({"role": role, "content": content})
        return valid_messages

    def _prepare_mcp_tool_definitions(self) -> List[Dict[str, Any]]:
        """Prepare tool definitions in Anthropic API standard format"""
        return get_mcp_tools("code_implementation")

    def _check_tool_results_for_errors(self, tool_results: List[Dict]) -> bool:
        """Check tool results for errors"""
        for result in tool_results:
            try:
                if hasattr(result['result'], 'content') and result['result'].content:
                    content_text = result['result'].content[0].text
                    parsed_result = json.loads(content_text)
                    if parsed_result.get('status') == 'error':
                        return True
                elif isinstance(result['result'], str):
                    if "error" in result['result'].lower():
                        return True
            except (json.JSONDecodeError, AttributeError, IndexError):
                result_str = str(result['result'])
                if "error" in result_str.lower():
                    return True
        return False

    def _should_trigger_summary(self, state: CodeImplementationState) -> bool:
        """Determine if summary should be triggered for memory management"""
        return (
            len(state.messages) > 15 and 
            state.iteration_count % 8 == 0 and
            state.summary_agent is not None
        )

    async def _apply_sliding_window_to_state(self, state: CodeImplementationState):
        """Apply sliding window memory optimization to state"""
        if not state.summary_agent:
            return
        
        self.logger.info("Applying sliding window memory optimization...")
        
        # Keep first 2 messages (system + initial task)
        initial_messages = state.messages[:2]
        
        # Generate summary of middle messages
        middle_messages = state.messages[2:-3] if len(state.messages) > 5 else []
        if middle_messages:
            # Create a simple summary
            summary = f"Progress Summary: {state.files_implemented} files implemented, {state.iteration_count} iterations completed. Implementation proceeding systematically."
            
            # Keep recent messages
            recent_messages = state.messages[-3:]
            
            # Reconstruct messages
            state.messages = initial_messages + [
                {"role": "assistant", "content": summary}
            ] + recent_messages
            
            self.logger.info(f"Memory optimized: kept {len(state.messages)} messages")

    def _check_completion_keywords(self, content: str) -> bool:
        """Check if content contains completion keywords"""
        completion_keywords = [
            "implementation complete", 
            "all files implemented",
            "reproduction complete",
            "ready to submit",
            "finished implementing"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in completion_keywords)

    # ==================== 8. User Feedback Generation ====================

    def _generate_success_guidance(self, files_count: int) -> str:
        """Generate success guidance"""
        return f"""✅ File implementation completed successfully! 

📊 **Progress Status:** {files_count} files implemented

🎯 **Next Action:** Continue with dependency-aware implementation workflow.

⚡ **Quick Reminder:**
- Use search_reference_code for unfamiliar file types
- Read related files before implementing dependent files
- Implement exactly ONE complete file per response

📋 **When ready to submit:** Call end_task with comprehensive completion summary."""

    def _generate_error_guidance(self) -> str:
        """Generate error guidance"""
        return """❌ Error detected during file implementation.

🔧 **Action Required:**
1. Review the error details above
2. Fix the identified issue
3. Continue with the next file implementation
4. Ensure proper error handling in future implementations"""

    def _generate_no_tools_guidance(self, files_count: int) -> str:
        """Generate guidance when no tools are called"""
        return f"""⚠️ No tool calls detected in your response.

📊 **Current Progress:** {files_count} files implemented

🚨 **Action Required:** You must use tools to implement the next file.

⚡ **Essential Tools:**
- search_reference_code → read_file → write_file → continue
- end_task (when completely finished)

🚨 **Critical:** Use tools to implement files, not just explanations!"""

    def _compile_user_response(self, tool_results: List[Dict], guidance: str) -> str:
        """Compile tool results and guidance into response"""
        response_parts = []
        
        if tool_results:
            response_parts.append("🔧 **Tool Execution Results:**")
            for tool_result in tool_results:
                tool_name = tool_result['tool_name']
                result_content = tool_result['result']
                response_parts.append(f"```\nTool: {tool_name}\nResult: {result_content}\n```")
        
        if guidance:
            response_parts.append("\n" + guidance)
        
        return "\n\n".join(response_parts)

    # ==================== 9. Reporting ====================

    async def _generate_state_final_report(self, state: CodeImplementationState) -> str:
        """Generate final report from state"""
        try:
            elapsed_time = time.time() - state.start_time
            
            # Get implementation statistics
            code_stats = state.code_agent.get_implementation_statistics() if state.code_agent else {}
            summary_stats = state.summary_agent.get_summary_statistics() if state.summary_agent else {}
            
            # Get operation history if available
            history_data = {"total_operations": 0, "history": []}
            if state.mcp_agent:
                try:
                    history_result = await state.mcp_agent.call_tool("get_operation_history", {"last_n": 30})
                    history_data = json.loads(history_result) if isinstance(history_result, str) else history_result
                except:
                    pass
            
            report = f"""
# State-Managed Code Implementation Completion Report

## Execution Summary
- Implementation iterations: {state.iteration_count}
- Total elapsed time: {elapsed_time:.2f} seconds
- Messages exchanged: {len(state.messages)}
- Files implemented: {state.files_implemented}
- Submission attempts: {state.submission_attempts}
- Final status: {'COMPLETED' if state.completed else 'INCOMPLETE'}

## State Management Performance
- Message limit: {state.message_limit or 'None'}
- Time limit: {state.real_time_limit or 'None'} seconds
- Memory optimizations: {state.iteration_count // 8} sliding window applications
- Total MCP operations: {history_data.get('total_operations', 0)}

## Submission Information
- Submission made: {'Yes' if state.submission_content else 'No'}
- Submission content: {state.submission_content[:200] + '...' if state.submission_content and len(state.submission_content) > 200 else state.submission_content or 'None'}

## Agent Performance
- Code Agent: {len(code_stats)} statistics tracked
- Summary Agent: {len(summary_stats)} statistics tracked

## Architecture Features
✅ State-managed message handling (aligned with paper2code benchmark)
✅ MCP-compliant tool execution
✅ Submission and evaluation framework
✅ Memory optimization with sliding window
✅ Specialized agent separation
✅ Production-grade error handling
✅ Compatible with paper2code evaluation pipeline

## Working Directory
{state.current_directory}/generate_code
"""
            
            if state.submission_content:
                report += f"\n## Final Submission\n{state.submission_content}\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate final report: {e}")
            return f"Failed to generate final report: {str(e)}"

    async def _cleanup_mcp_agent(self, mcp_agent: Agent):
        """Clean up MCP agent resources"""
        if mcp_agent:
            try:
                await mcp_agent.__aexit__(None, None, None)
                self.logger.info("MCP agent connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing MCP agent: {e}")


# ==================== 10. Program Entry Point ====================

async def main():
    """Main function for running the state-managed workflow"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    workflow = CodeImplementationWorkflowWithState()
    
    print("=" * 60)
    print("State-Managed Code Implementation Workflow")
    print("Compatible with paper2code benchmark evaluation")
    print("=" * 60)
    
    plan_file = "/Users/lizongwei/Desktop/LLM_research/Code-Agent/deepcode-mcp/agent_folders/papers/1/initial_plan.txt"
    
    print("Running State-Managed Pure Code Implementation...")
    
    result = await workflow.run_workflow_with_state(
        plan_file, 
        pure_code_mode=True,
        message_limit=100,
        real_time_limit=2400  # 40 minutes
    )
    
    print("=" * 60)
    print("State-Managed Workflow Results:")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"Code Directory: {result['code_directory']}")
        print(f"Architecture: {result.get('architecture', 'unknown')}")
        
        final_state = result.get('final_state', {})
        print(f"Messages: {final_state.get('messages_count', 0)}")
        print(f"Files Implemented: {final_state.get('files_implemented', 0)}")
        print(f"Iterations: {final_state.get('iterations', 0)}")
        print(f"Submissions: {final_state.get('submission_attempts', 0)}")
        print(f"Elapsed Time: {final_state.get('elapsed_time', 0):.2f}s")
        
        if final_state.get('submission_content'):
            print("✅ Submission completed")
        else:
            print("⚠️ No submission made")
        
        print("Execution completed!")
    else:
        print(f"Error Message: {result['message']}")
    
    print("=" * 60)
    print("✅ State-Managed MCP Architecture with Submission Support")


if __name__ == "__main__":
    asyncio.run(main())