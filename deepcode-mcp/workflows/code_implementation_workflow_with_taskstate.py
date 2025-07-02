"""
Paper Code Implementation Workflow with TaskState - MCP + inspect_ai Compatible

Features:
1. Direct TaskState usage from inspect_ai (移植自 _basic_agent_plus.py)
2. MCP Architecture integration 
3. Full compatibility with paper2code benchmark evaluation
4. Submission and scoring support

Architecture:
- TaskState from inspect_ai.solver._task_state (完全兼容)
- MCP Server integration: tools/code_implementation_server.py  
- Submission mechanism identical to _basic_agent_plus.py
- Configuration: mcp_agent.config.yaml
"""

import asyncio
import json
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, cast
from json import JSONDecodeError

# Import inspect_ai components (移植自 _basic_agent_plus.py)
from inspect_ai._util.format import format_progress_time
from inspect_ai.model._cache import CachePolicy
from inspect_ai.model._call_tools import call_tools
from inspect_ai.model._chat_message import ChatMessageTool, ChatMessageUser
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import get_model
from inspect_ai.scorer._metric import Score, ValueToFloat, value_to_float
from inspect_ai.scorer._score import score
from inspect_ai.solver._chain import chain
from inspect_ai.solver._prompt import system_message
from inspect_ai.solver._solver import Generate, Solver, solver
from inspect_ai.solver._task_state import TaskState
from inspect_ai.solver._use_tools import use_tools
from inspect_ai.tool._tool import Tool, ToolResult, tool
from inspect_ai.tool._tool_with import tool_with
from openai import LengthFinishReasonError
from typing_extensions import TypedDict, Unpack

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

# Try to import utils from _basic_agent_plus.py path
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "paper2code_benchmark", "agent"))
    from utils import generate_patched, prune_messages
except ImportError:
    # Fallback implementations if utils not available
    def generate_patched(*args, **kwargs):
        pass
    def prune_messages(messages, prune_individual=False):
        # Simple pruning - keep first and last 10 messages
        if len(messages) <= 20:
            return messages
        return messages[:10] + messages[-10:]

logger = logging.getLogger(__name__)

# Constants from _basic_agent_plus.py
SYSTEM_MESSAGE_BASE = """
You are a helpful agent attempting to solve a task. You have
several functions available to help with solving the task. Each message may
may perform one function call. You will see the result of the function right
after sending the message. If you need to perform multiple actions, you can
always send more messages with subsequent function calls. Do some reasoning
before your actions, describing what function calls you are going to use and
how they fit into your plan.

Use as much of the available time as possible to refine and optimize your solution: Your goal is to achieve the best possible score, not just to submit your first working solution. Keep working until the time limit expires.

YOU MUST use the bash and python tools provided provided to create your replication. ACTUALLY REPLICATE the paper YOURSELF, do not give me instructions how I can do so.

Only execute a single command using each tool at once.

This is a multi-step task. DO THINGS BIT BY BIT by slowly building up your solution with multiple tool calls, don't try to do everything in one go. Just reason about what is the best next step and execute it. Then take the next step after you receive the output of the previous step.

Do not stop until you have replicated all results from the paper. Before calling end_task, YOU MUST have an entire codebase that can reproduce all metrics, figures, tables, and findings from the paper."""

REPRODUCE_SH_MESSAGE = """
Ensure that running reproduce.sh will reproduce all metrics, figures, tables, and findings from the paper! You should check reproduce.sh YOURSELF to ensure that it actually reproduces all results from the paper to help with your development."""

SYSTEM_MESSAGE_END = """
Remember, you are tasked with replicating a whole paper, not just a single part of it or a minimal example. The file read tool is PAGINATED, so you will need to CALL IT MULTIPLE TIMES to make sure that you have read all the relevant parts of the paper."""

DEFAULT_SYSTEM_MESSAGE = SYSTEM_MESSAGE_BASE + REPRODUCE_SH_MESSAGE + SYSTEM_MESSAGE_END
PB_CODE_ONLY_SYSTEM_MESSAGE = SYSTEM_MESSAGE_BASE + SYSTEM_MESSAGE_END

DEFAULT_INCORRECT_MESSAGE = """
Your submission was incorrect. Please continue implementing missing components to achieve complete paper reproduction.
"""
DEFAULT_CONTINUE_MESSAGE = "Please proceed to the next step using your best judgement. If you believe you are finished, double check your work to continue to refine and improve your submission."
DEFAULT_SUBMIT_NAME = "end_task"
DEFAULT_SUBMIT_DESCRIPTION = (
    "Signal that you are completely finished and that you would like to end the task."
)


class TaskStateWorkflowDeprecatedArgs(TypedDict, total=False):
    max_messages: int | None


class CodeImplementationWorkflowWithTaskState:
    """
    Enhanced Code Implementation Workflow with TaskState
    
    直接使用 inspect_ai 的 TaskState，完全兼容 paper2code benchmark
    结合 MCP 架构进行代码实现和工具调用
    """
    
    # ==================== 1. Class Initialization and Configuration ====================
    
    def __init__(self, config_path: str = "mcp_agent.secrets.yaml"):
        """Initialize workflow with configuration"""
        self.config_path = config_path
        self.api_config = self._load_api_config()
        self.logger = self._create_logger()
        self.mcp_agent = None
        self.code_agent = None
        self.summary_agent = None

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

    # ==================== 2. TaskState Management and Initialization ====================

    async def _initialize_task_state(
        self, 
        plan_content: str, 
        target_directory: str,
        message_limit: Optional[int] = 100,
        token_limit: Optional[int] = None,
        real_time_limit: Optional[int] = 2400  # 40 minutes
    ) -> TaskState:
        """Initialize TaskState for code implementation"""
        
        # Create a minimal TaskState instance
        # Note: TaskState is usually created by inspect_ai framework, 
        # but we create a compatible one for our MCP workflow
        state = TaskState()
        
        # Set basic properties
        state.messages = []
        state.completed = False
        state.message_limit = message_limit
        state.token_limit = token_limit
        
        # Initialize MCP agent and tools
        code_directory = os.path.join(target_directory, "generate_code")
        self.mcp_agent = await self._initialize_mcp_agent(code_directory)
        
        # Initialize specialized agents
        self.code_agent = CodeImplementationAgent(self.mcp_agent, self.logger)
        self.summary_agent = SummaryAgent(self.logger)
        
        # Convert MCP tools to inspect_ai Tool format
        state.tools = await self._convert_mcp_tools_to_inspect_ai()
        
        # Add initial system message (using inspect_ai format)
        system_msg = self._create_system_message_for_paper_code()
        
        # Add initial task message
        initial_message = self._create_initial_task_message(plan_content, code_directory)
        state.messages.append(ChatMessageUser(content=initial_message))
        
        # Add custom attributes for tracking
        state.plan_content = plan_content
        state.current_directory = target_directory
        state.files_implemented = 0
        state.iteration_count = 0
        state.submission_attempts = 0
        state.max_attempts = 3
        state.start_time = time.time()
        state.real_time_limit = real_time_limit
        
        return state

    def _create_system_message_for_paper_code(self) -> str:
        """Create system message optimized for paper code implementation"""
        return """You are a code implementation assistant that reproduces research papers through systematic development.

**Core Mission:** Implement complete, executable code that reproduces ALL paper results (metrics, figures, tables).

**Available MCP Tools:**
- write_file: Create/update code files
- read_file: Read existing files for context  
- search_reference_code: Find relevant reference implementations
- execute_commands: Run shell commands for testing
- get_operation_history: Track implementation progress

**Implementation Strategy:**
1. Analyze reproduction plan thoroughly
2. Implement foundation files first (core classes, utilities)
3. Build dependent files systematically with proper imports
4. Test implementations incrementally
5. Create main execution scripts and reproduce.sh

**Quality Standards:**
- Complete codebase that reproduces paper results
- All metrics, figures, and tables can be generated
- reproduce.sh script works correctly
- Code is well-structured with proper error handling

**Completion:** Call end_task when ALL paper reproduction is complete with comprehensive summary.

Focus on systematic, dependency-aware implementation. Read related files before implementing dependent components."""

    def _create_initial_task_message(self, plan_content: str, code_directory: str) -> str:
        """Create initial task message"""
        return f"""**PAPER CODE REPRODUCTION TASK**

**Implementation Plan:**
{plan_content}

**Working Directory:** {code_directory}

**Objective:** Complete systematic implementation to reproduce ALL paper results:

1. **Phase 1:** Foundation - Core classes, utilities, data processing
2. **Phase 2:** Models - Neural network architectures, training logic  
3. **Phase 3:** Experiments - Training scripts, evaluation, metrics
4. **Phase 4:** Reproduction - Main scripts, reproduce.sh, documentation

**Critical:** This is COMPLETE reproduction, not partial implementation. Every metric, figure, and table from the paper must be reproducible.

Begin with foundation files and build systematically."""

    async def _initialize_mcp_agent(self, code_directory: str) -> Agent:
        """Initialize MCP agent and connect to code-implementation server"""
        try:
            mcp_agent = Agent(
                name="CodeImplementationAgent",
                instruction="Code implementation assistant using MCP tools for paper reproduction.",
                server_names=["code-implementation", "code-reference-indexer"],
            )
            
            await mcp_agent.__aenter__()
            await mcp_agent.attach_llm(AnthropicAugmentedLLM)
            
            return mcp_agent
                
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP agent: {e}")
            raise

    async def _convert_mcp_tools_to_inspect_ai(self) -> List[Tool]:
        """Convert MCP tools to inspect_ai Tool format"""
        # Get MCP tool definitions
        mcp_tools = get_mcp_tools("code_implementation")
        
        inspect_ai_tools = []
        
        # Add end_task tool (from _basic_agent_plus.py)
        @tool
        def end_task() -> Tool:
            async def execute(end_message: str) -> ToolResult:
                """Signal that you are completely finished with the paper reproduction.

                Args:
                  end_message (str): Final summary of what was implemented and reproduction status.
                """
                return end_message
            return execute
        
        inspect_ai_tools.append(tool_with(end_task(), DEFAULT_SUBMIT_NAME, DEFAULT_SUBMIT_DESCRIPTION))
        
        # Convert MCP tools to inspect_ai tools
        for mcp_tool in mcp_tools:
            tool_name = mcp_tool["name"]
            tool_description = mcp_tool["description"]
            
            # Create wrapper tool that calls MCP agent
            def create_mcp_wrapper(name: str, description: str):
                @tool
                def mcp_wrapper() -> Tool:
                    async def execute(**kwargs) -> ToolResult:
                        """Wrapper for MCP tool call"""
                        try:
                            if self.mcp_agent:
                                result = await self.mcp_agent.call_tool(name, kwargs)
                                return ToolResult(content=str(result), success=True)
                            else:
                                return ToolResult(content=f"MCP agent not available for {name}", success=False)
                        except Exception as e:
                            return ToolResult(content=f"Error calling {name}: {str(e)}", success=False)
                    return execute
                
                return tool_with(mcp_wrapper(), name, description)
            
            inspect_ai_tools.append(create_mcp_wrapper(tool_name, tool_description))
        
        return inspect_ai_tools

    # ==================== 3. Main Workflow Execution ====================

    async def run_workflow_with_taskstate(
        self, 
        plan_file_path: str, 
        target_directory: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run complete workflow with TaskState management - Main public interface"""
        try:
            plan_content = self._read_plan_file(plan_file_path)
            
            if target_directory is None:
                target_directory = str(Path(plan_file_path).parent)
            
            self.logger.info(f"Starting TaskState-managed workflow: {plan_file_path}")
            self.logger.info(f"Target directory: {target_directory}")
            
            results = {}
            
            # Check if file tree exists
            if self._check_file_tree_exists(target_directory):
                self.logger.info("File tree exists, skipping creation")
                results["file_tree"] = "Already exists, skipped creation"
            else:
                self.logger.info("Creating file tree...")
                results["file_tree"] = await self.create_file_structure(plan_content, target_directory)
            
            # Initialize TaskState
            state = await self._initialize_task_state(plan_content, target_directory, **kwargs)
            
            # Run TaskState-managed implementation (similar to _basic_agent_plus.py)
            self.logger.info("Starting TaskState-managed code implementation...")
            results["code_implementation"] = await self._run_taskstate_implementation_loop(state)
            
            self.logger.info("TaskState-managed workflow execution successful")
            
            return {
                "status": "success",
                "plan_file": plan_file_path,
                "target_directory": target_directory,
                "code_directory": os.path.join(target_directory, "generate_code"),
                "results": results,
                "final_state": {
                    "messages_count": len(state.messages),
                    "files_implemented": getattr(state, 'files_implemented', 0),
                    "iterations": getattr(state, 'iteration_count', 0),
                    "submission_attempts": getattr(state, 'submission_attempts', 0),
                    "elapsed_time": time.time() - getattr(state, 'start_time', time.time()),
                    "completed": state.completed
                },
                "architecture": "taskstate_mcp_compatible"
            }
            
        except Exception as e:
            self.logger.error(f"TaskState-managed workflow execution failed: {e}")
            return {"status": "error", "message": str(e), "plan_file": plan_file_path}
        finally:
            await self._cleanup_mcp_agent()

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

    # ==================== 4. TaskState Implementation Loop (移植自 _basic_agent_plus.py) ====================

    async def _run_taskstate_implementation_loop(self, state: TaskState) -> str:
        """Run implementation loop using TaskState (移植并适配自 _basic_agent_plus.py)"""
        self.logger.info("Starting TaskState-managed implementation loop...")
        
        # Setup model and tracking (from _basic_agent_plus.py)
        model = get_model()
        setattr(model, "total_retry_time", 0)
        setattr(model, "generate", generate_patched)
        
        # Track attempts and timing
        attempts = 0
        num_steps = 0
        start_time = getattr(state, 'start_time', time.time())
        max_attempts = getattr(state, 'max_attempts', 3)
        real_time_limit = getattr(state, 'real_time_limit', None)
        
        # Main loop (state.completed checks message_limit and token_limit)
        while not state.completed:
            # Proactively prune messages if approaching limit (from _basic_agent_plus.py)
            if "sonnet" in model.name.lower() and len(state.messages) > 900:
                state.messages = prune_messages(state.messages)

            # Update iteration count
            num_steps += 1
            if hasattr(state, 'iteration_count'):
                state.iteration_count += 1

            # Check time limits (from _basic_agent_plus.py)
            productive_time = time.time() - start_time - model.total_retry_time
            self.logger.warning(f"total runtime: {round(time.time() - start_time, 2)}, productive runtime: {round(productive_time, 2)}, retry time: {round(model.total_retry_time, 2)}")
            
            over_time_limit = (productive_time > real_time_limit) if real_time_limit is not None else False
            if real_time_limit is not None and over_time_limit:
                state.completed = True
                break

            # Send progress update every 5 steps (from _basic_agent_plus.py)
            if num_steps % 5 == 0:
                if real_time_limit is not None:
                    elapsed_time = productive_time
                    periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed out of {format_progress_time(real_time_limit)}. Remember, you only have to stop working when the time limit has been reached."
                else:
                    elapsed_time = time.time() - start_time
                    periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed"
                periodic_msg += "\n\nNote: Don't forget to git commit regularly and create comprehensive reproduce.sh!"
                state.messages.append(ChatMessageUser(content=periodic_msg))

            # Generate response (from _basic_agent_plus.py)
            length_finish_error = False
            prune_individual = False
            try:
                # Calculate timeout
                generate_timeout = None
                if real_time_limit is not None:
                    generate_timeout = int(real_time_limit - productive_time)

                # Generate output and append assistant message
                state.output = await model.generate(
                    self=model,
                    input=state.messages,
                    tools=state.tools,
                    cache=False,
                    config=GenerateConfig(timeout=generate_timeout),
                )
                state.messages.append(state.output.message)
                
            except (LengthFinishReasonError, IndexError) as e:
                length_finish_error = True
                if "PRUNE_INDIVIDUAL_MESSAGES" in str(e):
                    prune_individual = True
            except JSONDecodeError:
                state.messages.append(ChatMessageUser(content="The JSON returned was invalid."))
                continue

            # Handle context length overflow (from _basic_agent_plus.py)
            if length_finish_error or (hasattr(state.output, 'stop_reason') and state.output.stop_reason == "model_length"):
                self.logger.warning("context length overflow")
                state.messages = prune_messages(state.messages, prune_individual=prune_individual)
                continue

            # Handle tool calls (from _basic_agent_plus.py)
            if hasattr(state.output.message, 'tool_calls') and state.output.message.tool_calls:
                # Calculate timeout for tool calls
                timeout = None
                if real_time_limit is not None:
                    timeout = int(real_time_limit - productive_time)

                # Call tool functions
                try:
                    async with asyncio.timeout(timeout):
                        tool_results = await call_tools(
                            state.output.message, state.tools, max_output=None
                        )
                except asyncio.TimeoutError:
                    state.messages.append(
                        ChatMessageUser(content="Timeout: The tool call timed out.")
                    )
                    state.completed = True
                    break

                state.messages.extend(tool_results)

                # Check for submission (from _basic_agent_plus.py)
                answer = self._extract_submission_from_tool_results(tool_results)
                if answer:
                    # Set the output to the answer for scoring
                    state.output.completion = answer
                    
                    # Track submission attempt
                    if hasattr(state, 'submission_attempts'):
                        state.submission_attempts += 1

                    # Exit if we are at max_attempts
                    attempts += 1
                    if attempts >= max_attempts:
                        state.completed = True
                        break

                    # Evaluate submission (basic check for our workflow)
                    if await self._evaluate_taskstate_submission(state, answer):
                        self.logger.info("Submission accepted - implementation completed")
                        state.completed = True
                        break
                    else:
                        # Provide feedback and continue
                        response_message = "Your submission was incomplete. Please continue implementing missing paper reproduction components."
                        state.messages.append(ChatMessageUser(content=response_message))

            # No tool calls - urge to continue (from _basic_agent_plus.py)
            else:
                continue_msg = "Please proceed to the next implementation step. Use available tools to continue building the paper reproduction codebase."
                state.messages.append(ChatMessageUser(content=continue_msg))

        return await self._generate_taskstate_final_report(state)

    # ==================== 5. Submission and Evaluation (兼容 _basic_agent_plus.py) ====================

    def _extract_submission_from_tool_results(self, tool_results: List[ChatMessageTool]) -> Optional[str]:
        """Extract submission from tool results (from _basic_agent_plus.py)"""
        return next(
            (result.text for result in tool_results if result.function == DEFAULT_SUBMIT_NAME),
            None,
        )

    async def _evaluate_taskstate_submission(self, state: TaskState, submission: str) -> bool:
        """Evaluate submission for TaskState (compatible with paper2code benchmark)"""
        if not submission:
            return False
        
        # Basic evaluation criteria
        evaluation_score = 0.0
        
        # Check if reproduce.sh exists
        reproduce_script = os.path.join(getattr(state, 'current_directory', ''), "generate_code", "reproduce.sh")
        if os.path.exists(reproduce_script):
            evaluation_score += 0.4
            self.logger.info("✅ reproduce.sh exists")
        else:
            self.logger.warning("❌ reproduce.sh not found")
        
        # Check number of files implemented
        files_implemented = getattr(state, 'files_implemented', 0)
        if self.code_agent:
            files_implemented = self.code_agent.get_files_implemented_count()
            
        if files_implemented >= 5:
            evaluation_score += 0.3
            self.logger.info(f"✅ {files_implemented} files implemented")
        else:
            self.logger.warning(f"❌ Only {files_implemented} files implemented")
        
        # Check submission content quality
        submission_lower = submission.lower()
        quality_keywords = [
            "reproduce", "complete", "implemented", "metrics", "figures", 
            "results", "paper", "all", "finished", "codebase"
        ]
        
        keyword_matches = sum(1 for keyword in quality_keywords if keyword in submission_lower)
        if keyword_matches >= 4:
            evaluation_score += 0.3
            self.logger.info(f"✅ Submission content quality: {keyword_matches}/10 keywords")
        else:
            self.logger.warning(f"❌ Low submission content quality: {keyword_matches}/10 keywords")
        
        # Final evaluation
        is_complete = evaluation_score >= 0.7
        self.logger.info(f"TaskState submission evaluation: {evaluation_score:.2f}, Complete: {is_complete}")
        
        return is_complete

    # ==================== 6. Reporting and Cleanup ====================

    async def _generate_taskstate_final_report(self, state: TaskState) -> str:
        """Generate final report from TaskState"""
        try:
            elapsed_time = time.time() - getattr(state, 'start_time', time.time())
            
            # Get implementation statistics
            code_stats = self.code_agent.get_implementation_statistics() if self.code_agent else {}
            
            # Get operation history if available
            history_data = {"total_operations": 0, "history": []}
            if self.mcp_agent:
                try:
                    history_result = await self.mcp_agent.call_tool("get_operation_history", {"last_n": 30})
                    history_data = json.loads(history_result) if isinstance(history_result, str) else history_result
                except:
                    pass
            
            files_implemented = getattr(state, 'files_implemented', 0)
            if self.code_agent:
                files_implemented = self.code_agent.get_files_implemented_count()
            
            report = f"""
# TaskState-Managed Code Implementation Report

## Execution Summary  
- Implementation iterations: {getattr(state, 'iteration_count', 0)}
- Total elapsed time: {elapsed_time:.2f} seconds
- Messages exchanged: {len(state.messages)}
- Files implemented: {files_implemented}
- Submission attempts: {getattr(state, 'submission_attempts', 0)}
- Final status: {'COMPLETED' if state.completed else 'INCOMPLETE'}

## TaskState Properties
- Message limit: {state.message_limit or 'None'}
- Token limit: {state.token_limit or 'None'}
- Completed: {state.completed}
- Tools available: {len(state.tools)}

## Compatibility
✅ Using inspect_ai TaskState (fully compatible with paper2code benchmark)
✅ MCP tool integration for advanced code implementation  
✅ Submission mechanism identical to _basic_agent_plus.py
✅ Memory management with prune_messages
✅ Time and resource limit handling
✅ Production-grade error handling

## Working Directory
{getattr(state, 'current_directory', 'Unknown')}/generate_code

## Architecture
- TaskState: inspect_ai.solver._task_state (移植自 _basic_agent_plus.py)
- Tools: MCP + inspect_ai hybrid approach
- Evaluation: Compatible with paper2code benchmark scoring
"""
            
            if hasattr(state, 'output') and hasattr(state.output, 'completion'):
                report += f"\n## Final Submission\n{state.output.completion}\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate TaskState final report: {e}")
            return f"Failed to generate final report: {str(e)}"

    async def _cleanup_mcp_agent(self):
        """Clean up MCP agent resources"""
        if self.mcp_agent:
            try:
                await self.mcp_agent.__aexit__(None, None, None)
                self.logger.info("MCP agent connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing MCP agent: {e}")
            finally:
                self.mcp_agent = None


# ==================== 7. Program Entry Point ====================

async def main():
    """Main function for running the TaskState-managed workflow"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    workflow = CodeImplementationWorkflowWithTaskState()
    
    print("=" * 60)
    print("TaskState-Managed Code Implementation Workflow")
    print("Using inspect_ai TaskState - Fully Compatible with paper2code benchmark")
    print("=" * 60)
    
    plan_file = "/Users/lizongwei/Desktop/LLM_research/Code-Agent/deepcode-mcp/agent_folders/papers/1/initial_plan.txt"
    
    print("Running TaskState-Managed Implementation...")
    
    result = await workflow.run_workflow_with_taskstate(
        plan_file,
        message_limit=100,
        real_time_limit=2400  # 40 minutes
    )
    
    print("=" * 60)
    print("TaskState Workflow Results:")
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
        print(f"Completed: {final_state.get('completed', False)}")
        
        print("✅ TaskState execution completed!")
    else:
        print(f"Error Message: {result['message']}")
    
    print("=" * 60)
    print("✅ TaskState + MCP Architecture with Full paper2code Compatibility")


if __name__ == "__main__":
    asyncio.run(main())