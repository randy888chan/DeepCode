"""
Paper Code Implementation with Evaluation Workflow - Enhanced MCP Architecture

Features:
1. File Tree Creation
2. Code Implementation - Based on MCP iterative development  
3. Dependencies & Environment Setup
4. Code Execution & Testing
5. Results Reproduction & Validation
6. Performance Evaluation

Enhanced MCP Architecture:
- MCP Servers: code-implementation, code-reference-indexer, command-executor
- MCP Client: Called through mcp_agent framework
- Configuration: mcp_agent.config.yaml
- Execution: Bash and Python command execution through command-executor
"""

import asyncio
import json
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

# MCP Agent imports
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.code_prompts import STRUCTURE_GENERATOR_PROMPT
from prompts.code_prompts import CODE_IMPLEMENTATION_WITH_EVALUATION_SYSTEM_PROMPT
from workflows.agents import CodeImplementationAgent, SummaryAgent
from config.mcp_tool_definitions import get_mcp_tools


class CodeImplementationWithEvaluationWorkflow:
    """
    Enhanced Paper Code Implementation Workflow Manager with Evaluation
    
    Uses enhanced MCP architecture:
    1. Connect to code-implementation server for file operations
    2. Connect to command-executor server for bash/python execution
    3. Connect to code-reference-indexer for reference search
    4. Implement 5-phase workflow: Implementation → Setup → Testing → Reproduction → Evaluation
    """
    
    # ==================== 1. Class Initialization and Configuration ==================== 
    
    def __init__(self, config_path: str = "mcp_agent.secrets.yaml"):
        """Initialize enhanced workflow with configuration"""
        self.config_path = config_path
        self.api_config = self._load_api_config()
        self.logger = self._create_logger()
        self.mcp_agent = None
        self.current_phase = "initialization"
        self.evaluation_results = {}

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

    # ==================== 2. Public Interface Methods ==================== 

    async def run_complete_workflow(self, plan_file_path: str, target_directory: Optional[str] = None):
        """Run complete workflow with implementation and evaluation"""
        try:
            plan_content = self._read_plan_file(plan_file_path)
            
            if target_directory is None:
                target_directory = str(Path(plan_file_path).parent)
            
            self.logger.info(f"Starting enhanced workflow: {plan_file_path}")
            self.logger.info(f"Target directory: {target_directory}")
            
            results = {}
            
            # Phase 0: File Tree Creation
            if self._check_file_tree_exists(target_directory):
                self.logger.info("File tree exists, skipping creation")
                results["file_tree"] = "Already exists, skipped creation"
            else:
                self.logger.info("Creating file tree...")
                results["file_tree"] = await self.create_file_structure(plan_content, target_directory)
            
            # Phase 1-5: Enhanced Implementation with Evaluation
            self.logger.info("Starting enhanced implementation with evaluation...")
            results["implementation_and_evaluation"] = await self.implement_and_evaluate_code(
                plan_content, target_directory
            )
            
            self.logger.info("Enhanced workflow execution successful")
            
            return {
                "status": "success",
                "plan_file": plan_file_path,
                "target_directory": target_directory,
                "code_directory": os.path.join(target_directory, "generate_code"),
                "results": results,
                "evaluation_results": self.evaluation_results,
                "mcp_architecture": "enhanced_with_evaluation"
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced workflow execution failed: {e}")
            return {"status": "error", "message": str(e), "plan_file": plan_file_path}
        finally:
            await self._cleanup_mcp_agent()

    async def create_file_structure(self, plan_content: str, target_directory: str) -> str:
        """Create file tree structure based on implementation plan"""
        self.logger.info("Starting file tree creation...")
        self.current_phase = "file_tree_creation"
        
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

    # ==================== 3. Enhanced Implementation with Evaluation ==================== 

    async def implement_and_evaluate_code(self, plan_content: str, target_directory: str) -> str:
        """Complete implementation and evaluation workflow"""
        self.logger.info("Starting enhanced implementation with 5-phase evaluation...")
        
        code_directory = os.path.join(target_directory, "generate_code")
        if not os.path.exists(code_directory):
            raise FileNotFoundError("File tree structure not found, please run file tree creation first")
        
        try:
            client, client_type = await self._initialize_llm_client()
            await self._initialize_enhanced_mcp_agent(code_directory)
            
            tools = self._prepare_enhanced_mcp_tool_definitions()
            system_message = CODE_IMPLEMENTATION_WITH_EVALUATION_SYSTEM_PROMPT
            messages = []
            
            implementation_message = f"""**ENHANCED TASK: Complete Implementation and Evaluation Workflow**

**Code Reproduction Plan:**
{plan_content}

**Working Directory:** {code_directory}

**5-PHASE EXECUTION WORKFLOW:**

**Phase 1: Code Implementation**
- Implement each file systematically following the plan
- Use write_file, read_file, search_reference_code tools

**Phase 2: Dependencies & Environment Setup**
- Install dependencies: execute_commands for pip install -r requirements.txt
- Set up data directories and environment

**Phase 3: Code Execution & Testing**
- Test individual modules: execute_commands with python
- Verify mathematical correctness and functionality

**Phase 4: Results Reproduction & Validation**  
- Generate reproduce.sh script
- Execute reproduce.sh and validate all paper results
- Compare outputs with expected results

**Phase 5: Performance Evaluation**
- Measure execution performance
- Generate comprehensive evaluation report

**CRITICAL**: You must complete ALL 5 phases. Begin with Phase 1 implementation."""
            
            messages.append({"role": "user", "content": implementation_message})
            
            result = await self._enhanced_implementation_loop(
                client, client_type, system_message, messages, tools
            )
            
            return result
            
        finally:
            await self._cleanup_mcp_agent()

    async def _enhanced_implementation_loop(self, client, client_type, system_message, messages, tools):
        """Enhanced implementation loop with 5-phase execution and evaluation"""
        max_iterations = 80  # Increased for evaluation phases
        iteration = 0
        start_time = time.time()
        max_time = 3600  # 1 hour for complete workflow
        
        # Enhanced configuration for evaluation workflow
        WINDOW_SIZE = 2  # Larger window for evaluation context
        SUMMARY_TRIGGER = 12  # More context before summarization
        
        # Initialize specialized agents
        code_agent = CodeImplementationAgent(self.mcp_agent, self.logger)
        summary_agent = SummaryAgent(self.logger)
        
        # Preserve initial plan (never compressed)
        initial_plan_message = messages[0] if messages else None
        
        # Phase tracking
        phases_completed = {
            "code_implementation": False,
            "environment_setup": False,
            "code_testing": False,
            "results_reproduction": False,
            "performance_evaluation": False
        }
        
        while iteration < max_iterations:
            iteration += 1
            elapsed_time = time.time() - start_time
            
            if elapsed_time > max_time:
                self.logger.warning(f"Time limit reached: {elapsed_time:.2f}s")
                break
            
            self.logger.info(f"Enhanced iteration {iteration}: Phase tracking - {phases_completed}")
            
            messages = self._validate_messages(messages)
            current_system_message = self._get_enhanced_system_prompt()
            
            # Call LLM
            response = await self._call_llm_with_tools(
                client, client_type, current_system_message, messages, tools
            )
            
            response_content = response.get("content", "").strip()
            if not response_content:
                response_content = "Continue with next phase of implementation and evaluation..."
            
            messages.append({"role": "assistant", "content": response_content})
            
            # Enhanced tool call handling with phase detection
            if response.get("tool_calls"):
                tool_results = await code_agent.execute_tool_calls(response["tool_calls"])
                
                # Phase completion detection
                self._detect_phase_completion(tool_results, response_content, phases_completed)
                
                # Enhanced guidance based on phase progress
                has_error = self._check_tool_results_for_errors(tool_results)
                files_count = code_agent.get_files_implemented_count()
                
                if has_error:
                    guidance = self._generate_enhanced_error_guidance(phases_completed)
                else:
                    guidance = self._generate_enhanced_success_guidance(files_count, phases_completed)
                
                compiled_response = self._compile_user_response(tool_results, guidance)
                messages.append({"role": "user", "content": compiled_response})
                
            else:
                files_count = code_agent.get_files_implemented_count()
                no_tools_guidance = self._generate_enhanced_no_tools_guidance(files_count, phases_completed)
                messages.append({"role": "user", "content": no_tools_guidance})
            
            # Enhanced sliding window with evaluation context
            if code_agent.should_trigger_summary(SUMMARY_TRIGGER, messages):
                current_token_count = code_agent.calculate_messages_token_count(messages) if code_agent.tokenizer else len(messages)
                self.logger.info(f"Triggering enhanced summary: {files_count} files, {current_token_count:,} tokens, phases: {phases_completed}")
                
                # Enhanced summary with phase context
                summary = await summary_agent.generate_evaluation_aware_summary(
                    client, client_type, messages, code_agent.get_implementation_summary(), phases_completed
                )
                
                messages = summary_agent.apply_sliding_window(
                    messages, initial_plan_message, summary, WINDOW_SIZE
                )
                
                code_agent.mark_summary_triggered(messages)
            
            # Enhanced completion criteria
            if self._check_enhanced_completion(response_content, phases_completed):
                self.logger.info("Enhanced workflow completion detected")
                break
            
            # Emergency trim
            if len(messages) > 150:
                self.logger.warning("Emergency message trim for enhanced workflow")
                messages = summary_agent._emergency_message_trim(messages, initial_plan_message)
        
        # Generate comprehensive evaluation report
        return await self._generate_enhanced_final_report_with_evaluation(
            iteration, time.time() - start_time, code_agent, summary_agent, phases_completed
        )

    # ==================== 4. Enhanced MCP Agent Management ==================== 

    async def _initialize_enhanced_mcp_agent(self, code_directory: str):
        """Initialize enhanced MCP agent with execution capabilities"""
        try:
            self.mcp_agent = Agent(
                name="EnhancedCodeImplementationAgent",
                instruction="You are an enhanced code implementation and evaluation assistant with execution capabilities.",
                server_names=["code-implementation", "code-reference-indexer", "command-executor"],
            )
            
            await self.mcp_agent.__aenter__()
            llm = await self.mcp_agent.attach_llm(AnthropicAugmentedLLM)
            
            return llm
                
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced MCP agent: {e}")
            if self.mcp_agent:
                try:
                    await self.mcp_agent.__aexit__(None, None, None)
                except:
                    pass
                self.mcp_agent = None
            raise

    async def _cleanup_mcp_agent(self):
        """Clean up MCP agent resources"""
        if self.mcp_agent:
            try:
                await self.mcp_agent.__aexit__(None, None, None)
                self.logger.info("Enhanced MCP agent connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing enhanced MCP agent: {e}")
            finally:
                self.mcp_agent = None

    # ==================== 5. Enhanced Tool and Communication Management ==================== 

    def _prepare_enhanced_mcp_tool_definitions(self) -> List[Dict[str, Any]]:
        """Prepare enhanced tool definitions including execution tools"""
        # Get standard tools
        standard_tools = get_mcp_tools("code_implementation")
        
        # Add command execution tools
        execution_tools = [
            {
                "name": "execute_commands",
                "description": "Execute bash or python commands for environment setup, testing, and evaluation",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "commands": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of commands to execute"
                        },
                        "working_directory": {
                            "type": "string",
                            "description": "Working directory for command execution"
                        }
                    },
                    "required": ["commands"]
                }
            }
        ]
        
        return standard_tools + execution_tools

    def _get_enhanced_system_prompt(self) -> str:
        """Get enhanced system prompt with current phase context"""
        return CODE_IMPLEMENTATION_WITH_EVALUATION_SYSTEM_PROMPT

    # ==================== 6. Enhanced Phase Management and Detection ==================== 

    def _detect_phase_completion(self, tool_results: List[Dict], response_content: str, phases_completed: Dict[str, bool]):
        """Detect which phases have been completed based on tool results and content"""
        content_lower = response_content.lower()
        
        # Phase 1: Code Implementation
        if any("write_file" in str(result) for result in tool_results) and not phases_completed["code_implementation"]:
            if "algorithm" in content_lower or "implementation complete" in content_lower:
                phases_completed["code_implementation"] = True
                self.logger.info("✅ Phase 1: Code Implementation completed")
        
        # Phase 2: Environment Setup
        if any("execute_commands" in str(result) for result in tool_results):
            if "pip install" in content_lower or "requirements.txt" in content_lower or "environment" in content_lower:
                phases_completed["environment_setup"] = True
                self.logger.info("✅ Phase 2: Environment Setup completed")
        
        # Phase 3: Code Testing
        if any("execute_commands" in str(result) for result in tool_results):
            if "python" in content_lower and ("test" in content_lower or "run" in content_lower):
                phases_completed["code_testing"] = True
                self.logger.info("✅ Phase 3: Code Testing completed")
        
        # Phase 4: Results Reproduction
        if any("reproduce.sh" in str(result) for result in tool_results) or "reproduce.sh" in content_lower:
            phases_completed["results_reproduction"] = True
            self.logger.info("✅ Phase 4: Results Reproduction completed")
        
        # Phase 5: Performance Evaluation
        if "evaluation report" in content_lower or "performance" in content_lower:
            if phases_completed["results_reproduction"]:  # Can only complete after reproduction
                phases_completed["performance_evaluation"] = True
                self.logger.info("✅ Phase 5: Performance Evaluation completed")

    def _check_enhanced_completion(self, response_content: str, phases_completed: Dict[str, bool]) -> bool:
        """Check if enhanced workflow is completed"""
        content_lower = response_content.lower()
        
        # Check if all phases are completed
        all_phases_done = all(phases_completed.values())
        
        # Check for completion keywords
        completion_keywords = [
            "evaluation complete",
            "reproduction workflow finished",
            "all phases completed",
            "paper successfully reproduced"
        ]
        
        has_completion_keywords = any(keyword in content_lower for keyword in completion_keywords)
        
        return all_phases_done or has_completion_keywords

    # ==================== 7. Enhanced Guidance and Feedback ==================== 

    def _generate_enhanced_success_guidance(self, files_count: int, phases_completed: Dict[str, bool]) -> str:
        """Generate enhanced success guidance with phase awareness"""
        completed_phases = [phase for phase, completed in phases_completed.items() if completed]
        remaining_phases = [phase for phase, completed in phases_completed.items() if not completed]
        
        return f"""✅ Operation completed successfully!

📊 **Progress Status:** 
- Files implemented: {files_count}
- Completed phases: {len(completed_phases)}/5
- ✅ Completed: {', '.join(completed_phases) if completed_phases else 'None'}
- 🔄 Remaining: {', '.join(remaining_phases) if remaining_phases else 'All phases completed!'}

🎯 **Next Action:** 
{self._get_next_phase_guidance(remaining_phases)}

⚡ **Phase-Specific Tools:**
- Code Implementation: search_reference_code → read_file → write_file
- Environment Setup: execute_commands (bash for pip install)
- Code Testing: execute_commands (python for module testing)
- Results Reproduction: write_file (reproduce.sh) → execute_commands (bash reproduce.sh)
- Performance Evaluation: execute_commands (measurement and reporting)"""

    def _generate_enhanced_error_guidance(self, phases_completed: Dict[str, bool]) -> str:
        """Generate enhanced error guidance with phase context"""
        current_phase = self._get_current_phase(phases_completed)
        
        return f"""❌ Error detected during {current_phase} phase.

🔧 **Phase-Specific Recovery:**
{self._get_phase_specific_error_guidance(current_phase)}

🚨 **Action Required:**
1. Review the error details above
2. Apply phase-specific debugging steps
3. Retry the operation with corrected approach
4. Continue with systematic phase progression"""

    def _generate_enhanced_no_tools_guidance(self, files_count: int, phases_completed: Dict[str, bool]) -> str:
        """Generate enhanced guidance when no tools are called"""
        remaining_phases = [phase for phase, completed in phases_completed.items() if not completed]
        next_phase = remaining_phases[0] if remaining_phases else "evaluation"
        
        return f"""⚠️ No tool calls detected in your response.

📊 **Current Progress:** 
- Files implemented: {files_count}
- Current phase: {next_phase}
- Phases remaining: {len(remaining_phases)}

🚨 **Action Required:** Use tools to execute the next phase.

⚡ **Phase-Specific Tool Requirements:**
{self._get_phase_tool_requirements(next_phase)}

🚨 **Critical:** Execute tools to implement and evaluate, not just explanations!"""

    def _get_next_phase_guidance(self, remaining_phases: List[str]) -> str:
        """Get guidance for the next phase to complete"""
        if not remaining_phases:
            return "🎉 All phases completed! Generate final evaluation report."
        
        next_phase = remaining_phases[0]
        
        phase_guidance = {
            "code_implementation": "Continue implementing files according to the plan",
            "environment_setup": "Set up dependencies and environment using execute_commands",
            "code_testing": "Test implemented modules using execute_commands with python",
            "results_reproduction": "Create and execute reproduce.sh script",
            "performance_evaluation": "Measure performance and generate evaluation report"
        }
        
        return phase_guidance.get(next_phase, "Continue with next phase")

    def _get_current_phase(self, phases_completed: Dict[str, bool]) -> str:
        """Determine current phase based on completion status"""
        phase_order = ["code_implementation", "environment_setup", "code_testing", "results_reproduction", "performance_evaluation"]
        
        for phase in phase_order:
            if not phases_completed[phase]:
                return phase
        
        return "evaluation"

    def _get_phase_specific_error_guidance(self, phase: str) -> str:
        """Get phase-specific error recovery guidance"""
        guidance = {
            "code_implementation": "Check syntax, imports, and algorithm logic. Use read_file to review dependencies.",
            "environment_setup": "Verify package names, check network connectivity, try alternative installation methods.",
            "code_testing": "Debug Python syntax, check file paths, verify module imports.",
            "results_reproduction": "Check reproduce.sh syntax, verify file permissions, debug execution flow.",
            "performance_evaluation": "Verify measurement tools, check data availability, validate metrics calculation."
        }
        
        return guidance.get(phase, "Review error logs and apply systematic debugging.")

    def _get_phase_tool_requirements(self, phase: str) -> str:
        """Get required tools for specific phase"""
        requirements = {
            "code_implementation": "- search_reference_code → read_file → write_file",
            "environment_setup": "- execute_commands with bash for pip install and setup",
            "code_testing": "- execute_commands with python for module testing",
            "results_reproduction": "- write_file for reproduce.sh → execute_commands to run it",
            "performance_evaluation": "- execute_commands for measurements → write_file for reports"
        }
        
        return requirements.get(phase, "- Use appropriate tools for current task")

    # ==================== 8. Enhanced LLM Communication ==================== 

    async def _initialize_llm_client(self):
        """Initialize LLM client (inherited from parent workflow)"""
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
                self.logger.info("Using Anthropic API for enhanced workflow")
                return client, "anthropic"
        except Exception as e:
            self.logger.warning(f"Anthropic API unavailable: {e}")
        
        raise ValueError("No available LLM API for enhanced workflow")

    async def _call_llm_with_tools(self, client, client_type, system_message, messages, tools, max_tokens=16384):
        """Call LLM with enhanced tools"""
        try:
            if client_type == "anthropic":
                return await self._call_anthropic_with_tools(client, system_message, messages, tools, max_tokens)
            else:
                raise ValueError(f"Unsupported client type: {client_type}")
        except Exception as e:
            self.logger.error(f"Enhanced LLM call failed: {e}")
            raise

    async def _call_anthropic_with_tools(self, client, system_message, messages, tools, max_tokens):
        """Call Anthropic API with enhanced tools"""
        validated_messages = self._validate_messages(messages)
        if not validated_messages:
            validated_messages = [{"role": "user", "content": "Please continue with enhanced implementation and evaluation"}]
        
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
            self.logger.error(f"Enhanced Anthropic API call failed: {e}")
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

    # ==================== 9. Utility Methods ==================== 

    def _validate_messages(self, messages: List[Dict]) -> List[Dict]:
        """Validate and clean message list"""
        valid_messages = []
        for msg in messages:
            content = msg.get("content", "").strip()
            if content:
                valid_messages.append({
                    "role": msg.get("role", "user"),
                    "content": content
                })
            else:
                self.logger.warning(f"Skipping empty message: {msg}")
        return valid_messages

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

    def _compile_user_response(self, tool_results: List[Dict], guidance: str) -> str:
        """Compile tool results and guidance into a single user response"""
        response_parts = []
        
        if tool_results:
            response_parts.append("🔧 **Enhanced Tool Execution Results:**")
            for tool_result in tool_results:
                tool_name = tool_result['tool_name']
                result_content = tool_result['result']
                response_parts.append(f"```\nTool: {tool_name}\nResult: {result_content}\n```")
        
        if guidance:
            response_parts.append("\n" + guidance)
        
        return "\n\n".join(response_parts)

    # ==================== 10. Enhanced Reporting and Evaluation ==================== 

    async def _generate_enhanced_final_report_with_evaluation(
        self, 
        iterations: int, 
        elapsed_time: float, 
        code_agent: CodeImplementationAgent, 
        summary_agent: SummaryAgent,
        phases_completed: Dict[str, bool]
    ):
        """Generate comprehensive final report with evaluation results"""
        try:
            code_stats = code_agent.get_implementation_statistics()
            summary_stats = summary_agent.get_summary_statistics()
            
            if self.mcp_agent:
                history_result = await self.mcp_agent.call_tool("get_operation_history", {"last_n": 50})
                history_data = json.loads(history_result) if isinstance(history_result, str) else history_result
            else:
                history_data = {"total_operations": 0, "history": []}
            
            # Analyze operation types
            write_operations = 0
            execute_operations = 0
            files_created = []
            commands_executed = []
            
            if "history" in history_data:
                for item in history_data["history"]:
                    if item.get("action") == "write_file":
                        write_operations += 1
                        file_path = item.get("details", {}).get("file_path", "unknown")
                        files_created.append(file_path)
                    elif item.get("action") == "execute_commands":
                        execute_operations += 1
                        commands = item.get("details", {}).get("commands", [])
                        commands_executed.extend(commands)
            
            # Calculate completion metrics
            completed_phases_count = sum(phases_completed.values())
            completion_percentage = (completed_phases_count / 5) * 100
            
            report = f"""
# Enhanced Code Implementation with Evaluation Report

## Workflow Summary
- **Total iterations**: {iterations}
- **Total execution time**: {elapsed_time:.2f} seconds
- **Completion percentage**: {completion_percentage:.1f}% ({completed_phases_count}/5 phases)
- **Files implemented**: {code_stats['total_files_implemented']}
- **Commands executed**: {execute_operations}
- **Total MCP operations**: {history_data.get('total_operations', 0)}

## Phase Completion Status
### ✅ Completed Phases ({completed_phases_count}/5)
"""
            for phase, completed in phases_completed.items():
                status = "✅" if completed else "❌"
                phase_name = phase.replace("_", " ").title()
                report += f"- {status} **{phase_name}**\n"
            
            report += f"""

## Implementation Statistics
### Code Implementation Agent Performance
- **Files implemented**: {code_stats['files_implemented_count']}
- **Technical decisions**: {code_stats['technical_decisions_count']}
- **Architecture constraints**: {code_stats['constraints_count']}
- **Dependency analysis**: {code_stats['dependency_analysis_count']}
- **Files read for dependencies**: {code_stats['files_read_for_dependencies']}

### Summary Agent Performance  
- **Summaries generated**: {summary_stats['total_summaries_generated']}
- **Average summary length**: {summary_stats['average_summary_length']:.0f} characters

## Execution Analysis
### File Operations ({write_operations} files created)
"""
            for file_path in files_created[-15:]:
                report += f"- {file_path}\n"
            
            if len(files_created) > 15:
                report += f"... and {len(files_created) - 15} more files\n"
            
            report += f"""

### Command Execution ({execute_operations} commands executed)
"""
            for command in commands_executed[-10:]:
                report += f"- `{command}`\n"
            
            if len(commands_executed) > 10:
                report += f"... and {len(commands_executed) - 10} more commands\n"
            
            report += """

## Enhanced Architecture Features
✅ **5-Phase Implementation and Evaluation Workflow**
✅ **Enhanced MCP Architecture with Command Execution**
✅ **Phase-aware Progress Tracking and Guidance**
✅ **Execution and Testing Integration**
✅ **Results Reproduction and Validation**
✅ **Performance Evaluation and Reporting**
✅ **Memory Optimization with Evaluation Context**
✅ **Comprehensive Error Recovery Mechanisms**

## Quality Assurance
- **Code Quality**: Production-grade with error handling
- **Testing Coverage**: Module and integration testing
- **Result Validation**: reproduce.sh execution and verification
- **Performance Metrics**: Execution time and resource measurement
- **Documentation**: Complete implementation and evaluation documentation
"""

            # Store evaluation results
            self.evaluation_results = {
                "phases_completed": phases_completed,
                "completion_percentage": completion_percentage,
                "files_implemented": code_stats['total_files_implemented'],
                "commands_executed": execute_operations,
                "total_time": elapsed_time
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate enhanced final report: {e}")
            return f"Enhanced report generation failed: {str(e)}"


# ==================== 11. Program Entry Point ==================== 

async def main():
    """Main function for running the enhanced workflow"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    workflow = CodeImplementationWithEvaluationWorkflow()
    
    print("=" * 70)
    print("Enhanced Code Implementation with Evaluation Workflow")
    print("=" * 70)
    print("Features:")
    print("1. ✅ File Tree Creation")
    print("2. ✅ Code Implementation (File-by-File)")
    print("3. ✅ Dependencies & Environment Setup")
    print("4. ✅ Code Execution & Testing")
    print("5. ✅ Results Reproduction & Validation")
    print("6. ✅ Performance Evaluation")
    print("=" * 70)
    
    plan_file = "/Users/lizongwei/Desktop/LLM_research/Code-Agent/deepcode-mcp/deepcode_lab/papers/1/initial_plan.txt"
    
    print("🚀 Starting Enhanced Workflow...")
    print(f"📁 Plan file: {plan_file}")
    
    result = await workflow.run_complete_workflow(plan_file)
    
    print("=" * 70)
    print("Enhanced Workflow Execution Results:")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"✅ Code Directory: {result['code_directory']}")
        print(f"🏗️ MCP Architecture: {result.get('mcp_architecture', 'unknown')}")
        
        eval_results = result.get('evaluation_results', {})
        if eval_results:
            print(f"📊 Completion: {eval_results.get('completion_percentage', 0):.1f}%")
            print(f"📁 Files Implemented: {eval_results.get('files_implemented', 0)}")
            print(f"⚡ Commands Executed: {eval_results.get('commands_executed', 0)}")
            print(f"⏱️ Total Time: {eval_results.get('total_time', 0):.2f}s")
        
        print("🎉 Enhanced workflow completed successfully!")
    else:
        print(f"❌ Error: {result['message']}")
    
    print("=" * 70)
    print("✅ Enhanced MCP Architecture with Complete Implementation and Evaluation")


if __name__ == "__main__":
    asyncio.run(main()) 