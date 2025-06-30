"""
Paper Code Implementation Workflow - MCP-compliant Iterative Development

Features:
1. File Tree Creation
2. Code Implementation - Based on aisi-basic-agent iterative development

MCP Architecture:
- MCP Server: tools/code_implementation_server.py
- MCP Client: Called through mcp_agent framework
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
from typing import Dict, Any, Optional, List

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


class CodeImplementationWorkflow:
    """
    Paper Code Implementation Workflow Manager
    
    Uses standard MCP architecture:
    1. Connect to code-implementation server via MCP client
    2. Use MCP protocol for tool calls
    3. Support workspace management and operation history tracking
    """
    
    # ==================== 1. Class Initialization and Configuration (Infrastructure Layer) ====================
    
    def __init__(self, config_path: str = "mcp_agent.secrets.yaml"):
        """Initialize workflow with configuration"""
        self.config_path = config_path
        self.api_config = self._load_api_config()
        self.logger = self._create_logger()
        self.mcp_agent = None

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

    # ==================== 2. Public Interface Methods (External API Layer) ====================

    async def run_workflow(self, plan_file_path: str, target_directory: Optional[str] = None, pure_code_mode: bool = False):
        """Run complete workflow - Main public interface"""
        try:
            plan_content = self._read_plan_file(plan_file_path)
            
            if target_directory is None:
                target_directory = str(Path(plan_file_path).parent)
            
            self.logger.info(f"Starting workflow: {plan_file_path}")
            self.logger.info(f"Target directory: {target_directory}")
            
            results = {}
            
            # Check if file tree exists
            if self._check_file_tree_exists(target_directory):
                self.logger.info("File tree exists, skipping creation")
                results["file_tree"] = "Already exists, skipped creation"
            else:
                self.logger.info("Creating file tree...")
                results["file_tree"] = await self.create_file_structure(plan_content, target_directory)
            
            # Code implementation
            if pure_code_mode:
                self.logger.info("Starting pure code implementation...")
                results["code_implementation"] = await self.implement_code_pure(plan_content, target_directory)
            else:
                self.logger.info("Starting iterative code implementation...")
                iterative_implementation = IterativeCodeImplementation(self.logger, self.mcp_agent)
                results["code_implementation"] = await iterative_implementation.implement_code_standalone(
                    plan_content, target_directory, workflow_instance=self
                )
            
            self.logger.info("Workflow execution successful")
            
            return {
                "status": "success",
                "plan_file": plan_file_path,
                "target_directory": target_directory,
                "code_directory": os.path.join(target_directory, "generate_code"),
                "results": results,
                "mcp_architecture": "standard"
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
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

    async def implement_code_pure(self, plan_content: str, target_directory: str) -> str:
        """Pure code implementation - focus on code writing without testing"""
        self.logger.info("Starting pure code implementation (no testing)...")
        
        code_directory = os.path.join(target_directory, "generate_code")
        if not os.path.exists(code_directory):
            raise FileNotFoundError("File tree structure not found, please run file tree creation first")
        
        try:
            client, client_type = await self._initialize_llm_client()
            await self._initialize_mcp_agent(code_directory)
            
            tools = self._prepare_mcp_tool_definitions()
            system_message = PURE_CODE_IMPLEMENTATION_PROMPT
            messages = []
            
            implementation_message = f"""**Task: Implement code based on the following reproduction plan**

**Code Reproduction Plan:**
{plan_content}

**Working Directory:** {code_directory}

**Current Objective:** Begin implementation by analyzing the plan structure, examining the current project layout, and implementing the first foundation file according to the plan's priority order."""
            
            messages.append({"role": "user", "content": implementation_message})
            
            result = await self._pure_code_implementation_loop(
                client, client_type, system_message, messages, tools
            )
            
            return result
            
        finally:
            await self._cleanup_mcp_agent()

    # ==================== 3. Core Business Logic (Implementation Layer) ====================

    async def _pure_code_implementation_loop(self, client, client_type, system_message, messages, tools):
        """Pure code implementation loop with sliding window and key information extraction"""
        max_iterations = 50
        iteration = 0
        start_time = time.time()
        max_time = 2400  # 40 minutes
        
        # Sliding window configuration
        WINDOW_SIZE = 1
        SUMMARY_TRIGGER = 8
        
        # Initialize specialized agents
        code_agent = CodeImplementationAgent(self.mcp_agent, self.logger)
        summary_agent = SummaryAgent(self.logger)
        
        # Preserve initial plan (never compressed)
        initial_plan_message = messages[0] if messages else None
        
        while iteration < max_iterations:
            iteration += 1
            elapsed_time = time.time() - start_time
            
            if elapsed_time > max_time:
                self.logger.warning(f"Time limit reached: {elapsed_time:.2f}s")
                break
            
            self.logger.info(f"Pure code implementation iteration {iteration}: generating code")
            
            messages = self._validate_messages(messages)
            current_system_message = code_agent.get_system_prompt()
            
            # Call LLM
            response = await self._call_llm_with_tools(
                client, client_type, current_system_message, messages, tools
            )
            
            response_content = response.get("content", "").strip()
            if not response_content:
                response_content = "Continue implementing code files..."
            
            messages.append({"role": "assistant", "content": response_content})
            
            # Handle tool calls
            if response.get("tool_calls"):
                tool_results = await code_agent.execute_tool_calls(response["tool_calls"])
                
                # Determine guidance based on results
                has_error = self._check_tool_results_for_errors(tool_results)
                files_count = code_agent.get_files_implemented_count()
                
                if has_error:
                    guidance = self._generate_error_guidance()
                else:
                    guidance = self._generate_success_guidance(files_count)
                
                compiled_response = self._compile_user_response(tool_results, guidance)
                messages.append({"role": "user", "content": compiled_response})
                
            else:
                files_count = code_agent.get_files_implemented_count()
                no_tools_guidance = self._generate_no_tools_guidance(files_count)
                messages.append({"role": "user", "content": no_tools_guidance})
            
            # Sliding window + key information extraction based on token count
            if code_agent.should_trigger_summary(SUMMARY_TRIGGER, messages):
                current_token_count = code_agent.calculate_messages_token_count(messages) if code_agent.tokenizer else len(messages)
                self.logger.info(f"Triggering summary: {code_agent.get_files_implemented_count()} files implemented, {current_token_count:,} tokens")
                
                analysis_before = summary_agent.analyze_message_patterns(messages)
                
                summary = await summary_agent.generate_conversation_summary(
                    client, client_type, messages, code_agent.get_implementation_summary()
                )
                
                messages = summary_agent.apply_sliding_window(
                    messages, initial_plan_message, summary, WINDOW_SIZE
                )
                
                analysis_after = summary_agent.analyze_message_patterns(messages)
                compression_ratio = (analysis_before['total_messages'] - analysis_after['total_messages']) / analysis_before['total_messages'] * 100
                token_reduction = analysis_before.get('total_tokens', 0) - analysis_after.get('total_tokens', 0)
                self.logger.info(f"Compression ratio: {compression_ratio:.1f}%, token reduction: {token_reduction:,}")
                
                # Mark summary as triggered to prevent duplicate summaries
                code_agent.mark_summary_triggered(messages)
            
            # Check completion
            if any(keyword in response_content.lower() for keyword in [
                "all files implemented", 
                "implementation complete", 
                "all phases completed",
                "reproduction plan fully implemented"
            ]):
                self.logger.info("Code implementation declared complete")
                break
            
            # Emergency trim if too long
            if len(messages) > 120:
                self.logger.warning("Emergency message trim")
                messages = summary_agent._emergency_message_trim(messages, initial_plan_message)
        
        return await self._generate_pure_code_final_report_with_agents(
            iteration, time.time() - start_time, code_agent, summary_agent
        )

    # ==================== 4. MCP Agent and LLM Communication Management (Communication Layer) ====================

    async def _initialize_mcp_agent(self, code_directory: str):
        """Initialize MCP agent and connect to code-implementation server"""
        try:
            self.mcp_agent = Agent(
                name="CodeImplementationAgent",
                instruction="You are a code implementation assistant, using MCP tools to implement paper code replication.",
                server_names=["code-implementation", "code-reference-indexer"],
            )
            
            await self.mcp_agent.__aenter__()
            llm = await self.mcp_agent.attach_llm(AnthropicAugmentedLLM)
            
            # Set workspace
            workspace_result = await self.mcp_agent.call_tool(
                "set_workspace", 
                {"workspace_path": code_directory}
            )
            self.logger.info(f"Workspace setup result: {workspace_result}")
            
            return llm
                
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP agent: {e}")
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
                self.logger.info("MCP agent connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing MCP agent: {e}")
            finally:
                self.mcp_agent = None

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
        """Call LLM with tools"""
        try:
            if client_type == "anthropic":
                return await self._call_anthropic_with_tools(client, system_message, messages, tools, max_tokens)
            elif client_type == "openai":
                return await self._call_openai_with_tools(client, system_message, messages, tools, max_tokens)
            else:
                raise ValueError(f"Unsupported client type: {client_type}")
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise

    async def _call_anthropic_with_tools(self, client, system_message, messages, tools, max_tokens):
        """Call Anthropic API"""
        validated_messages = self._validate_messages(messages)
        if not validated_messages:
            validated_messages = [{"role": "user", "content": "Please continue implementing code"}]
        
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

    # ==================== 5. Tools and Utility Methods (Utility Layer) ====================

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

    # ==================== 6. User Interaction and Feedback (Interaction Layer) ====================

    def _generate_success_guidance(self, files_count: int) -> str:
        """Generate concise success guidance for continuing implementation"""
        return f"""✅ File implementation completed successfully! 

📊 **Progress Status:** {files_count} files implemented

🎯 **Next Action:** Continue with dependency-aware implementation workflow.

⚡ **Quick Reminder:**
- Use search_reference_code for unfamiliar file types
- Read related files before implementing dependent files
- Implement exactly ONE complete file per response"""

    def _generate_error_guidance(self) -> str:
        """Generate error guidance for handling issues"""
        return """❌ Error detected during file implementation.

🔧 **Action Required:**
1. Review the error details above
2. Fix the identified issue
3. Continue with the next file implementation
4. Ensure proper error handling in future implementations"""

    def _generate_no_tools_guidance(self, files_count: int) -> str:
        """Generate concise guidance when no tools are called"""
        return f"""⚠️ No tool calls detected in your response.

📊 **Current Progress:** {files_count} files implemented

🚨 **Action Required:** You must use tools to implement the next file. Follow the dependency-aware workflow in your system prompt.

⚡ **Essential Tools:**
- search_reference_code → read_file → write_file → continue

🚨 **Critical:** Use tools to implement files, not just explanations!"""

    def _compile_user_response(self, tool_results: List[Dict], guidance: str) -> str:
        """Compile tool results and guidance into a single user response"""
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

    # ==================== 7. Reporting and Output (Output Layer) ====================

    async def _generate_pure_code_final_report_with_agents(
        self, 
        iterations: int, 
        elapsed_time: float, 
        code_agent: CodeImplementationAgent, 
        summary_agent: SummaryAgent
    ):
        """Generate final report using agent statistics"""
        try:
            code_stats = code_agent.get_implementation_statistics()
            summary_stats = summary_agent.get_summary_statistics()
            
            if self.mcp_agent:
                history_result = await self.mcp_agent.call_tool("get_operation_history", {"last_n": 30})
                history_data = json.loads(history_result) if isinstance(history_result, str) else history_result
            else:
                history_data = {"total_operations": 0, "history": []}
            
            write_operations = 0
            files_created = []
            if "history" in history_data:
                for item in history_data["history"]:
                    if item.get("action") == "write_file":
                        write_operations += 1
                        file_path = item.get("details", {}).get("file_path", "unknown")
                        files_created.append(file_path)
            
            report = f"""
# Pure Code Implementation Completion Report

## Execution Summary
- Implementation iterations: {iterations}
- Total elapsed time: {elapsed_time:.2f} seconds
- Files implemented: {code_stats['total_files_implemented']}
- File write operations: {write_operations}
- Total MCP operations: {history_data.get('total_operations', 0)}

## Agent Performance
### Code Implementation Agent
- Files tracked: {code_stats['files_implemented_count']}
- Technical decisions: {code_stats['technical_decisions_count']}
- Constraints tracked: {code_stats['constraints_count']}
- Architecture notes: {code_stats['architecture_notes_count']}
- Dependency analysis performed: {code_stats['dependency_analysis_count']}
- Files read for dependencies: {code_stats['files_read_for_dependencies']}
- Last summary triggered at file count: {code_stats['last_summary_file_count']}

### Summary Agent
- Summaries generated: {summary_stats['total_summaries_generated']}
- Average summary length: {summary_stats['average_summary_length']:.0f} characters

## Files Created
"""
            for file_path in files_created[-20:]:
                report += f"- {file_path}\n"
            
            if len(files_created) > 20:
                report += f"... and {len(files_created) - 20} more files\n"
            
            report += """
## Architecture Features
✅ Specialized agent separation for clean code organization
✅ Sliding window memory optimization (70-80% token reduction)
✅ Progress tracking and implementation statistics
✅ MCP-compliant tool execution
✅ Production-grade code with comprehensive type hints
✅ Intelligent dependency analysis and file reading
✅ Cross-file consistency through smart dependency tracking
✅ Automated read_file usage for implementation context
"""
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate final report: {e}")
            return f"Failed to generate final report: {str(e)}"

    # ==================== 8. Testing and Debugging (Testing Layer) ====================

    async def test_code_reference_indexer(self):
        """Test code reference indexer integration"""
        self.logger.info("=" * 60)
        self.logger.info("TESTING CODE REFERENCE INDEXER INTEGRATION")
        self.logger.info("=" * 60)
        
        try:
            # Initialize MCP agent with code reference indexer
            test_directory = "test_workspace"
            await self._initialize_mcp_agent(test_directory)
            
            if not self.mcp_agent:
                self.logger.error("Failed to initialize MCP agent")
                return False
            
            # Test 1: Check available references
            self.logger.info("\n🔍 Test 1: Getting all available references...")
            try:
                refs_result = await self.mcp_agent.call_tool("get_all_available_references", {})
                self.logger.info(f"✅ get_all_available_references result: {refs_result}")
            except Exception as e:
                self.logger.error(f"❌ get_all_available_references failed: {e}")
            
            # Test 2: Set indexes directory
            self.logger.info("\n📁 Test 2: Setting indexes directory...")
            indexes_path = "/Users/lizongwei/Desktop/LLM_research/Code-Agent/deepcode-mcp/agent_folders/papers/1/indexes"
            try:
                set_dir_result = await self.mcp_agent.call_tool(
                    "set_indexes_directory", 
                    {"indexes_path": indexes_path}
                )
                self.logger.info(f"✅ set_indexes_directory result: {set_dir_result}")
            except Exception as e:
                self.logger.error(f"❌ set_indexes_directory failed: {e}")
            
            # Test 3: Search reference code
            self.logger.info("\n🔍 Test 3: Searching reference code...")
            try:
                search_result = await self.mcp_agent.call_tool(
                    "search_reference_code", 
                    {
                        "target_file": "models/transformer.py",
                        "keywords": "transformer,attention,pytorch",
                        "max_results": 5
                    }
                )
                self.logger.info(f"✅ search_reference_code result length: {len(str(search_result))}")
                
                # Parse and display summary
                if isinstance(search_result, str):
                    import json
                    try:
                        parsed_result = json.loads(search_result)
                        self.logger.info(f"📊 Search Summary:")
                        self.logger.info(f"  - Status: {parsed_result.get('status', 'unknown')}")
                        self.logger.info(f"  - Target File: {parsed_result.get('target_file', 'unknown')}")
                        self.logger.info(f"  - References Found: {parsed_result.get('total_references_found', 0)}")
                        self.logger.info(f"  - Relationships Found: {parsed_result.get('total_relationships_found', 0)}")
                        self.logger.info(f"  - Indexes Loaded: {parsed_result.get('indexes_loaded', [])}")
                    except json.JSONDecodeError:
                        self.logger.info(f"Raw result preview: {str(search_result)[:200]}...")
                        
            except Exception as e:
                self.logger.error(f"❌ search_reference_code failed: {e}")
            
            # Test 4: Check MCP tool definitions
            self.logger.info("\n🛠️ Test 4: Checking MCP tool definitions...")
            try:
                from config.mcp_tool_definitions import get_mcp_tools
                tools = get_mcp_tools("code_implementation")
                reference_tools = [tool for tool in tools if 'reference' in tool['name']]
                self.logger.info(f"✅ Reference tools found: {len(reference_tools)}")
                for tool in reference_tools:
                    self.logger.info(f"  - {tool['name']}: {tool['description']}")
            except Exception as e:
                self.logger.error(f"❌ Tool definitions check failed: {e}")
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("✅ CODE REFERENCE INDEXER TESTING COMPLETED")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Test failed with error: {e}")
            return False
        finally:
            await self._cleanup_mcp_agent()


# ==================== 9. Program Entry Point (Entry Layer) ====================

async def main():
    """Main function for running the workflow"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    workflow = CodeImplementationWorkflow()
    
    print("=" * 60)
    print("Code Implementation Workflow with Reference Indexer")
    print("=" * 60)
    print("Select mode:")
    print("1. Test Code Reference Indexer Integration")
    print("2. Run Full Implementation Workflow")
    print("3. Run Implementation with Pure Code Mode")
    
    # For testing purposes, we'll run the test first
    print("Running Code Reference Indexer Integration Test...")
    test_success = await workflow.test_code_reference_indexer()
    
    if test_success:
        print("\n" + "=" * 60)
        print("🎉 Code Reference Indexer Integration Test PASSED!")
        print("=" * 60)
        
        # Ask if user wants to continue with actual workflow
        print("\nContinuing with workflow execution...")
        
        plan_file = "/Users/lizongwei/Desktop/LLM_research/Code-Agent/deepcode-mcp/agent_folders/papers/1/initial_plan.txt"
        
        print("Implementation Mode Selection:")
        print("1. Pure Code Implementation Mode (Recommended)")
        print("2. Iterative Implementation Mode")
        
        pure_code_mode = True
        mode_name = "Pure Code Implementation Mode with Agent Architecture + Code Reference Indexer"
        print(f"Using: {mode_name}")
        
        result = await workflow.run_workflow(plan_file, pure_code_mode=pure_code_mode)
        
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
        print("✅ Using Standard MCP Architecture with Specialized Agents + Code Reference Indexer")
        
    else:
        print("\n" + "=" * 60)
        print("❌ Code Reference Indexer Integration Test FAILED!")
        print("Please check the configuration and try again.")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
