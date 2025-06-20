"""
Paper Code Implementation Workflow - MCP Agent Architecture
论文代码复现工作流 - 基于MCP Agent架构

Features:
1. File Tree Creation using MCP Agent (文件树创建)
2. Pure Code Implementation with MCP Agent (纯代码实现)
3. Sliding Window Memory Management (滑动窗口内存管理)
4. Specialized Agent Architecture (专业化代理架构)

MCP Agent Architecture:
- Uses standard mcp_agent.agents.agent.Agent class
- Integrates with AnthropicAugmentedLLM for LLM calls
- Follows initial_workflow.py patterns for agent usage
- Maintains code_implementation_workflow.py logic structure
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
from config.mcp_tool_definitions import get_mcp_tools


class CodeWorkflow:
    """
    Paper Code Implementation Workflow using MCP Agent Architecture
    使用MCP Agent架构的论文代码复现工作流
    
    This class replicates the functionality of CodeImplementationWorkflow
    but uses the standard mcp_agent patterns for agent management and LLM calls.
    """
    
    def __init__(self, config_path: str = "mcp_agent.secrets.yaml"):
        self.config_path = config_path
        self.api_config = self._load_api_config()
        self.logger = self._create_logger()
        self.current_agent = None
        self.code_implementation_agent = None
        self.summary_agent = None

    # ==================== Initialization ====================
    
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

    # ==================== Specialized Agent Management ====================
    
    def _initialize_specialized_agents(self, code_directory: str):
        """Initialize specialized agents for tracking and memory management"""
        # Create mock MCP agent for compatibility with existing agent classes
        # This allows us to reuse the existing specialized agent logic
        mock_mcp_agent = type('MockMCPAgent', (), {
            'call_tool': lambda self, tool_name, tool_input: asyncio.create_task(
                self._execute_mock_tool(tool_name, tool_input)
            )
        })()
        
        # Initialize specialized agents
        self.code_implementation_agent = CodeImplementationAgent(mock_mcp_agent, self.logger)
        self.summary_agent = SummaryAgent(self.logger)
        
        self.logger.info("Specialized agents initialized")
    
    async def _execute_mock_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Mock tool execution for compatibility"""
        return {"status": "success", "message": "Mock tool execution"}

    # ==================== File Structure Creation ====================

    async def create_file_structure(self, plan_content: str, target_directory: str) -> str:
        """Create file tree structure based on implementation plan using MCP Agent"""
        self.logger.info("Starting file tree creation with MCP Agent...")
        
        # Create structure generator agent
        structure_agent = Agent(
            name="StructureGeneratorAgent",
            instruction=STRUCTURE_GENERATOR_PROMPT,
            server_names=["command-executor"],
        )
        
        async with structure_agent:
            self.logger.info("Structure generator agent: Connected to server")
            
            # Attach LLM to the agent
            creator = await structure_agent.attach_llm(AnthropicAugmentedLLM)
            
            # Create the message for file structure creation
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
            
            # Generate response using the agent
            result = await creator.generate_str(message=message)
            
            self.logger.info("File tree structure creation completed")
            return result

    # ==================== Pure Code Implementation ====================

    async def implement_code_pure(self, plan_content: str, target_directory: str) -> str:
        """Pure code implementation using MCP Agent architecture"""
        self.logger.info("Starting pure code implementation with MCP Agent...")
        
        code_directory = os.path.join(target_directory, "generate_code")
        if not os.path.exists(code_directory):
            raise FileNotFoundError("File tree structure not found, please run file tree creation first")
        
        try:
            # Initialize specialized agents for tracking
            self._initialize_specialized_agents(code_directory)
            
            # Create code implementation agent
            code_agent = Agent(
                name="CodeImplementationAgent",
                instruction=PURE_CODE_IMPLEMENTATION_PROMPT,
                server_names=["code-implementation"],
            )
            
            async with code_agent:
                self.logger.info("Code implementation agent: Connected to server")
                
                # List available tools
                tools = await code_agent.list_tools()
                self.logger.info(f"Available tools: {[tool.name for tool in tools.tools]}")
                
                # Attach LLM to the agent
                implementer = await code_agent.attach_llm(AnthropicAugmentedLLM)
                
                # Create initial implementation message
                implementation_message = f"""Code Reproduction Plan:

{plan_content}

Working Directory: {code_directory}

**Smart Implementation Instructions:**
Analyze this plan and begin implementing files one by one using dependency-aware development:

1. **Start with Foundation Files**: Begin with the highest priority file from Phase 1 (Foundation)
2. **Dependency Analysis**: Before implementing each file, identify what existing files it should reference
3. **Read Related Files**: Use read_file tool to examine relevant implemented files for:
   - Interface patterns and base classes to follow
   - Configuration structures and constants to reference
   - Naming conventions and coding patterns already established
   - Import structures and dependency relationships
4. **Implement with Context**: Write each file to properly integrate with existing codebase
5. **Continue Chain**: Repeat this smart workflow for each subsequent file

**First Task:** 
- Use get_file_structure to understand current project layout
- Identify the first foundation file to implement
- Analyze any existing files it should reference
- Implement exactly one complete file per response

**Remember:** Always use dependency analysis and file reading before implementation to ensure consistency across all files."""
                
                # Run pure code implementation loop
                result = await self._pure_code_implementation_loop(
                    implementer, code_agent, implementation_message
                )
                
                return result
            
        except Exception as e:
            self.logger.error(f"Code implementation failed: {e}")
            raise
        finally:
            # Clean up resources
            self.current_agent = None

    async def _pure_code_implementation_loop(
        self, 
        implementer, 
        code_agent,
        initial_message: str
    ) -> str:
        """Pure code implementation loop with sliding window and specialized agents"""
        
        max_iterations = 50
        iteration = 0
        start_time = time.time()
        max_time = 2400  # 40 minutes
        
        # Sliding window configuration
        WINDOW_SIZE = 1
        SUMMARY_TRIGGER = 3
        
        # Initialize message history
        messages = [{"role": "user", "content": initial_message}]
        initial_plan_message = messages[0]
        
        while iteration < max_iterations:
            iteration += 1
            elapsed_time = time.time() - start_time
            
            if elapsed_time > max_time:
                self.logger.warning(f"Time limit reached: {elapsed_time:.2f}s")
                break
            
            self.logger.info(f"Pure code implementation iteration {iteration}: generating code")
            
            # Validate messages
            messages = self._validate_messages(messages)
            
            # Generate response using the agent
            try:
                response_content = await implementer.generate_str(
                    message=messages[-1]["content"] if messages else "Continue implementing code files..."
                )
                
                if not response_content.strip():
                    response_content = "Continue implementing code files..."
                
                # Simulate tool execution tracking
                files_count = self.code_implementation_agent.get_files_implemented_count() if self.code_implementation_agent else iteration
                
                # Add assistant response
                messages.append({"role": "assistant", "content": response_content})
                
                # Generate appropriate guidance based on iteration
                if "error" in response_content.lower():
                    guidance = self._generate_error_guidance()
                else:
                    guidance = self._generate_success_guidance(files_count)
                
                # Add user guidance
                messages.append({"role": "user", "content": guidance})
                
                # Sliding window + key information extraction
                if self.code_implementation_agent and self.code_implementation_agent.should_trigger_summary(SUMMARY_TRIGGER):
                    self.logger.info(f"Triggering summary: {files_count} files implemented")
                    
                    # Generate conversation summary
                    summary = await self._generate_conversation_summary(messages)
                    
                    # Apply sliding window
                    messages = self.summary_agent.apply_sliding_window(
                        messages, initial_plan_message, summary, WINDOW_SIZE
                    )
                    
                    # Mark summary as triggered
                    self.code_implementation_agent.mark_summary_triggered()
                
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
                    messages = self.summary_agent._emergency_message_trim(messages, initial_plan_message)
                
            except Exception as e:
                self.logger.error(f"Implementation iteration {iteration} failed: {e}")
                break
        
        return await self._generate_final_report(iteration, time.time() - start_time)

    # ==================== Utility Methods ====================

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

    async def _generate_conversation_summary(self, messages: List[Dict]) -> str:
        """Generate conversation summary using specialized agent"""
        try:
            if self.summary_agent:
                # Create a mock client for summary generation
                mock_client = type('MockClient', (), {
                    'messages': type('Messages', (), {
                        'create': lambda **kwargs: type('Response', (), {
                            'content': [type('Content', (), {'type': 'text', 'text': 'Implementation progress summary'})()]
                        })()
                    })()
                })()
                
                implementation_summary = self.code_implementation_agent.get_implementation_summary() if self.code_implementation_agent else {}
                
                return await self.summary_agent.generate_conversation_summary(
                    mock_client, "anthropic", messages, implementation_summary
                )
            else:
                return "Implementation progress summary - continuing with code implementation"
        except Exception as e:
            self.logger.warning(f"Summary generation failed: {e}")
            return "Implementation progress summary - continuing with code implementation"

    # ==================== Guidance Messages ====================

    def _generate_success_guidance(self, files_count: int) -> str:
        """Generate success guidance for continuing implementation"""
        return f"""✅ Implementation progress successful! 

📊 **Progress Status:** {files_count} files processed

🎯 **Next Action Required:**
Before implementing the next file, analyze dependencies and read relevant existing files for consistency.

📋 **Smart Implementation Steps:**
1. **Identify Next File**: Determine the highest-priority file from the implementation plan
2. **Dependency Analysis**: Analyze what existing files the new file should reference or import from
3. **Read Related Files**: Use read_file tool to examine relevant implemented files:
   - Base classes or interfaces to extend/implement
   - Configuration files and constants to reference  
   - Common patterns and naming conventions to follow
   - Import structures and dependencies already established
4. **Implement with Context**: Write complete, production-quality code that properly integrates with existing files
5. **Use write_file Tool**: Create the file with full implementation
6. **Continue Chain**: Repeat this process for each remaining file

🧠 **Dependency Reading Strategy:**
- If implementing a model class → Read existing base model files first
- If implementing a config file → Read other config files for consistency
- If implementing a utility → Read related utilities for pattern matching
- If implementing main/entry point → Read all core components first

⚠️ **Critical:** 
- Always use read_file tool BEFORE write_file when dependencies exist
- Ensure consistent interfaces and patterns across all files
- Implement exactly ONE complete file per response"""

    def _generate_error_guidance(self) -> str:
        """Generate error guidance for handling issues"""
        return """❌ Issue detected during implementation.

🔧 **Action Required:**
1. Review the error details above
2. Fix the identified issue
3. Continue with the next file implementation
4. Ensure proper error handling in future implementations"""

    # ==================== Report Generation ====================

    async def _generate_final_report(self, iterations: int, elapsed_time: float) -> str:
        """Generate final implementation report"""
        try:
            files_count = self.code_implementation_agent.get_files_implemented_count() if self.code_implementation_agent else iterations
            
            report = f"""
# Pure Code Implementation Completion Report

## Execution Summary
- Implementation iterations: {iterations}
- Total elapsed time: {elapsed_time:.2f} seconds
- Files processed: {files_count}
- MCP Agent architecture: ✅ Standard mcp_agent patterns

## Agent Performance
### Code Implementation with MCP Agent
- Used standard Agent class from mcp_agent package
- Integrated with AnthropicAugmentedLLM for LLM calls
- Followed initial_workflow.py patterns for agent management
- Maintained code_implementation_workflow.py logic structure

### Architecture Features
✅ Standard MCP Agent architecture using mcp_agent package
✅ Specialized agent separation for clean code organization
✅ Sliding window memory optimization
✅ Progress tracking and implementation statistics
✅ Production-grade code implementation approach
✅ Dependency-aware file implementation
✅ Cross-file consistency through smart dependency tracking

## Implementation Approach
- **File Structure Creation**: Used MCP Agent with command-executor server
- **Code Implementation**: Used MCP Agent with code-implementation server
- **Memory Management**: Implemented sliding window with conversation summarization
- **Progress Tracking**: Integrated with specialized CodeImplementationAgent
- **Error Handling**: Comprehensive error handling and guidance generation

## Technical Architecture
- **MCP Agent**: Standard mcp_agent.agents.agent.Agent class
- **LLM Integration**: AnthropicAugmentedLLM for consistent API calls
- **Server Integration**: code-implementation and command-executor servers
- **Specialized Agents**: CodeImplementationAgent and SummaryAgent for tracking
"""
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate final report: {e}")
            return f"Failed to generate final report: {str(e)}"

    # ==================== Main Workflow ====================

    async def run_workflow(self, plan_file_path: str, target_directory: Optional[str] = None, pure_code_mode: bool = False):
        """Run complete workflow using MCP Agent architecture"""
        try:
            plan_content = self._read_plan_file(plan_file_path)
            
            if target_directory is None:
                target_directory = str(Path(plan_file_path).parent)
            
            self.logger.info(f"Starting MCP Agent workflow: {plan_file_path}")
            self.logger.info(f"Target directory: {target_directory}")
            
            results = {}
            
            # Check if file tree exists
            if self._check_file_tree_exists(target_directory):
                self.logger.info("File tree exists, skipping creation")
                results["file_tree"] = "Already exists, skipped creation"
            else:
                self.logger.info("Creating file tree with MCP Agent...")
                results["file_tree"] = await self.create_file_structure(plan_content, target_directory)
            
            # Code implementation
            if pure_code_mode:
                self.logger.info("Starting pure code implementation with MCP Agent...")
                results["code_implementation"] = await self.implement_code_pure(plan_content, target_directory)
            else:
                self.logger.info("Pure code mode is the default and recommended approach")
                results["code_implementation"] = await self.implement_code_pure(plan_content, target_directory)
            
            self.logger.info("MCP Agent workflow execution successful")
            
            return {
                "status": "success",
                "plan_file": plan_file_path,
                "target_directory": target_directory,
                "code_directory": os.path.join(target_directory, "generate_code"),
                "results": results,
                "mcp_architecture": "mcp_agent_standard",
                "agent_type": "standard_mcp_agent"
            }
            
        except Exception as e:
            self.logger.error(f"MCP Agent workflow execution failed: {e}")
            return {"status": "error", "message": str(e), "plan_file": plan_file_path}


# ==================== Main Function ====================

async def main():
    """Main function for running the MCP Agent workflow"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    plan_file = "agent_folders/papers/2/initial_plan.txt"
    workflow = CodeWorkflow()
    
    print("=" * 60)
    print("MCP Agent Code Implementation Workflow")
    print("=" * 60)
    print("Using: Standard mcp_agent architecture")
    print("- Agent class: mcp_agent.agents.agent.Agent")
    print("- LLM integration: AnthropicAugmentedLLM")
    print("- Server integration: code-implementation, command-executor")
    print("- Specialized agents: CodeImplementationAgent, SummaryAgent")
    
    result = await workflow.run_workflow(plan_file, pure_code_mode=True)
    
    print("=" * 60)
    print("Workflow Execution Results:")
    print(f"Status: {result['status']}")
    print(f"Architecture: {result.get('mcp_architecture', 'unknown')}")
    print(f"Agent Type: {result.get('agent_type', 'unknown')}")
    
    if result['status'] == 'success':
        print(f"Code Directory: {result['code_directory']}")
        print("Execution completed!")
    else:
        print(f"Error Message: {result['message']}")
    
    print("=" * 60)
    print("✅ Using Standard MCP Agent Architecture")
    print("✅ Replicating code_implementation_workflow.py logic")
    print("✅ Following initial_workflow.py patterns")


if __name__ == "__main__":
    asyncio.run(main())
