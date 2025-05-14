import asyncio
from mcp_agent.core.fastagent import FastAgent
from Deepcode.prompts.code_prompts import (
    ROUTER_AGENT,
    CODE_GENERATION_AGENT,
    CODE_UNDERSTANDING_AGENT,
    DEBUGGING_AGENT,
    CODE_OPTIMIZATION_AGENT,
    CODE_REFACTORING_AGENT,
    REQUIREMENTS_ANALYSIS_AGENT,
    ERROR_ANALYSIS_AGENT,
    DOCUMENTATION_GENERATOR_AGENT,
    TESTING_STRATEGY_AGENT,
    IMPLEMENTATION_AGENT,
    ARCHITECTURE_DESIGN_AGENT
)

agents = FastAgent(name="CodeAssistant")

# Main Router Agent
@agents.agent(
    name="RouterAgent",
    model="sonnet",
    instruction=ROUTER_AGENT,
    servers=["interpreter", "filesystem"]
)

# Code Generation Agent and its sub-agents
@agents.agent(
    name="CodeGenerationAgent",
    model="sonnet",
    instruction=CODE_GENERATION_AGENT,
    servers=["interpreter", "filesystem"]
)

@agents.agent(
    name="RequirementsAnalysisAgent",
    model="sonnet",
    instruction=REQUIREMENTS_ANALYSIS_AGENT,
    servers=["interpreter", "filesystem"]
)

# Code Understanding Agent and its sub-agents
@agents.agent(
    name="CodeUnderstandingAgent",
    model="sonnet",
    instruction=CODE_UNDERSTANDING_AGENT,
    servers=["interpreter", "filesystem"]
)

# Debugging Agent and its sub-agents
@agents.agent(
    name="DebuggingAgent",
    model="sonnet",
    instruction=DEBUGGING_AGENT,
    servers=["interpreter", "filesystem"]
)

@agents.agent(
    name="ErrorAnalysisAgent",
    model="sonnet",
    instruction=ERROR_ANALYSIS_AGENT,
    servers=["interpreter", "filesystem"]
)

# Code Optimization Agent and its sub-agents
@agents.agent(
    name="CodeOptimizationAgent",
    model="sonnet",
    instruction=CODE_OPTIMIZATION_AGENT,
    servers=["interpreter", "filesystem"]
)

# Code Refactoring Agent and its sub-agents
@agents.agent(
    name="CodeRefactoringAgent",
    model="sonnet",
    instruction=CODE_REFACTORING_AGENT,
    servers=["interpreter", "filesystem"]
)

# Utility Agents
@agents.agent(
    name="DocumentationGeneratorAgent",
    model="sonnet",
    instruction=DOCUMENTATION_GENERATOR_AGENT,
    servers=["interpreter", "filesystem"]
)
@agents.agent(
    name="ImplementationAgent",
    model="sonnet",
    instruction=IMPLEMENTATION_AGENT,
    servers=["interpreter", "filesystem","brave"]
)
@agents.agent(
    name="TestingStrategyAgent",
    model="sonnet",
    instruction=TESTING_STRATEGY_AGENT,
    servers=["interpreter", "filesystem"]
)
@agents.agent(
    name="ArchitectureDesignAgent",
    model="sonnet",
    instruction=ARCHITECTURE_DESIGN_AGENT,
    servers=["interpreter", "filesystem","brave"]
)

# Define workflows for different code assistance tasks

# Code Generation Workflow
@agents.parallel(
  name="CG_Resoning",                       # name of the parallel workflow
  fan_out=["RequirementsAnalysisAgent", "ArchitectureDesignAgent"],          # list of agents to run in parallel
  fan_in="ImplementationAgent",                   # name of agent that combines results (optional)
  instruction="To generate code, first understand the requirements and the architecture of the code",             # instruction to describe the parallel for other workflows
  include_request=True,                  # include original request in fan-in message
)

@agents.chain(
    name="CodeGenerationWorkflow",
    sequence=["CodeGenerationAgent", "CG_Resoning", "CodeGenerationAgent", "DocumentationGeneratorAgent"],
    instruction="A comprehensive workflow for generating new code based on requirements",
    cumulative=True
)

@agents.chain(
    name="CodeUnderstandingWorkflow",
    sequence=["CodeUnderstandingAgent", "DocumentationGeneratorAgent"],
    instruction="A workflow for analyzing and understanding existing code",
    cumulative=True
)

@agents.chain(
    name="DebuggingWorkflow",
    sequence=["DebuggingAgent", "ErrorAnalysisAgent", "CodeGenerationAgent"],
    instruction="A workflow for identifying and fixing code issues",
    cumulative=True
)

@agents.chain(
    name="OptimizationWorkflow",
    sequence=["CodeOptimizationAgent", "TestingStrategyAgent"],
    instruction="A workflow for improving code performance and efficiency",
    cumulative=True
)

@agents.chain(
    name="RefactoringWorkflow",
    sequence=["CodeRefactoringAgent", "TestingStrategyAgent", "DocumentationGeneratorAgent"],
    instruction="A workflow for restructuring code while maintaining functionality",
    cumulative=True
)
@agents.router(
  name="CodeAssistantRouter",                          # name of the router
  agents=["CodeGenerationWorkflow", "CodeUnderstandingWorkflow", "DebuggingWorkflow", "OptimizationWorkflow", "RefactoringWorkflow"], # list of agent names router can delegate to
  model="o3-mini.high",                  # specify routing model
  use_history=False,                     # router maintains conversation history
  human_input=False,                     # whether router can request human input
)

@agents.chain(
    name="Deepcodeworkflow",
    sequence=["RouterAgent","CodeAssistantRouter"],
    instruction="A comprehensive workflow for Deepcode",
    cumulative=True
)

async def main() -> None:
    async with agents.run() as agent:
        # Initial greeting and explanation of capabilities
        await agent.RouterAgent.send(
            """I'm a comprehensive code assistance system that can help you with:
            1. Generating new code from requirements
            2. Understanding existing code
            3. Debugging and fixing issues
            4. Optimizing code performance
            5. Refactoring code structure
            
            What would you like help with today?"""
        )

        # Start the main workflow based on user input-+
        await agent.prompt("RouterAgent")

        print("\nWould you like to ask follow-up questions? (Type 'STOP' to end)")
        await agent.prompt("RouterAgent", default_prompt="STOP")

if __name__ == "__main__":
    asyncio.run(main())  # type: ignore
