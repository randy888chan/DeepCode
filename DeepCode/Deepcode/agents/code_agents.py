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
    ARCHITECTURE_DESIGN_AGENT,
    CODE_QUALITY_ANALYZER_AGENT,
    TEST_COVERAGE_ANALYZER_AGENT,
    CODE_ANALYSIS_DOCUMENT_AGENT,
)

agents = FastAgent(name="Deepcode")

# ==================== Code Generation Workflow Agents ====================
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

@agents.agent(
    name="ArchitectureDesignAgent",
    model="sonnet",
    instruction=ARCHITECTURE_DESIGN_AGENT,
    servers=["interpreter", "filesystem", "brave"]
)

@agents.agent(
    name="ImplementationAgent",
    model="sonnet",
    instruction=IMPLEMENTATION_AGENT,
    servers=["interpreter", "filesystem", "brave"]
)

# ==================== Code Understanding Workflow Agents ====================
@agents.agent(
    name="CodeUnderstandingAgent",
    model="sonnet",
    instruction=CODE_UNDERSTANDING_AGENT,
    servers=["filesystem", "brave", "fetch"]
)

@agents.agent(
    name="CodeQualityAnalyzerAgent",
    model="sonnet",
    instruction=CODE_QUALITY_ANALYZER_AGENT,
    servers=["interpreter", "filesystem", "brave"]
)

@agents.agent(
    name="TestCoverageAnalyzerAgent",
    model="sonnet",
    instruction=TEST_COVERAGE_ANALYZER_AGENT,
    servers=["filesystem","fetch","brave"]
)

@agents.agent(
    name="CodeAnalysisDocumentAgent",
    model="sonnet",
    instruction=CODE_ANALYSIS_DOCUMENT_AGENT,
    servers=["interpreter", "filesystem"]
)

# ==================== Debugging Workflow Agents ====================
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

# ==================== Optimization Workflow Agents ====================
@agents.agent(
    name="CodeOptimizationAgent",
    model="sonnet",
    instruction=CODE_OPTIMIZATION_AGENT,
    servers=["interpreter", "filesystem"]
)

@agents.agent(
    name="TestingStrategyAgent",
    model="sonnet",
    instruction=TESTING_STRATEGY_AGENT,
    servers=["interpreter", "filesystem"]
)

# ==================== Refactoring Workflow Agents ====================
@agents.agent(
    name="CodeRefactoringAgent",
    model="sonnet",
    instruction=CODE_REFACTORING_AGENT,
    servers=["interpreter", "filesystem"]
)

# ==================== Utility Agents ====================
@agents.agent(
    name="DocumentationGeneratorAgent",
    model="sonnet",
    instruction=DOCUMENTATION_GENERATOR_AGENT,
    servers=["interpreter", "filesystem"]
)

# ==================== Workflow Definitions ====================

# Code Generation Workflow
@agents.parallel(
    name="CG_Resoning",
    fan_out=["RequirementsAnalysisAgent", "ArchitectureDesignAgent"],
    fan_in="ImplementationAgent",
    instruction="To generate code, first understand the requirements and the architecture of the code",
    include_request=True
)

@agents.chain(
    name="CodeGenerationWorkflow",
    sequence=["CodeGenerationAgent", "CG_Resoning", "CodeGenerationAgent", "DocumentationGeneratorAgent"],
    instruction="A comprehensive workflow for generating new code based on requirements",
    cumulative=True
)

# Code Understanding Workflow
@agents.parallel(
    name="CodeUnderstandingWorkflow",
    fan_out=["CodeUnderstandingAgent", "CodeQualityAnalyzerAgent"],
    fan_in="CodeAnalysisDocumentAgent",
    instruction="Comprehensive code analysis in parallel",
    include_request=True
)

# Debugging Workflow
@agents.chain(
    name="DebuggingWorkflow",
    sequence=["DebuggingAgent", "ErrorAnalysisAgent", "CodeGenerationAgent"],
    instruction="A workflow for identifying and fixing code issues",
    cumulative=True
)

# Optimization Workflow
@agents.chain(
    name="OptimizationWorkflow",
    sequence=["CodeOptimizationAgent", "TestingStrategyAgent"],
    instruction="A workflow for improving code performance and efficiency",
    cumulative=True
)

# Refactoring Workflow
@agents.chain(
    name="RefactoringWorkflow",
    sequence=["CodeRefactoringAgent", "TestingStrategyAgent", "DocumentationGeneratorAgent"],
    instruction="A workflow for restructuring code while maintaining functionality",
    cumulative=True
)

# Main Router
@agents.router(
    name="Deepcodeworkflow",
    instruction=ROUTER_AGENT,
    agents=["CodeGenerationWorkflow", "CodeUnderstandingWorkflow", "DebuggingWorkflow", "OptimizationWorkflow", "RefactoringWorkflow"],
    model="o3-mini.high",
    use_history=False,
    human_input=False
)

async def main() -> None:
    async with agents.run() as agent:
        await agent.prompt("Deepcodeworkflow")

# if __name__ == "__main__":
#     asyncio.run(main())  # type: ignore
