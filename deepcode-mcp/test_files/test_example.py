
from mcp_agent.app import MCPApp
import time
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from utils.file_processor import FileProcessor
from tools.github_downloader import GitHubDownloader
from workflows.code_implementation_workflow import execute_code_implementation
import os
import asyncio
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # 禁止生成.pyc文件
from prompts.code_prompts import (
    PAPER_INPUT_ANALYZER_PROMPT,
    PAPER_DOWNLOADER_PROMPT,
    PAPER_REFERENCE_ANALYZER_PROMPT,
    PAPER_ALGORITHM_ANALYSIS_PROMPT,
    PAPER_CONCEPT_ANALYSIS_PROMPT,
    CODE_PLANNING_PROMPT,
    GITHUB_DOWNLOAD_PROMPT,
    INTEGRATION_VALIDATION_PROMPT
)
app = MCPApp(name="code generator")

async def run_code_generator(prompt_text, logger):
    """
    Run the paper analysis workflow using PaperInputAnalyzerAgent.
    
    Args:
        prompt_text (str): The input prompt text containing paper information
        logger: The logger instance for logging information
        
    Returns:
        str: The analysis result from the agent
    """
    analyzer_agent = Agent(
        name="PaperInputAnalyzerAgent",
        instruction="Please anlysis the command and generate the code",
        server_names=["code-generator"],
    )
    
    async with analyzer_agent:
        logger.info("analyzer: Connected to server, calling list_tools...")
        tools = await analyzer_agent.list_tools()
        logger.info("Tools available:", data=tools.model_dump())
        
        analyzer = await analyzer_agent.attach_llm(AnthropicAugmentedLLM)
        return await analyzer.generate_str(message=prompt_text, request_params=RequestParams(maxTokens=8192))
async def main():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
        message = """You are a code generator. Please generate a complete Python implementation for recdiff.py.

The target file path is: /Users/lizongwei/Desktop/LLM_research/Code-Agent/deepcode-mcp/agent_folders/papers/paper_3/generate_code/src/models/recdiff.py

Requirements:
1. Use the write_code_file tool to create the complete recdiff.py file
2. The file should implement a RecDiff model class for recommendation systems using diffusion models
3. Include proper imports, class definition, and key methods like __init__, forward, etc.
4. Make it a complete, runnable Python file

Please use the write_code_file tool to generate the complete code."""
        implementation_result = await run_code_generator(message, logger)
        logger.info(f"Code implementation completed. Result: {implementation_result}")
    
if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Total run time: {end - start:.2f}s")
    os.system('find . -type d -name "__pycache__" -exec rm -r {} +')