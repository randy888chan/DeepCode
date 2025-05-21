import asyncio
import os
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from prompts.code_prompts import (
    PAPER_INPUT_ANALYZER_PROMPT,
    PAPER_DOWNLOADER_PROMPT,
    PAPER_CONTENT_ANALYZER_PROMPT,
    CODE_REPLICATION_PROMPT,
    CODE_VERIFICATION_PROMPT,
)

# Initialize the MCP application
app = MCPApp(name="paper_to_code")

async def run_paper_analyzer(prompt_text, logger):
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
        instruction=PAPER_INPUT_ANALYZER_PROMPT,
    )
    
    async with analyzer_agent:
        logger.info("analyzer: Connected to server, calling list_tools...")
        tools = await analyzer_agent.list_tools()
        logger.info("Tools available:", data=tools.model_dump())
        
        analyzer = await analyzer_agent.attach_llm(OpenAIAugmentedLLM)
        return await analyzer.generate_str(message=prompt_text)

async def run_paper_downloader(analysis_result, logger):
    """
    Run the paper download workflow using PaperDownloaderAgent.
    
    Args:
        analysis_result (str): The result from the paper analyzer
        logger: The logger instance for logging information
        
    Returns:
        str: The download result from the agent
    """
    downloader_agent = Agent(
        name="PaperDownloaderAgent",
        instruction=PAPER_DOWNLOADER_PROMPT,
        server_names=["filesystem", "interpreter"],
    )
    
    async with downloader_agent:
        logger.info("downloader: Connected to server, calling list_tools...")
        tools = await downloader_agent.list_tools()
        logger.info("Tools available:", data=tools.model_dump())
        
        downloader = await downloader_agent.attach_llm(OpenAIAugmentedLLM)
        return await downloader.generate_str(message=analysis_result)

async def main():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
        
        prompt_text = f"""
Input Data:
Paper Path: /Users/lizongwei/Desktop/LLM_research/DeepCode/deepcode-mcp/agent_folders/2406.01629.pdf
Title: RecDiff: Diffusion Model for Social Recommendation
Authors: Unknown Author
Year: D:20
Additional Input: None
"""
        # Run paper analysis workflow
        analysis_result = await run_paper_analyzer(prompt_text, logger)
        
        # Run paper download workflow
        download_result = await run_paper_downloader(analysis_result, logger)
        
        print(download_result)

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Total run time: {end - start:.2f}s") 