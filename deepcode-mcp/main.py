import asyncio
import os
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from workflows.initial_workflows import run_paper_analyzer, run_paper_downloader,paper_code_analyzer

# Initialize the MCP application
app = MCPApp(name="paper_to_code")

async def main():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
        
        prompt_text = """
Input Data:
Paper Path: https://arxiv.org/pdf/2406.01629.pdf
Title: RecDiff: Diffusion Model for Social Recommendation
Authors: Unknown Author
Year: D:20
Additional Input: None
"""
        # Run paper analysis workflow
        # analysis_result = await run_paper_analyzer(prompt_text, logger)
        
        # Run paper download workflow
        # download_result = await run_paper_downloader(analysis_result, logger)

        file_input_info = """
{
    "status": "success",
    "paper_path": "./agent_folders/papers/paper_3/paper_3.md",
    "metadata": {
        "title": "RecDiff: Diffusion Model for Social Recommendation",
        "authors": ["Unknown Author"],
        "year": "D:20"
    }
}
"""     
        paper_code_result = await paper_code_analyzer(file_input_info, logger)
        
        print(paper_code_result)

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Total run time: {end - start:.2f}s") 