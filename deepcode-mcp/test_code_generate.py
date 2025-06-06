import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # 禁止生成.pyc文件

import asyncio
import time
import json

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


from workflows.code_implementation_workflow import execute_code_implementation
from utils.file_processor import FileProcessor

# Initialize the MCP application
app = MCPApp(name="paper_to_code")

async def main():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        paper_dir = "/Users/lizongwei/Desktop/LLM_research/Code-Agent/deepcode-mcp/agent_folders/papers/paper_3"
        implementation_result = await execute_code_implementation(paper_dir, logger)
        logger.info(f"Code implementation completed with status: {implementation_result.get('status')}")

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Total run time: {end - start:.2f}s")
    os.system('find . -type d -name "__pycache__" -exec rm -r {} +')