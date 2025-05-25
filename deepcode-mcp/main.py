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

from workflows.initial_workflows import (
    paper_code_preparation,
)
from utils.file_processor import FileProcessor

# Initialize the MCP application
app = MCPApp(name="paper_to_code")

def format_sections(sections, indent=0):
    """格式化章节输出"""
    result = []
    for section in sections:
        # 添加标题
        result.append("  " * indent + f"{'#' * section['level']} {section['title']}")
        # 添加内容（如果有）
        if section['content']:
            result.append("  " * indent + section['content'])
        # 递归处理子章节
        if section['subsections']:
            result.extend(format_sections(section['subsections'], indent + 1))
    return result

async def main():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
        
        file_input_info = """
                            {
                                "status": "success",
                                "paper_path": "./agent_folders/papers/paper_1/paper_1.md",
                                "metadata": {
                                    "title": "RecDiff: Diffusion Model for Social Recommendation",
                                    "authors": ["Unknown Author"],
                                    "year": "D:20"
                                }
                            }
                            """         
        await paper_code_preparation(file_input_info, logger)

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Total run time: {end - start:.2f}s")
    os.system('find . -type d -name "__pycache__" -exec rm -r {} +')
    os.system('Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force')