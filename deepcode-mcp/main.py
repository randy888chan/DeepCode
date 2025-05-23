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
    run_paper_analyzer, 
    run_paper_downloader,
    paper_code_analyzer,
    paper_reference_analyzer,
    github_repo_search,
    github_repo_download
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
    "paper_path": "./agent_folders/papers/paper_3/paper_3.md",
    "metadata": {
        "title": "RecDiff: Diffusion Model for Social Recommendation",
        "authors": ["Unknown Author"],
        "year": "D:20"
    }
}
"""         
        try:
            # 使用FileProcessor处理文件
            result = await FileProcessor.process_file_input(file_input_info)
            paper_dir = os.path.dirname(result['file_path'])
            reference_path = os.path.join(paper_dir, 'reference.txt')
            
            # 1. 分析论文引用或读取已有结果
            if os.path.exists(reference_path):
                logger.info(f"Found existing reference analysis at {reference_path}")
                with open(reference_path, 'r', encoding='utf-8') as f:
                    reference_result = f.read()
            else:
                # 执行论文引用分析
                reference_result = await paper_reference_analyzer(result['standardized_text'], logger)
                # 将reference结果写入文件
                with open(reference_path, 'w', encoding='utf-8') as f:
                    f.write(reference_result)
                logger.info(f"Reference analysis has been saved to {reference_path}")
            
            # 2. 搜索GitHub仓库或读取已有结果
            search_path = os.path.join(paper_dir, 'github_search.txt')
            if os.path.exists(search_path):
                logger.info(f"Found existing GitHub search results at {search_path}")
                with open(search_path, 'r', encoding='utf-8') as f:
                    search_result = f.read()
            else:
                # 执行GitHub仓库搜索
                search_result = await github_repo_search(reference_result, logger)
                # 将搜索结果写入文件
                with open(search_path, 'w', encoding='utf-8') as f:
                    f.write(search_result)
                logger.info(f"GitHub search results have been saved to {search_path}")
            
            # 3. 下载GitHub仓库
            download_result = await github_repo_download(search_result, logger)
            download_path = os.path.join(paper_dir, 'github_download.txt')
            with open(download_path, 'w', encoding='utf-8') as f:
                f.write(download_result)
            logger.info(f"GitHub download results have been saved to {download_path}")
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Total run time: {end - start:.2f}s")
    os.system('find . -type d -name "__pycache__" -exec rm -r {} +')