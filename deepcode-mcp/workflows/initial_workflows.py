from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from utils.file_processor import FileProcessor
from tools.github_downloader import GitHubDownloader
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
        server_names=["filesystem", "file-downloader"],
    )
    
    async with downloader_agent:
        logger.info("downloader: Connected to server, calling list_tools...")
        tools = await downloader_agent.list_tools()
        logger.info("Tools available:", data=tools.model_dump())
        
        downloader = await downloader_agent.attach_llm(AnthropicAugmentedLLM)
        return await downloader.generate_str(message=analysis_result)

async def paper_code_analyzer(document, logger):
    """
    Run the paper code analysis workflow using multiple agents.
    
    Args:
        document (str): The document to analyze
        logger: The logger instance for logging information
        
    Returns:
        str: The analysis result from the agents
    """
    concept_analysis_agent = Agent(
        name="ConceptAnalysisAgent",
        instruction=PAPER_CONCEPT_ANALYSIS_PROMPT,
        server_names=["filesystem"],
    )
    algorithm_analysis_agent = Agent(
        name="AlgorithmAnalysisAgent",
        instruction=PAPER_ALGORITHM_ANALYSIS_PROMPT,
        server_names=["filesystem","brave"],
    )
    code_planner_agent = Agent(
        name="CodePlannerAgent",
        instruction=CODE_PLANNING_PROMPT,
        server_names=["brave"],
    )
    # code_validation_agent = Agent(
    #     name="CodeValidationAgent",
    #     instruction=INTEGRATION_VALIDATION_PROMPT,
    # )
    code_aggregator_agent = ParallelLLM(
            fan_in_agent=code_planner_agent,
            fan_out_agents=[concept_analysis_agent, algorithm_analysis_agent],
            llm_factory=AnthropicAugmentedLLM,
        )
    result = await code_aggregator_agent.generate_str(message=document)
    logger.info(f"Code analysis result: {result}")
    return result
    # async with code_validation_agent:
    #     logger.info("code_validation_agent: Connected to server, calling list_tools...")
    #     code_validation = await code_validation_agent.attach_llm(AnthropicAugmentedLLM)
    #     return await code_validation.generate_str(message=result)

async def github_repo_download(search_result, paper_dir, logger):
    """
    Download GitHub repositories based on search results.
    
    Args:
        search_result (str): The result from GitHub repository search
        paper_dir (str): The directory where the paper and its code will be stored
        logger: The logger instance for logging information
        
    Returns:
        str: The download result
    """
    github_download_agent = Agent(
        name="GithubDownloadAgent",
        instruction="Download github repo to the directory {paper_dir}/code_base".format(paper_dir=paper_dir),
        server_names=["filesystem", "github-downloader"],
    )
    
    async with github_download_agent:
        logger.info("GitHub downloader: Downloading repositories...")
        downloader = await github_download_agent.attach_llm(AnthropicAugmentedLLM)
        return await downloader.generate_str(message=search_result)

async def paper_reference_analyzer(analysis_result, logger):
    """
    Run the paper reference analysis and GitHub repository workflow.
    
    Args:
        analysis_result (str): The result from the paper analyzer
        logger: The logger instance for logging information
        
    Returns:
        tuple: (reference_result, search_result, download_result)
    """
    # 1. Analyze references
    reference_analysis_agent = Agent(
        name="ReferenceAnalysisAgent",
        instruction=PAPER_REFERENCE_ANALYZER_PROMPT,
        server_names=["filesystem", "brave", "fetch"],
    )
    
    async with reference_analysis_agent:
        logger.info("Reference analyzer: Connected to server, analyzing references...")
        analyzer = await reference_analysis_agent.attach_llm(AnthropicAugmentedLLM)
        reference_result = await analyzer.generate_str(message=analysis_result)
        return reference_result

    
async def paper_code_preparation(download_result, logger):
    """
    Prepare the paper code for analysis.
    
    Args:
        download_result (str): The result from the paper downloader containing file information
        logger: The logger instance for logging information
        
    Returns:
        str: The preparation result
    """ 
    try:
        # 解析download_result以获取文件信息
        # download_result应该包含paper_path信息
        result = await FileProcessor.process_file_input(download_result)
        paper_dir = result['paper_dir']  # 直接使用返回的paper_dir
        reference_path = os.path.join(paper_dir, 'reference.txt')
        initial_plan_path = os.path.join(paper_dir, 'initial_plan.txt')
        # 1. 分析论文引用或读取已有结果
        if os.path.exists(reference_path):
            logger.info(f"Found existing reference analysis at {reference_path}")
            with open(reference_path, 'r', encoding='utf-8') as f:
                reference_result = f.read()
        else:
            # 执行论文引用分析
            reference_result = await paper_reference_analyzer(result['standardized_text'], logger)
            initial_plan_result = await paper_code_analyzer(result['standardized_text'], logger)
            # 将reference结果写入文件
            with open(reference_path, 'w', encoding='utf-8') as f:
                f.write(reference_result)
            logger.info(f"Reference analysis has been saved to {reference_path}")
            with open(initial_plan_path, 'w', encoding='utf-8') as f:
                f.write(initial_plan_result)
            logger.info(f"Initial plan has been saved to {initial_plan_path}")
        # 2. 下载GitHub仓库
        await asyncio.sleep(5) 
        download_result = await github_repo_download(reference_result, paper_dir, logger)
        download_path = os.path.join(paper_dir, 'github_download.txt')
        with open(download_path, 'w', encoding='utf-8') as f:
            f.write(download_result)
        logger.info(f"GitHub download results have been saved to {download_path}")
        return download_result

    except Exception as e:
        logger.error(f"Error in paper_code_preparation: {e}")
        raise e
