from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from prompts.code_prompts import (
    PAPER_INPUT_ANALYZER_PROMPT,
    PAPER_DOWNLOADER_PROMPT,
    PAPER_REFERENCE_ANALYZER_PROMPT,
    PAPER_ALGORITHM_ANALYZER_PROMPT,
    PAPER_CONCEPT_ANALYZER_PROMPT,
    CODE_PLANNING_PROMPT,
    GITHUB_SEARCH_PROMPT,
    GITHUB_DOWNLOAD_PROMPT
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
        server_names=["filesystem", "interpreter"],
    )
    
    async with downloader_agent:
        logger.info("downloader: Connected to server, calling list_tools...")
        tools = await downloader_agent.list_tools()
        logger.info("Tools available:", data=tools.model_dump())
        
        downloader = await downloader_agent.attach_llm(OpenAIAugmentedLLM)
        return await downloader.generate_str(message=analysis_result)

async def paper_code_analyzer(analysis_result, logger):
    """
    Run the paper download workflow using PaperDownloaderAgent.
    
    Args:
        analysis_result (str): The result from the paper analyzer
        logger: The logger instance for logging information
        
    Returns:
        str: The download result from the agent
    """
    reference_analysis_agent = Agent(
        name="ReferenceAnalysisAgent",
        instruction=PAPER_REFERENCE_ANALYZER_PROMPT,
        server_names=["filesystem", "brave","fetch"],
    )
    concept_analysis_agent = Agent(
        name="ConceptAnalysisAgent",
        instruction=PAPER_CONCEPT_ANALYZER_PROMPT,
        server_names=["filesystem"],
    )
    algorithm_analysis_agent = Agent(
        name="AlgorithmAnalysisAgent",
        instruction=PAPER_ALGORITHM_ANALYZER_PROMPT,
        server_names=["filesystem","brave"],
    )
    code_planner_agent = Agent(
        name="CodePlannerAgent",
        instruction=CODE_PLANNING_PROMPT,
        server_names=["brave"],
    )
    code_aggregator_agent = ParallelLLM(
            fan_in_agent=code_planner_agent,
            fan_out_agents=[reference_analysis_agent, concept_analysis_agent, algorithm_analysis_agent],
            llm_factory=OpenAIAugmentedLLM,
        )

async def github_repo_search(reference_result, logger):
    """
    Search for GitHub repositories based on paper references.
    
    Args:
        reference_result (str): The result from reference analysis
        logger: The logger instance for logging information
        
    Returns:
        str: The search result containing repository information
    """
    github_search_agent = Agent(
        name="GithubSearchAgent",
        instruction=GITHUB_SEARCH_PROMPT,
        server_names=["brave", "fetch"],
    )
    
    async with github_search_agent:
        logger.info("GitHub searcher: Searching for repositories...")
        searcher = await github_search_agent.attach_llm(OpenAIAugmentedLLM)
        return await searcher.generate_str(message=reference_result)

async def github_repo_download(search_result, logger):
    """
    Download GitHub repositories based on search results.
    
    Args:
        search_result (str): The result from GitHub repository search
        logger: The logger instance for logging information
        
    Returns:
        str: The download result
    """
    github_download_agent = Agent(
        name="GithubDownloadAgent",
        instruction=GITHUB_DOWNLOAD_PROMPT,
        server_names=["github", "filesystem"],
    )
    
    async with github_download_agent:
        logger.info("GitHub downloader: Downloading repositories...")
        downloader = await github_download_agent.attach_llm(OpenAIAugmentedLLM)
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
    
  
