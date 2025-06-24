from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from utils.file_processor import FileProcessor
from tools.github_downloader import GitHubDownloader
# 导入代码实现工作流 / Import code implementation workflow
from workflows.code_implementation_workflow import CodeImplementationWorkflow
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
import json
import re

def extract_clean_json(llm_output: str) -> str:
    """
    从LLM输出中提取纯净的JSON，移除所有额外的文本和格式化
    
    Args:
        llm_output: LLM的原始输出
        
    Returns:
        纯净的JSON字符串
    """
    try:
        # 1. 首先尝试直接解析整个输出为JSON
        json.loads(llm_output.strip())
        return llm_output.strip()
    except json.JSONDecodeError:
        pass
    
    # 2. 移除markdown代码块
    if '```json' in llm_output:
        pattern = r'```json\s*(.*?)\s*```'
        match = re.search(pattern, llm_output, re.DOTALL)
        if match:
            json_text = match.group(1).strip()
            try:
                json.loads(json_text)
                return json_text
            except json.JSONDecodeError:
                pass
    
    # 3. 查找以{开始的JSON对象
    lines = llm_output.split('\n')
    json_lines = []
    in_json = False
    brace_count = 0
    
    for line in lines:
        stripped = line.strip()
        if not in_json and stripped.startswith('{'):
            in_json = True
            json_lines = [line]
            brace_count = stripped.count('{') - stripped.count('}')
        elif in_json:
            json_lines.append(line)
            brace_count += stripped.count('{') - stripped.count('}')
            if brace_count == 0:
                break
    
    if json_lines:
        json_text = '\n'.join(json_lines).strip()
        try:
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            pass
    
    # 4. 最后的尝试：使用正则表达式查找JSON
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(pattern, llm_output, re.DOTALL)
    for match in matches:
        try:
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue
    
    # 如果所有方法都失败，返回原始输出
    return llm_output

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
        server_names=["brave"],
    )
    
    async with analyzer_agent:
        logger.info("analyzer: Connected to server, calling list_tools...")
        tools = await analyzer_agent.list_tools()
        logger.info("Tools available:", data=tools.model_dump())
        
        
        analyzer = await analyzer_agent.attach_llm(AnthropicAugmentedLLM)
        raw_result = await analyzer.generate_str(message=prompt_text)
        
        # 清理LLM输出，确保只返回纯净的JSON
        clean_result = extract_clean_json(raw_result)
        logger.info(f"Raw LLM output: {raw_result}")
        logger.info(f"Cleaned JSON output: {clean_result}")
        
        return clean_result

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

    
async def _process_input_source(input_source: str, logger) -> str:
    """
    Process and validate input source (file path or URL).
    处理和验证输入源（文件路径或URL）
    
    Args:
        input_source: The input source (file path or analysis result)
        logger: Logger instance
        
    Returns:
        str: Processed input source
    """
    if input_source.startswith("file://"):
        file_path = input_source[7:]
        if os.name == 'nt' and file_path.startswith('/'):
            file_path = file_path.lstrip('/')
        return file_path
    return input_source

async def _execute_paper_analysis_phase(input_source: str, logger, progress_callback=None) -> tuple:
    """
    Execute paper analysis and download phase.
    执行论文分析和下载阶段
    
    Args:
        input_source: Input source
        logger: Logger instance
        progress_callback: Progress callback function
        
    Returns:
        tuple: (analysis_result, download_result)
    """
    # Step 1: Paper Analysis
    if progress_callback:
        progress_callback(10, "📊 Analyzing paper content and extracting key information...")
    analysis_result = await run_paper_analyzer(input_source, logger)
    
    # Add brief pause for system stability
    await asyncio.sleep(5)
    
    # Step 2: Download Processing
    if progress_callback:
        progress_callback(25, "📥 Processing downloads and preparing document structure...")
    download_result = await run_paper_downloader(analysis_result, logger)
    
    return analysis_result, download_result

async def _setup_paper_directory_structure(download_result: str, logger) -> dict:
    """
    Setup paper directory structure and prepare file paths.
    设置论文目录结构并准备文件路径
    
    Args:
        download_result: Download result from previous phase
        logger: Logger instance
        
    Returns:
        dict: Directory structure information
    """
    # Parse download result to get file information
    result = await FileProcessor.process_file_input(download_result)
    paper_dir = result['paper_dir']
    
    return {
        'paper_dir': paper_dir,
        'standardized_text': result['standardized_text'],
        'reference_path': os.path.join(paper_dir, 'reference.txt'),
        'initial_plan_path': os.path.join(paper_dir, 'initial_plan.txt'),
        'download_path': os.path.join(paper_dir, 'github_download.txt'),
        'index_report_path': os.path.join(paper_dir, 'codebase_index_report.txt'),
        'implementation_report_path': os.path.join(paper_dir, 'code_implementation_report.txt')
    }

async def _execute_reference_analysis_phase(dir_info: dict, logger, progress_callback=None) -> str:
    """
    Execute reference analysis phase.
    执行引用分析阶段
    
    Args:
        dir_info: Directory structure information
        logger: Logger instance
        progress_callback: Progress callback function
        
    Returns:
        str: Reference analysis result
    """
    if progress_callback:
        progress_callback(45, "🔍 Analyzing paper references and related work...")
    
    reference_path = dir_info['reference_path']
    
    # Check if reference analysis already exists
    if os.path.exists(reference_path):
        logger.info(f"Found existing reference analysis at {reference_path}")
        with open(reference_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Execute reference analysis
    reference_result = await paper_reference_analyzer(dir_info['standardized_text'], logger)
    
    # Save reference analysis result
    with open(reference_path, 'w', encoding='utf-8') as f:
        f.write(reference_result)
    logger.info(f"Reference analysis saved to {reference_path}")
    
    return reference_result

async def _execute_code_planning_phase(dir_info: dict, logger, progress_callback=None):
    """
    Execute code planning phase.
    执行代码规划阶段
    
    Args:
        dir_info: Directory structure information
        logger: Logger instance
        progress_callback: Progress callback function
    """
    if progress_callback:
        progress_callback(50, "📋 Generating initial code implementation plan...")
    
    initial_plan_path = dir_info['initial_plan_path']
    
    # Check if initial plan already exists
    if not os.path.exists(initial_plan_path):
        initial_plan_result = await paper_code_analyzer(dir_info['standardized_text'], logger)
        with open(initial_plan_path, 'w', encoding='utf-8') as f:
            f.write(initial_plan_result)
        logger.info(f"Initial plan saved to {initial_plan_path}")

async def _execute_github_download_phase(reference_result: str, dir_info: dict, logger, progress_callback=None):
    """
    Execute GitHub repository download phase.
    执行GitHub仓库下载阶段
    
    Args:
        reference_result: Reference analysis result
        dir_info: Directory structure information
        logger: Logger instance
        progress_callback: Progress callback function
    """
    if progress_callback:
        progress_callback(60, "📦 Downloading relevant GitHub repositories...")
    
    await asyncio.sleep(5)  # Brief pause for stability
    
    try:
        download_result = await github_repo_download(reference_result, dir_info['paper_dir'], logger)
        
        # Save download results
        with open(dir_info['download_path'], 'w', encoding='utf-8') as f:
            f.write(download_result)
        logger.info(f"GitHub download results saved to {dir_info['download_path']}")
        
        # Verify if any repositories were actually downloaded
        code_base_path = os.path.join(dir_info['paper_dir'], 'code_base')
        if os.path.exists(code_base_path):
            downloaded_repos = [
                d for d in os.listdir(code_base_path) 
                if os.path.isdir(os.path.join(code_base_path, d)) and not d.startswith(".")
            ]
            
            if downloaded_repos:
                logger.info(f"Successfully downloaded {len(downloaded_repos)} repositories: {downloaded_repos}")
            else:
                logger.warning("GitHub download phase completed, but no repositories were found in the code_base directory")
                logger.info("This might indicate:")
                logger.info("1. No relevant repositories were identified in the reference analysis")
                logger.info("2. Repository downloads failed due to access permissions or network issues")
                logger.info("3. The download agent encountered errors during the download process")
        else:
            logger.warning(f"Code base directory was not created: {code_base_path}")
            
    except Exception as e:
        logger.error(f"Error during GitHub repository download: {e}")
        # Still save the error information
        error_message = f"GitHub download failed: {str(e)}"
        with open(dir_info['download_path'], 'w', encoding='utf-8') as f:
            f.write(error_message)
        logger.info(f"GitHub download error saved to {dir_info['download_path']}")
        raise e  # Re-raise to be handled by the main pipeline

async def _execute_codebase_indexing_phase(dir_info: dict, logger, progress_callback=None) -> dict:
    """
    Execute codebase indexing phase.
    执行代码库索引阶段
    
    Args:
        dir_info: Directory structure information
        logger: Logger instance
        progress_callback: Progress callback function
        
    Returns:
        dict: Indexing result
    """
    if progress_callback:
        progress_callback(70, "🗂️ Building codebase index and analyzing relationships...")
    
    logger.info("Starting codebase indexing to build relationships between downloaded code and target structure...")
    await asyncio.sleep(2)  # Brief pause before starting indexing
    
    # Check if code_base directory exists and has content
    code_base_path = os.path.join(dir_info['paper_dir'], 'code_base')
    if not os.path.exists(code_base_path):
        logger.warning(f"Code base directory not found: {code_base_path}")
        return {'status': 'skipped', 'message': 'No code base directory found - skipping indexing'}
    
    # Check if there are any repositories in the code_base directory
    try:
        repo_dirs = [
            d for d in os.listdir(code_base_path) 
            if os.path.isdir(os.path.join(code_base_path, d)) and not d.startswith(".")
        ]
        
        if not repo_dirs:
            logger.warning(f"No repositories found in {code_base_path}")
            logger.info("This might be because:")
            logger.info("1. GitHub download phase didn't complete successfully")
            logger.info("2. No relevant repositories were identified for download")
            logger.info("3. Repository download failed due to access issues")
            logger.info("Continuing with code implementation without codebase indexing...")
            
            # Save a report about the skipped indexing
            skip_report = {
                'status': 'skipped',
                'reason': 'no_repositories_found',
                'message': f'No repositories found in {code_base_path}',
                'suggestions': [
                    'Check if GitHub download phase completed successfully',
                    'Verify if relevant repositories were identified in reference analysis',
                    'Check network connectivity and GitHub access permissions'
                ]
            }
            
            with open(dir_info['index_report_path'], 'w', encoding='utf-8') as f:
                f.write(str(skip_report))
            logger.info(f"Indexing skip report saved to {dir_info['index_report_path']}")
            
            return skip_report
            
    except Exception as e:
        logger.error(f"Error checking code base directory: {e}")
        return {'status': 'error', 'message': f'Error checking code base directory: {str(e)}'}
    
    try:
        from workflows.codebase_index_workflow import run_codebase_indexing
        
        logger.info(f"Found {len(repo_dirs)} repositories to index: {repo_dirs}")
        
        # Run codebase index workflow
        index_result = await run_codebase_indexing(
            paper_dir=dir_info['paper_dir'],
            initial_plan_path=dir_info['initial_plan_path'],
            config_path="mcp_agent.secrets.yaml",
            logger=logger
        )
        
        # Log indexing results
        if index_result['status'] == 'success':
            logger.info("Code indexing completed successfully!")
            logger.info(f"Indexed {index_result['statistics']['total_repositories'] if index_result.get('statistics') else len(index_result['output_files'])} repositories")
            logger.info(f"Generated {len(index_result['output_files'])} index files")
            
            # Save indexing results to file
            with open(dir_info['index_report_path'], 'w', encoding='utf-8') as f:
                f.write(str(index_result))
            logger.info(f"Indexing report saved to {dir_info['index_report_path']}")
            
        elif index_result['status'] == 'warning':
            logger.warning(f"Code indexing completed with warnings: {index_result['message']}")
        else:
            logger.error(f"Code indexing failed: {index_result['message']}")
            
        return index_result
        
    except Exception as e:
        logger.error(f"Error during codebase indexing workflow: {e}")
        logger.info("Continuing with code implementation despite indexing failure...")
        
        # Save error report
        error_report = {
            'status': 'error',
            'message': str(e),
            'phase': 'codebase_indexing',
            'recovery_action': 'continuing_with_code_implementation'
        }
        
        with open(dir_info['index_report_path'], 'w', encoding='utf-8') as f:
            f.write(str(error_report))
        logger.info(f"Indexing error report saved to {dir_info['index_report_path']}")
        
        return error_report

async def _execute_code_implementation_phase(dir_info: dict, logger, progress_callback=None) -> dict:
    """
    Execute code implementation phase.
    执行代码实现阶段
    
    Args:
        dir_info: Directory structure information
        logger: Logger instance
        progress_callback: Progress callback function
        
    Returns:
        dict: Implementation result
    """
    if progress_callback:
        progress_callback(85, "⚙️ Implementing code based on the generated plan...")
    
    logger.info("Starting code implementation based on the initial plan...")
    await asyncio.sleep(3)  # Brief pause before starting implementation
    
    try:
        # Create code implementation workflow instance
        code_workflow = CodeImplementationWorkflow()
        
        # Check if initial plan file exists
        if os.path.exists(dir_info['initial_plan_path']):
            logger.info(f"Using initial plan from {dir_info['initial_plan_path']}")
            
            # Run code implementation workflow with pure code mode
            implementation_result = await code_workflow.run_workflow(
                plan_file_path=dir_info['initial_plan_path'],
                target_directory=dir_info['paper_dir'],
                pure_code_mode=True  # Focus on code implementation, skip testing
            )
            
            # Log implementation results
            if implementation_result['status'] == 'success':
                logger.info("Code implementation completed successfully!")
                logger.info(f"Code directory: {implementation_result['code_directory']}")
                
                # Save implementation results to file
                with open(dir_info['implementation_report_path'], 'w', encoding='utf-8') as f:
                    f.write(str(implementation_result))
                logger.info(f"Implementation report saved to {dir_info['implementation_report_path']}")
                
            else:
                logger.error(f"Code implementation failed: {implementation_result.get('message', 'Unknown error')}")
                
            return implementation_result
        else:
            logger.warning(f"Initial plan file not found at {dir_info['initial_plan_path']}, skipping code implementation")
            return {'status': 'warning', 'message': 'Initial plan not found - code implementation skipped'}
            
    except Exception as e:
        logger.error(f"Error during code implementation workflow: {e}")
        return {'status': 'error', 'message': str(e)}

async def execute_multi_agent_research_pipeline(input_source, logger, progress_callback=None):
    """
    Execute the complete multi-agent research pipeline from paper input to code implementation.
    执行从论文输入到代码实现的完整多智能体研究流水线
    
    This is the main orchestration function that coordinates all research workflow phases:
    - Paper analysis and content extraction
    - Reference analysis and GitHub repository discovery
    - Code planning and structure design
    - Codebase indexing and relationship analysis
    - Final code implementation
    
    Args:
        input_source (str): The input source (file path, URL, or analysis result)
        logger: The logger instance for comprehensive logging
        progress_callback (callable, optional): Progress callback function for UI updates
        
    Returns:
        str: The comprehensive pipeline execution result with status and outcomes
    """ 
    try:
        # Phase 1: Input Processing and Validation
        # 阶段1：输入处理和验证
        input_source = await _process_input_source(input_source, logger)
        
        # Phase 2: Paper Analysis and Download (if needed)
        # 阶段2：论文分析和下载（如需要）
        if isinstance(input_source, str) and (input_source.endswith(('.pdf', '.docx', '.txt', '.html', '.md')) or 
                                            input_source.startswith(('http', 'file://'))):
            analysis_result, download_result = await _execute_paper_analysis_phase(input_source, logger, progress_callback)
        else:
            download_result = input_source  # Use input directly if already processed
        
        # Phase 3: Directory Structure Setup
        # 阶段3：目录结构设置
        if progress_callback:
            progress_callback(40, "🔧 Starting comprehensive code preparation workflow...")
        
        dir_info = await _setup_paper_directory_structure(download_result, logger)
        
        # Phase 4: Reference Analysis
        # 阶段4：引用分析
        reference_result = await _execute_reference_analysis_phase(dir_info, logger, progress_callback)
        
        # Phase 5: Code Planning
        # 阶段5：代码规划
        await _execute_code_planning_phase(dir_info, logger, progress_callback)
        
        # Phase 6: GitHub Repository Download
        # 阶段6：GitHub仓库下载
        await _execute_github_download_phase(reference_result, dir_info, logger, progress_callback)
        
        # Phase 7: Codebase Indexing
        # 阶段7：代码库索引
        index_result = await _execute_codebase_indexing_phase(dir_info, logger, progress_callback)
        
        # Phase 8: Code Implementation
        # 阶段8：代码实现
        implementation_result = await _execute_code_implementation_phase(dir_info, logger, progress_callback)
        
        # Final Status Report
        # 最终状态报告
        pipeline_summary = f"Multi-agent research pipeline completed for {dir_info['paper_dir']}"
        
        # Add indexing status to summary
        if index_result['status'] == 'skipped':
            pipeline_summary += f"\n🔶 Codebase indexing: {index_result['message']}"
        elif index_result['status'] == 'error':
            pipeline_summary += f"\n❌ Codebase indexing failed: {index_result['message']}"
        elif index_result['status'] == 'success':
            pipeline_summary += f"\n✅ Codebase indexing completed successfully"
        
        # Add implementation status to summary
        if implementation_result['status'] == 'success':
            pipeline_summary += f"\n🎉 Code implementation completed successfully!"
            pipeline_summary += f"\n📁 Code generated in: {implementation_result['code_directory']}"
            return pipeline_summary
        elif implementation_result['status'] == 'warning':
            pipeline_summary += f"\n⚠️ Code implementation: {implementation_result['message']}"
            return pipeline_summary
        else:
            pipeline_summary += f"\n❌ Code implementation failed: {implementation_result['message']}"
            return pipeline_summary
        
    except Exception as e:
        logger.error(f"Error in execute_multi_agent_research_pipeline: {e}")
        raise e

# Backward compatibility alias (deprecated)
# 向后兼容别名（已弃用）
async def paper_code_preparation(input_source, logger, progress_callback=None):
    """
    Deprecated: Use execute_multi_agent_research_pipeline instead.
    已弃用：请使用 execute_multi_agent_research_pipeline 替代。
    """
    logger.warning("paper_code_preparation is deprecated. Use execute_multi_agent_research_pipeline instead.")
    return await execute_multi_agent_research_pipeline(input_source, logger, progress_callback)
