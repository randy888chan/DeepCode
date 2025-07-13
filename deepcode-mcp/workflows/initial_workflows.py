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
# 导入Docker同步管理器 / Import Docker sync manager
from utils.docker_sync_manager import setup_docker_sync, get_sync_directory
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
    try:
        # Log input information for debugging
        print(f"📊 Starting paper analysis...")
        print(f"Input prompt length: {len(prompt_text) if prompt_text else 0}")
        print(f"Input preview: {prompt_text[:200] if prompt_text else 'None'}...")
        
        if not prompt_text or prompt_text.strip() == "":
            raise ValueError("Empty or None prompt_text provided to run_paper_analyzer")
        
        analyzer_agent = Agent(
            name="PaperInputAnalyzerAgent",
            instruction=PAPER_INPUT_ANALYZER_PROMPT,
            server_names=["brave"],
        )
        
        async with analyzer_agent:
            print("analyzer: Connected to server, calling list_tools...")
            try:
                tools = await analyzer_agent.list_tools()
                print("Tools available:", tools.model_dump() if hasattr(tools, 'model_dump') else str(tools))
            except Exception as e:
                print(f"Failed to list tools: {e}")
            
            try:
                analyzer = await analyzer_agent.attach_llm(AnthropicAugmentedLLM)
                print("✅ LLM attached successfully")
            except Exception as e:
                print(f"❌ Failed to attach LLM: {e}")
                raise
            
            # 为论文分析设置更高的token输出量 / Set higher token output for paper analysis
            analysis_params = RequestParams(
                max_tokens=6144,  # 为论文分析设置6144 tokens
                temperature=0.3,  # 适中的创造性用于分析
            )
            
            print(f"🔄 Making LLM request with params: max_tokens={analysis_params.max_tokens}, temperature={analysis_params.temperature}")
            
            try:
                raw_result = await analyzer.generate_str(
                    message=prompt_text,
                    request_params=analysis_params
                )
                
                print(f"✅ LLM request completed")
                print(f"Raw result type: {type(raw_result)}")
                print(f"Raw result length: {len(raw_result) if raw_result else 0}")
                
                if not raw_result:
                    print("❌ CRITICAL: raw_result is empty or None!")
                    print("This could indicate:")
                    print("1. LLM API call failed silently")
                    print("2. API rate limiting or quota exceeded")
                    print("3. Network connectivity issues")
                    print("4. MCP server communication problems")
                    raise ValueError("LLM returned empty result")
                
            except Exception as e:
                print(f"❌ LLM generation failed: {e}")
                print(f"Exception type: {type(e)}")
                raise
            
            # 清理LLM输出，确保只返回纯净的JSON
            try:
                clean_result = extract_clean_json(raw_result)
                print(f"Raw LLM output: {raw_result}")
                print(f"Cleaned JSON output: {clean_result}")
                
                # Log to SimpleLLMLogger
                if hasattr(logger, 'log_response'):
                    logger.log_response(clean_result, model="PaperInputAnalyzer", agent="PaperInputAnalyzerAgent")
                
                if not clean_result or clean_result.strip() == "":
                    print("❌ CRITICAL: clean_result is empty after JSON extraction!")
                    print(f"Original raw_result was: {raw_result}")
                    raise ValueError("JSON extraction resulted in empty output")
                
                return clean_result
                
            except Exception as e:
                print(f"❌ JSON extraction failed: {e}")
                print(f"Raw result was: {raw_result}")
                raise
            
    except Exception as e:
        print(f"❌ run_paper_analyzer failed: {e}")
        print(f"Exception details: {type(e).__name__}: {str(e)}")
        raise

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
        print("downloader: Connected to server, calling list_tools...")
        tools = await downloader_agent.list_tools()
        print("Tools available:", tools.model_dump() if hasattr(tools, 'model_dump') else str(tools))
        
        downloader = await downloader_agent.attach_llm(AnthropicAugmentedLLM)
        
        # 为下载器设置更高的token输出量 / Set higher token output for downloader
        downloader_params = RequestParams(
            max_tokens=4096,  # 为下载器设置4096 tokens
            temperature=0.2,  # 下载任务需要精确性
        )
        
        return await downloader.generate_str(
            message=analysis_result,
            request_params=downloader_params
        )

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
        server_names=[],
    )
    algorithm_analysis_agent = Agent(
        name="AlgorithmAnalysisAgent",
        instruction=PAPER_ALGORITHM_ANALYSIS_PROMPT,
        server_names=["brave"],
    )
    code_planner_agent = Agent(
        name="CodePlannerAgent",
        instruction=CODE_PLANNING_PROMPT,
        server_names=["brave"],
    )

    code_aggregator_agent = ParallelLLM(
            fan_in_agent=code_planner_agent,
            fan_out_agents=[concept_analysis_agent, algorithm_analysis_agent],
            llm_factory=AnthropicAugmentedLLM,
        )
    
    # 设置更高的token输出量 / Set higher token output limit
    enhanced_params = RequestParams(
        max_tokens=26384,  # 增加到16384 tokens以确保完整输出
        temperature=0.3,   # 保持适中的创造性
    )
    
    result = await code_aggregator_agent.generate_str(
        message=document,
        request_params=enhanced_params
    )
    print(f"Code analysis result: {result}")
    return result

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
        print("GitHub downloader: Downloading repositories...")
        downloader = await github_download_agent.attach_llm(AnthropicAugmentedLLM)
        
        # 为GitHub下载设置更高的token输出量 / Set higher token output for GitHub download
        github_params = RequestParams(
            max_tokens=4096,  # 为GitHub下载设置4096 tokens
            temperature=0.1,  # GitHub下载需要高精确性
        )
        
        return await downloader.generate_str(
            message=search_result,
            request_params=github_params
        )

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
        print("Reference analyzer: Connected to server, analyzing references...")
        analyzer = await reference_analysis_agent.attach_llm(AnthropicAugmentedLLM)
        
        # 为引用分析设置更高的token输出量 / Set higher token output for reference analysis
        reference_params = RequestParams(
            max_tokens=30000,  # 为引用分析设置6144 tokens
            temperature=0.2,  # 引用分析需要更精确
        )
        
        reference_result = await analyzer.generate_str(
            message=analysis_result,
            request_params=reference_params
        )
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

async def _setup_paper_directory_structure(download_result: str, logger, sync_directory: str = None) -> dict:
    """
    Setup paper directory structure and prepare file paths.
    设置论文目录结构并准备文件路径
    
    Args:
        download_result: Download result from previous phase
        logger: Logger instance
        sync_directory: Optional sync directory path override
        
    Returns:
        dict: Directory structure information
    """
    # Parse download result to get file information
    result = await FileProcessor.process_file_input(download_result, base_dir=sync_directory)
    paper_dir = result['paper_dir']
    
    # Log directory structure setup
    print(f"📁 Paper directory structure:")
    print(f"   Base sync directory: {sync_directory or 'default'}")
    print(f"   Paper directory: {paper_dir}")
    
    return {
        'paper_dir': paper_dir,
        'standardized_text': result['standardized_text'],
        'reference_path': os.path.join(paper_dir, 'reference.txt'),
        'initial_plan_path': os.path.join(paper_dir, 'initial_plan.txt'),
        'download_path': os.path.join(paper_dir, 'github_download.txt'),
        'index_report_path': os.path.join(paper_dir, 'codebase_index_report.txt'),
        'implementation_report_path': os.path.join(paper_dir, 'code_implementation_report.txt'),
        'sync_directory': sync_directory
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
        progress_callback(50, "🔍 Analyzing paper references and related work...")
    
    reference_path = dir_info['reference_path']
    
    # Check if reference analysis already exists
    if os.path.exists(reference_path):
        print(f"Found existing reference analysis at {reference_path}")
        with open(reference_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Execute reference analysis
    reference_result = await paper_reference_analyzer(dir_info['standardized_text'], logger)
    
    # Save reference analysis result
    with open(reference_path, 'w', encoding='utf-8') as f:
        f.write(reference_result)
    print(f"Reference analysis saved to {reference_path}")
    
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
        progress_callback(40, "📋 Generating initial code implementation plan...")
    
    initial_plan_path = dir_info['initial_plan_path']
    
    # Check if initial plan already exists
    if not os.path.exists(initial_plan_path):
        initial_plan_result = await paper_code_analyzer(dir_info['standardized_text'], logger)
        with open(initial_plan_path, 'w', encoding='utf-8') as f:
            f.write(initial_plan_result)
        print(f"Initial plan saved to {initial_plan_path}")

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
        print(f"GitHub download results saved to {dir_info['download_path']}")
        
        # Verify if any repositories were actually downloaded
        code_base_path = os.path.join(dir_info['paper_dir'], 'code_base')
        if os.path.exists(code_base_path):
            downloaded_repos = [
                d for d in os.listdir(code_base_path) 
                if os.path.isdir(os.path.join(code_base_path, d)) and not d.startswith(".")
            ]
            
            if downloaded_repos:
                print(f"Successfully downloaded {len(downloaded_repos)} repositories: {downloaded_repos}")
            else:
                print("GitHub download phase completed, but no repositories were found in the code_base directory")
                print("This might indicate:")
                print("1. No relevant repositories were identified in the reference analysis")
                print("2. Repository downloads failed due to access permissions or network issues")
                print("3. The download agent encountered errors during the download process")
        else:
            print(f"Code base directory was not created: {code_base_path}")
            
    except Exception as e:
        print(f"Error during GitHub repository download: {e}")
        # Still save the error information
        error_message = f"GitHub download failed: {str(e)}"
        with open(dir_info['download_path'], 'w', encoding='utf-8') as f:
            f.write(error_message)
        print(f"GitHub download error saved to {dir_info['download_path']}")
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
    
    print("Starting codebase indexing to build relationships between downloaded code and target structure...")
    await asyncio.sleep(2)  # Brief pause before starting indexing
    
    # Check if code_base directory exists and has content
    code_base_path = os.path.join(dir_info['paper_dir'], 'code_base')
    if not os.path.exists(code_base_path):
        print(f"Code base directory not found: {code_base_path}")
        return {'status': 'skipped', 'message': 'No code base directory found - skipping indexing'}
    
    # Check if there are any repositories in the code_base directory
    try:
        repo_dirs = [
            d for d in os.listdir(code_base_path) 
            if os.path.isdir(os.path.join(code_base_path, d)) and not d.startswith(".")
        ]
        
        if not repo_dirs:
            print(f"No repositories found in {code_base_path}")
            print("This might be because:")
            print("1. GitHub download phase didn't complete successfully")
            print("2. No relevant repositories were identified for download")
            print("3. Repository download failed due to access issues")
            print("Continuing with code implementation without codebase indexing...")
            
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
            print(f"Indexing skip report saved to {dir_info['index_report_path']}")
            
            return skip_report
            
    except Exception as e:
        print(f"Error checking code base directory: {e}")
        return {'status': 'error', 'message': f'Error checking code base directory: {str(e)}'}
    
    try:
        from workflows.codebase_index_workflow import run_codebase_indexing
        
        print(f"Found {len(repo_dirs)} repositories to index: {repo_dirs}")
        
        # Run codebase index workflow
        index_result = await run_codebase_indexing(
            paper_dir=dir_info['paper_dir'],
            initial_plan_path=dir_info['initial_plan_path'],
            config_path="mcp_agent.secrets.yaml",
            logger=logger
        )
        
        # Log indexing results
        if index_result['status'] == 'success':
            print("Code indexing completed successfully!")
            print(f"Indexed {index_result['statistics']['total_repositories'] if index_result.get('statistics') else len(index_result['output_files'])} repositories")
            print(f"Generated {len(index_result['output_files'])} index files")
            
            # Save indexing results to file
            with open(dir_info['index_report_path'], 'w', encoding='utf-8') as f:
                f.write(str(index_result))
            print(f"Indexing report saved to {dir_info['index_report_path']}")
            
        elif index_result['status'] == 'warning':
            print(f"Code indexing completed with warnings: {index_result['message']}")
        else:
            print(f"Code indexing failed: {index_result['message']}")
            
        return index_result
        
    except Exception as e:
        print(f"Error during codebase indexing workflow: {e}")
        print("Continuing with code implementation despite indexing failure...")
        
        # Save error report
        error_report = {
            'status': 'error',
            'message': str(e),
            'phase': 'codebase_indexing',
            'recovery_action': 'continuing_with_code_implementation'
        }
        
        with open(dir_info['index_report_path'], 'w', encoding='utf-8') as f:
            f.write(str(error_report))
        print(f"Indexing error report saved to {dir_info['index_report_path']}")
        
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
    
    print("Starting code implementation based on the initial plan...")
    await asyncio.sleep(3)  # Brief pause before starting implementation
    
    try:
        # Create code implementation workflow instance
        code_workflow = CodeImplementationWorkflow()
        
        # Check if initial plan file exists
        if os.path.exists(dir_info['initial_plan_path']):
            print(f"Using initial plan from {dir_info['initial_plan_path']}")
            
            # Run code implementation workflow with pure code mode
            implementation_result = await code_workflow.run_workflow(
                plan_file_path=dir_info['initial_plan_path'],
                target_directory=dir_info['paper_dir'],
                pure_code_mode=True  # Focus on code implementation, skip testing
            )
            
            # Log implementation results
            if implementation_result['status'] == 'success':
                print("Code implementation completed successfully!")
                print(f"Code directory: {implementation_result['code_directory']}")
                
                # Save implementation results to file
                with open(dir_info['implementation_report_path'], 'w', encoding='utf-8') as f:
                    f.write(str(implementation_result))
                print(f"Implementation report saved to {dir_info['implementation_report_path']}")
                
            else:
                print(f"Code implementation failed: {implementation_result.get('message', 'Unknown error')}")
                
            return implementation_result
        else:
            print(f"Initial plan file not found at {dir_info['initial_plan_path']}, skipping code implementation")
            return {'status': 'warning', 'message': 'Initial plan not found - code implementation skipped'}
            
    except Exception as e:
        print(f"Error during code implementation workflow: {e}")
        return {'status': 'error', 'message': str(e)}

async def execute_multi_agent_research_pipeline(input_source, logger, progress_callback=None, enable_indexing=True):
    """
    Execute the complete multi-agent research pipeline from paper input to code implementation.
    执行从论文输入到代码实现的完整多智能体研究流水线
    
    This is the main orchestration function that coordinates all research workflow phases:
    - Docker synchronization setup for seamless file access
    - Paper analysis and content extraction
    - Code planning and structure design
    - Reference analysis and GitHub repository discovery (optional)
    - Codebase indexing and relationship analysis (optional)
    - Final code implementation
    
    Args:
        input_source (str): The input source (file path, URL, or analysis result)
        logger: The logger instance for comprehensive logging
        progress_callback (callable, optional): Progress callback function for UI updates
        enable_indexing (bool, optional): Whether to enable codebase indexing (default: True)
        
    Returns:
        str: The comprehensive pipeline execution result with status and outcomes
    """ 
    try:
        # Phase 0: Docker Synchronization Setup
        # 阶段0：Docker同步设置
        if progress_callback:
            progress_callback(5, "🔄 Setting up Docker synchronization for seamless file access...")
        
        print("🚀 Starting multi-agent research pipeline with Docker sync support")
        
        # Setup Docker synchronization
        sync_result = await setup_docker_sync(logger=logger)
        sync_directory = get_sync_directory()
        
        print(f"📁 Sync environment: {sync_result['environment']}")
        print(f"📂 Sync directory: {sync_directory}")
        print(f"✅ Sync status: {sync_result['message']}")
        
        # 记录索引功能状态
        if enable_indexing:
            print("🗂️ Codebase indexing enabled - full workflow")
        else:
            print("⚡ Fast mode - codebase indexing disabled")
        
        # Update file processor to use sync directory
        if sync_result['environment'] == 'docker':
            print("🐳 Running in Docker container - files will sync to local machine")
        else:
            print("💻 Running locally - use Docker container for full sync experience")
            print("💡 Tip: Run 'python start_docker_sync.py' for Docker sync mode")
        
        # Continue with original pipeline phases...
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
        
        dir_info = await _setup_paper_directory_structure(download_result, logger, sync_directory)
        await asyncio.sleep(30)
        
        # Phase 4: Code Planning
        # 阶段4：代码规划
        await _execute_code_planning_phase(dir_info, logger, progress_callback)
        
        # Phase 5: Reference Analysis (仅在启用索引时执行)
        # 阶段5：引用分析（仅在启用索引时执行）
        if enable_indexing:
            reference_result = await _execute_reference_analysis_phase(dir_info, logger, progress_callback)
        else:
            print("🔶 Skipping reference analysis (indexing disabled)")
            # 创建一个空的引用分析结果以保持文件结构一致性
            reference_result = "Reference analysis skipped - indexing disabled for faster processing"
            with open(dir_info['reference_path'], 'w', encoding='utf-8') as f:
                f.write(reference_result)
        
        # Phase 6: GitHub Repository Download (可选)
        # 阶段6：GitHub仓库下载（可选）
        if enable_indexing:
            await _execute_github_download_phase(reference_result, dir_info, logger, progress_callback)
        else:
            print("🔶 Skipping GitHub repository download (indexing disabled)")
            # 创建一个空的下载结果文件以保持文件结构一致性
            with open(dir_info['download_path'], 'w', encoding='utf-8') as f:
                f.write("GitHub repository download skipped - indexing disabled for faster processing")
        
        # Phase 7: Codebase Indexing (可选)
        # 阶段7：代码库索引（可选）
        if enable_indexing:
            index_result = await _execute_codebase_indexing_phase(dir_info, logger, progress_callback)
        else:
            print("🔶 Skipping codebase indexing (indexing disabled)")
            # 创建一个跳过索引的结果
            index_result = {
                'status': 'skipped',
                'reason': 'indexing_disabled',
                'message': 'Codebase indexing skipped for faster processing'
            }
            with open(dir_info['index_report_path'], 'w', encoding='utf-8') as f:
                f.write(str(index_result))
        
        # Phase 8: Code Implementation
        # 阶段8：代码实现
        implementation_result = await _execute_code_implementation_phase(dir_info, logger, progress_callback)
        
        # Final Status Report
        # 最终状态报告
        if enable_indexing:
            pipeline_summary = f"Multi-agent research pipeline completed for {dir_info['paper_dir']}"
        else:
            pipeline_summary = f"Multi-agent research pipeline completed (fast mode) for {dir_info['paper_dir']}"
        
        # Add indexing status to summary
        if not enable_indexing:
            pipeline_summary += f"\n⚡ Fast mode: GitHub download and codebase indexing skipped"
        elif index_result['status'] == 'skipped':
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
        print(f"Error in execute_multi_agent_research_pipeline: {e}")
        raise e

# Backward compatibility alias (deprecated)
# 向后兼容别名（已弃用）
async def paper_code_preparation(input_source, logger, progress_callback=None):
    """
    Deprecated: Use execute_multi_agent_research_pipeline instead.
    已弃用：请使用 execute_multi_agent_research_pipeline 替代。
    """
    print("paper_code_preparation is deprecated. Use execute_multi_agent_research_pipeline instead.")
    return await execute_multi_agent_research_pipeline(input_source, logger, progress_callback)
