from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from utils.file_processor import FileProcessor

# 导入代码实现工作流 / Import code implementation workflow
from workflows.code_implementation_workflow import CodeImplementationWorkflow
import os
import asyncio

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # 禁止生成.pyc文件
from prompts.code_prompts import (
    PAPER_INPUT_ANALYZER_PROMPT,
    PAPER_DOWNLOADER_PROMPT,
    PAPER_REFERENCE_ANALYZER_PROMPT,
    PAPER_ALGORITHM_ANALYSIS_PROMPT,
    PAPER_CONCEPT_ANALYSIS_PROMPT,
    CODE_PLANNING_PROMPT,
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
    if "```json" in llm_output:
        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, llm_output, re.DOTALL)
        if match:
            json_text = match.group(1).strip()
            try:
                json.loads(json_text)
                return json_text
            except json.JSONDecodeError:
                pass

    # 3. 查找以{开始的JSON对象
    lines = llm_output.split("\n")
    json_lines = []
    in_json = False
    brace_count = 0

    for line in lines:
        stripped = line.strip()
        if not in_json and stripped.startswith("{"):
            in_json = True
            json_lines = [line]
            brace_count = stripped.count("{") - stripped.count("}")
        elif in_json:
            json_lines.append(line)
            brace_count += stripped.count("{") - stripped.count("}")
            if brace_count == 0:
                break

    if json_lines:
        json_text = "\n".join(json_lines).strip()
        try:
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            pass

    # 4. 最后的尝试：使用正则表达式查找JSON
    pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
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
        server_names=["filesystem", "brave"],
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
        instruction="Download github repo to the directory {paper_dir}/code_base".format(
            paper_dir=paper_dir
        ),
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
        paper_dir = result["paper_dir"]  # 直接使用返回的paper_dir
        reference_path = os.path.join(paper_dir, "reference.txt")
        initial_plan_path = os.path.join(paper_dir, "initial_plan.txt")

        # 1. 分析论文引用或读取已有结果
        if os.path.exists(reference_path):
            logger.info(f"Found existing reference analysis at {reference_path}")
            with open(reference_path, "r", encoding="utf-8") as f:
                reference_result = f.read()
        else:
            # 执行论文引用分析
            reference_result = await paper_reference_analyzer(
                result["standardized_text"], logger
            )
            initial_plan_result = await paper_code_analyzer(
                result["standardized_text"], logger
            )
            # 将reference结果写入文件
            with open(reference_path, "w", encoding="utf-8") as f:
                f.write(reference_result)
            logger.info(f"Reference analysis has been saved to {reference_path}")
            with open(initial_plan_path, "w", encoding="utf-8") as f:
                f.write(initial_plan_result)
            logger.info(f"Initial plan has been saved to {initial_plan_path}")

        # 2. 下载GitHub仓库
        await asyncio.sleep(5)
        download_result = await github_repo_download(
            reference_result, paper_dir, logger
        )
        download_path = os.path.join(paper_dir, "github_download.txt")
        with open(download_path, "w", encoding="utf-8") as f:
            f.write(download_result)
        logger.info(f"GitHub download results have been saved to {download_path}")

        # 3. 执行代码复现
        logger.info("Starting code implementation based on the initial plan...")
        await asyncio.sleep(3)  # Brief pause before starting implementation

        # 步骤4: 代码实现工作流 / Step 4: Code Implementation Workflow
        try:
            # 创建代码实现工作流实例 / Create code implementation workflow instance
            code_workflow = CodeImplementationWorkflow()

            # 检查初始计划文件是否存在 / Check if initial plan file exists
            if os.path.exists(initial_plan_path):
                logger.info(f"Using initial plan from {initial_plan_path}")

                # 运行代码实现工作流 / Run code implementation workflow
                # 使用纯代码模式进行实现 / Use pure code mode for implementation
                implementation_result = await code_workflow.run_workflow(
                    plan_file_path=initial_plan_path,
                    target_directory=paper_dir,
                    pure_code_mode=True,  # 专注于代码实现，跳过测试
                )

                # 记录实现结果 / Log implementation results
                if implementation_result["status"] == "success":
                    logger.info("Code implementation completed successfully!")
                    logger.info(
                        f"Code directory: {implementation_result['code_directory']}"
                    )

                    # 保存实现结果到文件 / Save implementation results to file
                    implementation_report_path = os.path.join(
                        paper_dir, "code_implementation_report.txt"
                    )
                    with open(implementation_report_path, "w", encoding="utf-8") as f:
                        f.write(str(implementation_result))
                    logger.info(
                        f"Implementation report saved to {implementation_report_path}"
                    )

                    return f"Paper code preparation and implementation completed successfully for {paper_dir}. Code generated in: {implementation_result['code_directory']}"
                else:
                    logger.error(
                        f"Code implementation failed: {implementation_result.get('message', 'Unknown error')}"
                    )
                    return f"Paper code preparation completed, but code implementation failed: {implementation_result.get('message', 'Unknown error')}"
            else:
                logger.warning(
                    f"Initial plan file not found at {initial_plan_path}, skipping code implementation"
                )
                return f"Paper code preparation completed for {paper_dir}, but initial plan not found - code implementation skipped"

        except Exception as e:
            logger.error(f"Error during code implementation workflow: {e}")
            return f"Paper code preparation completed for {paper_dir}, but code implementation failed: {str(e)}"

    except Exception as e:
        logger.error(f"Error in paper_code_preparation: {e}")
        raise e
