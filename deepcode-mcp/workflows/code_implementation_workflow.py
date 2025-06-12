"""
通用论文代码复现工作流 - 基于MCP代理实现
Universal Paper Code Implementation Workflow - MCP Agent Based

这个模块实现了论文代码复现的完整工作流：
1. 文件树创建 (File Tree Creation)
2. 代码实现 (Code Implementation)
3. 测试生成 (Test Generation) - 待实现
4. 文档生成 (Documentation Generation) - 待实现

This module implements the complete workflow for paper code reproduction:
1. File Tree Creation
2. Code Implementation
3. Test Generation - To be implemented
4. Documentation Generation - To be implemented
"""

import asyncio
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# 导入MCP代理相关模块 / Import MCP agent related modules
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM

# 导入提示词 / Import prompts
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.code_prompts import STRUCTURE_GENERATOR_PROMPT, CODE_IMPLEMENTATION_PROMPT


class CodeImplementationWorkflow:
    """
    论文代码复现工作流管理器 / Paper Code Implementation Workflow Manager
    
    基于MCP代理模式，负责完整的代码复现流程
    Based on MCP agent pattern, responsible for complete code reproduction process
    """
    
    def __init__(self, config_path: str = "mcp_agent.secrets.yaml"):
        """
        初始化工作流 / Initialize workflow
        
        Args:
            config_path: API配置文件路径 / API configuration file path
        """
        self.config_path = config_path
        self.api_config = self._load_api_config()
    
    def _load_api_config(self) -> Dict[str, Any]:
        """
        加载API配置 / Load API configuration
        
        Returns:
            配置字典 / Configuration dictionary
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"无法加载API配置文件 / Unable to load API configuration file: {e}")

    def _create_logger(self, name: str = __name__) -> logging.Logger:
        """
        创建日志记录器 / Create logger
        
        Args:
            name: 日志记录器名称 / Logger name
            
        Returns:
            日志记录器 / Logger
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _read_plan_file(self, plan_file_path: str) -> str:
        """
        读取计划文件 / Read plan file
        
        Args:
            plan_file_path: 计划文件路径 / Plan file path
            
        Returns:
            计划内容 / Plan content
        """
        plan_path = Path(plan_file_path)
        if not plan_path.exists():
            raise FileNotFoundError(f"实现计划文件不存在 / Implementation plan file does not exist: {plan_file_path}")
        
        with open(plan_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _determine_target_directory(self, plan_file_path: str, target_directory: Optional[str] = None) -> str:
        """
        确定目标目录 / Determine target directory
        
        Args:
            plan_file_path: 计划文件路径 / Plan file path
            target_directory: 目标目录 / Target directory
            
        Returns:
            目标目录路径 / Target directory path
        """
        if target_directory is None:
            return str(Path(plan_file_path).parent)
        return target_directory

    # ==================== 步骤1: 文件树创建 / Step 1: File Tree Creation ====================
    
    async def step1_create_file_structure(self, plan_content: str, target_directory: str, logger: logging.Logger) -> str:
        """
        步骤1: 创建文件树结构 / Step 1: Create file tree structure
        
        Args:
            plan_content: 实现计划内容 / Implementation plan content
            target_directory: 目标目录路径 / Target directory path
            logger: 日志记录器 / Logger
            
        Returns:
            创建结果 / Creation result
        """
        logger.info("步骤1开始: 文件树结构创建 / Step 1 Starting: File tree structure creation")
        
        # 创建文件结构生成代理 / Create file structure generation agent
        structure_agent = Agent(
            name="StructureGeneratorAgent",
            instruction=STRUCTURE_GENERATOR_PROMPT,
            server_names=["command-executor"],
        )
        
        async with structure_agent:
            # 连接LLM / Connect to LLM
            creator = await structure_agent.attach_llm(AnthropicAugmentedLLM)
            
            # 构建分析消息 / Build analysis message
            message = f"""Analyze the following implementation plan and generate shell commands to create the file tree structure.

Target directory: {target_directory}/generate_code

Implementation Plan:
{plan_content}

TASK:
1. Find the file tree structure in the implementation plan
2. Generate shell commands (mkdir -p, touch) to create that structure
3. Use execute_commands tool to run the commands and create the files

Requirements:
- Create directories with mkdir -p
- Create files with touch
- Include __init__.py files for Python packages
- Use relative paths from the target directory
- Execute the commands to actually create the file structure

Please generate and execute the commands to create the complete project structure."""
            
            # 生成并执行命令 / Generate and execute commands
            result = await creator.generate_str(message=message)
            
            logger.info("步骤1完成: 文件树结构创建成功 / Step 1 Completed: File tree structure creation successful")
            return result

    # ==================== 步骤2: 代码实现 / Step 2: Code Implementation ====================
    
    async def step2_implement_code(self, plan_content: str, target_directory: str, logger: logging.Logger) -> str:
        """
        步骤2: 代码实现 / Step 2: Code implementation
        
        Args:
            plan_content: 实现计划内容 / Implementation plan content
            target_directory: 目标目录路径 / Target directory path  
            logger: 日志记录器 / Logger
            
        Returns:
            代码实现结果 / Code implementation result
        """
        logger.info("步骤2开始: 代码实现 / Step 2 Starting: Code implementation")
        
        # 检查文件树是否已存在 / Check if file tree exists
        code_directory = os.path.join(target_directory, "generate_code")
        if not os.path.exists(code_directory):
            raise FileNotFoundError("文件树结构不存在，请先运行文件树创建 / File tree structure doesn't exist, please run file tree creation first")
        
        # 创建代码实现代理 / Create code implementation agent
        code_agent = Agent(
            name="CodeImplementationAgent",
            instruction=CODE_IMPLEMENTATION_PROMPT,
            server_names=["command-executor"],
        )
        
        async with code_agent:
            # 连接LLM / Connect to LLM
            creator = await code_agent.attach_llm(AnthropicAugmentedLLM)
            
            # 获取文件结构信息 / Get file structure information
            file_structure = self._get_file_structure(code_directory)
            
            # 构建代码实现消息 / Build code implementation message
            message = f"""Target directory: {code_directory}

{file_structure}

Implementation Plan:
{plan_content}"""
            
            # 生成代码实现 / Generate code implementation
            result = await creator.generate_str(message=message)
            
            logger.info("步骤2完成: 代码实现成功 / Step 2 Completed: Code implementation successful")
            return result

    def _get_file_structure(self, code_directory: str) -> str:
        """
        获取已生成的文件结构 / Get generated file structure
        
        Args:
            code_directory: 代码目录 / Code directory
            
        Returns:
            文件结构字符串 / File structure string
        """
        try:
            if not os.path.exists(code_directory):
                return "No file structure found - please run file tree creation first"
            
            # 使用find命令获取文件结构 / Use find command to get file structure
            import subprocess
            result = subprocess.run(
                ["find", ".", "-type", "f"], 
                cwd=code_directory,
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                return f"Existing file structure:\n" + '\n'.join(sorted(files))
            else:
                return "Error reading file structure"
                
        except Exception as e:
            return f"Error getting file structure: {str(e)}"

    # ==================== 步骤3: 待实现功能 / Step 3: Future Features ====================
    
    async def step3_generate_tests(self, plan_content: str, target_directory: str, logger: logging.Logger) -> str:
        """
        步骤3: 生成测试代码 (待实现) / Step 3: Generate test code (To be implemented)
        
        Args:
            plan_content: 实现计划内容 / Implementation plan content
            target_directory: 目标目录路径 / Target directory path
            logger: 日志记录器 / Logger
            
        Returns:
            测试生成结果 / Test generation result
        """
        logger.info("步骤3: 测试生成功能待实现 / Step 3: Test generation feature to be implemented")
        return "Step 3: Test generation - Feature to be implemented"

    async def step4_generate_documentation(self, plan_content: str, target_directory: str, logger: logging.Logger) -> str:
        """
        步骤4: 生成文档 (待实现) / Step 4: Generate documentation (To be implemented)
        
        Args:
            plan_content: 实现计划内容 / Implementation plan content
            target_directory: 目标目录路径 / Target directory path
            logger: 日志记录器 / Logger
            
        Returns:
            文档生成结果 / Documentation generation result
        """
        logger.info("步骤4: 文档生成功能待实现 / Step 4: Documentation generation feature to be implemented")
        return "Step 4: Documentation generation - Feature to be implemented"

    # ==================== 主工作流 / Main Workflow ====================
    
    async def run_complete_workflow(
        self, 
        plan_file_path: str, 
        target_directory: Optional[str] = None,
        steps: Optional[list] = None,
        logger: Optional[logging.Logger] = None
    ) -> Dict[str, Any]:
        """
        运行完整的代码实现工作流 / Run complete code implementation workflow
        
        Args:
            plan_file_path: 实现计划文件路径 / Implementation plan file path
            target_directory: 目标目录 / Target directory
            steps: 要执行的步骤列表 / List of steps to execute ["structure", "code", "test", "docs"]
            logger: 日志记录器 / Logger
            
        Returns:
            完整工作流执行结果 / Complete workflow execution result
        """
        # 创建日志记录器（如果没有提供）/ Create logger (if not provided)
        if logger is None:
            logger = self._create_logger()
        
        # 默认执行步骤 / Default steps to execute
        if steps is None:
            steps = ["structure", "code"]  # 只执行已实现的步骤 / Only execute implemented steps
        
        try:
            # 读取实现计划 / Read implementation plan
            plan_content = self._read_plan_file(plan_file_path)
            
            # 确定目标目录 / Determine target directory
            target_directory = self._determine_target_directory(plan_file_path, target_directory)
            
            logger.info(f"开始完整工作流 / Starting complete workflow: {plan_file_path}")
            logger.info(f"目标目录 / Target directory: {target_directory}")
            logger.info(f"执行步骤 / Executing steps: {steps}")
            
            results = {}
            
            # 步骤1: 文件树创建 / Step 1: File tree creation
            if "structure" in steps:
                results["step1_structure"] = await self.step1_create_file_structure(
                    plan_content, target_directory, logger
                )
            
            # 步骤2: 代码实现 / Step 2: Code implementation  
            if "code" in steps:
                results["step2_code"] = await self.step2_implement_code(
                    plan_content, target_directory, logger
                )
            
            # 步骤3: 测试生成 (待实现) / Step 3: Test generation (To be implemented)
            if "test" in steps:
                results["step3_test"] = await self.step3_generate_tests(
                    plan_content, target_directory, logger
                )
            
            # 步骤4: 文档生成 (待实现) / Step 4: Documentation generation (To be implemented)
            if "docs" in steps:
                results["step4_docs"] = await self.step4_generate_documentation(
                    plan_content, target_directory, logger
                )
            
            logger.info("完整工作流执行成功 / Complete workflow execution successful")
            
            return {
                "status": "success",
                "plan_file": plan_file_path,
                "target_directory": target_directory,
                "code_directory": os.path.join(target_directory, "generate_code"),
                "executed_steps": steps,
                "results": results,
                "method": "mcp_complete_workflow"
            }
            
        except Exception as e:
            logger.error(f"完整工作流执行失败 / Complete workflow execution failed: {e}")
            return {
                "status": "error", 
                "message": str(e),
                "plan_file": plan_file_path
            }

    # ==================== 便捷接口 / Convenience Interfaces ====================
    
    async def run_file_tree_creation_only(
        self, 
        plan_file_path: str, 
        target_directory: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ) -> Dict[str, Any]:
        """
        仅运行文件树创建 / Run file tree creation only
        """
        return await self.run_complete_workflow(
            plan_file_path=plan_file_path,
            target_directory=target_directory,
            steps=["structure"],
            logger=logger
        )

    async def run_code_implementation_only(
        self, 
        plan_file_path: str, 
        target_directory: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ) -> Dict[str, Any]:
        """
        仅运行代码实现 / Run code implementation only
        """
        return await self.run_complete_workflow(
            plan_file_path=plan_file_path,
            target_directory=target_directory,
            steps=["code"],
            logger=logger
        )


# ==================== 便捷函数接口 / Convenience Function Interface ====================

async def create_project_structure(paper_dir: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    创建项目结构 - 便捷函数接口 / Create project structure - convenience function interface
    """
    plan_file_path = os.path.join(paper_dir, "initial_plan.txt")
    workflow = CodeImplementationWorkflow()
    return await workflow.run_file_tree_creation_only(
        plan_file_path=plan_file_path,
        target_directory=paper_dir,
        logger=logger
    )

async def implement_project_code(paper_dir: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    实现项目代码 - 便捷函数接口 / Implement project code - convenience function interface
    """
    plan_file_path = os.path.join(paper_dir, "initial_plan.txt")
    workflow = CodeImplementationWorkflow()
    return await workflow.run_code_implementation_only(
        plan_file_path=plan_file_path,
        target_directory=paper_dir,
        logger=logger
    )

async def run_full_implementation_workflow(paper_dir: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    运行完整实现工作流 - 便捷函数接口 / Run full implementation workflow - convenience function interface
    """
    plan_file_path = os.path.join(paper_dir, "initial_plan.txt")
    workflow = CodeImplementationWorkflow()
    return await workflow.run_complete_workflow(
        plan_file_path=plan_file_path,
        target_directory=paper_dir,
        steps=["structure", "code"],  # 目前只执行已实现的步骤 / Currently only execute implemented steps
        logger=logger
    )


# ==================== 主函数示例 / Main Function Example ====================

async def main():
    """主函数示例 / Main function example"""
    # 设置日志 / Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    logger = logging.getLogger(__name__)
    
    # 示例用法 / Example usage
    plan_file = "agent_folders/papers/1/initial_plan.txt"
    
    workflow = CodeImplementationWorkflow()
    
    # 选择运行模式 / Choose run mode
    import sys
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "structure":
            # 仅文件树创建模式 / File tree creation only mode
            logger.info("运行文件树创建工作流 / Running file tree creation workflow...")
            result = await workflow.run_file_tree_creation_only(plan_file, logger=logger)
        
        elif mode == "code":
            # 仅代码实现模式 / Code implementation only mode
            logger.info("运行代码实现工作流 / Running code implementation workflow...")
            result = await workflow.run_code_implementation_only(plan_file, logger=logger)
        
        elif mode == "full":
            # 完整工作流模式 / Full workflow mode
            logger.info("运行完整工作流 / Running complete workflow...")
            result = await workflow.run_complete_workflow(plan_file, logger=logger)
        
        else:
            print("无效模式 / Invalid mode. 使用 / Use: structure, code, full")
            return
    else:
        # 默认运行完整工作流 / Default to run complete workflow
        logger.info("运行完整工作流 / Running complete workflow...")
        result = await workflow.run_complete_workflow(plan_file, logger=logger)
    
    # 显示结果 / Display results
    print("=" * 80)
    print("工作流执行结果 / Workflow Execution Result:")
    print(f"状态 / Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"执行步骤 / Executed Steps: {result['executed_steps']}")
        print(f"代码目录 / Code Directory: {result['code_directory']}")
        print(f"使用方法 / Method: {result['method']}")
    else:
        print(f"错误信息 / Error Message: {result['message']}")
    
    print("\n" + "=" * 80)
    print("使用说明 / Usage Instructions:")
    print("文件树创建 / File tree creation: python workflows/code_implementation_workflow.py structure")
    print("代码实现 / Code implementation: python workflows/code_implementation_workflow.py code")
    print("完整工作流 / Complete workflow: python workflows/code_implementation_workflow.py full")


if __name__ == "__main__":
    asyncio.run(main())
