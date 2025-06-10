"""
通用论文代码复现工作流 - 文件树创建模块

这个模块实现了论文代码复现的第一步：文件树创建功能
从实现计划中提取文件结构信息，并在目标目录创建完整的文件树
"""

import asyncio
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import openai
from anthropic import Anthropic

# 导入提示词
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.code_prompts import (
    FILE_TREE_EXTRACTOR_PROMPT,
    FILE_TREE_CREATOR_PROMPT
)

# 导入简化的文件创建工具
from tools.simple_file_creator import create_file_tree, parse_file_tree_to_list

class CodeImplementationWorkflow:
    """论文代码复现工作流管理器"""
    
    def __init__(self, config_path: str = "mcp_agent.secrets.yaml"):
        """
        初始化工作流
        
        Args:
            config_path: API配置文件路径
        """
        self.config_path = config_path
        self.api_config = self._load_api_config()
        self.openai_client = None
        self.anthropic_client = None
        self._setup_llm_clients()
    
    def _load_api_config(self) -> Dict[str, Any]:
        """加载API配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"无法加载API配置文件: {e}")
    
    def _setup_llm_clients(self):
        """设置LLM客户端"""
        try:
            # 设置OpenAI客户端
            if 'openai' in self.api_config:
                openai.api_key = self.api_config['openai']['api_key']
                self.openai_client = openai.OpenAI(
                    api_key=self.api_config['openai']['api_key']
                )
            
            # 设置Anthropic客户端  
            if 'anthropic' in self.api_config:
                self.anthropic_client = Anthropic(
                    api_key=self.api_config['anthropic']['api_key']
                )
                
        except Exception as e:
            print(f"警告: LLM客户端设置失败: {e}")
    
    async def extract_file_tree_from_plan(self, plan_content: str, use_anthropic: bool = True) -> str:
        """
        从实现计划中提取文件树结构
        
        Args:
            plan_content: 实现计划内容
            use_anthropic: 是否使用Anthropic API
            
        Returns:
            提取的文件树结构
        """
        prompt = f"{FILE_TREE_EXTRACTOR_PROMPT}\n\n实现计划内容:\n{plan_content}"
        
        try:
            if use_anthropic and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "你是一个专业的文件结构分析专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000
                )
                return response.choices[0].message.content
                
            else:
                raise Exception("没有可用的LLM客户端")
                
        except Exception as e:
            raise Exception(f"文件树提取失败: {e}")
    
    def create_file_tree_directly(self, file_tree: str, target_directory: str) -> str:
        """
        直接创建文件树，不使用MCP
        
        Args:
            file_tree: 文件树结构描述
            target_directory: 目标目录路径
            
        Returns:
            创建结果
        """
        try:
            # 创建generate_code目录路径
            generate_code_path = os.path.join(target_directory, "generate_code")
            
            # 使用简化的文件创建工具
            result = create_file_tree(generate_code_path, file_tree)
            
            return result
                    
        except Exception as e:
            return f"文件树创建失败: {e}"
    
    async def generate_shell_commands_with_llm(self, file_tree: str, use_anthropic: bool = True) -> str:
        """
        使用LLM生成shell命令
        
        Args:
            file_tree: 文件树结构
            use_anthropic: 是否使用Anthropic API
            
        Returns:
            shell命令字符串
        """
        prompt = f"{FILE_TREE_CREATOR_PROMPT}\n\n文件树结构:\n{file_tree}"
        
        try:
            if use_anthropic and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                content = response.content[0].text
            
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4", 
                    messages=[
                        {"role": "system", "content": "你是一个专业的文件系统操作专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000
                )
                content = response.choices[0].message.content
            else:
                raise Exception("没有可用的LLM客户端")
            
            # 提取命令块
            if "```commands" in content:
                start = content.find("```commands") + len("```commands")
                end = content.find("```", start)
                commands = content[start:end].strip()
            elif "```" in content:
                # 备用解析
                start = content.find("```") + 3
                end = content.find("```", start)
                commands = content[start:end].strip()
            else:
                # 直接使用内容
                commands = content.strip()
            
            return commands
            
        except Exception as e:
            raise Exception(f"命令生成失败: {e}")
            
    def execute_commands_directly(self, commands: str, target_directory: str) -> str:
        """
        直接执行shell命令，不使用MCP
        
        Args:
            commands: shell命令字符串
            target_directory: 目标目录
            
        Returns:
            执行结果
        """
        try:
            from tools.simple_shell_executor import execute_shell_commands
            
            # 修正命令中的路径 - 替换项目名为generate_code
            corrected_commands = commands.replace('recdiff_project/', 'generate_code/')
            corrected_commands = corrected_commands.replace('project/', 'generate_code/')
            
            result = execute_shell_commands(corrected_commands, target_directory)
            return result
                    
        except Exception as e:
            return f"命令执行失败: {e}"
    
    def parse_file_tree_to_file_list(self, file_tree: str) -> list:
        """
        直接解析文件树为文件列表，不使用LLM
        
        Args:
            file_tree: 文件树结构
            
        Returns:
            文件路径列表
        """
        return parse_file_tree_to_list(file_tree)
    
    async def run_file_tree_creation(
        self, 
        plan_file_path: str, 
        target_directory: Optional[str] = None,
        use_anthropic: bool = True,
        use_llm_for_extraction: bool = True
    ) -> Dict[str, Any]:
        """
        运行文件树创建流程
        
        Args:
            plan_file_path: 实现计划文件路径
            target_directory: 目标目录，如果为None则从plan_file_path推导
            use_anthropic: 是否使用Anthropic API
            use_llm_for_extraction: 是否使用LLM提取文件树
            
        Returns:
            运行结果
        """
        try:
            # 读取实现计划
            plan_path = Path(plan_file_path)
            if not plan_path.exists():
                return {"status": "error", "message": f"计划文件不存在: {plan_file_path}"}
            
            with open(plan_path, 'r', encoding='utf-8') as f:
                plan_content = f.read()
            
            # 确定目标目录
            if target_directory is None:
                target_directory = str(plan_path.parent)
            
            print(f"开始处理计划文件: {plan_file_path}")
            print(f"目标目录: {target_directory}")
            
            # 步骤1: 提取文件树结构
            print("步骤1: 提取文件树结构...")
            if use_llm_for_extraction:
                file_tree = await self.extract_file_tree_from_plan(plan_content, use_anthropic)
            else:
                # 从计划中直接提取文件树（查找4. Code Organization部分）
                file_tree = self._extract_file_tree_from_text(plan_content)
            print("文件树提取完成")
            
            # 步骤2: LLM生成shell命令
            print("步骤2: LLM生成创建命令...")
            if use_llm_for_extraction:
                commands = await self.generate_shell_commands_with_llm(file_tree, use_anthropic)
            else:
                # 简化版本：直接解析文件列表然后创建
                file_list = self.parse_file_tree_to_file_list(file_tree)
                creation_result = self.create_file_tree_directly(file_tree, target_directory)
                print("文件结构创建完成")
                
                return {
                    "status": "success",
                    "plan_file": plan_file_path,
                    "target_directory": target_directory,
                    "file_tree": file_tree,
                    "file_list": file_list,
                    "creation_result": creation_result,
                    "files_created": len(file_list),
                    "commands": "直接创建（未使用LLM生成命令）"
                }
            
            print(f"LLM生成命令完成")
            
            # 步骤3: 执行shell命令
            print("步骤3: 执行shell命令...")
            creation_result = self.execute_commands_directly(commands, target_directory)
            print("命令执行完成")
            
            # 解析文件数量（用于显示）
            file_count = commands.count('touch') + commands.count('mkdir')
            
            return {
                "status": "success",
                "plan_file": plan_file_path,
                "target_directory": target_directory,
                "file_tree": file_tree,
                "commands": commands,
                "creation_result": creation_result,
                "files_created": file_count
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "message": str(e),
                "plan_file": plan_file_path
            }
    
    def _extract_file_tree_from_text(self, plan_content: str) -> str:
        """
        从计划文本中直接提取文件树结构
        
        Args:
            plan_content: 计划内容
            
        Returns:
            提取的文件树结构
        """
        lines = plan_content.split('\n')
        file_tree_lines = []
        in_file_tree = False
        
        for line in lines:
            if 'Code Organization' in line or 'File Tree' in line or 'project/' in line:
                in_file_tree = True
                continue
            
            if in_file_tree:
                if line.strip() == '---' or (line.startswith('5.') or line.startswith('Phase')):
                    break
                if ('├──' in line or '└──' in line or '│' in line or 
                    line.strip().endswith('/') or line.strip().endswith('.py') or 
                    line.strip().endswith('.txt') or line.strip().endswith('.md') or
                    line.strip().endswith('.yaml')):
                    file_tree_lines.append(line)
        
        return '\n'.join(file_tree_lines) if file_tree_lines else ""

# 便捷函数
async def create_project_structure(plan_file_path: str, target_directory: str = None) -> Dict[str, Any]:
    """
    便捷函数：创建项目结构
    
    Args:
        plan_file_path: 实现计划文件路径
        target_directory: 目标目录
        
    Returns:
        运行结果
    """
    workflow = CodeImplementationWorkflow()
    return await workflow.run_file_tree_creation(plan_file_path, target_directory, use_llm_for_extraction=False)

# 主函数示例
async def main():
    """主函数示例"""
    # 示例用法
    plan_file = r"agent_folders\papers\paper_3\initial_plan.txt"
    
    workflow = CodeImplementationWorkflow()
    result = await workflow.run_file_tree_creation(plan_file, use_llm_for_extraction=False)
    
    print("=" * 50)
    print("文件树创建结果:")
    print(f"状态: {result['status']}")
    if result['status'] == 'success':
        print(f"创建文件数: {result['files_created']}")
        print(f"目标目录: {result['target_directory']}")
        print("创建结果:")
        print(result['creation_result'])
    else:
        print(f"错误信息: {result['message']}")

if __name__ == "__main__":
    asyncio.run(main())
