#!/usr/bin/env python3
"""
Paper to Code - CLI Application Main Program
论文到代码 - CLI应用主程序

🧬 Command Line Interface for AI Research Engine
⚡ Transform research papers into working code via CLI
"""

import os
import sys
import asyncio
import time
import json
from pathlib import Path
from typing import Optional

# 禁止生成.pyc文件
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入MCP应用和工作流
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from workflows.initial_workflows import (
    execute_multi_agent_research_pipeline,
    run_paper_analyzer,
    run_paper_downloader
)
from utils.file_processor import FileProcessor
from cli.cli_interface import CLIInterface, Colors

class CLIApp:
    """CLI应用主类"""
    
    def __init__(self):
        self.cli = CLIInterface()
        self.app = MCPApp(name="paper_to_code_cli")
        self.logger = None
        self.context = None
        
    async def initialize_mcp_app(self):
        """初始化MCP应用"""
        self.cli.show_spinner("🚀 Initializing MCP application", 2.0)
        
        # 启动MCP应用
        self.app_context = self.app.run()
        self.agent_app = await self.app_context.__aenter__()
        
        self.logger = self.agent_app.logger
        self.context = self.agent_app.context
        
        # 配置文件系统
        self.context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
        
        self.cli.print_status("MCP application initialized successfully", "success")
        
    async def cleanup_mcp_app(self):
        """清理MCP应用"""
        if hasattr(self, 'app_context'):
            await self.app_context.__aexit__(None, None, None)
            
    async def process_input(self, input_source: str, input_type: str):
        """处理输入源（URL或文件）"""
        try:
            self.cli.print_separator()
            self.cli.print_status("Starting paper analysis...", "processing")
            
            # 显示处理阶段
            self.cli.display_processing_stages(0)
            
            # 处理输入源路径
            if input_source.startswith("file://"):
                file_path = input_source[7:]
                if os.name == 'nt' and file_path.startswith('/'):
                    file_path = file_path.lstrip('/')
                input_source = file_path
                
            # 阶段1: 论文分析
            self.cli.print_status("📊 Analyzing paper content...", "analysis")
            self.cli.display_processing_stages(1)
            
            analysis_result = await run_paper_analyzer(input_source, self.logger)
            self.cli.print_status("Paper analysis completed", "success")
            
            # 阶段2: 文档下载处理
            self.cli.print_status("📥 Processing downloads...", "download")
            self.cli.display_processing_stages(2)
            
            # 添加短暂暂停以显示进度
            await asyncio.sleep(2)
            
            download_result = await run_paper_downloader(analysis_result, self.logger)
            self.cli.print_status("Download processing completed", "success")
            
            # 阶段3-8: 多智能体研究管道
            self.cli.print_status("🔄 Executing multi-agent research pipeline...", "implementation")
            self.cli.display_processing_stages(3)
            
            repo_result = await execute_multi_agent_research_pipeline(download_result, self.logger)
            
            # 显示完成状态
            self.cli.display_processing_stages(8)
            self.cli.print_status("All operations completed successfully! 🎉", "complete")
            
            # 显示结果
            self.display_results(analysis_result, download_result, repo_result)
            
            # 添加到历史记录
            result = {
                'status': 'success',
                'analysis_result': analysis_result,
                'download_result': download_result,
                'repo_result': repo_result
            }
            self.cli.add_to_history(input_source, result)
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            self.cli.print_error_box("Processing Error", error_msg)
            self.cli.print_status(f"Error during processing: {error_msg}", "error")
            
            # 添加错误到历史记录
            error_result = {
                'status': 'error',
                'error': error_msg
            }
            self.cli.add_to_history(input_source, error_result)
            
            return error_result
            
    def display_results(self, analysis_result: str, download_result: str, repo_result: str):
        """显示处理结果"""
        self.cli.print_results_header()
        
        print(f"{Colors.BOLD}{Colors.OKCYAN}📊 ANALYSIS PHASE RESULTS:{Colors.ENDC}")
        self.cli.print_separator("─", 79, Colors.CYAN)
        
        # 尝试解析并格式化分析结果
        try:
            if analysis_result.strip().startswith('{'):
                parsed_analysis = json.loads(analysis_result)
                print(json.dumps(parsed_analysis, indent=2, ensure_ascii=False))
            else:
                print(analysis_result[:1000] + "..." if len(analysis_result) > 1000 else analysis_result)
        except:
            print(analysis_result[:1000] + "..." if len(analysis_result) > 1000 else analysis_result)
            
        print(f"\n{Colors.BOLD}{Colors.PURPLE}📥 DOWNLOAD PHASE RESULTS:{Colors.ENDC}")
        self.cli.print_separator("─", 79, Colors.PURPLE)
        print(download_result[:1000] + "..." if len(download_result) > 1000 else download_result)
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}⚙️  IMPLEMENTATION PHASE RESULTS:{Colors.ENDC}")
        self.cli.print_separator("─", 79, Colors.GREEN)
        print(repo_result[:1000] + "..." if len(repo_result) > 1000 else repo_result)
        
        # 尝试提取生成的代码目录信息
        if "Code generated in:" in repo_result:
            code_dir = repo_result.split("Code generated in:")[-1].strip().split('\n')[0]
            print(f"\n{Colors.BOLD}{Colors.YELLOW}📁 Generated Code Directory: {Colors.ENDC}{code_dir}")
            
        # 显示处理完成的工作流阶段
        print(f"\n{Colors.BOLD}{Colors.OKCYAN}🔄 COMPLETED WORKFLOW STAGES:{Colors.ENDC}")
        stages = [
            "📄 Document Processing",
            "🔍 Reference Analysis", 
            "📋 Plan Generation",
            "📦 Repository Download",
            "🗂️ Codebase Indexing",
            "⚙️ Code Implementation"
        ]
        
        for stage in stages:
            print(f"  ✅ {stage}")
            
        self.cli.print_separator()
        
    async def run_interactive_session(self):
        """运行交互式会话"""
        # 清屏并显示启动界面
        self.cli.clear_screen()
        self.cli.print_logo()
        self.cli.print_welcome_banner()
        
        # 初始化MCP应用
        await self.initialize_mcp_app()
        
        try:
            # 主交互循环
            while self.cli.is_running:
                self.cli.create_menu()
                choice = self.cli.get_user_input()
                
                if choice in ['q', 'quit', 'exit']:
                    self.cli.print_goodbye()
                    break
                    
                elif choice in ['u', 'url']:
                    url = self.cli.get_url_input()
                    if url:
                        await self.process_input(url, 'url')
                        
                elif choice in ['f', 'file']:
                    file_path = self.cli.upload_file_gui()
                    if file_path:
                        await self.process_input(f"file://{file_path}", 'file')
                        
                elif choice in ['h', 'history']:
                    self.cli.show_history()
                    
                else:
                    self.cli.print_status("Invalid choice. Please select U, F, H, or Q.", "warning")
                
                # 询问是否继续
                if self.cli.is_running and choice in ['u', 'f']:
                    if not self.cli.ask_continue():
                        self.cli.is_running = False
                        self.cli.print_status("Session ended by user", "info")
                        
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}⚠️  Process interrupted by user{Colors.ENDC}")
        except Exception as e:
            print(f"\n{Colors.FAIL}❌ Unexpected error: {str(e)}{Colors.ENDC}")
        finally:
            # 清理资源
            await self.cleanup_mcp_app()

async def main():
    """主函数"""
    start_time = time.time()
    
    try:
        # 创建并运行CLI应用
        app = CLIApp()
        await app.run_interactive_session()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠️  Application interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Application error: {str(e)}{Colors.ENDC}")
    finally:
        end_time = time.time()
        print(f"\n{Colors.BOLD}{Colors.CYAN}⏱️  Total runtime: {end_time - start_time:.2f} seconds{Colors.ENDC}")
        
        # 清理缓存文件
        print(f"{Colors.YELLOW}🧹 Cleaning up cache files...{Colors.ENDC}")
        if os.name == 'nt':  # Windows
            os.system('powershell -Command "Get-ChildItem -Path . -Filter \'__pycache__\' -Recurse -Directory | Remove-Item -Recurse -Force" 2>nul')
        else:  # Unix/Linux/macOS
            os.system('find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null')
        
        print(f"{Colors.OKGREEN}✨ Goodbye! Thanks for using Paper-to-Code CLI! ✨{Colors.ENDC}")

if __name__ == "__main__":
    asyncio.run(main()) 