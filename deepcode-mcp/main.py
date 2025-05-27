import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # 禁止生成.pyc文件

import asyncio
import time
import json
import sys
from pathlib import Path
from typing import Optional

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from workflows.initial_workflows import (
    paper_code_preparation,
    run_paper_analyzer,
    run_paper_downloader
)
from utils.file_processor import FileProcessor
from utils.cli_interface import CLIInterface, Colors

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

async def process_input(input_source: str, cli: CLIInterface, logger):
    """Process either URL or file path"""
    cli.print_separator()
    cli.print_status("Starting paper analysis...", "processing")
    cli.show_progress_bar("🔍 Initializing analysis engine")
    
    try:
        # 处理输入源路径
        if input_source.startswith("file://"):
            # 移除file://前缀并转换为正确的文件路径
            file_path = input_source[7:]  # 跳过"file://"
            if os.name == 'nt' and file_path.startswith('/'):
                # Windows下处理路径格式
                file_path = file_path.lstrip('/')
            input_source = file_path
            
        # Run paper analyzer
        cli.print_status("📊 Analyzing paper content...", "analysis")
        analysis_result = await run_paper_analyzer(input_source, logger)
        cli.print_status("Paper analysis completed", "success")
        
        # Run paper downloader
        cli.print_status("📥 Processing downloads...", "download")
        download_result = await run_paper_downloader(analysis_result, logger)
        cli.print_status("Download processing completed", "success")
        # Display results with beautiful formatting
        cli.print_results_header()
        print(f"{Colors.CYAN}{download_result}{Colors.ENDC}")

        repo_result = await paper_code_preparation(download_result, logger)
        
        cli.print_separator()
        cli.print_status("All operations completed successfully! 🎉", "success")
        
    except Exception as e:
        cli.print_error_box("Processing Error", str(e))
        cli.print_status(f"Error during processing: {str(e)}", "error")

async def main():
    """Enhanced main function with professional CLI interface"""
    cli = CLIInterface()
    
    # Clear screen and show startup sequence
    cli.clear_screen()
    cli.print_logo()
    cli.print_welcome_banner()
    
    # Initialize MCP application
    cli.show_spinner("🚀 Initializing ReproAI", 2.0)
    
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
        
        cli.print_status("Engine initialized successfully", "success")
        cli.print_separator()
        
        # Main interaction loop
        while cli.is_running:
            cli.create_menu()
            choice = cli.get_user_input()
            
            if choice in ['q', 'quit', 'exit']:
                cli.print_goodbye()
                break
                
            elif choice in ['u', 'url']:
                url = cli.get_url_input()
                if url:
                    await process_input(url, cli, logger)
                    
            elif choice in ['f', 'file']:
                file_path = cli.upload_file_gui()
                if file_path:
                    await process_input(f"file://{file_path}", cli, logger)
                    
            else:
                cli.print_status("Invalid choice. Please select U, F, or Q.", "warning")
            
            # Ask if user wants to continue
            if cli.is_running:
                if not cli.ask_continue():
                    cli.is_running = False
                    cli.print_status("Session ended by user", "info")

if __name__ == "__main__":
    start = time.time()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠️  Process interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Unexpected error: {str(e)}{Colors.ENDC}")
    finally:
        end = time.time()
        print(f"\n{Colors.BOLD}{Colors.CYAN}⏱️  Total runtime: {end - start:.2f} seconds{Colors.ENDC}")
        
        # Clean up cache files
        print(f"{Colors.YELLOW}🧹 Cleaning up cache files...{Colors.ENDC}")
        if os.name == 'nt':  # Windows
            os.system('powershell -Command "Get-ChildItem -Path . -Filter \'__pycache__\' -Recurse -Directory | Remove-Item -Recurse -Force"')
        else:  # Unix/Linux/macOS
            os.system('find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null')
        
        print(f"{Colors.OKGREEN}✨ Goodbye! Thanks for using Paper-to-Code Engine! ✨{Colors.ENDC}")