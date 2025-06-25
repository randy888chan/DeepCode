#!/usr/bin/env python3
"""
Paper to Code - Main CLI Entry Point
论文到代码 - 主CLI入口点

🧬 This is the main entry point for the CLI version of Paper to Code
⚡ Provides the same functionality as paper_to_code.py but via command line

Usage:
    python main_cli.py                 # Interactive CLI mode
    python main_cli.py --help          # Show help information
"""

import os
import sys
import argparse
from pathlib import Path

# 确保项目根目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Paper to Code CLI - Transform research papers into working code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main_cli.py                          # Interactive mode
    python main_cli.py --version                # Show version
    
For more information, visit: https://github.com/your-repo/paper-to-code
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='Paper to Code CLI v2.0.0'
    )
    
    parser.add_argument(
        '--interactive', 
        action='store_true', 
        default=True,
        help='Run in interactive mode (default)'
    )
    
    return parser.parse_args()

def cleanup_cache():
    """清理Python缓存文件 / Clean up Python cache files"""
    try:
        print("🧹 Cleaning up cache files...")
        # 清理__pycache__目录
        os.system('find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null')
        # 清理.pyc文件
        os.system('find . -name "*.pyc" -delete 2>/dev/null')
        print("✅ Cache cleanup completed")
    except Exception as e:
        print(f"⚠️  Cache cleanup failed: {e}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 打印欢迎信息
    print("🧬 Paper to Code CLI - Starting...")
    
    try:
        # 导入并运行CLI启动器
        from cli.paper_to_code_cli import main as cli_launcher_main
        cli_launcher_main()
        
    except ImportError as e:
        print(f"❌ Failed to import CLI components: {e}")
        print("Please ensure all CLI files are properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        # 清理缓存文件
        cleanup_cache()

if __name__ == "__main__":
    main() 