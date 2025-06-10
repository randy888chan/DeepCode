#!/usr/bin/env python3
"""
Paper to Code - AI Research Engine Launcher
论文到代码 - AI研究引擎启动器

🧬 Next-Generation AI Research Automation Platform
⚡ Transform research papers into working code automatically
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查必要的依赖是否已安装 / Check if necessary dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        missing_deps.append("streamlit>=1.28.0")
    
    try:
        import yaml
        print("✅ PyYAML is installed")
    except ImportError:
        missing_deps.append("pyyaml")
    
    try:
        import asyncio
        print("✅ Asyncio is available")
    except ImportError:
        missing_deps.append("asyncio")
    
    if missing_deps:
        print("\n❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies using:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    print("✅ All dependencies satisfied")
    return True

def print_banner():
    """显示启动横幅 / Display startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🧬 Paper to Code - AI Research Engine                     ║
║                                                              ║
║    ⚡ NEURAL • AUTONOMOUS • REVOLUTIONARY ⚡                ║
║                                                              ║
║    Transform research papers into working code               ║
║    Next-generation AI automation platform                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    """主函数 / Main function"""
    print_banner()
    
    # 检查依赖 / Check dependencies
    if not check_dependencies():
        print("\n🚨 Please install missing dependencies and try again.")
        sys.exit(1)
    
    # 获取当前脚本目录 / Get current script directory
    current_dir = Path(__file__).parent
    streamlit_app_path = current_dir / "ui" / "streamlit_app.py"
    
    # 检查streamlit_app.py是否存在 / Check if streamlit_app.py exists
    if not streamlit_app_path.exists():
        print(f"❌ UI application file not found: {streamlit_app_path}")
        print("Please ensure the ui/streamlit_app.py file exists.")
        sys.exit(1)
    
    print(f"\n📁 UI App location: {streamlit_app_path}")
    print("🌐 Starting Paper to Code web interface...")
    print("🚀 Launching on http://localhost:8501")
    print("=" * 70)
    print("💡 Tip: Keep this terminal open while using the application")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 70)
    
    # 启动Streamlit应用 / Launch Streamlit application
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "dark",
            "--theme.primaryColor", "#4dd0e1",
            "--theme.backgroundColor", "#0a0e27",
            "--theme.secondaryBackgroundColor", "#1a1f3a"
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start Paper to Code: {e}")
        print("Please check if Streamlit is properly installed.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Paper to Code server stopped by user")
        print("Thank you for using Paper to Code! 🧬")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your Python environment and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 