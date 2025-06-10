#!/usr/bin/env python3
"""
Streamlit App Launcher for ReproAI
启动ReproAI的Streamlit Web界面
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import streamlit
        print("✅ Streamlit is installed")
        return True
    except ImportError:
        print("❌ Streamlit is not installed")
        print("Please install it using: pip install streamlit>=1.28.0")
        return False

def main():
    """主函数"""
    print("🚀 ReproAI Streamlit Launcher")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 获取当前脚本目录
    current_dir = Path(__file__).parent
    streamlit_app_path = current_dir / "streamlit_app.py"
    
    # 检查streamlit_app.py是否存在
    if not streamlit_app_path.exists():
        print(f"❌ Streamlit app file not found: {streamlit_app_path}")
        sys.exit(1)
    
    print(f"📁 App location: {streamlit_app_path}")
    print("🌐 Starting Streamlit server...")
    print("=" * 50)
    
    # 启动Streamlit应用
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit server stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 