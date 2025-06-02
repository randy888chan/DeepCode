@echo off
echo 🚀 ReproAI Streamlit Launcher
echo ================================

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

REM 检查Streamlit是否安装
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ❌ Streamlit is not installed
    echo Installing Streamlit...
    pip install streamlit>=1.28.0
)

REM 启动Streamlit应用
echo 🌐 Starting Streamlit server...
echo ================================
python -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false

pause 