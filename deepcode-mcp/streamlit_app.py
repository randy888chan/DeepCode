import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # 禁止生成.pyc文件

import asyncio
import time
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional
import streamlit as st
from datetime import datetime

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

# 页面配置
st.set_page_config(
    page_title="Paper to Code - AI Research Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式 - 重新设计的高端科技感配色
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Inter:wght@300;400;600;700&display=swap');
    
    :root {
        --primary-bg: #0a0e27;
        --secondary-bg: #1a1f3a;
        --accent-bg: #2d3748;
        --card-bg: rgba(45, 55, 72, 0.9);
        --glass-bg: rgba(255, 255, 255, 0.08);
        --glass-border: rgba(255, 255, 255, 0.12);
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --neon-blue: #64b5f6;
        --neon-cyan: #4dd0e1;
        --neon-green: #81c784;
        --neon-purple: #ba68c8;
        --text-primary: #ffffff;
        --text-secondary: #e3f2fd;
        --text-muted: #90caf9;
        --border-color: rgba(100, 181, 246, 0.2);
    }
    
    /* 全局应用背景和文字 */
    .stApp {
        background: linear-gradient(135deg, var(--primary-bg) 0%, var(--secondary-bg) 100%);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* 强制所有文本使用高对比度 */
    .stApp * {
        color: var(--text-primary) !important;
    }
    
    /* 侧边栏重新设计 - 深色科技风 */
    .css-1d391kg {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 50%, #21262d 100%) !important;
        border-right: 2px solid var(--neon-cyan) !important;
        box-shadow: 0 0 20px rgba(77, 208, 225, 0.3) !important;
    }
    
    .css-1d391kg * {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    .css-1d391kg h3 {
        color: var(--neon-cyan) !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        text-shadow: 0 0 15px rgba(77, 208, 225, 0.6) !important;
        border-bottom: 1px solid rgba(77, 208, 225, 0.3) !important;
        padding-bottom: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    .css-1d391kg p, .css-1d391kg div {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    /* 侧边栏信息框 - 深色科技风格 */
    .css-1d391kg .stAlert, 
    .css-1d391kg .stInfo, 
    .css-1d391kg .stSuccess, 
    .css-1d391kg .stWarning, 
    .css-1d391kg .stError {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
        border: 2px solid var(--neon-cyan) !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        box-shadow: 0 0 15px rgba(77, 208, 225, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        margin: 0.5rem 0 !important;
        padding: 1rem !important;
    }
    
    /* 侧边栏信息框文字强制白色 */
    .css-1d391kg .stInfo div,
    .css-1d391kg .stInfo p,
    .css-1d391kg .stInfo span {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
    }
    
    /* 侧边栏按钮 - 科技风格 */
    .css-1d391kg .stButton button {
        background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-blue) 100%) !important;
        color: #000000 !important;
        font-weight: 800 !important;
        border: 2px solid var(--neon-cyan) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 20px rgba(77, 208, 225, 0.4) !important;
        text-shadow: none !important;
        transition: all 0.3s ease !important;
    }
    
    .css-1d391kg .stButton button:hover {
        box-shadow: 0 0 30px rgba(77, 208, 225, 0.6) !important;
        transform: translateY(-2px) !important;
    }
    
    /* 侧边栏展开器 - 深色科技风 */
    .css-1d391kg .streamlit-expanderHeader {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--neon-purple) !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        box-shadow: 0 0 10px rgba(186, 104, 200, 0.3) !important;
    }
    
    .css-1d391kg .streamlit-expanderContent {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
        border: 2px solid var(--neon-purple) !important;
        color: var(--text-primary) !important;
        border-radius: 0 0 10px 10px !important;
        box-shadow: 0 0 10px rgba(186, 104, 200, 0.2) !important;
    }
    
    /* 侧边栏所有文字元素强制高对比度 */
    .css-1d391kg span, 
    .css-1d391kg p, 
    .css-1d391kg div, 
    .css-1d391kg label,
    .css-1d391kg strong,
    .css-1d391kg b {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* 侧边栏markdown内容 */
    .css-1d391kg [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-weight: 600 !important;
        background: none !important;
    }
    
    /* 侧边栏特殊样式 - 系统信息框 */
    .css-1d391kg .element-container {
        background: none !important;
    }
    
    .css-1d391kg .element-container div {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
        border: 1px solid var(--neon-cyan) !important;
        border-radius: 8px !important;
        padding: 0.8rem !important;
        box-shadow: 0 0 10px rgba(77, 208, 225, 0.2) !important;
        margin: 0.3rem 0 !important;
    }
    
    /* Processing History特殊处理 */
    .css-1d391kg .stExpander {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
        border: 2px solid var(--neon-green) !important;
        border-radius: 12px !important;
        box-shadow: 0 0 15px rgba(129, 199, 132, 0.3) !important;
        margin: 0.5rem 0 !important;
    }
    
    /* 确保所有文字在深色背景上可见 */
    .css-1d391kg .stExpander div,
    .css-1d391kg .stExpander p,
    .css-1d391kg .stExpander span {
        color: #ffffff !important;
        font-weight: 600 !important;
        background: none !important;
    }
    
    /* 主标题区域 */
    .main-header {
        background: linear-gradient(135deg, 
            rgba(100, 181, 246, 0.1) 0%, 
            rgba(77, 208, 225, 0.08) 50%, 
            rgba(129, 199, 132, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        box-shadow: 
            0 8px 32px rgba(100, 181, 246, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-blue) 50%, var(--neon-purple) 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-shadow: 0 0 30px rgba(77, 208, 225, 0.5) !important;
        margin-bottom: 1rem !important;
        animation: titleGlow 3s ease-in-out infinite alternate !important;
    }
    
    @keyframes titleGlow {
        from { filter: drop-shadow(0 0 10px rgba(77, 208, 225, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(186, 104, 200, 0.7)); }
    }
    
    .main-header h3 {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.2rem !important;
        font-weight: 400 !important;
        color: var(--text-secondary) !important;
        letter-spacing: 2px !important;
        margin-bottom: 0.5rem !important;
    }
    
    .main-header p {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
        color: var(--neon-green) !important;
        letter-spacing: 1px !important;
        font-weight: 600 !important;
    }
    
    /* 功能卡片重新设计 */
    .feature-card {
        background: var(--card-bg);
        backdrop-filter: blur(15px);
        border: 1px solid var(--border-color);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: var(--neon-cyan);
        box-shadow: 0 8px 30px rgba(77, 208, 225, 0.3);
    }
    
    .feature-card h4 {
        font-family: 'Inter', sans-serif !important;
        color: var(--neon-cyan) !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.8rem !important;
    }
    
    .feature-card p {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-secondary) !important;
        line-height: 1.6 !important;
        font-weight: 400 !important;
    }
    
    /* Streamlit 组件样式重写 */
    .stMarkdown h3 {
        color: var(--neon-cyan) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        text-shadow: 0 0 10px rgba(77, 208, 225, 0.3) !important;
    }
    
    /* 单选按钮样式 */
    .stRadio > div {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stRadio label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stRadio > div > div > div > label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    
    /* 文件上传器 */
    .stFileUploader > div {
        background: var(--card-bg) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 15px !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--neon-cyan) !important;
        box-shadow: 0 0 20px rgba(77, 208, 225, 0.3) !important;
    }
    
    .stFileUploader label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    .stFileUploader span {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    
    /* 文本输入框 */
    .stTextInput > div > div > input {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--neon-cyan) !important;
        box-shadow: 0 0 0 1px var(--neon-cyan) !important;
    }
    
    .stTextInput label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    /* 按钮样式 */
    .stButton > button {
        width: 100% !important;
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* 状态消息样式 */
    .status-success, .stSuccess {
        background: linear-gradient(135deg, rgba(129, 199, 132, 0.15) 0%, rgba(129, 199, 132, 0.05) 100%) !important;
        color: var(--neon-green) !important;
        padding: 1rem 1.5rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(129, 199, 132, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        font-weight: 600 !important;
    }
    
    .status-error, .stError {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.15) 0%, rgba(244, 67, 54, 0.05) 100%) !important;
        color: #ff8a80 !important;
        padding: 1rem 1.5rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(244, 67, 54, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        font-weight: 600 !important;
    }
    
    .status-warning, .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 193, 7, 0.05) 100%) !important;
        color: #ffcc02 !important;
        padding: 1rem 1.5rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 193, 7, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        font-weight: 600 !important;
    }
    
    .status-info, .stInfo {
        background: linear-gradient(135deg, rgba(77, 208, 225, 0.15) 0%, rgba(77, 208, 225, 0.05) 100%) !important;
        color: var(--neon-cyan) !important;
        padding: 1rem 1.5rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(77, 208, 225, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        font-weight: 600 !important;
    }
    
    /* 进度条 */
    .progress-container {
        margin: 1.5rem 0;
        padding: 2rem;
        background: var(--card-bg);
        backdrop-filter: blur(15px);
        border: 1px solid var(--border-color);
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .stProgress > div > div > div {
        background: var(--accent-gradient) !important;
        border-radius: 10px !important;
    }
    
    /* 文本区域 */
    .stTextArea > div > div > textarea {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* 确保所有Markdown内容可见 */
    [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    
    /* 分隔线 */
    hr {
        border-color: var(--border-color) !important;
        opacity: 0.5 !important;
    }
    
    /* 滚动条 */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--accent-bg);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-gradient);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-gradient);
    }
    
    /* 占位符文本 */
    ::placeholder {
        color: var(--text-muted) !important;
        opacity: 0.7 !important;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'mcp_app' not in st.session_state:
    st.session_state.mcp_app = None

def display_header():
    """显示应用头部"""
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Paper to Code</h1>
        <h3>NEXT-GENERATION AI RESEARCH AUTOMATION PLATFORM</h3>
        <p>⚡ NEURAL • AUTONOMOUS • REVOLUTIONARY ⚡</p>
    </div>
    """, unsafe_allow_html=True)

def display_features():
    """显示功能特性"""
    st.markdown("### 🔮 Advanced Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🧠 Quantum Document Analysis</h4>
            <p>Advanced neural networks with deep semantic understanding and multi-modal content extraction</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>📡 Universal Format Intelligence</h4>
            <p>PDF • DOCX • PPTX • HTML • TXT • LaTeX • arXiv</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🚀 Autonomous Code Genesis</h4>
            <p>Intelligent repository creation with automated dependency management and architecture design</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>⚛️ Cutting-Edge Tech Stack</h4>
            <p>Python • Transformer Models • MCP Protocol • Docling • Multi-Agent Systems</p>
        </div>
        """, unsafe_allow_html=True)

def display_status(message: str, status_type: str = "info"):
    """显示状态消息"""
    status_classes = {
        "success": "status-success",
        "error": "status-error", 
        "warning": "status-warning",
        "info": "status-info"
    }
    
    icons = {
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️"
    }
    
    css_class = status_classes.get(status_type, "status-info")
    icon = icons.get(status_type, "ℹ️")
    
    st.markdown(f"""
    <div class="{css_class}">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_mcp_app():
    """获取MCP应用实例（使用缓存）"""
    return MCPApp(name="paper_to_code")

async def initialize_app():
    """初始化MCP应用"""
    if not st.session_state.app_initialized:
        try:
            # 创建MCP应用实例
            st.session_state.mcp_app = get_mcp_app()
            st.session_state.app_initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize application: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")
            return False
    return True

async def process_input_async(input_source: str, input_type: str):
    """异步处理输入"""
    progress_container = st.container()
    
    try:
        # 获取MCP应用实例
        app = st.session_state.mcp_app
        
        async with app.run() as agent_app:
            logger = agent_app.logger
            context = agent_app.context
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            
            with progress_container:
                st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                
                # 创建进度条和状态文本
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 步骤1: 论文分析
                status_text.markdown("**📊 Step 1/3: Analyzing paper content...**")
                progress_bar.progress(10)
                
                # 处理输入源路径
                if input_source.startswith("file://"):
                    file_path = input_source[7:]
                    if os.name == 'nt' and file_path.startswith('/'):
                        file_path = file_path.lstrip('/')
                    input_source = file_path
                
                progress_bar.progress(20)
                analysis_result = await run_paper_analyzer(input_source, logger)
                progress_bar.progress(35) 
                
                # 添加5秒停顿
                await asyncio.sleep(5)
                
                # 步骤2: 下载处理
                status_text.markdown("**📥 Step 2/3: Processing downloads...**")
                progress_bar.progress(40)
                
                download_result = await run_paper_downloader(analysis_result, logger)
                progress_bar.progress(65)
                
                # 步骤3: 代码准备
                status_text.markdown("**🔧 Step 3/3: Preparing code repository...**")
                progress_bar.progress(70)
                
                repo_result = await paper_code_preparation(download_result, logger)
                progress_bar.progress(95)
                
                # 完成
                status_text.markdown("**✅ Processing completed successfully!**")
                progress_bar.progress(100)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            return {
                "analysis_result": analysis_result,
                "download_result": download_result,
                "repo_result": repo_result,
                "status": "success"
            }
            
    except Exception as e:
        error_msg = str(e)
        traceback_msg = traceback.format_exc()
        
        with progress_container:
            st.error(f"❌ Processing failed: {error_msg}")
            with st.expander("🔍 View detailed error information"):
                st.code(traceback_msg, language="python")
        
        return {
            "error": error_msg,
            "traceback": traceback_msg,
            "status": "error"
        }

def run_async_task(coro):
    """运行异步任务的辅助函数"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环正在运行，创建新的任务
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

def format_file_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def main():
    """主函数"""
    display_header()
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # 应用状态
        if st.session_state.app_initialized:
            st.success("🟢 Engine Ready")
        else:
            st.warning("🟡 Engine Initializing...")
        
        # 系统信息
        st.markdown("### 📊 System Info")
        st.info(f"**Python:** {sys.version.split()[0]}")
        st.info(f"**Platform:** {sys.platform}")
        
        st.markdown("---")
        
        # 处理历史
        st.markdown("### 📊 Processing History")
        if st.session_state.results:
            for i, result in enumerate(st.session_state.results):
                status_icon = "✅" if result.get('status') == 'success' else "❌"
                with st.expander(f"{status_icon} Task {i+1} - {result.get('timestamp', 'Unknown')}"):
                    st.write(f"**Status:** {result.get('status', 'Unknown')}")
                    if result.get('input_type'):
                        st.write(f"**Type:** {result['input_type']}")
                    if result.get('error'):
                        st.error(f"Error: {result['error']}")
        else:
            st.info("No processing history yet")
        
        # 清除历史按钮
        if st.session_state.results:
            if st.button("🗑️ Clear History"):
                st.session_state.results = []
                st.rerun()
    
    # 主内容区域
    display_features()
    
    st.markdown("---")
    st.markdown("""
    <h3 style="color: var(--text-primary) !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; font-size: 1.5rem !important; margin-bottom: 1rem !important;">
        🚀 Start Processing
    </h3>
    """, unsafe_allow_html=True)
    
    # 输入选项
    st.markdown("""
    <p style="color: var(--text-secondary) !important; font-family: 'Inter', sans-serif !important; font-weight: 500 !important; margin-bottom: 1rem !important;">
        Choose input method:
    </p>
    """, unsafe_allow_html=True)
    
    input_method = st.radio(
        "Choose your input method:",
        ["📁 Upload File", "🌐 Enter URL"],
        horizontal=True,
        label_visibility="hidden"
    )
    
    input_source = None
    input_type = None
    
    if input_method == "📁 Upload File":
        uploaded_file = st.file_uploader(
            "Upload research paper file",
            type=['pdf', 'docx', 'doc', 'html', 'htm', 'txt', 'md'],
            help="Supported formats: PDF, Word, PowerPoint, HTML, Text"
        )
        
        if uploaded_file is not None:
            # 显示文件信息
            file_size = len(uploaded_file.getvalue())
            st.info(f"📄 **File:** {uploaded_file.name} ({format_file_size(file_size)})")
            
            # 保存上传的文件到临时目录
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    input_source = tmp_file.name
                    input_type = "file"
                
                st.success(f"✅ File uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to save uploaded file: {str(e)}")
            
    else:  # URL输入
        url_input = st.text_input(
            "Enter paper URL",
            placeholder="https://arxiv.org/abs/..., https://ieeexplore.ieee.org/..., etc.",
            help="Enter a direct link to a research paper (arXiv, IEEE, ACM, etc.)"
        )
        
        if url_input:
            # 简单的URL验证
            if url_input.startswith(('http://', 'https://')):
                input_source = url_input
                input_type = "url"
                st.success(f"✅ URL entered: {url_input}")
            else:
                st.warning("⚠️ Please enter a valid URL starting with http:// or https://")
    
    # 处理按钮
    if input_source and not st.session_state.processing:
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            st.session_state.processing = True
            
            # 初始化应用
            with st.spinner("🚀 Initializing ReproAI Engine..."):
                init_success = run_async_task(initialize_app())
            
            if init_success:
                display_status("Engine initialized successfully", "success")
                
                # 处理输入
                st.markdown("### 📊 Processing Progress")
                
                result = run_async_task(process_input_async(input_source, input_type))
                
                if result["status"] == "success":
                    display_status("All operations completed successfully! 🎉", "success")
                    
                    # 显示结果
                    st.markdown("### 📋 Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        with st.expander("📊 Analysis Result", expanded=True):
                            st.text_area("Analysis Output", result["analysis_result"], height=200, key="analysis")
                    
                    with col2:
                        with st.expander("📥 Download Result"):
                            st.text_area("Download Output", result["download_result"], height=200, key="download")
                    
                    with col3:
                        with st.expander("🔧 Repository Result"):
                            st.text_area("Repository Output", result.get("repo_result", ""), height=200, key="repo")
                    
                    # 保存到历史记录
                    st.session_state.results.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "input_type": input_type,
                        "status": "success",
                        "result": result
                    })
                    
                else:
                    display_status(f"Error during processing", "error")
                    
                    # 保存错误到历史记录
                    st.session_state.results.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "input_type": input_type,
                        "status": "error",
                        "error": result.get("error", "Unknown error")
                    })
            else:
                display_status("Failed to initialize engine", "error")
            
            st.session_state.processing = False
            
            # 清理临时文件
            if input_type == "file" and input_source and os.path.exists(input_source):
                try:
                    os.unlink(input_source)
                except:
                    pass
    
    elif st.session_state.processing:
        st.info("🔄 Processing in progress... Please wait.")
        st.warning("⚠️ Do not refresh the page or close the browser during processing.")
    
    elif not input_source:
        st.info("👆 Please upload a file or enter a URL to start processing.")
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🧬 <strong>Paper to Code v3.0</strong> | Next-Gen AI Research Platform | 
        <a href="https://github.com/your-repo" target="_blank" style="color: var(--neon-blue);">GitHub</a></p>
        <p>⚡ Powered by Neural Networks • Quantum Computing • Multi-Agent AI • Advanced NLP</p>
        <p><small>💡 Tip: Experience the future of research automation - keep this tab active for optimal performance</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 