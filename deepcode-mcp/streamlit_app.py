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
    page_title="ReproAI - Paper to Code Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .status-info {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .progress-container {
        margin: 1rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
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
        <h1>🤖 ReproAI - Paper to Code Engine</h1>
        <h3>AI-POWERED RESEARCH PAPER REPRODUCTION ENGINE 🚀</h3>
        <p>⚡ INTELLIGENT • AUTOMATED • CUTTING-EDGE ⚡</p>
    </div>
    """, unsafe_allow_html=True)

def display_features():
    """显示功能特性"""
    st.markdown("### 💎 Core Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🔍 Neural PDF Analysis</h4>
            <p>Advanced document processing with AI-powered content extraction</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>📁 Multi-Format Support</h4>
            <p>PDF • DOCX • PPTX • HTML • TXT</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🚀 Automated Repository Management</h4>
            <p>Smart GitHub integration and code organization</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🔬 Advanced Tech Stack</h4>
            <p>Python • AI • MCP • Docling • LLM</p>
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
    st.markdown("### 🚀 Start Processing")
    
    # 输入选项
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload File", "🌐 Enter URL"],
        horizontal=True
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
        <p>🔬 <strong>ReproAI v2.0.0</strong> | Built with Streamlit & AI | 
        <a href="https://github.com/your-repo" target="_blank">GitHub</a></p>
        <p>⚡ Powered by Python • AI • MCP • Docling • LLM</p>
        <p><small>💡 Tip: Keep this tab open during processing for best results</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 