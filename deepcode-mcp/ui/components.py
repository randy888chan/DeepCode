"""
Streamlit UI组件模块 / Streamlit UI Components Module

包含所有可复用的UI组件
Contains all reusable UI components
"""

import streamlit as st
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime


def display_header():
    """
    显示应用头部 / Display application header
    """
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Paper to Code</h1>
        <h3>NEXT-GENERATION AI RESEARCH AUTOMATION PLATFORM</h3>
        <p>⚡ NEURAL • AUTONOMOUS • REVOLUTIONARY ⚡</p>
    </div>
    """, unsafe_allow_html=True)


def display_features():
    """
    显示功能特性 / Display application features
    """
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
    """
    显示状态消息 / Display status message
    
    Args:
        message: 状态消息 / Status message
        status_type: 状态类型 / Status type (success, error, warning, info)
    """
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


def sidebar_control_panel() -> Dict[str, Any]:
    """
    侧边栏控制面板 / Sidebar control panel
    
    Returns:
        控制面板状态 / Control panel state
    """
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # 应用状态
        if st.session_state.processing:
            st.warning("🟡 Engine Processing...")
        else:
            st.info("⚪ Engine Ready")
        
        # 系统信息
        st.markdown("### 📊 System Info")
        st.info(f"**Python:** {sys.version.split()[0]}")
        st.info(f"**Platform:** {sys.platform}")
        
        st.markdown("---")
        
        # 处理历史
        history_info = display_processing_history()
        
        return {
            "processing": st.session_state.processing,
            "history_count": history_info["count"],
            "has_history": history_info["has_history"]
        }


def display_processing_history() -> Dict[str, Any]:
    """
    显示处理历史 / Display processing history
    
    Returns:
        历史信息 / History information
    """
    st.markdown("### 📊 Processing History")
    
    has_history = bool(st.session_state.results)
    history_count = len(st.session_state.results)
    
    if has_history:
        # 只显示最近10条记录
        recent_results = st.session_state.results[-10:]
        for i, result in enumerate(reversed(recent_results)):
            status_icon = "✅" if result.get('status') == 'success' else "❌"
            with st.expander(f"{status_icon} Task - {result.get('timestamp', 'Unknown')}"):
                st.write(f"**Status:** {result.get('status', 'Unknown')}")
                if result.get('input_type'):
                    st.write(f"**Type:** {result['input_type']}")
                if result.get('error'):
                    st.error(f"Error: {result['error']}")
    else:
        st.info("No processing history yet")
    
    # 清除历史按钮
    if has_history:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.results = []
                st.rerun()
        with col2:
            st.info(f"Total: {history_count} tasks")
    
    return {
        "has_history": has_history,
        "count": history_count
    }


def file_input_component(task_counter: int) -> Optional[str]:
    """
    文件输入组件 / File input component
    
    Args:
        task_counter: 任务计数器 / Task counter
        
    Returns:
        文件路径或None / File path or None
    """
    uploaded_file = st.file_uploader(
        "Upload research paper file",
        type=['pdf', 'docx', 'doc', 'html', 'htm', 'txt', 'md'],
        help="Supported formats: PDF, Word, PowerPoint, HTML, Text",
        key=f"file_uploader_{task_counter}"
    )
    
    if uploaded_file is not None:
        # 显示文件信息
        file_size = len(uploaded_file.getvalue())
        st.info(f"📄 **File:** {uploaded_file.name} ({format_file_size(file_size)})")
        
        # 保存上传的文件到临时目录
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.success(f"✅ File uploaded successfully!")
                return tmp_file.name
        except Exception as e:
            st.error(f"❌ Failed to save uploaded file: {str(e)}")
            return None
    
    return None


def url_input_component(task_counter: int) -> Optional[str]:
    """
    URL输入组件 / URL input component
    
    Args:
        task_counter: 任务计数器 / Task counter
        
    Returns:
        URL或None / URL or None
    """
    url_input = st.text_input(
        "Enter paper URL",
        placeholder="https://arxiv.org/abs/..., https://ieeexplore.ieee.org/..., etc.",
        help="Enter a direct link to a research paper (arXiv, IEEE, ACM, etc.)",
        key=f"url_input_{task_counter}"
    )
    
    if url_input:
        # 简单的URL验证
        if url_input.startswith(('http://', 'https://')):
            st.success(f"✅ URL entered: {url_input}")
            return url_input
        else:
            st.warning("⚠️ Please enter a valid URL starting with http:// or https://")
            return None
    
    return None


def input_method_selector(task_counter: int) -> tuple[Optional[str], Optional[str]]:
    """
    输入方法选择器 / Input method selector
    
    Args:
        task_counter: 任务计数器 / Task counter
        
    Returns:
        (input_source, input_type) / (输入源, 输入类型)
    """
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
        label_visibility="hidden",
        key=f"input_method_{task_counter}"
    )
    
    input_source = None
    input_type = None
    
    if input_method == "📁 Upload File":
        input_source = file_input_component(task_counter)
        input_type = "file" if input_source else None
    else:  # URL输入
        input_source = url_input_component(task_counter)
        input_type = "url" if input_source else None
    
    return input_source, input_type


def results_display_component(result: Dict[str, Any], task_counter: int):
    """
    结果显示组件 / Results display component
    
    Args:
        result: 处理结果 / Processing result
        task_counter: 任务计数器 / Task counter
    """
    st.markdown("### 📋 Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("📊 Analysis Result", expanded=True):
            st.text_area("Analysis Output", result["analysis_result"], height=200, key=f"analysis_{task_counter}")
    
    with col2:
        with st.expander("📥 Download Result"):
            st.text_area("Download Output", result["download_result"], height=200, key=f"download_{task_counter}")
    
    with col3:
        with st.expander("🔧 Repository Result"):
            st.text_area("Repository Output", result.get("repo_result", ""), height=200, key=f"repo_{task_counter}")
    
    # 提供新任务按钮
    if st.button("🔄 Start New Task", type="primary", use_container_width=True):
        st.session_state.show_results = False
        st.session_state.last_result = None
        st.session_state.last_error = None
        st.session_state.task_counter += 1
        st.rerun()


def progress_display_component():
    """
    进度显示组件 / Progress display component
    
    Returns:
        (progress_bar, status_text) / (进度条, 状态文本)
    """
    # 显示处理进度标题
    st.markdown("### 📊 Processing Progress")
    
    # 创建进度容器
    progress_container = st.container()
    
    with progress_container:
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        
        # 创建进度条和状态文本
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return progress_bar, status_text


def footer_component():
    """
    页脚组件 / Footer component
    """
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🧬 <strong>Paper to Code v3.0</strong> | Next-Gen AI Research Platform | 
        <a href="https://github.com/your-repo" target="_blank" style="color: var(--neon-blue);">GitHub</a></p>
        <p>⚡ Powered by Neural Networks • Quantum Computing • Multi-Agent AI • Advanced NLP</p>
        <p><small>💡 Tip: Experience the future of research automation - keep this tab active for optimal performance</small></p>
    </div>
    """, unsafe_allow_html=True)


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小 / Format file size
    
    Args:
        size_bytes: 字节大小 / Size in bytes
        
    Returns:
        格式化的文件大小 / Formatted file size
    """
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}" 