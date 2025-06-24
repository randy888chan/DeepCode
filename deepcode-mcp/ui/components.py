"""
Streamlit UI组件模块 / Streamlit UI Components Module

包含所有可复用的UI组件
Contains all reusable UI components
"""

import streamlit as st
import sys
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


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


def system_status_component():
    """
    系统状态检查组件 / System status check component
    """
    st.markdown("### 🔧 System Status & Diagnostics")
    
    # 基本系统信息
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Environment")
        st.info(f"**Python:** {sys.version.split()[0]}")
        st.info(f"**Platform:** {sys.platform}")
        
        # 检查关键模块
        modules_to_check = [
            ("streamlit", "Streamlit UI Framework"),
            ("asyncio", "Async Processing"),
            ("nest_asyncio", "Nested Event Loops"),
            ("concurrent.futures", "Threading Support"),
        ]
        
        st.markdown("#### 📦 Module Status")
        for module_name, description in modules_to_check:
            try:
                __import__(module_name)
                st.success(f"✅ {description}")
            except ImportError:
                st.error(f"❌ {description} - Missing")
    
    with col2:
        st.markdown("#### ⚙️ Threading & Context")
        
        # 检查 Streamlit 上下文
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()
            if ctx:
                st.success("✅ Streamlit Context Available")
            else:
                st.warning("⚠️ Streamlit Context Not Found")
        except Exception as e:
            st.error(f"❌ Context Check Failed: {e}")
        
        # 检查事件循环
        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    st.info("🔄 Event Loop Running")
                else:
                    st.info("⏸️ Event Loop Not Running")
            except RuntimeError:
                st.info("🆕 No Event Loop (Normal)")
        except Exception as e:
            st.error(f"❌ Event Loop Check Failed: {e}")


def error_troubleshooting_component():
    """
    错误诊断组件 / Error troubleshooting component
    """
    with st.expander("🛠️ Troubleshooting Tips", expanded=False):
        st.markdown("""
        ### Common Issues & Solutions / 常见问题和解决方案
        
        #### 1. ScriptRunContext Warnings / ScriptRunContext 警告
        - **What it means:** Threading context warnings in Streamlit
        - **Solution:** These warnings are usually safe to ignore
        - **Prevention:** Restart the application if persistent
        
        #### 2. Async Processing Errors / 异步处理错误
        - **Symptoms:** "Event loop" or "Thread" errors
        - **Solution:** The app uses multiple fallback methods
        - **Action:** Try refreshing the page or restarting
        
        #### 3. File Upload Issues / 文件上传问题
        - **Check:** File size < 200MB
        - **Formats:** PDF, DOCX, TXT, HTML, MD
        - **Action:** Try a different file format
        
        #### 4. Processing Timeout / 处理超时
        - **Normal:** Large papers may take 5-10 minutes
        - **Action:** Wait patiently, check progress indicators
        - **Limit:** 5-minute maximum processing time
        
        #### 5. Memory Issues / 内存问题
        - **Symptoms:** "Out of memory" errors
        - **Solution:** Close other applications
        - **Action:** Try smaller/simpler papers first
        """)
        
        if st.button("🔄 Reset Application State"):
            # 清理所有session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Application state reset! Please refresh the page.")
            st.rerun()


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
        
        # 添加系统状态检查
        with st.expander("🔧 System Status"):
            system_status_component()
        
        # 添加错误诊断
        error_troubleshooting_component()
        
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
    st.markdown("### 📋 Processing Results")
    
    # 显示整体状态
    if result.get("status") == "success":
        st.success("🎉 **All workflows completed successfully!**")
    else:
        st.error("❌ **Processing encountered errors**")
    
    # 创建标签页来组织不同阶段的结果
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analysis Phase", 
        "📥 Download Phase", 
        "🔧 Implementation Phase",
        "📁 Generated Files"
    ])
    
    with tab1:
        st.markdown("#### 📊 Paper Analysis Results")
        with st.expander("Analysis Output Details", expanded=True):
            analysis_result = result.get("analysis_result", "No analysis result available")
            try:
                # 尝试解析JSON结果进行格式化显示
                if analysis_result.strip().startswith('{'):
                    parsed_analysis = json.loads(analysis_result)
                    st.json(parsed_analysis)
                else:
                    st.text_area("Raw Analysis Output", analysis_result, height=300, key=f"analysis_{task_counter}")
            except:
                st.text_area("Analysis Output", analysis_result, height=300, key=f"analysis_{task_counter}")
    
    with tab2:
        st.markdown("#### 📥 Download & Preparation Results")
        with st.expander("Download Process Details", expanded=True):
            download_result = result.get("download_result", "No download result available")
            st.text_area("Download Output", download_result, height=300, key=f"download_{task_counter}")
    
            # 尝试提取文件路径信息
            if "paper_dir" in download_result or "path" in download_result.lower():
                st.info("💡 **Tip:** Look for file paths in the output above to locate generated files")
    
    with tab3:
        st.markdown("#### 🔧 Code Implementation Results")
        repo_result = result.get("repo_result", "No implementation result available")
        
        # 分析实现结果以提取关键信息
        if "successfully" in repo_result.lower():
            st.success("✅ Code implementation completed successfully!")
        elif "failed" in repo_result.lower():
            st.warning("⚠️ Code implementation encountered issues")
        else:
            st.info("ℹ️ Code implementation status unclear")
        
        with st.expander("Implementation Details", expanded=True):
            st.text_area("Repository & Code Generation Output", repo_result, height=300, key=f"repo_{task_counter}")
        
        # 尝试提取生成的代码目录信息
        if "Code generated in:" in repo_result:
            code_dir = repo_result.split("Code generated in:")[-1].strip()
            st.markdown(f"**📁 Generated Code Directory:** `{code_dir}`")
        
        # 显示工作流阶段详情
        st.markdown("#### 🔄 Workflow Stages Completed")
        stages = [
            ("📄 Document Processing", "✅"),
            ("🔍 Reference Analysis", "✅"),
            ("📋 Plan Generation", "✅"),
            ("📦 Repository Download", "✅"),
            ("🗂️ Codebase Indexing", "✅" if "indexing" in repo_result.lower() else "⚠️"),
            ("⚙️ Code Implementation", "✅" if "successfully" in repo_result.lower() else "⚠️")
        ]
        
        for stage_name, status in stages:
            st.markdown(f"- {stage_name}: {status}")
    
    with tab4:
        st.markdown("#### 📁 Generated Files & Reports")
        
        # 尝试从结果中提取文件路径
        all_results = f"{result.get('download_result', '')} {result.get('repo_result', '')}"
        
        # 查找可能的文件路径模式
        import re
        file_patterns = [
            r'([^\s]+\.txt)',
            r'([^\s]+\.json)',
            r'([^\s]+\.py)',
            r'([^\s]+\.md)',
            r'paper_dir[:\s]+([^\s]+)',
            r'saved to ([^\s]+)',
            r'generated in[:\s]+([^\s]+)'
        ]
        
        found_files = set()
        for pattern in file_patterns:
            matches = re.findall(pattern, all_results, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    found_files.update(match)
                else:
                    found_files.add(match)
        
        if found_files:
            st.markdown("**📄 Detected Generated Files:**")
            for file_path in sorted(found_files):
                if file_path and len(file_path) > 3:  # 过滤掉太短的匹配
                    st.markdown(f"- `{file_path}`")
        else:
            st.info("No specific file paths detected in the output. Check the detailed results above for file locations.")
    
        # 提供查看原始结果的选项
        with st.expander("View Raw Processing Results"):
            st.json({
                "analysis_result": result.get("analysis_result", ""),
                "download_result": result.get("download_result", ""),
                "repo_result": result.get("repo_result", ""),
                "status": result.get("status", "unknown")
            })
    
    # 操作按钮
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Process New Paper", type="primary", use_container_width=True):
            st.session_state.show_results = False
            st.session_state.last_result = None
            st.session_state.last_error = None
            st.session_state.task_counter += 1
            st.rerun()
    
    with col2:
        if st.button("💾 Export Results", type="secondary", use_container_width=True):
            # 创建结果导出
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "processing_results": result,
                "status": result.get("status", "unknown")
            }
            st.download_button(
                label="📄 Download Results JSON",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"paper_processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )


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
        # 添加自定义CSS样式
        st.markdown("""
        <style>
        .progress-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .progress-steps {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .progress-step {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 8px 12px;
            margin: 2px;
            color: white;
            font-size: 0.8rem;
            font-weight: 500;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }
        .progress-step.active {
            background: rgba(255,255,255,0.3);
            border-color: #00ff88;
            box-shadow: 0 0 15px rgba(0,255,136,0.3);
        }
        .progress-step.completed {
            background: rgba(0,255,136,0.2);
            border-color: #00ff88;
        }
        .status-text {
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            margin: 10px 0;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        
        # 创建步骤指示器
        st.markdown("""
        <div class="progress-steps">
            <div class="progress-step" id="step-init">🚀 Initialize</div>
            <div class="progress-step" id="step-analyze">📊 Analyze</div>
            <div class="progress-step" id="step-download">📥 Download</div>
            <div class="progress-step" id="step-references">🔍 References</div>
            <div class="progress-step" id="step-plan">📋 Plan</div>
            <div class="progress-step" id="step-repos">📦 Repos</div>
            <div class="progress-step" id="step-index">🗂️ Index</div>
            <div class="progress-step" id="step-implement">⚙️ Implement</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 创建进度条和状态文本
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return progress_bar, status_text


def enhanced_progress_display_component():
    """
    增强版进度显示组件 / Enhanced progress display component
    
    Returns:
        (progress_bar, status_text, step_indicators, workflow_steps, communication_container) / (进度条, 状态文本, 步骤指示器, 工作流步骤, 通信容器)
    """
    # 显示处理进度标题
    st.markdown("### 🚀 AI Research Engine - Processing Workflow")
    
    # 创建进度容器
    progress_container = st.container()
    
    with progress_container:
        # 工作流步骤定义
        workflow_steps = [
            ("🚀", "Initialize", "Setting up AI engine"),
            ("📊", "Analyze", "Analyzing paper content"),
            ("📥", "Download", "Processing document"),
            ("🔍", "References", "Analyzing references"),
            ("📋", "Plan", "Generating code plan"),
            ("📦", "Repos", "Downloading repositories"),
            ("🗂️", "Index", "Building code index"),
            ("⚙️", "Implement", "Implementing code")
        ]
        
        # 创建步骤指示器容器
        step_container = st.container()
        
        # 显示步骤网格
        cols = st.columns(len(workflow_steps))
        step_indicators = []
        
        for i, (icon, title, desc) in enumerate(workflow_steps):
            with cols[i]:
                step_placeholder = st.empty()
                step_indicators.append(step_placeholder)
                step_placeholder.markdown(f"""
                <div style="
                    text-align: center;
                    padding: 10px;
                    border-radius: 10px;
                    background: rgba(255,255,255,0.05);
                    margin: 5px 0;
                    border: 2px solid transparent;
                ">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div style="font-size: 0.8rem; font-weight: 600;">{title}</div>
                    <div style="font-size: 0.6rem; color: #888;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # 创建主进度条
        st.markdown("#### Overall Progress")
        progress_bar = st.progress(0)
        
        # 状态文本显示
        status_text = st.empty()
        
        # 实时信息显示
        info_text = st.empty()
    
    # 创建Agent与LLM通信窗口容器
    communication_container = create_communication_windows_container(workflow_steps)
    
    return progress_bar, status_text, step_indicators, workflow_steps, communication_container


def create_communication_windows_container(workflow_steps):
    """
    创建Agent与LLM通信窗口容器 / Create Agent-LLM communication windows container
    
    Args:
        workflow_steps: 工作流步骤定义 / Workflow steps definition
        
    Returns:
        communication_container: 通信容器 / Communication container
    """
    st.markdown("---")
    st.markdown("### 🤖 Agent & LLM Communication")
    
    # 初始化session state中的通信日志
    if 'stage_communications' not in st.session_state:
        st.session_state.stage_communications = {}
    
    # 确保所有阶段都已初始化，并处理workflow_steps长度变化的情况
    for i, (icon, title, desc) in enumerate(workflow_steps):
        if i not in st.session_state.stage_communications:
            st.session_state.stage_communications[i] = {
                'title': f"{icon} {title}",
                'messages': [],
                'is_active': False,
                'is_completed': False
            }
    
    if 'current_communication_stage' not in st.session_state:
        st.session_state.current_communication_stage = -1
    
    # 创建通信窗口容器
    communication_container = st.container()
    
    with communication_container:
        # 为每个阶段创建可折叠的通信窗口
        for stage_id in range(len(workflow_steps)):
            # 安全地获取阶段信息，如果不存在则创建
            if stage_id not in st.session_state.stage_communications:
                icon, title, desc = workflow_steps[stage_id]
                st.session_state.stage_communications[stage_id] = {
                    'title': f"{icon} {title}",
                    'messages': [],
                    'is_active': False,
                    'is_completed': False
                }
            
            stage_info = st.session_state.stage_communications[stage_id]
            
            # 确定窗口状态
            is_current = stage_id == st.session_state.current_communication_stage
            is_completed = stage_info['is_completed']
            has_messages = len(stage_info['messages']) > 0
            
            # 设置展开状态：当前阶段默认展开，已完成的可以展开查看
            expanded = is_current or (not is_current and has_messages)
            
            # 窗口标题和状态指示
            if is_current:
                status_indicator = "🔴 ACTIVE"
                title_style = "🔥"
            elif is_completed:
                status_indicator = "✅ COMPLETED"
                title_style = "✨"
            elif has_messages:
                status_indicator = "⏸️ PAUSED"
                title_style = "📋"
            else:
                status_indicator = "⏳ PENDING"
                title_style = "⭕"
            
            window_title = f"{title_style} {stage_info['title']} - {status_indicator}"
            
            # 创建可折叠窗口
            with st.expander(window_title, expanded=expanded):
                stage_container = st.container()
                
                # 创建消息显示区域
                if has_messages:
                    # 创建一个固定高度的滚动区域
                    message_container = st.container()
                    with message_container:
                        # 显示消息历史
                        for msg in stage_info['messages']:
                            timestamp = msg.get('timestamp', '')
                            msg_type = msg.get('type', 'info')
                            content = msg.get('content', '')
                            
                            # 根据消息类型选择样式
                            if msg_type == 'agent_request':
                                st.markdown(f"""
                                <div style="background: rgba(0,123,255,0.1); border-left: 4px solid #007bff; padding: 10px; margin: 5px 0; border-radius: 5px;">
                                    <small style="color: #888;">🤖 Agent Request [{timestamp}]</small><br>
                                    <span style="color: #007bff;">{content}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            elif msg_type == 'llm_response':
                                st.markdown(f"""
                                <div style="background: rgba(40,167,69,0.1); border-left: 4px solid #28a745; padding: 10px; margin: 5px 0; border-radius: 5px;">
                                    <small style="color: #888;">🧠 LLM Response [{timestamp}]</small><br>
                                    <span style="color: #28a745;">{content}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            elif msg_type == 'system_info':
                                st.markdown(f"""
                                <div style="background: rgba(255,193,7,0.1); border-left: 4px solid #ffc107; padding: 10px; margin: 5px 0; border-radius: 5px;">
                                    <small style="color: #888;">⚙️ System Info [{timestamp}]</small><br>
                                    <span style="color: #ffc107;">{content}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="background: rgba(108,117,125,0.1); border-left: 4px solid #6c757d; padding: 10px; margin: 5px 0; border-radius: 5px;">
                                    <small style="color: #888;">ℹ️ Info [{timestamp}]</small><br>
                                    <span style="color: #6c757d;">{content}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # 如果是当前活跃阶段，显示实时状态
                        if is_current:
                            st.markdown("""
                            <div style="background: rgba(255,0,0,0.1); border: 2px dashed #ff0000; padding: 10px; margin: 10px 0; border-radius: 5px; text-align: center;">
                                <span style="color: #ff0000; font-weight: bold;">🔴 Live Communication - Agent & LLM are actively working...</span>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    # 空状态显示
                    if is_current:
                        st.info("🚀 Stage starting... Communication will appear here.")
                    else:
                        st.info("📭 No communication recorded for this stage yet.")
    
    return communication_container


def add_communication_message(stage_id: int, msg_type: str, content: str):
    """
    添加通信消息到指定阶段 / Add communication message to specified stage
    
    Args:
        stage_id: 阶段ID / Stage ID
        msg_type: 消息类型 ('agent_request', 'llm_response', 'system_info') / Message type
        content: 消息内容 / Message content
    """
    if 'stage_communications' not in st.session_state:
        st.session_state.stage_communications = {}
    
    from datetime import datetime
    
    # 确保阶段存在，如果不存在则创建基本结构
    if stage_id not in st.session_state.stage_communications:
        st.session_state.stage_communications[stage_id] = {
            'title': f"Stage {stage_id}",
            'messages': [],
            'is_active': False,
            'is_completed': False
        }
    
    message = {
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'type': msg_type,
        'content': content
    }
    st.session_state.stage_communications[stage_id]['messages'].append(message)
    
    # 限制每个阶段最多保存50条消息
    if len(st.session_state.stage_communications[stage_id]['messages']) > 50:
        st.session_state.stage_communications[stage_id]['messages'] = \
            st.session_state.stage_communications[stage_id]['messages'][-50:]


def set_communication_stage(stage_id: int, status: str = 'active'):
    """
    设置当前通信阶段状态 / Set current communication stage status
    
    Args:
        stage_id: 阶段ID / Stage ID
        status: 状态 ('active', 'completed', 'error') / Status
    """
    if 'stage_communications' not in st.session_state:
        st.session_state.stage_communications = {}
    
    # 确保阶段存在，如果不存在则创建基本结构
    if stage_id not in st.session_state.stage_communications:
        st.session_state.stage_communications[stage_id] = {
            'title': f"Stage {stage_id}",
            'messages': [],
            'is_active': False,
            'is_completed': False
        }
    
    # 更新当前阶段
    if status == 'active':
        st.session_state.current_communication_stage = stage_id
        st.session_state.stage_communications[stage_id]['is_active'] = True
    elif status == 'completed':
        st.session_state.stage_communications[stage_id]['is_active'] = False
        st.session_state.stage_communications[stage_id]['is_completed'] = True


def create_persistent_processing_state():
    """
    创建持久化处理状态 / Create persistent processing state
    
    This function ensures that processing continues even if the UI is refreshed
    """
    if 'persistent_task_id' not in st.session_state:
        st.session_state.persistent_task_id = None
    
    if 'persistent_task_status' not in st.session_state:
        st.session_state.persistent_task_status = 'idle'  # idle, running, completed, error
    
    if 'persistent_task_progress' not in st.session_state:
        st.session_state.persistent_task_progress = 0
    
    if 'persistent_task_stage' not in st.session_state:
        st.session_state.persistent_task_stage = -1
    
    if 'persistent_task_message' not in st.session_state:
        st.session_state.persistent_task_message = ""
    
    if 'task_start_time' not in st.session_state:
        st.session_state.task_start_time = None


def update_persistent_processing_state(task_id: str, status: str, progress: int, stage: int, message: str):
    """
    更新持久化处理状态 / Update persistent processing state
    
    Args:
        task_id: 任务ID / Task ID
        status: 状态 / Status
        progress: 进度 / Progress
        stage: 阶段 / Stage
        message: 消息 / Message
    """
    st.session_state.persistent_task_id = task_id
    st.session_state.persistent_task_status = status
    st.session_state.persistent_task_progress = progress
    st.session_state.persistent_task_stage = stage
    st.session_state.persistent_task_message = message
    
    if status == 'running' and st.session_state.task_start_time is None:
        from datetime import datetime
        st.session_state.task_start_time = datetime.now()


def display_refresh_warning():
    """
    显示刷新警告和状态恢复信息 / Display refresh warning and status recovery info
    """
    if st.session_state.persistent_task_status == 'running':
        # 计算运行时间
        if st.session_state.task_start_time:
            from datetime import datetime
            elapsed = datetime.now() - st.session_state.task_start_time
            elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        else:
            elapsed_str = "Unknown"
        
        st.warning(f"""
        🔄 **Task Recovery Mode**  
        A processing task is currently running in the background.  
        - **Task ID:** {st.session_state.persistent_task_id}  
        - **Status:** {st.session_state.persistent_task_status.upper()}  
        - **Progress:** {st.session_state.persistent_task_progress}%  
        - **Current Stage:** {st.session_state.persistent_task_stage + 1}/8  
        - **Elapsed Time:** {elapsed_str}  
        - **Last Message:** {st.session_state.persistent_task_message}
        
        📱 **UI Refresh Safe**: You can refresh this page without affecting the running task.
        """)


def update_step_indicator(step_indicators, workflow_steps, current_step: int, status: str = "active"):
    """
    更新步骤指示器 / Update step indicator
    
    Args:
        step_indicators: 步骤指示器列表 / Step indicator list
        workflow_steps: 工作流步骤定义 / Workflow steps definition
        current_step: 当前步骤索引 / Current step index
        status: 状态 ("active", "completed", "error") / Status
    """
    status_colors = {
        "pending": ("rgba(255,255,255,0.05)", "transparent", "#888"),
        "active": ("rgba(255,215,0,0.2)", "#ffd700", "#fff"),
        "completed": ("rgba(0,255,136,0.2)", "#00ff88", "#fff"),
        "error": ("rgba(255,99,99,0.2)", "#ff6363", "#fff")
    }
    
    for i, (icon, title, desc) in enumerate(workflow_steps):
        if i < current_step:
            bg_color, border_color, text_color = status_colors["completed"]
            display_icon = "✅"
        elif i == current_step:
            bg_color, border_color, text_color = status_colors[status]
            display_icon = icon
        else:
            bg_color, border_color, text_color = status_colors["pending"]
            display_icon = icon
        
        step_indicators[i].markdown(f"""
        <div style="
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            background: {bg_color};
            margin: 5px 0;
            border: 2px solid {border_color};
            color: {text_color};
            transition: all 0.3s ease;
            box-shadow: {f'0 0 15px {border_color}30' if i == current_step else 'none'};
        ">
            <div style="font-size: 1.5rem;">{display_icon}</div>
            <div style="font-size: 0.8rem; font-weight: 600;">{title}</div>
            <div style="font-size: 0.6rem; opacity: 0.8;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


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