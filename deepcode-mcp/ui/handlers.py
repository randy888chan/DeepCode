"""
Streamlit 事件处理模块 / Streamlit Event Handlers Module

包含所有事件处理和业务逻辑
Contains all event handling and business logic
"""

import asyncio
import time
import os
import sys
import traceback
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st
import nest_asyncio
import concurrent.futures

# 导入必要的模块
from mcp_agent.app import MCPApp
from workflows.initial_workflows import (
    paper_code_preparation,
    run_paper_analyzer,
    run_paper_downloader
)


async def process_input_async(input_source: str, input_type: str) -> Dict[str, Any]:
    """
    异步处理输入 / Process input asynchronously
    
    Args:
        input_source: 输入源 / Input source
        input_type: 输入类型 / Input type
        
    Returns:
        处理结果 / Processing result
    """
    try:
        # 在同一个异步上下文中创建和使用 MCP 应用
        app = MCPApp(name="paper_to_code")
        
        async with app.run() as agent_app:
            logger = agent_app.logger
            context = agent_app.context
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            
            # 处理输入源路径
            if input_source.startswith("file://"):
                file_path = input_source[7:]
                if os.name == 'nt' and file_path.startswith('/'):
                    file_path = file_path.lstrip('/')
                input_source = file_path
            
            # 步骤1: 论文分析
            analysis_result = await run_paper_analyzer(input_source, logger)
            
            # 添加5秒停顿
            await asyncio.sleep(5)
            
            # 步骤2: 下载处理
            download_result = await run_paper_downloader(analysis_result, logger)
            
            # 步骤3: 代码准备
            repo_result = await paper_code_preparation(download_result, logger)
            
            return {
                "analysis_result": analysis_result,
                "download_result": download_result,
                "repo_result": repo_result,
                "status": "success"
            }
            
    except Exception as e:
        error_msg = str(e)
        traceback_msg = traceback.format_exc()
        
        return {
            "error": error_msg,
            "traceback": traceback_msg,
            "status": "error"
        }


def run_async_task(coro):
    """
    运行异步任务的辅助函数 / Helper function to run async tasks
    
    Args:
        coro: 协程对象 / Coroutine object
        
    Returns:
        任务结果 / Task result
    """
    # 应用 nest_asyncio 来支持嵌套的事件循环
    nest_asyncio.apply()
    
    def run_in_new_loop():
        """在新的事件循环中运行协程 / Run coroutine in new event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    
    # 使用线程池来运行异步任务，避免事件循环冲突
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_new_loop)
        return future.result()


def handle_processing_workflow(input_source: str, input_type: str) -> Dict[str, Any]:
    """
    处理工作流的主要处理函数 / Main processing function for workflow
    
    Args:
        input_source: 输入源 / Input source
        input_type: 输入类型 / Input type
        
    Returns:
        处理结果 / Processing result
    """
    from .components import progress_display_component, display_status
    
    # 显示进度组件
    progress_bar, status_text = progress_display_component()
    
    # 步骤1: 开始处理
    status_text.markdown("**🚀 Initializing AI engine...**")
    progress_bar.progress(5)
    time.sleep(0.5)
    
    # 步骤2: 分析论文
    status_text.markdown("**📊 Step 1/3: Analyzing paper content...**")
    progress_bar.progress(15)
    
    # 开始异步处理
    with st.spinner("Processing..."):
        result = run_async_task(process_input_async(input_source, input_type))
    
    # 根据结果模拟进度更新
    if result["status"] == "success":
        # 步骤3: 下载处理
        status_text.markdown("**📥 Step 2/3: Processing downloads...**")
        progress_bar.progress(50)
        time.sleep(0.5)
        
        # 步骤4: 代码准备
        status_text.markdown("**🔧 Step 3/3: Preparing code repository...**")
        progress_bar.progress(80)
        time.sleep(0.5)
        
        # 完成
        progress_bar.progress(100)
        status_text.markdown("**✅ Processing completed successfully!**")
    else:
        status_text.markdown("**❌ Processing failed**")
    
    # 等待一下让用户看到完成状态
    time.sleep(1.5)
    
    return result


def update_session_state_with_result(result: Dict[str, Any], input_type: str):
    """
    用结果更新session state / Update session state with result
    
    Args:
        result: 处理结果 / Processing result
        input_type: 输入类型 / Input type
    """
    if result["status"] == "success":
        # 保存结果到session state
        st.session_state.last_result = result
        st.session_state.show_results = True
        
        # 保存到历史记录
        st.session_state.results.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_type": input_type,
            "status": "success",
            "result": result
        })
    else:
        # 保存错误信息到session state用于显示
        st.session_state.last_error = result.get("error", "Unknown error")
        
        # 保存错误到历史记录
        st.session_state.results.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_type": input_type,
            "status": "error",
            "error": result.get("error", "Unknown error")
        })
    
    # 限制历史记录最多保存50条
    if len(st.session_state.results) > 50:
        st.session_state.results = st.session_state.results[-50:]


def cleanup_temp_file(input_source: str, input_type: str):
    """
    清理临时文件 / Cleanup temporary file
    
    Args:
        input_source: 输入源 / Input source
        input_type: 输入类型 / Input type
    """
    if input_type == "file" and input_source and os.path.exists(input_source):
        try:
            os.unlink(input_source)
        except:
            pass


def handle_start_processing_button(input_source: str, input_type: str):
    """
    处理开始处理按钮点击 / Handle start processing button click
    
    Args:
        input_source: 输入源 / Input source
        input_type: 输入类型 / Input type
    """
    from .components import display_status
    
    st.session_state.processing = True
    
    # 处理工作流
    result = handle_processing_workflow(input_source, input_type)
    
    # 显示结果状态
    if result["status"] == "success":
        display_status("All operations completed successfully! 🎉", "success")
    else:
        display_status(f"Error during processing", "error")
    
    # 更新session state
    update_session_state_with_result(result, input_type)
    
    # 处理完成后重置状态
    st.session_state.processing = False
    
    # 清理临时文件
    cleanup_temp_file(input_source, input_type)
    
    # 重新运行以显示结果或错误
    st.rerun()


def handle_error_display():
    """
    处理错误显示 / Handle error display
    """
    if hasattr(st.session_state, 'last_error') and st.session_state.last_error:
        st.error(f"❌ Error: {st.session_state.last_error}")
        if st.button("🔄 Try Again", type="secondary", use_container_width=True):
            st.session_state.last_error = None
            st.session_state.task_counter += 1
            st.rerun()


def initialize_session_state():
    """
    初始化session state / Initialize session state
    """
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'task_counter' not in st.session_state:
        st.session_state.task_counter = 0
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None 