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
import uuid

# 导入必要的模块
from mcp_agent.app import MCPApp
from workflows.initial_workflows import (
    execute_multi_agent_research_pipeline,
    run_paper_analyzer,
    run_paper_downloader
)


async def process_input_async(input_source: str, input_type: str, progress_callback=None) -> Dict[str, Any]:
    """
    异步处理输入 / Process input asynchronously
    
    Args:
        input_source: 输入源 / Input source
        input_type: 输入类型 / Input type
        progress_callback: 进度回调函数 / Progress callback function
        
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
            
            # 初始化进度 / Initialize Progress
            if progress_callback:
                progress_callback(5, "🚀 Initializing AI research engine...")
            
            # 调用完整的多智能体研究流水线 / Call complete multi-agent research pipeline
            # 现在execute_multi_agent_research_pipeline包含了所有步骤：分析、下载、代码准备和实现
            repo_result = await execute_multi_agent_research_pipeline(input_source, logger, progress_callback)
            
            return {
                "analysis_result": "Integrated into complete workflow",
                "download_result": "Integrated into complete workflow", 
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
    
    # 保存当前的 Streamlit 上下文
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        from streamlit.runtime.scriptrunner.script_run_context import SCRIPT_RUN_CONTEXT_ATTR_NAME
        import threading
        
        current_ctx = get_script_run_ctx()
        context_available = True
    except ImportError:
        # 如果无法导入 Streamlit 上下文相关模块，使用备用方法
        current_ctx = None
        context_available = False
    
    def run_in_new_loop():
        """在新的事件循环中运行协程 / Run coroutine in new event loop"""
        # 在新线程中设置 Streamlit 上下文（如果可用）
        if context_available and current_ctx:
            try:
                import threading
                setattr(threading.current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME, current_ctx)
            except Exception:
                pass  # 忽略上下文设置错误
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
            # 清理线程上下文（如果可用）
            if context_available:
                try:
                    import threading
                    if hasattr(threading.current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME):
                        delattr(threading.current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME)
                except Exception:
                    pass  # 忽略清理错误
    
    # 使用线程池来运行异步任务，避免事件循环冲突
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()
    except Exception as e:
        # 如果线程池执行失败，尝试直接运行
        st.error(f"Async task execution error: {e}")
        try:
            # 备用方法：直接在当前线程中运行
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
                return result
            finally:
                loop.close()
        except Exception as backup_error:
            st.error(f"Backup async execution also failed: {backup_error}")
            raise backup_error


def run_async_task_simple(coro):
    """
    简单的异步任务运行器，避免多线程问题 / Simple async task runner avoiding threading issues
    
    Args:
        coro: 协程对象 / Coroutine object
        
    Returns:
        任务结果 / Task result
    """
    # 应用 nest_asyncio 来支持嵌套的事件循环
    nest_asyncio.apply()
    
    try:
        # 尝试在当前事件循环中运行
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果当前循环正在运行，创建新循环
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=300)  # 5分钟超时
        else:
            # 直接在当前循环中运行
            return loop.run_until_complete(coro)
    except:
        # 创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def handle_processing_workflow(input_source: str, input_type: str) -> Dict[str, Any]:
    """
    处理工作流的主要处理函数 / Main processing function for workflow
    
    Args:
        input_source: 输入源 / Input source
        input_type: 输入类型 / Input type
        
    Returns:
        处理结果 / Processing result
    """
    from .components import (
        enhanced_progress_display_component, 
        update_step_indicator, 
        display_status,
        add_communication_message,
        set_communication_stage,
        create_persistent_processing_state,
        update_persistent_processing_state
    )
    
    # 创建任务ID和初始化持久化状态
    task_id = str(uuid.uuid4())[:8]
    create_persistent_processing_state()
    
    # 显示增强版进度组件
    progress_bar, status_text, step_indicators, workflow_steps, communication_container = enhanced_progress_display_component()
    
    # 步骤映射：将进度百分比映射到步骤索引
    step_mapping = {
        5: 0,   # Initialize
        10: 1,  # Analyze
        25: 2,  # Download
        45: 3,  # References
        50: 4,  # Plan
        60: 5,  # Repos
        70: 6,  # Index
        85: 7,  # Implement
        100: 7  # Complete
    }
    
    current_step = 0
    
    # 定义增强版进度回调函数 / Define enhanced progress callback function
    def update_progress(progress: int, message: str):
        nonlocal current_step
        
        # 更新进度条
        progress_bar.progress(progress)
        status_text.markdown(f"**{message}**")
        
        # 确定当前步骤
        new_step = step_mapping.get(progress, current_step)
        if new_step != current_step:
            # 完成上一个阶段
            if current_step >= 0:
                set_communication_stage(current_step, 'completed')
                add_communication_message(current_step, 'system_info', f"Stage completed: {workflow_steps[current_step][1]}")
            
            current_step = new_step
            update_step_indicator(step_indicators, workflow_steps, current_step, "active")
            
            # 激活新阶段的通信窗口
            set_communication_stage(current_step, 'active')
            add_communication_message(current_step, 'system_info', f"Stage started: {workflow_steps[current_step][1]}")
        
        # 更新持久化状态
        update_persistent_processing_state(task_id, 'running', progress, current_step, message)
        
        # 添加进度消息到当前通信窗口
        if current_step >= 0:
            add_communication_message(current_step, 'system_info', message)
        
        time.sleep(0.3)  # 短暂停顿以便用户看到进度变化
    
    # 自定义进度回调，包含Agent-LLM通信模拟
    def enhanced_update_progress(progress: int, message: str):
        update_progress(progress, message)
        
        # 模拟Agent与LLM的对话
        stage_names = {
            0: "Initialization",
            1: "Analysis", 
            2: "Download",
            3: "References",
            4: "Planning", 
            5: "Repositories",
            6: "Indexing",
            7: "Implementation"
        }
        
        if current_step in stage_names:
            stage_name = stage_names[current_step]
            
            # 模拟Agent请求
            agent_requests = {
                0: "Initializing system components and loading AI models...",
                1: "Analyzing paper structure, extracting key concepts and methodologies...",
                2: "Processing document downloads and preparing file structures...", 
                3: "Examining references and building knowledge graph...",
                4: "Creating implementation plan based on paper findings...",
                5: "Searching and downloading relevant code repositories...",
                6: "Building comprehensive codebase index and relationships...",
                7: "Implementing code based on analysis and planning..."
            }
            
            llm_responses = {
                0: "System ready. Neural networks loaded. Proceeding with task initialization.",
                1: "Paper analysis complete. Identified key algorithms and implementation patterns.",
                2: "Document processing successful. File structure organized and ready.",
                3: "Reference analysis complete. Knowledge connections established.",
                4: "Implementation plan generated with detailed step-by-step approach.",
                5: "Repository search complete. Relevant codebases identified and downloaded.",
                6: "Codebase indexing complete. Dependencies and relationships mapped.",
                7: "Code implementation in progress. Following best practices and patterns."
            }
            
            if current_step in agent_requests:
                add_communication_message(current_step, 'agent_request', agent_requests[current_step])
                time.sleep(0.5)  # Small delay for realism
                add_communication_message(current_step, 'llm_response', llm_responses[current_step])
    
    # 步骤1: 初始化 / Step 1: Initialization
    enhanced_update_progress(5, "🚀 Initializing AI research engine and loading models...")
    update_step_indicator(step_indicators, workflow_steps, 0, "active")
    
    # 开始异步处理，使用增强的进度回调
    with st.spinner("🔄 Processing workflow stages..."):
        try:
            # 首先尝试使用简单的异步处理方法
            result = run_async_task_simple(process_input_async(input_source, input_type, enhanced_update_progress))
        except Exception as e:
            st.warning(f"Primary async method failed: {e}")
            # 备用方法：使用原始的线程池方法
            try:
                result = run_async_task(process_input_async(input_source, input_type, enhanced_update_progress))
            except Exception as backup_error:
                st.error(f"Both async methods failed. Error: {backup_error}")
                # 更新持久化状态为错误
                update_persistent_processing_state(task_id, 'error', 0, current_step, f"Processing failed: {backup_error}")
                return {
                    "status": "error",
                    "error": str(backup_error),
                    "traceback": traceback.format_exc()
                }
    
    # 根据结果更新最终状态
    if result["status"] == "success":
        # 完成所有步骤
        enhanced_update_progress(100, "✅ All processing stages completed successfully!")
        update_step_indicator(step_indicators, workflow_steps, len(workflow_steps), "completed")
        
        # 完成最后一个阶段
        if current_step >= 0:
            set_communication_stage(current_step, 'completed')
            add_communication_message(current_step, 'system_info', "🎉 All stages completed successfully!")
            add_communication_message(current_step, 'llm_response', "Task execution completed. All objectives achieved successfully.")
        
        # 更新持久化状态为完成
        update_persistent_processing_state(task_id, 'completed', 100, current_step, "Processing completed successfully!")
        
        # 显示成功信息
        st.balloons()  # 添加庆祝动画
        display_status("🎉 Workflow completed! Your research paper has been successfully processed and code has been generated.", "success")
        
    else:
        # 处理失败
        enhanced_update_progress(0, "❌ Processing failed - see error details below")
        update_step_indicator(step_indicators, workflow_steps, current_step, "error")
        
        # 在当前阶段添加错误信息
        if current_step >= 0:
            set_communication_stage(current_step, 'error')
            add_communication_message(current_step, 'system_info', f"❌ Error occurred: {result.get('error', 'Unknown error')}")
        
        # 更新持久化状态为错误
        update_persistent_processing_state(task_id, 'error', 0, current_step, f"Processing failed: {result.get('error', 'Unknown error')}")
        
        display_status(f"❌ Processing encountered an error: {result.get('error', 'Unknown error')}", "error")
    
    # 等待一下让用户看到完成状态
    time.sleep(2.5)
    
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
    
    # 新增：通信窗口相关状态
    if 'stage_communications' not in st.session_state:
        st.session_state.stage_communications = {}
    if 'current_communication_stage' not in st.session_state:
        st.session_state.current_communication_stage = -1
    
    # 新增：持久化处理状态
    if 'persistent_task_id' not in st.session_state:
        st.session_state.persistent_task_id = None
    if 'persistent_task_status' not in st.session_state:
        st.session_state.persistent_task_status = 'idle'
    if 'persistent_task_progress' not in st.session_state:
        st.session_state.persistent_task_progress = 0
    if 'persistent_task_stage' not in st.session_state:
        st.session_state.persistent_task_stage = -1
    if 'persistent_task_message' not in st.session_state:
        st.session_state.persistent_task_message = ""
    if 'task_start_time' not in st.session_state:
        st.session_state.task_start_time = None 