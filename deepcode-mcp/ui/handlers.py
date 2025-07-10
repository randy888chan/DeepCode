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
import atexit
import signal
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st
import nest_asyncio
import concurrent.futures

# 导入必要的模块
from mcp_agent.app import MCPApp
from workflows.initial_workflows import (
    execute_multi_agent_research_pipeline,
    run_paper_analyzer,
    run_paper_downloader
)


def _emergency_cleanup():
    """
    应急资源清理函数 / Emergency resource cleanup function
    在程序异常退出时调用 / Called when program exits abnormally
    """
    try:
        cleanup_resources()
    except Exception:
        pass  # 静默处理，避免在退出时抛出新异常


def _signal_handler(signum, frame):
    """
    信号处理器 / Signal handler
    处理程序终止信号 / Handle program termination signals
    """
    try:
        cleanup_resources()
    except Exception:
        pass
    finally:
        # 恢复默认信号处理并重新发送信号
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


# 注册退出清理函数 / Register exit cleanup function
atexit.register(_emergency_cleanup)

# 注册信号处理器 / Register signal handlers
# 在某些环境中（如 Streamlit），信号处理可能受限，需要更加小心
def _safe_register_signal_handlers():
    """安全地注册信号处理器 / Safely register signal handlers"""
    try:
        # 检查是否在主线程中
        import threading
        if threading.current_thread() is not threading.main_thread():
            return  # 信号处理器只能在主线程中注册
        
        # 尝试注册信号处理器
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, _signal_handler)
    except (AttributeError, OSError, ValueError) as e:
        # 某些信号在某些平台上不可用，或者在某些运行环境中被禁用
        # 这在 Streamlit 等 Web 框架中很常见
        pass

# 延迟注册信号处理器，避免在模块导入时出错
try:
    _safe_register_signal_handlers()
except Exception:
    # 如果注册失败，静默忽略，不影响应用启动
    pass


async def process_input_async(input_source: str, input_type: str, enable_indexing: bool = True, progress_callback=None) -> Dict[str, Any]:
    """
    异步处理输入 / Process input asynchronously
    
    Args:
        input_source: 输入源 / Input source
        input_type: 输入类型 / Input type
        enable_indexing: 是否启用索引功能 / Whether to enable indexing
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
            repo_result = await execute_multi_agent_research_pipeline(
                input_source, 
                logger, 
                progress_callback,
                enable_indexing=enable_indexing  # 传递索引控制参数
            )
            
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
        
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            return result
        except Exception as e:
            raise e
        finally:
            # 清理资源
            if loop:
                try:
                    loop.close()
                except Exception:
                    pass
            asyncio.set_event_loop(None)
            
            # 清理线程上下文（如果可用）
            if context_available:
                try:
                    import threading
                    if hasattr(threading.current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME):
                        delattr(threading.current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME)
                except Exception:
                    pass  # 忽略清理错误
            
            # 强制垃圾回收
            import gc
            gc.collect()
    
    # 使用线程池来运行异步任务，避免事件循环冲突
    executor = None
    try:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
                            thread_name_prefix="deepcode_ctx_async"
        )
        future = executor.submit(run_in_new_loop)
        result = future.result(timeout=300)  # 5分钟超时
        return result
    except concurrent.futures.TimeoutError:
        st.error("Processing timeout after 5 minutes. Please try again.")
        raise TimeoutError("Processing timeout")
    except Exception as e:
        # 如果线程池执行失败，尝试直接运行
        st.warning(f"Threaded async execution failed: {e}, trying direct execution...")
        try:
            # 备用方法：直接在当前线程中运行
            loop = None
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(coro)
                return result
            finally:
                if loop:
                    try:
                        loop.close()
                    except Exception:
                        pass
                asyncio.set_event_loop(None)
                import gc
                gc.collect()
        except Exception as backup_error:
            st.error(f"All execution methods failed: {backup_error}")
            raise backup_error
    finally:
        # 确保线程池被正确关闭
        if executor:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
        # 强制垃圾回收
        import gc
        gc.collect()


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
            # 如果当前循环正在运行，使用改进的线程池方法
            import concurrent.futures
            import threading
            import gc
            
            def run_in_thread():
                # 创建新的事件循环并设置为当前线程的循环
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(coro)
                    return result
                except Exception as e:
                    # 确保异常信息被正确传递
                    raise e
                finally:
                    # 确保循环被正确关闭
                    try:
                        new_loop.close()
                    except Exception:
                        pass
                    # 清除当前线程的事件循环引用
                    asyncio.set_event_loop(None)
                    # 强制垃圾回收
                    gc.collect()
            
            # 使用上下文管理器确保线程池被正确关闭
            executor = None
            try:
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="deepcode_async"
                )
                future = executor.submit(run_in_thread)
                result = future.result(timeout=300)  # 5分钟超时
                return result
            except concurrent.futures.TimeoutError:
                st.error("Processing timeout after 5 minutes. Please try again with a smaller file.")
                raise TimeoutError("Processing timeout")
            except Exception as e:
                st.error(f"Async processing error: {e}")
                raise e
            finally:
                # 确保线程池被正确关闭
                if executor:
                    try:
                        executor.shutdown(wait=True, cancel_futures=True)
                    except Exception:
                        pass
                # 强制垃圾回收
                gc.collect()
        else:
            # 直接在当前循环中运行
            return loop.run_until_complete(coro)
    except Exception as e:
        # 最后的备用方法：创建新的事件循环
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            return result
        except Exception as backup_error:
            st.error(f"All async methods failed: {backup_error}")
            raise backup_error
        finally:
            if loop:
                try:
                    loop.close()
                except Exception:
                    pass
            asyncio.set_event_loop(None)
            # 强制垃圾回收
            import gc
            gc.collect()


def handle_processing_workflow(input_source: str, input_type: str, enable_indexing: bool = True) -> Dict[str, Any]:
    """
    处理工作流的主要处理函数 / Main processing function for workflow
    
    Args:
        input_source: 输入源 / Input source
        input_type: 输入类型 / Input type
        enable_indexing: 是否启用索引功能 / Whether to enable indexing
        
    Returns:
        处理结果 / Processing result
    """
    from .components import enhanced_progress_display_component, update_step_indicator, display_status
    
    # 显示增强版进度组件
    progress_bar, status_text, step_indicators, workflow_steps = enhanced_progress_display_component(enable_indexing)
    
    # 步骤映射：将进度百分比映射到步骤索引 - 根据索引开关调整
    if not enable_indexing:
        # 跳过索引相关步骤的进度映射 - 快速模式顺序：Initialize -> Analyze -> Download -> Plan -> Implement
        step_mapping = {
            5: 0,   # Initialize
            10: 1,  # Analyze
            25: 2,  # Download
            40: 3,  # Plan (现在优先于References，40%)
            85: 4,  # Implement (跳过 References, Repos 和 Index)
            100: 4  # Complete
        }
    else:
        # 完整工作流的步骤映射 - 新顺序：Initialize -> Analyze -> Download -> Plan -> References -> Repos -> Index -> Implement
        step_mapping = {
            5: 0,   # Initialize
            10: 1,  # Analyze
            25: 2,  # Download
            40: 3,  # Plan (现在在第4位，40%)
            50: 4,  # References (现在在第5位，条件性，50%)
            60: 5,  # Repos (GitHub下载)
            70: 6,  # Index (代码索引)
            85: 7,  # Implement (代码实现)
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
            current_step = new_step
            update_step_indicator(step_indicators, workflow_steps, current_step, "active")
        
        time.sleep(0.3)  # 短暂停顿以便用户看到进度变化
    
    # 步骤1: 初始化 / Step 1: Initialization
    if enable_indexing:
        update_progress(5, "🚀 Initializing AI research engine and loading models...")
    else:
        update_progress(5, "🚀 Initializing AI research engine (Fast mode - indexing disabled)...")
    update_step_indicator(step_indicators, workflow_steps, 0, "active")
    
    # 开始异步处理，使用进度回调
    with st.spinner("🔄 Processing workflow stages..."):
        try:
            # 首先尝试使用简单的异步处理方法
            result = run_async_task_simple(process_input_async(input_source, input_type, enable_indexing, update_progress))
        except Exception as e:
            st.warning(f"Primary async method failed: {e}")
            # 备用方法：使用原始的线程池方法
            try:
                result = run_async_task(process_input_async(input_source, input_type, enable_indexing, update_progress))
            except Exception as backup_error:
                st.error(f"Both async methods failed. Error: {backup_error}")
                return {
                    "status": "error",
                    "error": str(backup_error),
                    "traceback": traceback.format_exc()
                }
    
    # 根据结果更新最终状态
    if result["status"] == "success":
        # 完成所有步骤
        update_progress(100, "✅ All processing stages completed successfully!")
        update_step_indicator(step_indicators, workflow_steps, len(workflow_steps), "completed")
        
        # 显示成功信息
        st.balloons()  # 添加庆祝动画
        if enable_indexing:
            display_status("🎉 Workflow completed! Your research paper has been successfully processed and code has been generated.", "success")
        else:
            display_status("🎉 Fast workflow completed! Your research paper has been processed (indexing skipped for faster processing).", "success")
        
    else:
        # 处理失败
        update_progress(0, "❌ Processing failed - see error details below")
        update_step_indicator(step_indicators, workflow_steps, current_step, "error")
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
    
    # 获取索引开关状态
    enable_indexing = st.session_state.get("enable_indexing", True)
    
    try:
        # 处理工作流
        result = handle_processing_workflow(input_source, input_type, enable_indexing)
        
        # 显示结果状态
        if result["status"] == "success":
            display_status("All operations completed successfully! 🎉", "success")
        else:
            display_status(f"Error during processing", "error")
        
        # 更新session state
        update_session_state_with_result(result, input_type)
        
    except Exception as e:
        # 处理异常情况
        st.error(f"Unexpected error during processing: {e}")
        result = {"status": "error", "error": str(e)}
        update_session_state_with_result(result, input_type)
    
    finally:
        # 处理完成后重置状态和清理资源
        st.session_state.processing = False
        
        # 清理临时文件
        cleanup_temp_file(input_source, input_type)
        
        # 清理系统资源
        cleanup_resources()
        
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
    if 'enable_indexing' not in st.session_state:
        st.session_state.enable_indexing = True  # 默认启用索引功能 


def cleanup_resources():
    """
    清理系统资源，防止内存泄露 / Clean up system resources to prevent memory leaks
    """
    try:
        import gc
        import threading
        import multiprocessing
        import asyncio
        import sys
        
        # 1. 清理asyncio相关资源
        try:
            # 获取当前事件循环（如果存在）
            try:
                loop = asyncio.get_running_loop()
                # 取消所有挂起的任务
                if loop and not loop.is_closed():
                    pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
                    if pending_tasks:
                        for task in pending_tasks:
                            if not task.cancelled():
                                task.cancel()
                        # 等待任务取消完成
                        try:
                            if pending_tasks:
                                # 使用超时避免阻塞太久
                                import time
                                time.sleep(0.1)
                        except Exception:
                            pass
            except RuntimeError:
                # 没有运行中的事件循环，继续其他清理
                pass
        except Exception:
            pass
        
        # 2. 强制垃圾回收
        gc.collect()
        
        # 3. 清理活跃线程（除主线程外）
        active_threads = threading.active_count()
        if active_threads > 1:
            # 等待一段时间让线程自然结束
            import time
            time.sleep(0.5)
        
        # 4. 清理multiprocessing资源
        try:
            # 清理可能的多进程资源
            if hasattr(multiprocessing, 'active_children'):
                for child in multiprocessing.active_children():
                    if child.is_alive():
                        child.terminate()
                        child.join(timeout=1.0)
                        # 如果join超时，强制kill
                        if child.is_alive():
                            try:
                                child.kill()
                                child.join(timeout=0.5)
                            except Exception:
                                pass
            
            # 清理multiprocessing相关的资源追踪器
            try:
                import multiprocessing.resource_tracker
                if hasattr(multiprocessing.resource_tracker, '_resource_tracker'):
                    tracker = multiprocessing.resource_tracker._resource_tracker
                    if tracker and hasattr(tracker, '_stop'):
                        tracker._stop()
            except Exception:
                pass
                
        except Exception:
            pass
        
        # 5. 强制清理Python内部缓存
        try:
            # 清理模块缓存中的一些临时对象
            import sys
            # 不删除关键模块，只清理可能的临时资源
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
        except Exception:
            pass
        
        # 6. 最终垃圾回收
        gc.collect()
            
    except Exception as e:
        # 静默处理清理错误，避免影响主流程
        # 但在调试模式下可以记录错误
        try:
            import logging
            logging.getLogger(__name__).debug(f"Resource cleanup warning: {e}")
        except Exception:
            pass 