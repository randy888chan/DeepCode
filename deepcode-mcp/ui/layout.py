"""
Streamlit 页面布局模块 / Streamlit Page Layout Module

包含主要的页面布局和流程控制
Contains main page layout and flow control
"""

import streamlit as st
from typing import Dict, Any

from .components import (
    display_header,
    display_features,
    sidebar_control_panel,
    input_method_selector,
    results_display_component,
    footer_component
)
from .handlers import (
    initialize_session_state,
    handle_start_processing_button,
    handle_error_display
)
from .styles import get_main_styles


def setup_page_config():
    """
    设置页面配置 / Setup page configuration
    """
    st.set_page_config(
        page_title="Paper to Code - AI Research Engine",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def apply_custom_styles():
    """
    应用自定义样式 / Apply custom styles
    """
    st.markdown(get_main_styles(), unsafe_allow_html=True)


def render_main_content():
    """
    渲染主要内容区域 / Render main content area
    """
    # 显示头部和功能特性
    display_header()
    display_features()
    st.markdown("---")
    
    # 如果有结果显示，先显示结果
    if st.session_state.show_results and st.session_state.last_result:
        results_display_component(st.session_state.last_result, st.session_state.task_counter)
        st.markdown("---")
        return
    
    # 只有在不显示结果时才显示输入界面
    if not st.session_state.show_results:
        render_input_interface()
    
    # 显示错误信息（如果有）
    handle_error_display()


def render_input_interface():
    """
    渲染输入界面 / Render input interface
    """
    # 获取输入源和类型
    input_source, input_type = input_method_selector(st.session_state.task_counter)
    
    # 处理按钮
    if input_source and not st.session_state.processing:
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            handle_start_processing_button(input_source, input_type)
    
    elif st.session_state.processing:
        st.info("🔄 Processing in progress... Please wait.")
        st.warning("⚠️ Do not refresh the page or close the browser during processing.")
    
    elif not input_source:
        st.info("👆 Please upload a file or enter a URL to start processing.")


def render_sidebar():
    """
    渲染侧边栏 / Render sidebar
    """
    return sidebar_control_panel()


def main_layout():
    """
    主布局函数 / Main layout function
    """
    # 初始化session state
    initialize_session_state()
    
    # 设置页面配置
    setup_page_config()
    
    # 应用自定义样式
    apply_custom_styles()
    
    # 渲染侧边栏
    sidebar_info = render_sidebar()
    
    # 渲染主要内容
    render_main_content()
    
    # 显示页脚
    footer_component()
    
    return sidebar_info 