"""
UI模块 / UI Module

Streamlit应用的用户界面组件模块
User interface components module for Streamlit application

包含以下子模块 / Contains the following submodules:
- styles: CSS样式 / CSS styles
- components: UI组件 / UI components  
- layout: 页面布局 / Page layout
- handlers: 事件处理 / Event handlers
- streamlit_app: 主应用 / Main application
- app: 应用入口 / Application entry
"""

__version__ = "1.0.0"
__author__ = "Paper to Code Team"

# 导入主要组件 / Import main components
from .layout import main_layout
from .components import display_header, display_features, display_status
from .handlers import initialize_session_state
from .styles import get_main_styles

# 导入应用主函数 / Import application main function
try:
    from .streamlit_app import main as streamlit_main
except ImportError:
    # 如果相对导入失败，尝试绝对导入 / If relative import fails, try absolute import
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from streamlit_app import main as streamlit_main

__all__ = [
    "main_layout",
    "display_header", 
    "display_features",
    "display_status",
    "initialize_session_state",
    "get_main_styles",
    "streamlit_main"
] 