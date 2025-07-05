"""
DeepCode UI Application Entry Point
论文到代码 UI应用程序入口

这个文件作为UI模块的统一入口点
This file serves as the unified entry point for the UI module
"""

from .streamlit_app import main

# 直接导出main函数，使外部可以直接调用
# Directly export main function for external calls
__all__ = ["main"]

if __name__ == "__main__":
    main()
