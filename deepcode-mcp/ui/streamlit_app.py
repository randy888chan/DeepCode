"""
Paper to Code - AI Research Engine
论文到代码 - AI研究引擎

Streamlit Web界面主应用文件
Main Streamlit web interface application file
"""

import os
import sys
# 禁止生成.pyc文件 / Disable .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# 添加父目录到路径，确保可以导入项目模块 / Add parent directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入UI模块 / Import UI modules
from ui.layout import main_layout


def main():
    """
    主函数 - Streamlit应用入口 / Main function - Streamlit application entry
    
    所有的UI逻辑都已经模块化到ui/文件夹中
    All UI logic has been modularized into ui/ folder
    """
    # 运行主布局 / Run main layout
    sidebar_info = main_layout()
    
    # 这里可以添加额外的全局逻辑（如果需要）
    # Additional global logic can be added here (if needed)
    
    return sidebar_info


if __name__ == "__main__":
    main() 