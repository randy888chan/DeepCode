"""
Streamlit UI样式模块 / Streamlit UI Styles Module

包含应用程序的所有CSS样式定义
Contains all CSS style definitions for the application
"""


def get_main_styles() -> str:
    """
    获取主要的CSS样式 / Get main CSS styles

    Returns:
        CSS样式字符串 / CSS styles string
    """
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Inter:wght@300;400;600;700&display=swap');

        :root {
            --primary-bg: #0a0e27;
            --secondary-bg: #1a1f3a;
            --accent-bg: #2d3748;
            --card-bg: rgba(45, 55, 72, 0.9);
            --glass-bg: rgba(255, 255, 255, 0.08);
            --glass-border: rgba(255, 255, 255, 0.12);
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --neon-blue: #64b5f6;
            --neon-cyan: #4dd0e1;
            --neon-green: #81c784;
            --neon-purple: #ba68c8;
            --text-primary: #ffffff;
            --text-secondary: #e3f2fd;
            --text-muted: #90caf9;
            --border-color: rgba(100, 181, 246, 0.2);
        }

        /* 全局应用背景和文字 */
        .stApp {
            background: linear-gradient(135deg, var(--primary-bg) 0%, var(--secondary-bg) 100%);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
        }

        /* 强制所有文本使用高对比度 */
        .stApp * {
            color: var(--text-primary) !important;
        }

        /* 侧边栏重新设计 - 深色科技风 */
        .css-1d391kg {
            background: linear-gradient(180deg, #0d1117 0%, #161b22 50%, #21262d 100%) !important;
            border-right: 2px solid var(--neon-cyan) !important;
            box-shadow: 0 0 20px rgba(77, 208, 225, 0.3) !important;
        }

        .css-1d391kg * {
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }

        .css-1d391kg h3 {
            color: var(--neon-cyan) !important;
            font-weight: 700 !important;
            font-size: 1.2rem !important;
            text-shadow: 0 0 15px rgba(77, 208, 225, 0.6) !important;
            border-bottom: 1px solid rgba(77, 208, 225, 0.3) !important;
            padding-bottom: 0.5rem !important;
            margin-bottom: 1rem !important;
        }

        .css-1d391kg p, .css-1d391kg div {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }

        /* 侧边栏信息框 - 深色科技风格 */
        .css-1d391kg .stAlert,
        .css-1d391kg .stInfo,
        .css-1d391kg .stSuccess,
        .css-1d391kg .stWarning,
        .css-1d391kg .stError {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
            border: 2px solid var(--neon-cyan) !important;
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            border-radius: 12px !important;
            box-shadow: 0 0 15px rgba(77, 208, 225, 0.3) !important;
            backdrop-filter: blur(10px) !important;
            margin: 0.5rem 0 !important;
            padding: 1rem !important;
        }

        /* 侧边栏信息框文字强制白色 */
        .css-1d391kg .stInfo div,
        .css-1d391kg .stInfo p,
        .css-1d391kg .stInfo span {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
        }

        /* 侧边栏按钮 - 科技风格 */
        .css-1d391kg .stButton button {
            background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-blue) 100%) !important;
            color: #000000 !important;
            font-weight: 800 !important;
            border: 2px solid var(--neon-cyan) !important;
            border-radius: 10px !important;
            box-shadow: 0 0 20px rgba(77, 208, 225, 0.4) !important;
            text-shadow: none !important;
            transition: all 0.3s ease !important;
        }

        .css-1d391kg .stButton button:hover {
            box-shadow: 0 0 30px rgba(77, 208, 225, 0.6) !important;
            transform: translateY(-2px) !important;
        }

        /* 侧边栏展开器 - 深色科技风 */
        .css-1d391kg .streamlit-expanderHeader {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
            color: var(--text-primary) !important;
            border: 2px solid var(--neon-purple) !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
            box-shadow: 0 0 10px rgba(186, 104, 200, 0.3) !important;
        }

        .css-1d391kg .streamlit-expanderContent {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
            border: 2px solid var(--neon-purple) !important;
            color: var(--text-primary) !important;
            border-radius: 0 0 10px 10px !important;
            box-shadow: 0 0 10px rgba(186, 104, 200, 0.2) !important;
        }

        /* 侧边栏所有文字元素强制高对比度 */
        .css-1d391kg span,
        .css-1d391kg p,
        .css-1d391kg div,
        .css-1d391kg label,
        .css-1d391kg strong,
        .css-1d391kg b {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* 侧边栏markdown内容 */
        .css-1d391kg [data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
            font-weight: 600 !important;
            background: none !important;
        }

        /* 侧边栏特殊样式 - 系统信息框 */
        .css-1d391kg .element-container {
            background: none !important;
        }

        .css-1d391kg .element-container div {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
            border: 1px solid var(--neon-cyan) !important;
            border-radius: 8px !important;
            padding: 0.8rem !important;
            box-shadow: 0 0 10px rgba(77, 208, 225, 0.2) !important;
            margin: 0.3rem 0 !important;
        }

        /* Processing History特殊处理 */
        .css-1d391kg .stExpander {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
            border: 2px solid var(--neon-green) !important;
            border-radius: 12px !important;
            box-shadow: 0 0 15px rgba(129, 199, 132, 0.3) !important;
            margin: 0.5rem 0 !important;
        }

        /* 确保所有文字在深色背景上可见 */
        .css-1d391kg .stExpander div,
        .css-1d391kg .stExpander p,
        .css-1d391kg .stExpander span {
            color: #ffffff !important;
            font-weight: 600 !important;
            background: none !important;
        }

        /* 主标题区域 */
        .main-header {
            background: linear-gradient(135deg,
                rgba(100, 181, 246, 0.1) 0%,
                rgba(77, 208, 225, 0.08) 50%,
                rgba(129, 199, 132, 0.1) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            text-align: center;
            position: relative;
            box-shadow:
                0 8px 32px rgba(100, 181, 246, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .main-header h1 {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 3.5rem !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-blue) 50%, var(--neon-purple) 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            text-shadow: 0 0 30px rgba(77, 208, 225, 0.5) !important;
            margin-bottom: 1rem !important;
            animation: titleGlow 3s ease-in-out infinite alternate !important;
        }

        @keyframes titleGlow {
            from { filter: drop-shadow(0 0 10px rgba(77, 208, 225, 0.5)); }
            to { filter: drop-shadow(0 0 20px rgba(186, 104, 200, 0.7)); }
        }

        .main-header h3 {
            font-family: 'Inter', sans-serif !important;
            font-size: 1.2rem !important;
            font-weight: 400 !important;
            color: var(--text-secondary) !important;
            letter-spacing: 2px !important;
            margin-bottom: 0.5rem !important;
        }

        .main-header p {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.9rem !important;
            color: var(--neon-green) !important;
            letter-spacing: 1px !important;
            font-weight: 600 !important;
        }

        /* 功能卡片重新设计 */
        .feature-card {
            background: var(--card-bg);
            backdrop-filter: blur(15px);
            border: 1px solid var(--border-color);
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        .feature-card:hover {
            transform: translateY(-5px);
            border-color: var(--neon-cyan);
            box-shadow: 0 8px 30px rgba(77, 208, 225, 0.3);
        }

        .feature-card h4 {
            font-family: 'Inter', sans-serif !important;
            color: var(--neon-cyan) !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.8rem !important;
        }

        .feature-card p {
            font-family: 'Inter', sans-serif !important;
            color: var(--text-secondary) !important;
            line-height: 1.6 !important;
            font-weight: 400 !important;
        }

        /* Streamlit 组件样式重写 */
        .stMarkdown h3 {
            color: var(--neon-cyan) !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1.5rem !important;
            text-shadow: 0 0 10px rgba(77, 208, 225, 0.3) !important;
        }

        /* 单选按钮样式 */
        .stRadio > div {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            backdrop-filter: blur(10px) !important;
        }

        .stRadio label {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }

        .stRadio > div > div > div > label {
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
        }

        /* 文件上传器 */
        .stFileUploader > div {
            background: var(--card-bg) !important;
            border: 2px dashed var(--border-color) !important;
            border-radius: 15px !important;
            transition: all 0.3s ease !important;
            backdrop-filter: blur(10px) !important;
        }

        .stFileUploader > div:hover {
            border-color: var(--neon-cyan) !important;
            box-shadow: 0 0 20px rgba(77, 208, 225, 0.3) !important;
        }

        .stFileUploader label {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }

        .stFileUploader span {
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
        }

        /* 文本输入框 */
        .stTextInput > div > div > input {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
            font-weight: 500 !important;
            backdrop-filter: blur(10px) !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: var(--neon-cyan) !important;
            box-shadow: 0 0 0 1px var(--neon-cyan) !important;
        }

        .stTextInput label {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }

        /* 按钮样式 */
        .stButton > button {
            width: 100% !important;
            background: var(--primary-gradient) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.8rem 2rem !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            letter-spacing: 0.5px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        }

        /* 状态消息样式 */
        .status-success, .stSuccess {
            background: linear-gradient(135deg, rgba(129, 199, 132, 0.15) 0%, rgba(129, 199, 132, 0.05) 100%) !important;
            color: var(--neon-green) !important;
            padding: 1rem 1.5rem !important;
            border-radius: 10px !important;
            border: 1px solid rgba(129, 199, 132, 0.3) !important;
            backdrop-filter: blur(10px) !important;
            font-weight: 600 !important;
        }

        .status-error, .stError {
            background: linear-gradient(135deg, rgba(244, 67, 54, 0.15) 0%, rgba(244, 67, 54, 0.05) 100%) !important;
            color: #ff8a80 !important;
            padding: 1rem 1.5rem !important;
            border-radius: 10px !important;
            border: 1px solid rgba(244, 67, 54, 0.3) !important;
            backdrop-filter: blur(10px) !important;
            font-weight: 600 !important;
        }

        .status-warning, .stWarning {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 193, 7, 0.05) 100%) !important;
            color: #ffcc02 !important;
            padding: 1rem 1.5rem !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 193, 7, 0.3) !important;
            backdrop-filter: blur(10px) !important;
            font-weight: 600 !important;
        }

        .status-info, .stInfo {
            background: linear-gradient(135deg, rgba(77, 208, 225, 0.15) 0%, rgba(77, 208, 225, 0.05) 100%) !important;
            color: var(--neon-cyan) !important;
            padding: 1rem 1.5rem !important;
            border-radius: 10px !important;
            border: 1px solid rgba(77, 208, 225, 0.3) !important;
            backdrop-filter: blur(10px) !important;
            font-weight: 600 !important;
        }

        /* 进度条 */
        .progress-container {
            margin: 1.5rem 0;
            padding: 2rem;
            background: var(--card-bg);
            backdrop-filter: blur(15px);
            border: 1px solid var(--border-color);
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        .stProgress > div > div > div {
            background: var(--accent-gradient) !important;
            border-radius: 10px !important;
        }

        /* 文本区域 */
        .stTextArea > div > div > textarea {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
            font-family: 'JetBrains Mono', monospace !important;
            backdrop-filter: blur(10px) !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            font-weight: 600 !important;
        }

        .streamlit-expanderContent {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
        }

        /* 确保所有Markdown内容可见 */
        [data-testid="stMarkdownContainer"] p {
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
        }

        /* 分隔线 */
        hr {
            border-color: var(--border-color) !important;
            opacity: 0.5 !important;
        }

        /* 滚动条 */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--accent-bg);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--accent-gradient);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary-gradient);
        }

        /* 占位符文本 */
        ::placeholder {
            color: var(--text-muted) !important;
            opacity: 0.7 !important;
        }
    </style>
    """
