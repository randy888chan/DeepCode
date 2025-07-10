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

        /* 主标题区域 - 增强版 */
        .main-header {
            position: relative;
            background: linear-gradient(135deg,
                rgba(100, 181, 246, 0.12) 0%,
                rgba(77, 208, 225, 0.10) 30%,
                rgba(186, 104, 200, 0.12) 70%,
                rgba(129, 199, 132, 0.10) 100%);
            backdrop-filter: blur(25px);
            border: 1px solid transparent;
            background-clip: padding-box;
            padding: 4rem 2rem;
            border-radius: 25px;
            margin-bottom: 3rem;
            text-align: center;
            overflow: hidden;
            box-shadow:
                0 20px 60px rgba(0, 0, 0, 0.4),
                0 8px 32px rgba(100, 181, 246, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg,
                var(--neon-cyan) 0%,
                var(--neon-purple) 25%,
                var(--neon-blue) 50%,
                var(--neon-green) 75%,
                var(--neon-cyan) 100%);
            background-size: 400% 400%;
            border-radius: 25px;
            padding: 1px;
            margin: -1px;
            z-index: -1;
            animation: borderGlow 6s ease-in-out infinite;
        }

        .main-header::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 300%;
            height: 300%;
            background: radial-gradient(circle, transparent 30%, rgba(77, 208, 225, 0.03) 60%, transparent 70%);
            transform: translate(-50%, -50%);
            animation: headerPulse 8s ease-in-out infinite;
            pointer-events: none;
        }

        @keyframes headerPulse {
            0%, 100% { 
                opacity: 0.3;
                transform: translate(-50%, -50%) scale(1);
            }
            50% { 
                opacity: 0.7;
                transform: translate(-50%, -50%) scale(1.1);
            }
        }

        .main-header h1 {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 3.8rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-blue) 30%, var(--neon-purple) 70%, var(--neon-green) 100%) !important;
            background-size: 200% 200% !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            text-shadow: 0 0 40px rgba(77, 208, 225, 0.6) !important;
            margin-bottom: 1.2rem !important;
            animation: titleGlow 4s ease-in-out infinite alternate, gradientShift 3s ease-in-out infinite !important;
            position: relative;
            z-index: 2;
        }

        @keyframes titleGlow {
            0% { 
                filter: drop-shadow(0 0 15px rgba(77, 208, 225, 0.5)) drop-shadow(0 0 25px rgba(77, 208, 225, 0.3));
                text-shadow: 0 0 40px rgba(77, 208, 225, 0.6);
            }
            50% {
                filter: drop-shadow(0 0 25px rgba(186, 104, 200, 0.7)) drop-shadow(0 0 35px rgba(186, 104, 200, 0.5));
                text-shadow: 0 0 50px rgba(186, 104, 200, 0.8);
            }
            100% { 
                filter: drop-shadow(0 0 20px rgba(129, 199, 132, 0.6)) drop-shadow(0 0 30px rgba(129, 199, 132, 0.4));
                text-shadow: 0 0 45px rgba(129, 199, 132, 0.7);
            }
        }

        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
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

        /* 旧的功能卡片样式已移除，使用新的AI Agent卡片样式 */

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

        /* ================================
           🤖 AI AGENT 功能展示样式
           ================================ */

        /* AI Capabilities 区域 - 简化版，避免与主header冲突 */
        .ai-capabilities-section {
            position: relative;
            background: linear-gradient(135deg,
                rgba(77, 208, 225, 0.08) 0%,
                rgba(186, 104, 200, 0.06) 50%,
                rgba(129, 199, 132, 0.08) 100%);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(77, 208, 225, 0.2);
            padding: 2rem 1.5rem;
            border-radius: 20px;
            margin: 2rem 0;
            text-align: center;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }

        .ai-capabilities-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg,
                transparent 0%,
                rgba(77, 208, 225, 0.1) 50%,
                transparent 100%);
            animation: shimmer 3s ease-in-out infinite;
        }

        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        @keyframes borderGlow {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        /* 神经网络动画 */
        .neural-network {
            position: absolute;
            top: 1rem;
            right: 2rem;
            display: flex;
            gap: 0.5rem;
        }

        .neuron {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--neon-cyan);
            box-shadow: 0 0 10px var(--neon-cyan);
            animation-duration: 2s;
            animation-iteration-count: infinite;
            animation-timing-function: ease-in-out;
        }

        .pulse-1 { animation-name: neuronPulse; animation-delay: 0s; }
        .pulse-2 { animation-name: neuronPulse; animation-delay: 0.3s; }
        .pulse-3 { animation-name: neuronPulse; animation-delay: 0.6s; }

        @keyframes neuronPulse {
            0%, 100% {
                transform: scale(1);
                opacity: 0.7;
                box-shadow: 0 0 10px var(--neon-cyan);
            }
            50% {
                transform: scale(1.3);
                opacity: 1;
                box-shadow: 0 0 20px var(--neon-cyan), 0 0 30px var(--neon-cyan);
            }
        }

        .capabilities-title {
            font-family: 'Inter', sans-serif !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple), var(--neon-green));
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(77, 208, 225, 0.3);
            margin-bottom: 0.5rem !important;
            letter-spacing: -0.5px;
        }

        .capabilities-subtitle {
            font-family: 'JetBrains Mono', monospace !important;
            color: var(--neon-cyan) !important;
            font-size: 0.9rem !important;
            letter-spacing: 1.5px !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            opacity: 0.8;
        }

        /* 增强的功能卡片系统 - 确保对齐 */
        .feature-card {
            position: relative;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            padding: 2.5rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3);
            overflow: hidden;
            /* 确保卡片对齐 */
            height: 400px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        /* 卡片发光效果 */
        .card-glow {
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, transparent 20%, rgba(77, 208, 225, 0.05) 50%, transparent 70%);
            opacity: 0;
            transition: opacity 0.4s ease;
            pointer-events: none;
        }

        .feature-card:hover .card-glow {
            opacity: 1;
            animation: glowRotate 3s linear infinite;
        }

        @keyframes glowRotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* 不同主题的卡片样式 */
        .feature-card.primary {
            border-color: var(--neon-cyan);
            background: linear-gradient(135deg, 
                rgba(77, 208, 225, 0.1) 0%, 
                rgba(45, 55, 72, 0.95) 30%);
        }

        .feature-card.primary:hover {
            transform: translateY(-8px) scale(1.02);
            border-color: var(--neon-cyan);
            box-shadow: 
                0 20px 60px rgba(77, 208, 225, 0.3),
                0 0 50px rgba(77, 208, 225, 0.2);
        }

        .feature-card.secondary {
            border-color: var(--neon-purple);
            background: linear-gradient(135deg, 
                rgba(186, 104, 200, 0.1) 0%, 
                rgba(45, 55, 72, 0.95) 30%);
        }

        .feature-card.secondary:hover {
            transform: translateY(-8px) scale(1.02);
            border-color: var(--neon-purple);
            box-shadow: 
                0 20px 60px rgba(186, 104, 200, 0.3),
                0 0 50px rgba(186, 104, 200, 0.2);
        }

        .feature-card.accent {
            border-color: var(--neon-green);
            background: linear-gradient(135deg, 
                rgba(129, 199, 132, 0.1) 0%, 
                rgba(45, 55, 72, 0.95) 30%);
        }

        .feature-card.accent:hover {
            transform: translateY(-8px) scale(1.02);
            border-color: var(--neon-green);
            box-shadow: 
                0 20px 60px rgba(129, 199, 132, 0.3),
                0 0 50px rgba(129, 199, 132, 0.2);
        }

        .feature-card.tech {
            border-color: var(--neon-blue);
            background: linear-gradient(135deg, 
                rgba(100, 181, 246, 0.1) 0%, 
                rgba(45, 55, 72, 0.95) 30%);
        }

        .feature-card.tech:hover {
            transform: translateY(-8px) scale(1.02);
            border-color: var(--neon-blue);
            box-shadow: 
                0 20px 60px rgba(100, 181, 246, 0.3),
                0 0 50px rgba(100, 181, 246, 0.2);
        }

        /* 功能图标 */
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            text-align: center;
            filter: drop-shadow(0 0 10px rgba(77, 208, 225, 0.5));
            flex-shrink: 0;
        }

        /* 功能标题 */
        .feature-title {
            font-family: 'Inter', sans-serif !important;
            color: var(--text-primary) !important;
            font-size: 1.3rem !important;
            font-weight: 700 !important;
            margin-bottom: 1rem !important;
            text-align: center;
            text-shadow: 0 0 15px rgba(255, 255, 255, 0.3);
            flex-shrink: 0;
        }

        /* 功能描述 */
        .feature-description {
            color: var(--text-secondary) !important;
            line-height: 1.6 !important;
            font-weight: 500 !important;
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        /* 打字机效果 */
        .typing-text {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.95rem !important;
            margin-bottom: 1.5rem;
            border-right: 2px solid var(--neon-cyan);
            white-space: nowrap;
            overflow: hidden;
            animation: typing 3s steps(60, end), blink 1s infinite;
        }

        @keyframes typing {
            from { width: 0; }
            to { width: 100%; }
        }

        @keyframes blink {
            0%, 50% { border-color: var(--neon-cyan); }
            51%, 100% { border-color: transparent; }
        }

        /* 技术标签 */
        .tech-specs {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .spec-tag {
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-blue));
            color: #000 !important;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 10px rgba(77, 208, 225, 0.3);
        }

        /* 进度条动画 */
        .progress-bar {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 1rem;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--neon-purple), var(--neon-cyan), var(--neon-green));
            background-size: 200% 100%;
            border-radius: 3px;
            animation: progressMove 2s ease-in-out infinite;
            width: 75%;
        }

        @keyframes progressMove {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        /* 代码预览区域 */
        .code-preview {
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--neon-green);
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
            font-family: 'JetBrains Mono', monospace;
        }

        .code-line {
            font-size: 0.85rem;
            line-height: 1.6;
            margin-bottom: 0.5rem;
            color: var(--neon-green) !important;
        }

        .code-line.generating {
            color: var(--neon-cyan) !important;
            animation: textGlow 2s ease-in-out infinite;
        }

        @keyframes textGlow {
            0%, 100% { text-shadow: 0 0 5px var(--neon-cyan); }
            50% { text-shadow: 0 0 15px var(--neon-cyan), 0 0 25px var(--neon-cyan); }
        }

        /* 进度点 */
        .code-progress {
            margin-top: 1rem;
        }

        .progress-dots {
            display: flex;
            gap: 0.5rem;
            justify-content: center;
        }

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
        }

        .dot.active {
            background: var(--neon-green);
            box-shadow: 0 0 10px var(--neon-green);
            animation: dotPulse 1.5s ease-in-out infinite;
        }

        @keyframes dotPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.3); }
        }

        /* 技术栈展示 */
        .tech-stack {
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }

        .stack-item {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.8rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            transition: all 0.3s ease;
        }

        .stack-item:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--neon-blue);
            transform: translateX(5px);
        }

        .stack-icon {
            font-size: 1.2rem;
            filter: drop-shadow(0 0 8px rgba(100, 181, 246, 0.6));
        }

        .stack-name {
            font-family: 'JetBrains Mono', monospace !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 0.9rem;
        }

        /* 响应式设计 */
        @media (max-width: 768px) {
            .main-header {
                padding: 2.5rem 1.5rem;
                margin-bottom: 2rem;
                border-radius: 20px;
            }
            
            .main-header h1 {
                font-size: 2.5rem !important;
            }
            
            .main-header h3 {
                font-size: 1rem !important;
                letter-spacing: 1.5px !important;
            }
            
            .main-header p {
                font-size: 0.8rem !important;
                letter-spacing: 0.5px !important;
            }
            
            .ai-capabilities-section {
                padding: 1.5rem 1rem;
                margin: 1.5rem 0;
                border-radius: 15px;
            }
            
            .capabilities-title {
                font-size: 1.6rem !important;
            }
            
            .capabilities-subtitle {
                font-size: 0.8rem !important;
                letter-spacing: 1px !important;
            }
            
            .feature-card {
                padding: 1.5rem;
                margin: 1rem 0;
                height: auto;
                min-height: 350px;
                border-radius: 15px;
            }
            
            .neural-network {
                top: 0.5rem;
                right: 1rem;
            }
            
            .typing-text {
                white-space: normal;
                border-right: none;
                animation: none;
                font-size: 0.85rem !important;
            }
            
            .feature-icon {
                font-size: 2.5rem;
            }
            
            .feature-title {
                font-size: 1.1rem !important;
            }
        }
    </style>
    """
