# 环境变量配置文件
# 当代码复制到新环境时，可以通过修改这个文件来配置环境变量

import os
import subprocess
import socket
from pathlib import Path

# 动态获取项目根目录
def get_project_root():
    """获取项目根目录路径"""
    # 从当前文件路径向上查找，直到找到包含特定标识文件的目录
    current_file = Path(__file__).resolve()
    # 从 agent/env_config.py 向上两级到达项目根目录
    project_root = current_file.parent.parent
    return str(project_root)

# 获取项目根目录
PROJECT_ROOT = get_project_root()

# 默认环境变量配置
DEFAULT_ENV_VARS = {
    # 路径相关（动态配置）
    "WORKSPACE_BASE": PROJECT_ROOT,
    "CODE_DIR": os.path.join(PROJECT_ROOT, "code"), 
    "AGENT_DIR": os.path.join(PROJECT_ROOT, "agent"),
    "LOGS_DIR": os.path.join(PROJECT_ROOT, "logs"),
    "SUBMISSION_DIR": os.path.join(PROJECT_ROOT, "submission"),
    "PAPER_DIR": os.path.join(PROJECT_ROOT, "paper"),
    
    # AI模型配置
    "MODEL": "openai/gpt-4o",  # 默认使用较稳定的模型
    "MAX_TIME_IN_HOURS": "36000",
    "DISALLOW_SUBMIT": "False",
    "ITERATIVE_AGENT": "False",
    
    # API密钥（从agent.env读取的值）
    "OPENAI_API_KEY": "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",
    "ANTHROPIC_API_KEY": "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",
    "HF_TOKEN": "",
    "OPENROUTER_API_KEY": "",
    "GOOGLE_API_KEY": "",
    "PB_CODE_ONLY": "true",
    
    # 网络代理配置（香港大学宿主机代理：127.0.0.1:7890 -> Docker内：172.17.0.1:7890）
    "USE_PROXY": "True",   # 启用宿主机代理以解决地区限制
    "PROXY_HOST": "172.17.0.1",  # Docker容器内访问宿主机代理
    "PROXY_PORT": "7890",
    
    # 其他配置
    "CONDA_ENV_NAME": "agent",
    "PYTHON_VERSION": "3.12",
    
    # Docker网络配置
    "AUTO_PROXY": "True",   # 启用自动代理检测和配置
    "NETWORK_MODE": "bridge",  # 当前使用Docker Bridge模式
}

def load_env_config():
    """
    加载环境变量配置
    优先级：系统环境变量 > 本配置文件 > 默认值
    """
    for key, default_value in DEFAULT_ENV_VARS.items():
        if key not in os.environ:
            os.environ[key] = default_value
            print(f"设置环境变量: {key} = {default_value}")
    
    # 处理代理设置
    if os.environ.get("USE_PROXY", "False").lower() == "true":
        proxy_host = os.environ.get("PROXY_HOST", "127.0.0.1")
        proxy_port = os.environ.get("PROXY_PORT", "7890")
        proxy_url = f"http://{proxy_host}:{proxy_port}"
        
        os.environ['https_proxy'] = proxy_url
        os.environ['http_proxy'] = proxy_url
        print(f"已启用代理: {proxy_url}")
    else:
        # 确保删除代理设置（如果之前设置过）
        os.environ.pop('https_proxy', None)
        os.environ.pop('http_proxy', None)
        print("未启用代理")


def substitute_path_variables(text):
    """
    替换文本中的路径变量占位符
    将 ${VARIABLE_NAME} 替换为对应的环境变量值
    """
    import re
    
    def replace_var(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))  # 如果变量不存在，保持原样
    
    # 匹配 ${VARIABLE_NAME} 格式的占位符
    return re.sub(r'\$\{(\w+)\}', replace_var, text)

def get_env_info():
    """
    获取当前环境变量信息（用于调试）
    """
    print("\n=== 当前环境变量配置 ===")
    for key in DEFAULT_ENV_VARS.keys():
        value = os.environ.get(key, "未设置")
        # 对于API密钥，只显示前8位和后4位
        if "API_KEY" in key and value != "未设置" and len(value) > 12:
            masked_value = value[:8] + "***" + value[-4:]
            print(f"{key}: {masked_value}")
        else:
            print(f"{key}: {value}")
    print("========================\n")

if __name__ == "__main__":
    # 如果直接运行此文件，显示配置信息
    load_env_config()
    get_env_info()
