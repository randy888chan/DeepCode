#!/bin/bash

# 智能网络配置启动脚本
# 自动检测网络环境并配置最佳设置

echo "🚀 启动AI代理 - 智能网络配置"
echo "================================"

# 检查基础环境
echo "🔍 检查环境..."
python --version
echo "当前工作目录: $(pwd)"
echo "容器IP: $(hostname -I)"

# 加载和检测网络配置
echo "🌐 配置网络环境..."
python -c "
from env_config import load_env_config, configure_proxy_for_docker, test_network_connectivity
print('正在加载环境配置...')
load_env_config()
print('正在检测和配置代理...')
configure_proxy_for_docker()
print('正在测试网络连通性...')
test_network_connectivity()
"

# 检查返回状态
if [ $? -eq 0 ]; then
    echo "✅ 网络配置完成"
else
    echo "⚠️  网络配置可能有问题，但继续启动..."
fi

echo ""
echo "🤖 启动AI代理..."
echo "================================"

# 启动主程序
python start.py

# 检查程序退出状态
if [ $? -eq 0 ]; then
    echo "✅ 程序正常退出"
else
    echo "❌ 程序异常退出，退出码: $?"
fi 