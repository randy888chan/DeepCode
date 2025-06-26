#!/usr/bin/env python3
"""
代理控制工具 - 快速管理Docker环境中的宿主机代理设置
"""

import os
import sys
import subprocess
import socket
import json
import argparse

def test_proxy_connection(host="172.17.0.1", port=7890, timeout=5):
    """测试代理连接"""
    print(f"🔍 测试代理连接: {host}:{port}")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ 代理端口可达")
            return True
        else:
            print(f"❌ 代理端口不可达 (错误码: {result})")
            return False
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False

def test_proxy_functionality(proxy_url):
    """测试代理功能"""
    print(f"🌐 测试代理功能: {proxy_url}")
    try:
        # 测试HTTP代理
        result = subprocess.run([
            'curl', '-s', '--connect-timeout', '10', 
            '--proxy', proxy_url, 'http://httpbin.org/ip'
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                proxy_ip = data.get('origin', '未知')
                print(f"✅ 代理功能正常，通过代理的IP: {proxy_ip}")
                return True, proxy_ip
            except:
                print(f"✅ 代理连接成功，但响应格式异常")
                return True, None
        else:
            print(f"❌ 代理功能测试失败: {result.stderr}")
            return False, None
    except subprocess.TimeoutExpired:
        print(f"❌ 代理功能测试超时")
        return False, None
    except Exception as e:
        print(f"❌ 代理功能测试异常: {e}")
        return False, None

def enable_proxy(host="172.17.0.1", port=7890, force=False):
    """启用代理"""
    proxy_url = f"http://{host}:{port}"
    print(f"🔧 启用代理: {proxy_url}")
    
    if not force:
        # 先测试连接
        if not test_proxy_connection(host, port):
            print("⚠️  代理端口不可达，使用 --force 强制启用")
            return False
    
    # 设置环境变量
    os.environ['https_proxy'] = proxy_url
    os.environ['http_proxy'] = proxy_url
    
    print(f"✅ 代理已启用: {proxy_url}")
    print("环境变量:")
    print(f"  https_proxy = {os.environ.get('https_proxy')}")
    print(f"  http_proxy = {os.environ.get('http_proxy')}")
    
    # 测试功能
    success, proxy_ip = test_proxy_functionality(proxy_url)
    if success and proxy_ip:
        print(f"🎯 代理工作正常，外部IP: {proxy_ip}")
    
    return success

def disable_proxy():
    """禁用代理"""
    print("🚫 禁用代理")
    
    # 清除环境变量
    os.environ.pop('https_proxy', None)
    os.environ.pop('http_proxy', None)
    
    print("✅ 代理已禁用")
    print("环境变量:")
    print(f"  https_proxy = {os.environ.get('https_proxy', '未设置')}")
    print(f"  http_proxy = {os.environ.get('http_proxy', '未设置')}")

def show_status():
    """显示当前代理状态"""
    print("📊 当前代理状态")
    print("================")
    
    https_proxy = os.environ.get('https_proxy')
    http_proxy = os.environ.get('http_proxy')
    
    if https_proxy or http_proxy:
        print("🔧 代理已启用:")
        if https_proxy:
            print(f"  https_proxy = {https_proxy}")
        if http_proxy:
            print(f"  http_proxy = {http_proxy}")
            
        # 测试当前代理
        if http_proxy:
            test_proxy_functionality(http_proxy)
    else:
        print("🚫 代理未启用 (直连模式)")
        
        # 测试直连
        print("\n🌐 测试直连:")
        try:
            result = subprocess.run([
                'curl', '-s', '--connect-timeout', '5', 'http://httpbin.org/ip'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    direct_ip = data.get('origin', '未知')
                    print(f"✅ 直连正常，IP: {direct_ip}")
                except:
                    print("✅ 直连成功，但响应格式异常")
            else:
                print("❌ 直连失败")
        except:
            print("❌ 直连测试异常")

def test_api_access():
    """测试API访问"""
    print("🤖 测试AI API访问")
    print("=================")
    
    apis = [
        ("OpenAI", "https://api.openai.com"),
        ("Anthropic", "https://api.anthropic.com"),
    ]
    
    for name, url in apis:
        print(f"\n{name} API:")
        try:
            result = subprocess.run([
                'curl', '-s', '--connect-timeout', '5', '-I', url
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and 'HTTP' in result.stdout:
                status_line = result.stdout.split('\n')[0]
                print(f"  ✅ {status_line}")
            else:
                print(f"  ❌ 连接失败")
        except:
            print(f"  ❌ 测试异常")

def main():
    parser = argparse.ArgumentParser(description='Docker环境代理控制工具')
    parser.add_argument('action', choices=['enable', 'disable', 'status', 'test', 'api'], 
                       help='操作: enable(启用), disable(禁用), status(状态), test(测试), api(测试API)')
    parser.add_argument('--host', default='172.17.0.1', help='代理主机地址 (默认: 172.17.0.1)')
    parser.add_argument('--port', type=int, default=7890, help='代理端口 (默认: 7890)')
    parser.add_argument('--force', action='store_true', help='强制启用代理，即使连接测试失败')
    
    args = parser.parse_args()
    
    print("🐳 Docker环境代理控制工具")
    print("=" * 30)
    print(f"宿主机代理: {args.host}:{args.port}")
    print(f"对应宿主机地址: 127.0.0.1:{args.port}")
    print()
    
    if args.action == 'enable':
        enable_proxy(args.host, args.port, args.force)
    elif args.action == 'disable':
        disable_proxy()
    elif args.action == 'status':
        show_status()
    elif args.action == 'test':
        test_proxy_connection(args.host, args.port)
        proxy_url = f"http://{args.host}:{args.port}"
        test_proxy_functionality(proxy_url)
    elif args.action == 'api':
        test_api_access()

if __name__ == "__main__":
    main() 