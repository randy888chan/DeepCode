#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理工具
功能：加载和管理系统配置文件
作者：AI Assistant
创建时间：2024-01-01
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        if not os.path.exists(config_path):
            logging.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return get_default_config()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 合并默认配置
        default_config = get_default_config()
        merged_config = merge_configs(default_config, config)
        
        logging.info(f"成功加载配置文件: {config_path}")
        return merged_config
        
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
        logging.info("使用默认配置")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns:
        默认配置字典
    """
    return {
        'execution': {
            'timeout': 300,
            'memory_limit': '2GB',
            'cpu_limit': '2',
            'max_output_size': '100MB'
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'sandbox_test.log'
        },
        'isolation': {
            'use_docker': True,
            'network_access': False,
            'file_access': 'readonly',
            'temp_dir': '/tmp/sandbox'
        },
        'agents': {
            'code_analyzer': {
                'model': 'qwen-turbo-latest',
                'max_tokens': 4000,
                'temperature': 0.1
            },
            'test_writer': {
                'model': 'qwen-turbo-latest',
                'max_tokens': 8000,
                'temperature': 0.2
            },
            'execution_monitor': {
                'model': 'qwen-turbo-latest',
                'max_tokens': 2000,
                'temperature': 0.1
            },
            'review_agent': {
                'model': 'qwen-turbo-latest',
                'max_tokens': 6000,
                'temperature': 0.3
            }
        },
        'languages': {
            'python': {
                'enabled': True,
                'interpreter': 'python3',
                'package_manager': 'pip'
            },
            'javascript': {
                'enabled': True,
                'interpreter': 'node',
                'package_manager': 'npm'
            },
            'java': {
                'enabled': False,
                'interpreter': 'java',
                'compiler': 'javac'
            }
        },
        'output': {
            'temp_dir': 'temp'
        }
    }


def merge_configs(default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置字典
    
    Args:
        default: 默认配置
        custom: 自定义配置
        
    Returns:
        合并后的配置
    """
    result = default.copy()
    
    def merge_dict(target: Dict[str, Any], source: Dict[str, Any]):
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                merge_dict(target[key], value)
            else:
                target[key] = value
    
    merge_dict(result, custom)
    return result


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
        
    Returns:
        是否保存成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logging.info(f"配置已保存到: {config_path}")
        return True
        
    except Exception as e:
        logging.error(f"保存配置失败: {str(e)}")
        return False


def get_agent_config(config: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """
    获取指定代理的配置
    
    Args:
        config: 主配置字典
        agent_name: 代理名称
        
    Returns:
        代理配置字典
    """
    agents_config = config.get('agents', {})
    return agents_config.get(agent_name, {})


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置的有效性
    
    Args:
        config: 配置字典
        
    Returns:
        配置是否有效
    """
    try:
        # 检查必需的配置项
        required_keys = ['execution', 'logging', 'isolation', 'agents']
        for key in required_keys:
            if key not in config:
                logging.error(f"缺少必需的配置项: {key}")
                return False
        
        # 验证执行配置
        execution = config['execution']
        if execution.get('timeout', 0) <= 0:
            logging.error("执行超时时间必须大于0")
            return False
        
        # 验证代理配置
        agents = config['agents']
        required_agents = ['code_analyzer', 'test_writer', 'execution_monitor', 'review_agent']
        for agent in required_agents:
            if agent not in agents:
                logging.error(f"缺少代理配置: {agent}")
                return False
        
        logging.info("配置验证通过")
        return True
        
    except Exception as e:
        logging.error(f"配置验证失败: {str(e)}")
        return False 