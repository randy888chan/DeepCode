#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础代理类
功能：提供LLM代理的基础功能和配置

"""

import os
import logging
import openai
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """LLM代理基础类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化代理
        
        Args:
            config: 代理配置字典
        """
        self.config = config
        self.model_name = config.get('model', 'qwen-turbo-latest')
        self.max_tokens = config.get('max_tokens', 4000)
        self.temperature = config.get('temperature', 0.1)
        self.base_url = config.get('base_url', None)
        
        # 设置OpenAI客户端
        self._setup_openai_client()
        
        logging.info(f"代理初始化完成: {self.__class__.__name__}")
    
    def _setup_openai_client(self):
        """设置OpenAI客户端"""
        try:
            # 获取API密钥
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logging.warning("未找到OPENAI_API_KEY环境变量")
                api_key = "dummy_key"  # 用于测试
            
            # 获取base_url，优先使用环境变量
            base_url = os.getenv('OPENAI_BASE_URL') or self.base_url
            
            # 创建客户端（添加超时设置）
            if base_url:
                self.client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=120.0  # 设置API调用超时为120秒
                )
                logging.info(f"使用base_url: {base_url}")
            else:
                self.client = openai.OpenAI(
                    api_key=api_key,
                    timeout=120.0  # 设置API调用超时为120秒
                )
                logging.info("使用默认OpenAI API")
            
        except Exception as e:
            logging.error(f"设置OpenAI客户端失败: {str(e)}")
            self.client = None
    
    def call_llm(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        调用LLM
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            LLM响应内容
        """
        try:
            if not self.client:
                logging.warning("OpenAI客户端未初始化，跳过LLM调用")
                return None
            
            # 合并配置参数
            params = {
                'model': self.model_name,
                'messages': messages,
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature)
            }
            
            logging.info(f"[LLM] 🚀 发起API调用: {self.model_name}, tokens: {params['max_tokens']}")
            logging.info(f"[LLM] 📝 消息数量: {len(messages)}")
            
            # 显示第一条消息的长度（用于调试）
            if messages:
                first_msg_len = len(messages[0].get('content', ''))
                logging.info(f"[LLM] 📏 第一条消息长度: {first_msg_len} 字符")
            
            # 调用API
            logging.info(f"[LLM] ⏳ 正在等待API响应...")
            response = self.client.chat.completions.create(**params)
            logging.info(f"[LLM] 🎉 API响应成功")
            
            # 提取响应内容
            content = response.choices[0].message.content
            logging.debug(f"LLM响应: {content[:100]}...")
            
            return content
            
        except openai.APITimeoutError as e:
            logging.error(f"LLM调用超时 (120秒): {str(e)}")
            return None
        except openai.APIError as e:
            logging.error(f"LLM API错误: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"LLM调用失败: {str(e)}")
            return None
    
    def create_system_message(self, role: str, task: str) -> Dict[str, str]:
        """
        创建系统消息
        
        Args:
            role: 代理角色
            task: 任务描述
            
        Returns:
            系统消息字典
        """
        return {
            "role": "system",
            "content": f"你是一个{role}。{task}"
        }
    
    def create_user_message(self, content: str) -> Dict[str, str]:
        """
        创建用户消息
        
        Args:
            content: 消息内容
            
        Returns:
            用户消息字典
        """
        return {
            "role": "user",
            "content": content
        }
    
    def create_assistant_message(self, content: str) -> Dict[str, str]:
        """
        创建助手消息
        
        Args:
            content: 消息内容
            
        Returns:
            助手消息字典
        """
        return {
            "role": "assistant",
            "content": content
        }
    
    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        处理输入数据（抽象方法）
        
        Args:
            input_data: 输入数据
            
        Returns:
            处理结果
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据
            
        Returns:
            是否有效
        """
        return input_data is not None
    
    def format_output(self, result: Any) -> Dict[str, Any]:
        """
        格式化输出结果
        
        Args:
            result: 原始结果
            
        Returns:
            格式化后的结果
        """
        return {
            'status': 'success',
            'result': result,
            'agent': self.__class__.__name__
        }
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        处理错误
        
        Args:
            error: 异常对象
            
        Returns:
            错误信息
        """
        logging.error(f"代理处理错误: {str(error)}")
        return {
            'status': 'error',
            'error': str(error),
            'agent': self.__class__.__name__
        } 