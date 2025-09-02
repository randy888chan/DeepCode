#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
彩色日志工具
功能：提供带颜色的日志输出，便于调试和追踪程序执行流程
作者：AI Assistant
创建时间：2024-01-01
"""

import logging
import sys
import inspect
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
        'RESET': '\033[0m',      # 重置
        'BOLD': '\033[1m',       # 粗体
        'BLUE': '\033[34m',      # 蓝色
        'MAGENTA': '\033[35m',   # 品红
    }
    
    def format(self, record):
        # 获取位置信息
        filename = record.filename
        funcname = record.funcName
        lineno = record.lineno
        
        # 缩短文件名（只保留文件名，不要完整路径）
        short_filename = filename.split('/')[-1] if '/' in filename else filename
        
        # 构建位置信息
        location_info = f"{short_filename}:{funcname}:{lineno}"
        
        # 构建时间信息
        time_str = self.formatTime(record, '%H:%M:%S')
        
        # 根据日志级别添加颜色
        level_color = self.COLORS.get(record.levelname, '')
        reset_color = self.COLORS['RESET']
        
        # 特殊标记处理
        message = record.getMessage()
        if '[STEP]' in message:
            level_color = self.COLORS['BLUE'] + self.COLORS['BOLD']
            icon = '🔄'
        elif '[LLM]' in message:
            level_color = self.COLORS['MAGENTA'] + self.COLORS['BOLD']
            icon = '🤖'
        elif '[AGENT]' in message:
            level_color = self.COLORS['BLUE']
            icon = '🎯'
        elif record.levelname == 'ERROR':
            icon = '❌'
        elif record.levelname == 'WARNING':
            icon = '⚠️'
        elif record.levelname == 'INFO':
            icon = '📝'
        elif record.levelname == 'DEBUG':
            icon = '🔍'
        else:
            icon = '📋'
        
        # 格式化最终输出
        formatted_msg = (
            f"{level_color}{icon} {time_str} "
            f"[{self.COLORS['RESET']}{self.COLORS['BLUE']}{location_info}{self.COLORS['RESET']}{level_color}] "
            f"{record.levelname}: {message}{reset_color}"
        )
        
        return formatted_msg


def setup_colored_logging(level: str = 'INFO') -> logging.Logger:
    """
    设置彩色日志系统
    
    Args:
        level: 日志级别
        
    Returns:
        配置好的logger
    """
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有handlers
    logger.handlers.clear()
    
    # 创建控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # 创建彩色格式化器（格式在ColoredFormatter中自定义）
    formatter = ColoredFormatter()
    console_handler.setFormatter(formatter)
    
    # 添加handler到logger
    logger.addHandler(console_handler)
    
    return logger


def log_step(message: str, step_number: Optional[int] = None):
    """
    记录步骤日志
    
    Args:
        message: 日志消息
        step_number: 步骤编号
    """
    if step_number:
        logging.info(f"[STEP {step_number}] {message}")
    else:
        logging.info(f"[STEP] {message}")


def log_llm_call(message: str):
    """
    记录LLM调用日志
    
    Args:
        message: 日志消息
    """
    logging.info(f"[LLM] {message}")


def log_agent(agent_name: str, message: str):
    """
    记录Agent操作日志
    
    Args:
        agent_name: Agent名称
        message: 日志消息
    """
    logging.info(f"[AGENT] {agent_name}: {message}")


def log_success(message: str):
    """记录成功日志"""
    logging.info(f"✅ {message}")


def log_warning(message: str):
    """记录警告日志"""
    logging.warning(f"⚠️ {message}")


def log_error(message: str):
    """记录错误日志"""
    logging.error(f"❌ {message}")


def log_debug(message: str):
    """记录调试日志"""
    logging.debug(f"🔍 {message}")


# 增强的日志函数，包含调用者信息
def log_detailed(level: str, message: str, extra_info: str = ""):
    """
    记录详细日志，包含调用者信息
    
    Args:
        level: 日志级别 (INFO, ERROR, WARNING, DEBUG)
        message: 日志消息
        extra_info: 额外信息
    """
    # 获取调用者信息
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename.split('/')[-1]
    funcname = frame.f_code.co_name
    lineno = frame.f_lineno
    
    # 构建详细消息
    detailed_msg = f"{message}"
    if extra_info:
        detailed_msg += f" | {extra_info}"
    
    # 记录日志
    logger = logging.getLogger(f"{filename}:{funcname}")
    getattr(logger, level.lower())(detailed_msg)


def log_function_entry(func_name: str = None, args: dict = None):
    """记录函数入口"""
    if not func_name:
        func_name = inspect.currentframe().f_back.f_code.co_name
    
    msg = f"🚀 ENTER {func_name}"
    if args:
        args_str = ", ".join([f"{k}={v}" for k, v in args.items()])
        msg += f"({args_str})"
    
    log_detailed("INFO", msg)


def log_function_exit(func_name: str = None, result: any = None):
    """记录函数退出"""
    if not func_name:
        func_name = inspect.currentframe().f_back.f_code.co_name
    
    msg = f"✅ EXIT {func_name}"
    if result is not None:
        msg += f" -> {result}"
    
    log_detailed("INFO", msg)


def log_checkpoint(checkpoint_name: str, details: str = ""):
    """记录检查点"""
    msg = f"🎯 CHECKPOINT: {checkpoint_name}"
    if details:
        msg += f" | {details}"
    log_detailed("INFO", msg)


# 创建一些常用的日志函数
def print_separator(title: str = "", char: str = "=", length: int = 60):
    """打印分隔线"""
    if title:
        padding = (length - len(title) - 2) // 2
        separator = char * padding + f" {title} " + char * padding
        if len(separator) < length:
            separator += char
    else:
        separator = char * length
    
    print(f"\033[36m{separator}\033[0m")  # 青色分隔线 