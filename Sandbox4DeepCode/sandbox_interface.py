#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沙箱测试接口模块
功能：为WebUI提供独立的沙箱测试接口，避免UI依赖问题

主要功能：
- 封装沙箱系统的导入和初始化逻辑
- 处理相对导入问题
- 提供完整的错误处理和日志捕获
"""

import os
import sys
import logging
import io
from pathlib import Path
from typing import Dict, Any, List


def run_sandbox_test_standalone(code_directory: str, help_docs_paths: List[str] = None, timeout: int = 120, detailed_log: bool = True) -> Dict[str, Any]:
    """
    独立运行沙箱测试的接口函数
    
    功能：
    - 不依赖streamlit和其他UI模块
    - 直接调用沙箱系统
    - 提供完整的错误处理
    
    Args:
        code_directory: 代码目录路径
        help_docs_paths: 帮助文档文件路径列表
        timeout: 超时时间
        detailed_log: 是否启用详细日志
        
    Returns:
        沙箱测试结果字典
    """
    # 保存当前工作目录（尽早保存以确保能恢复）
    original_cwd = os.getcwd()
    
    # 初始化调试信息
    debug_info = {
        'original_cwd': original_cwd,
        'function_start': True
    }
    
    try:
        # 获取沙箱系统路径（当前文件就在沙箱目录中）
        sandbox_path = Path(__file__).parent
        
        # 检查沙箱主文件是否存在
        sandbox_main = sandbox_path / "main.py"
        if not sandbox_main.exists():
            return {
                'status': 'error',
                'error_type': 'sandbox_main_not_found',
                'error_message': f'沙箱主文件不存在: {sandbox_main}',
                'suggestion': '请确保main.py文件存在于Sandbox4DeepCode目录中'
            }
        
        # 切换到沙箱目录以确保相对导入正常工作
        os.chdir(str(sandbox_path))
        
        # 保存当前的sys.path状态用于调试
        debug_info = {
            'original_sys_path': sys.path.copy(),
            'working_directory': os.getcwd(),
            'sandbox_path': str(sandbox_path)
        }
        
        # 1. 确保所有必要的路径都在sys.path中
        paths_to_add = [
            str(sandbox_path),
            str(sandbox_path / "agents"),
            str(sandbox_path / "core"), 
            str(sandbox_path / "utils"),
            str(sandbox_path / "config")
        ]

        for path in paths_to_add:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)

        debug_info['added_paths'] = paths_to_add
        debug_info['final_sys_path'] = sys.path[:15]  # 前15个路径
        
        # 2. 预导入所有必需的模块以解决相对导入问题
        import importlib.util
        
        def preload_module(module_path, module_name):
            """预加载模块到sys.modules"""
            try:
                if module_name not in sys.modules:
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        return module
                return sys.modules[module_name]
            except Exception as e:
                print(f"预加载模块失败 {module_name}: {e}")
                return None
        
        # 预加载所有工具模块
        utils_modules = [
            ('file_utils', sandbox_path / "utils" / "file_utils.py"),
            ('colored_logging', sandbox_path / "utils" / "colored_logging.py"),
            ('config_utils', sandbox_path / "utils" / "config_utils.py"),
        ]
        
        for module_name, module_path in utils_modules:
            if module_path.exists():
                preload_module(module_path, module_name)
                preload_module(module_path, f"utils.{module_name}")
        
        # 预加载基础代理
        base_agent_path = sandbox_path / "agents" / "base_agent.py"
        if base_agent_path.exists():
            preload_module(base_agent_path, "base_agent")
            preload_module(base_agent_path, "agents.base_agent")
            
        # 3. 使用importlib动态导入main.py
        main_py_path = sandbox_path / "main.py"
        if not main_py_path.exists():
            raise ImportError(f"main.py not found at {main_py_path}")
        
        spec = importlib.util.spec_from_file_location("sandbox_main", main_py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create spec for {main_py_path}")
        
        sandbox_main_module = importlib.util.module_from_spec(spec)
        
        # 执行模块以加载所有定义
        spec.loader.exec_module(sandbox_main_module)
        
        # 获取SandboxTestSystem类
        if not hasattr(sandbox_main_module, 'SandboxTestSystem'):
            raise ImportError("SandboxTestSystem class not found in main.py")
        
        SandboxTestSystem = getattr(sandbox_main_module, 'SandboxTestSystem')
        
        # 保存调试信息，以防需要
        debug_info['success'] = True
        
        # 4. 创建沙箱测试系统实例
        sandbox_system = SandboxTestSystem()
        
        # 运行完整测试流程
        try:
            # 设置日志捕获
            log_capture_string = io.StringIO()
            log_handler = logging.StreamHandler(log_capture_string)
            log_handler.setLevel(logging.INFO)
            log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            log_handler.setFormatter(log_formatter)

            # 添加处理器到根日志记录器
            root_logger = logging.getLogger()
            original_level = root_logger.level
            root_logger.addHandler(log_handler)
            root_logger.setLevel(logging.INFO)

            try:
                # 确保在沙箱目录中运行测试
                result = sandbox_system.run_complete_analysis(code_directory, help_docs_paths or [])

                # 获取捕获的日志
                log_contents = log_capture_string.getvalue()

                # 将日志添加到结果中
                if 'debug_logs' not in result:
                    result['debug_logs'] = []
                result['debug_logs'].append({
                    'source': 'sandbox_execution',
                    'content': log_contents
                })

                return result

            finally:
                # 清理日志处理器
                root_logger.removeHandler(log_handler)
                root_logger.setLevel(original_level)
                log_handler.close()

        except Exception as execution_error:
            # 恢复原始工作目录
            os.chdir(original_cwd)
            
            import traceback
            detailed_traceback = traceback.format_exc()
            
            return {
                'status': 'error',
                'error_type': 'execution_error',
                'error_message': f'沙箱执行失败: {str(execution_error)}',
                'suggestion': '请检查代码目录和依赖是否正确',
                'detailed_error': str(execution_error),
                'full_traceback': detailed_traceback,
                'debug_info': debug_info
            }
            
    except ImportError as import_error:
        # 恢复原始工作目录
        os.chdir(original_cwd)
        
        # 收集详细的导入调试信息
        import traceback
        detailed_traceback = traceback.format_exc()
        
        return {
            'status': 'error',
            'error_type': 'import_error',
            'error_message': f'无法导入沙箱系统: {str(import_error)}',
            'suggestion': '请检查沙箱系统的依赖是否正确安装',
            'detailed_error': str(import_error),
            'full_traceback': detailed_traceback,
            'sys_path_info': debug_info.get('final_sys_path', sys.path[:10]),
            'current_dir': str(sandbox_path) if 'sandbox_path' in locals() else 'unknown',
            'sandbox_exists': sandbox_main.exists() if 'sandbox_main' in locals() else False,
            'main_py_exists': (sandbox_path / "main.py").exists() if 'sandbox_path' in locals() else False,
            'utils_dir_exists': (sandbox_path / "utils").exists() if 'sandbox_path' in locals() else False,
            'file_utils_exists': (sandbox_path / "utils" / "file_utils.py").exists() if 'sandbox_path' in locals() else False,
            'debug_info': debug_info,
            'added_paths': debug_info.get('added_paths', [])
        }
        
    except Exception as e:
        # 恢复原始工作目录
        os.chdir(original_cwd)
        
        import traceback
        detailed_traceback = traceback.format_exc()
        
        return {
            'status': 'error',
            'error_type': 'general_error',
            'error_message': f'沙箱测试出现意外错误: {str(e)}',
            'suggestion': '请检查系统环境和依赖',
            'detailed_error': str(e),
            'full_traceback': detailed_traceback,
            'debug_info': debug_info
        }
    
    finally:
        # 确保总是恢复原始工作目录
        try:
            os.chdir(original_cwd)
        except Exception:
            pass  # 如果连这个都失败了，我们无能为力 