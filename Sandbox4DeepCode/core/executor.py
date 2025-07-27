# 文件概述：
# 本文件负责沙箱测试的执行，包括自动创建虚拟环境、依赖安装、代码执行和清理。
# 主要函数：_create_virtualenv, _install_dependencies, _run_in_virtualenv, _cleanup_virtualenv
# 所有操作均在独立venv中完成，保证环境隔离和安全。
#
# 代码安全性优先，所有异常均有详细日志。
#
# Author: 自动生成

import os
import shutil
import subprocess
import logging
import sys
import time
from typing import Dict, Any
import venv

class SandboxExecutor:
    """
    沙箱执行器：负责在独立的虚拟环境中自动化执行测试代码，并在结束后清理环境。
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _create_virtualenv(self, venv_path: str) -> bool:
        """创建独立的Python虚拟环境"""
        try:
            builder = venv.EnvBuilder(with_pip=True)
            builder.create(venv_path)
            logging.info(f"[沙箱] ✅ 虚拟环境已创建: {venv_path}")
            return True
        except Exception as e:
            logging.error(f"[沙箱] ❌ 创建虚拟环境失败: {e}")
            return False

    def _install_dependencies(self, venv_path: str, requirements_file: str) -> bool:
        """在虚拟环境中安装依赖"""
        try:
            python_bin = os.path.join(venv_path, 'bin', 'python')
            pip_bin = os.path.join(venv_path, 'bin', 'pip')
            abs_requirements_file = os.path.abspath(requirements_file)
            
            # 读取requirements文件来显示要安装的包
            try:
                with open(abs_requirements_file, 'r') as f:
                    packages = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
                logging.info(f"[沙箱] 📦 开始安装 {len(packages)} 个依赖包...")
                logging.info(f"[沙箱] 📋 依赖列表: {', '.join(packages[:5])}{'...' if len(packages) > 5 else ''}")
            except:
                logging.info(f"[沙箱] 📦 开始安装依赖...")
            
            start_time = time.time()
            
            cmd = [pip_bin, 'install', '-r', abs_requirements_file]
            logging.info(f"[沙箱] 🔄 执行命令: {' '.join(cmd)}")
            
            # 使用Popen来实时显示进度
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1, 
                universal_newlines=True
            )
            
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_lines.append(output.strip())
                    # 只显示重要的进度信息
                    if any(keyword in output.lower() for keyword in ['collecting', 'downloading', 'installing', 'successfully installed']):
                        logging.info(f"[沙箱] 📦 {output.strip()}")
            
            returncode = process.poll()
            install_time = time.time() - start_time
            
            if returncode == 0:
                logging.info(f"[沙箱] ✅ 依赖安装成功 (耗时: {install_time:.1f}秒)")
                return True
            else:
                logging.error(f"[沙箱] ❌ 依赖安装失败 (返回码: {returncode})")
                # 显示最后几行错误信息
                for line in output_lines[-10:]:
                    if line.strip():
                        logging.error(f"[沙箱] {line}")
                return False
                
        except subprocess.TimeoutExpired:
            logging.error(f"[沙箱] ❌ 依赖安装超时 (>300秒)")
            return False
        except Exception as e:
            logging.error(f"[沙箱] ❌ 安装依赖异常: {e}")
            return False

    def _run_in_virtualenv(self, venv_path: str, entry_file: str, cwd: str, env: dict = None) -> tuple:
        """在虚拟环境中执行主程序，返回(stdout, stderr, returncode)"""
        # 查找虚拟环境中的 Python 可执行文件
        possible_paths = [
            os.path.join(venv_path, 'bin', 'python'),
            os.path.join(venv_path, 'bin', 'python3'),
            os.path.join(venv_path, 'Scripts', 'python.exe'),  # Windows
            os.path.join(venv_path, 'Scripts', 'python3.exe')  # Windows
        ]
        
        python_bin = None
        for path in possible_paths:
            # 检查文件是否存在，包括符号链接
            if os.path.lexists(path):
                # 如果是符号链接，验证它指向有效的文件
                if os.path.islink(path):
                    try:
                        # 尝试解析符号链接
                        real_path = os.path.realpath(path)
                        if os.path.exists(real_path):
                            python_bin = path
                            logging.info(f"[沙箱] 🔗 符号链接Python: {path} -> {real_path}")
                            break
                    except:
                        continue
                elif os.path.exists(path):
                    python_bin = path
                    break
        
        if not python_bin:
            logging.error(f"[沙箱] ❌ 找不到虚拟环境中的Python可执行文件: {venv_path}")
            # 详细列出bin目录内容进行调试
            bin_dir = os.path.join(venv_path, 'bin')
            if os.path.exists(bin_dir):
                files = os.listdir(bin_dir)
                logging.error(f"[沙箱] 📂 bin目录内容: {files}")
            return '', 'Python executable not found', -1
        
        logging.info(f"[沙箱] 🐍 使用Python: {python_bin}")
        
        # 在执行前再次验证Python可执行文件和入口文件
        if not os.path.exists(python_bin) and not os.path.islink(python_bin):
            logging.error(f"[沙箱] ❌ Python可执行文件在执行前消失: {python_bin}")
            return '', 'Python executable disappeared', -1
        
        entry_file_path = os.path.join(cwd, entry_file)
        if not os.path.exists(entry_file_path):
            logging.error(f"[沙箱] ❌ 入口文件不存在: {entry_file_path}")
            return '', 'Entry file not found', -1
        
        logging.info(f"[沙箱] 📄 入口文件: {entry_file}")
        logging.info(f"[沙箱] 📁 工作目录: {cwd}")
        
        # 关键修复：使用相对于当前工作目录的Python路径
        # 计算Python可执行文件相对于目标工作目录的路径
        current_dir = os.getcwd()
        python_relative_to_cwd = os.path.relpath(os.path.abspath(python_bin), os.path.abspath(cwd))
        
        try:
            logging.info(f"[沙箱] 🚀 执行命令: {python_relative_to_cwd} {entry_file}")
            logging.info(f"[沙箱] 📍 当前目录: {current_dir}")
            process = subprocess.Popen(
                [python_relative_to_cwd, entry_file],
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=600)
            logging.info(f"[沙箱] ✅ 代码执行完成，返回码: {process.returncode}")
            return stdout, stderr, process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            logging.error("[沙箱] ❌ 代码执行超时")
            return '', 'Timeout', -1
        except Exception as e:
            logging.error(f"[沙箱] ❌ 代码执行异常: {e}")
            return '', str(e), -1

    def _cleanup_virtualenv(self, venv_path: str):
        """删除虚拟环境目录"""
        try:
            shutil.rmtree(venv_path)
            logging.info(f"[沙箱] 🧹 虚拟环境已清理: {venv_path}")
        except Exception as e:
            logging.warning(f"[沙箱] ⚠️ 清理虚拟环境失败: {e}")

    def run_sandbox(self, test_repo_path: str, entry_file: str, requirements_file: str) -> dict:
        """主入口：自动创建venv，安装依赖，执行代码，清理环境，返回结果"""
        venv_path = os.path.join(test_repo_path, '.sandbox_venv')
        result = {'status': 'error', 'stdout': '', 'stderr': '', 'returncode': -1}
        cleanup_needed = False
        
        try:
            # 创建虚拟环境
            if not self._create_virtualenv(venv_path):
                result['stderr'] = '虚拟环境创建失败'
                return result
            cleanup_needed = True
            
            # 安装依赖
            if not self._install_dependencies(venv_path, requirements_file):
                result['stderr'] = '依赖安装失败'
                return result
            
            # 执行代码前，验证虚拟环境仍然存在
            if not os.path.exists(venv_path):
                logging.error(f"[沙箱] ❌ 虚拟环境在执行前消失: {venv_path}")
                result['stderr'] = '虚拟环境在执行前消失'
                return result
            
            logging.info(f"[沙箱] 🚀 准备执行代码: {entry_file}")
            stdout, stderr, returncode = self._run_in_virtualenv(
                venv_path, entry_file, test_repo_path
            )
            result.update({'status': 'ok' if returncode == 0 else 'error', 'stdout': stdout, 'stderr': stderr, 'returncode': returncode})
            return result
            
        finally:
            # 临时禁用清理，让用户可以检查沙箱环境
            # if cleanup_needed:
            #     self._cleanup_virtualenv(venv_path)
            logging.info(f"[沙箱] 🔍 调试模式: 保留虚拟环境用于检查: {venv_path}") 