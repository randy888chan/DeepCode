#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码重写代理
功能：按照原repo结构重写每个代码文件，添加详细log信息和必要的测试输入，生成依赖列表

"""

import os
import shutil
import logging
import re
import time
import json
from typing import Dict, Any, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.base_agent import BaseAgent
from utils.file_utils import read_file_content
from utils.colored_logging import log_checkpoint, log_detailed


class CodeRewriterAgent(BaseAgent):
    """代码重写代理 - 重写代码文件添加log和测试输入"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.test_repo_suffix = "_test_sandbox"
        self.dependencies = set()
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理代码重写请求
        
        Args:
            input_data: 包含repo_path, structure_analysis的字典
            
        Returns:
            代码重写结果
        """
        try:
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("输入数据无效"))
            
            repo_path = input_data['repo_path']
            structure_analysis = input_data['structure_analysis']
            
            # 创建测试repo目录
            test_repo_path = self._create_test_repo_directory(repo_path)
            
            # 复制和重写代码文件
            rewritten_files = self._rewrite_code_files(repo_path, test_repo_path, structure_analysis)
            
            # 生成requirements.txt
            requirements_path = self._generate_requirements_file(test_repo_path)
            
            # 生成项目摘要
            project_summary = self._generate_project_summary(rewritten_files, structure_analysis)
            
            result = {
                'test_repo_path': test_repo_path,
                'rewritten_files': rewritten_files,
                'requirements_path': requirements_path,
                'dependencies': list(self.dependencies),
                'total_files': len(rewritten_files),
                'log_points_added': sum(f.get('log_points', 0) for f in rewritten_files),
                'project_summary': project_summary
            }
            
            return self.format_output(result)
            
        except Exception as e:
            return self.handle_error(e)
    
    def _create_test_repo_directory(self, repo_path: str) -> str:
        """创建测试repo目录"""
        try:
            repo_name = os.path.basename(repo_path.rstrip('/\\'))
            # 将测试repo保存到output目录下
            output_dir = os.path.join(os.getcwd(), 'output')
            os.makedirs(output_dir, exist_ok=True)
            test_repo_path = os.path.join(output_dir, f"{repo_name}{self.test_repo_suffix}")
            
            # 如果目录已存在，强制删除（包括虚拟环境等顽固目录）
            if os.path.exists(test_repo_path):
                self._force_remove_directory(test_repo_path)
            
            # 创建新目录
            os.makedirs(test_repo_path)
            logging.info(f"创建测试repo目录: {test_repo_path}")
            
            return test_repo_path
            
        except Exception as e:
            logging.error(f"创建测试repo目录失败: {str(e)}")
            raise
    
    def _force_remove_directory(self, dir_path: str):
        """强制删除目录，处理虚拟环境等顽固目录"""
        import stat
        import time
        
        def handle_remove_readonly(func, path, exc):
            """处理只读文件删除"""
            if os.path.exists(path):
                os.chmod(path, stat.S_IWRITE)
                func(path)
        
        try:
            # 首先尝试正常删除
            shutil.rmtree(dir_path)
            logging.info(f"成功删除目录: {dir_path}")
        except Exception as e1:
            logging.warning(f"正常删除失败: {e1}，尝试强制删除...")
            try:
                # 处理只读文件，然后重试
                shutil.rmtree(dir_path, onerror=handle_remove_readonly)
                logging.info(f"强制删除成功: {dir_path}")
            except Exception as e2:
                logging.warning(f"强制删除失败: {e2}，尝试系统命令...")
                try:
                    # 最后尝试使用系统命令
                    import subprocess
                    if os.name == 'nt':  # Windows
                        subprocess.run(['rmdir', '/s', '/q', dir_path], check=True, shell=True)
                    else:  # Unix/Linux/Mac
                        subprocess.run(['rm', '-rf', dir_path], check=True)
                    logging.info(f"系统命令删除成功: {dir_path}")
                except Exception as e3:
                    logging.error(f"所有删除方法都失败: {e3}")
                    # 如果还是失败，添加时间戳避免冲突
                    import random
                    timestamp = int(time.time())
                    new_name = f"{dir_path}_backup_{timestamp}_{random.randint(1000,9999)}"
                    os.rename(dir_path, new_name)
                    logging.warning(f"无法删除，已重命名为: {new_name}")
    
    def _rewrite_code_files(self, repo_path: str, test_repo_path: str, structure_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """重写代码文件（并发执行Python文件重写）"""
        try:
            rewritten_files = []
            python_files = []
            other_files = []
            
            log_checkpoint("扫描项目文件")
            
            # 遍历原repo的所有文件，分类收集
            for root, dirs, files in os.walk(repo_path):
                # 跳过常见的非代码目录
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env', '.git']]
                
                for file in files:
                    source_path = os.path.join(root, file)
                    rel_path = os.path.relpath(source_path, repo_path)
                    target_path = os.path.join(test_repo_path, rel_path)
                    
                    # 确保目标目录存在
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    if file.endswith('.py'):
                        # Python文件添加到并发处理列表
                        python_files.append((source_path, target_path, rel_path))
                    else:
                        # 其他文件先收集起来
                        other_files.append((source_path, target_path, rel_path, file))
            
            log_detailed("INFO", f"发现 {len(python_files)} 个Python文件，{len(other_files)} 个其他文件")
            
            # 1. 并发处理Python文件
            if python_files:
                log_checkpoint("开始并发重写Python文件")
                python_results = self._concurrent_rewrite_python_files(python_files, structure_analysis)
                rewritten_files.extend(python_results)
                log_checkpoint("Python文件并发重写完成", f"处理了 {len(python_results)} 个文件")
            
            # 2. 串行处理其他文件（简单复制操作无需并发）
            if other_files:
                log_checkpoint("处理其他文件")
                for source_path, target_path, rel_path, file in other_files:
                    if file in ['requirements.txt', 'setup.py', 'pyproject.toml', 'setup.cfg']:
                        # 配置文件直接复制并分析依赖
                        shutil.copy2(source_path, target_path)
                        self._extract_dependencies_from_config(source_path, file)
                        rewritten_files.append({
                            'original_path': rel_path,
                            'target_path': target_path,
                            'type': 'config',
                            'copied': True
                        })
                    else:
                        # 其他文件直接复制
                        shutil.copy2(source_path, target_path)
                        rewritten_files.append({
                            'original_path': rel_path,
                            'target_path': target_path,
                            'type': 'other',
                            'copied': True
                        })
                log_checkpoint("其他文件处理完成", f"处理了 {len(other_files)} 个文件")
            
            return rewritten_files
            
        except Exception as e:
            logging.error(f"重写代码文件失败: {str(e)}")
            raise
    
    def _concurrent_rewrite_python_files(self, python_files: List[tuple], structure_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """并发重写Python文件"""
        try:
            results = []
            total_files = len(python_files)
            
            # 控制并发数量，避免API限制
            # 从配置中获取并发数量，默认为15
            execution_config = self.config.get('execution', {})
            config_workers = execution_config.get('concurrent_workers', 15)
            max_workers = min(config_workers, total_files)
            log_detailed("INFO", f"使用 {max_workers} 个并发线程处理 {total_files} 个Python文件")
            
            start_time = time.time()
            
            def process_single_file(file_info):
                """处理单个Python文件的包装函数"""
                source_path, target_path, rel_path = file_info
                try:
                    log_detailed("DEBUG", f"开始处理: {rel_path}")
                    result = self._rewrite_python_file(source_path, target_path, structure_analysis)
                    log_detailed("DEBUG", f"完成处理: {rel_path}")
                    return result
                except Exception as e:
                    log_detailed("ERROR", f"处理文件失败 {rel_path}: {str(e)}")
                    # 返回错误结果，而不是抛出异常
                    return {
                        'original_path': rel_path,
                        'target_path': target_path,
                        'type': 'python',
                        'error': str(e)
                    }
            
            # 使用ThreadPoolExecutor进行并发处理
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="PyRewriter") as executor:
                # 提交所有任务
                future_to_file = {
                    executor.submit(process_single_file, file_info): file_info[2]  # file_info[2] 是 rel_path
                    for file_info in python_files
                }
                
                completed_count = 0
                
                # 收集结果
                for future in as_completed(future_to_file):
                    rel_path = future_to_file[future]
                    completed_count += 1
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # 显示进度
                        if completed_count % 5 == 0 or completed_count == total_files:
                            progress = (completed_count / total_files) * 100
                            elapsed = time.time() - start_time
                            log_detailed("INFO", f"进度: {completed_count}/{total_files} ({progress:.1f}%) - 用时: {elapsed:.1f}s")
                        
                    except Exception as e:
                        log_detailed("ERROR", f"获取结果失败 {rel_path}: {str(e)}")
                        results.append({
                            'original_path': rel_path,
                            'target_path': '',
                            'type': 'python',
                            'error': str(e)
                        })
            
            total_time = time.time() - start_time
            successful_count = len([r for r in results if 'error' not in r])
            failed_count = total_files - successful_count
            
            log_detailed("INFO", f"并发处理完成: 成功 {successful_count}，失败 {failed_count}，总用时 {total_time:.2f}s")
            
            # 🆕 聚合所有Python文件的依赖
            self._aggregate_dependencies_from_results(results)
            
            return results
            
        except Exception as e:
            logging.error(f"并发重写Python文件失败: {str(e)}")
            # 如果并发处理失败，回退到串行处理
            log_detailed("WARNING", "并发处理失败，回退到串行处理")
            return self._fallback_serial_rewrite(python_files, structure_analysis)
    
    def _fallback_serial_rewrite(self, python_files: List[tuple], structure_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """并发处理失败时的串行处理回退方案"""
        try:
            results = []
            for source_path, target_path, rel_path in python_files:
                try:
                    result = self._rewrite_python_file(source_path, target_path, structure_analysis)
                    results.append(result)
                except Exception as e:
                    logging.error(f"串行处理文件失败 {rel_path}: {str(e)}")
                    results.append({
                        'original_path': rel_path,
                        'target_path': target_path,
                        'type': 'python',
                        'error': str(e),
                        'dependencies': []  # 🆕 错误情况下的空依赖列表
                    })
            
            # 🆕 聚合串行处理的依赖
            self._aggregate_dependencies_from_results(results)
            
            return results
        except Exception as e:
            logging.error(f"串行回退处理失败: {str(e)}")
            return []
    
    def _rewrite_python_file(self, source_path: str, target_path: str, structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """重写Python文件，添加log和测试输入"""
        try:
            original_content = read_file_content(source_path)
            if not original_content:
                # 空文件直接复制
                shutil.copy2(source_path, target_path)
                return {
                    'original_path': os.path.relpath(source_path, os.path.dirname(source_path)),
                    'target_path': target_path,
                    'type': 'python',
                    'empty': True
                }
            
            # 🆕 使用新的LLM重写方法（JSON格式）
            llm_result = self._llm_rewrite_code_with_deps(original_content, source_path, structure_analysis)
            
            if llm_result.get('success'):
                # LLM成功返回JSON格式结果
                rewritten_content = llm_result['code']
                
                # 后处理：确保导入正确
                rewritten_content = self._post_process_imports(rewritten_content, source_path)
                
                # 写入重写后的内容
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(rewritten_content)
                
                # 分析重写结果
                log_points = self._count_log_points(rewritten_content)
                
                return {
                    'original_path': os.path.relpath(source_path, os.path.dirname(source_path)),
                    'target_path': target_path,
                    'type': 'python',
                    'rewritten': True,
                    'log_points': log_points,
                    'dependencies': llm_result.get('packages', []),  # 🎯 LLM直接提供的准确包名
                    'explanation': llm_result.get('explanation', ''),
                    'llm_method': True,  # 标记使用了LLM方法
                    'json_attempt': llm_result.get('attempt', 1),
                    'original_lines': len(original_content.split('\n')),
                    'rewritten_lines': len(rewritten_content.split('\n'))
                }
            else:
                # LLM JSON方法失败，回退到fallback增强
                logging.warning(f"LLM JSON方法失败，回退到基础增强: {source_path}")
                enhanced_content = self._add_basic_logging(original_content)
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_content)
                
                # 使用简化的静态分析作为fallback（保留作为兜底）
                fallback_dependencies = self._analyze_file_dependencies_fallback(original_content, enhanced_content)
                
                return {
                    'original_path': os.path.relpath(source_path, os.path.dirname(source_path)),
                    'target_path': target_path,
                    'type': 'python',
                    'fallback_enhanced': True,
                    'log_points': self._count_log_points(enhanced_content),
                    'dependencies': list(fallback_dependencies),  # fallback依赖分析
                    'llm_error': llm_result.get('error', 'Unknown'),
                    'llm_method': False
                }
                
        except Exception as e:
            logging.error(f"重写Python文件失败 {source_path}: {str(e)}")
            # 失败时直接复制原文件
            shutil.copy2(source_path, target_path)
            
            # 🆕 即使失败也尝试分析原文件的依赖（fallback）
            try:
                original_content = read_file_content(source_path) or ""
                error_dependencies = self._analyze_file_dependencies_fallback(original_content, original_content)
            except:
                error_dependencies = set()
            
            return {
                'original_path': os.path.relpath(source_path, os.path.dirname(source_path)),
                'target_path': target_path,
                'type': 'python',
                'error': str(e),
                'dependencies': list(error_dependencies)  # 🆕 错误情况下的依赖列表
            }
    
    def _llm_rewrite_code_with_deps(self, original_content: str, file_path: str, structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """使用LLM重写代码并同时提供依赖包信息"""
        try:
            # 获取文件在项目中的角色
            file_role = self._determine_file_role(file_path, structure_analysis)
            
            # 预处理：修复相对导入
            preprocessed_content = self._fix_relative_imports(original_content, file_path)
            
            messages = [
                self.create_system_message(
                    "Python代码重写专家",
                    "重写Python代码并以JSON格式返回结果，包含重写后的代码和所需的依赖包信息。"
                ),
                self.create_user_message(
                    f"""请重写以下Python代码，并以JSON格式返回结果。

重写要求：
1. **保持原有功能完全不变**
2. **修复导入问题**：将相对导入转换为绝对导入
3. **添加详细的日志记录**：
   - 在每个函数入口添加 logging.info(f"[LOG] ENTER function_name: {{参数信息}}")
   - 在关键变量赋值处添加 logging.info(f"[LOG] VARIABLE variable_name = {{value}}")
   - 在函数退出前添加 logging.info(f"[LOG] EXIT function_name: {{返回值信息}}")
   - 在异常处理处添加 logging.error(f"[LOG] ERROR in function_name: {{错误信息}}")
4. **添加必要的测试输入**：为主入口文件添加合理的测试数据
5. **如果适合，可以添加性能监控**（如psutil监控内存/CPU）

文件路径：{file_path}
文件角色：{file_role}

原始代码：
```python
{preprocessed_content}
```

请严格按照以下JSON格式返回：

```json
{{
    "rewritten_code": "重写后的完整Python代码（包含所有import语句）",
    "required_packages": ["package1", "package2"],
    "explanation": "重写说明和新增功能描述"
}}
```

重要说明：
- required_packages必须是准确的pip安装包名（不是import名）
- 例如：cv2对应opencv-python，PIL对应Pillow，sklearn对应scikit-learn
- 只包含第三方包，不要包含标准库（如os, sys, logging等）
- rewritten_code必须是完整可运行的代码
""")
            ]
            
            response = self.call_llm(messages, max_tokens=8000)
            
            # 使用JSON提取和重试机制
            return self._extract_json_with_retry(response, file_path)
                
        except Exception as e:
            logging.error(f"LLM重写代码失败 {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'fallback_response': None
            }
    
    def _fix_relative_imports(self, content: str, file_path: str) -> str:
        """修复相对导入路径"""
        try:
            lines = content.split('\n')
            fixed_lines = []
            
            for line in lines:
                # 处理相对导入
                if re.match(r'^\s*from\s+\.+', line):
                    # 匹配类似 "from ..models import xxx" 或 "from .utils import xxx" 的模式
                    match = re.match(r'(\s*)from\s+(\.+)(\w+(?:\.\w+)*)\s+import\s+(.+)', line)
                    if match:
                        indent, dots, module_path, imports = match.groups()
                        
                        # 根据点的数量和文件路径计算绝对导入路径
                        abs_import = self._convert_to_absolute_import(file_path, dots, module_path)
                        if abs_import:
                            fixed_line = f"{indent}from {abs_import} import {imports}"
                            fixed_lines.append(fixed_line)
                            logging.info(f"修复相对导入: {line.strip()} -> {fixed_line.strip()}")
                            continue
                
                # 保持其他行不变
                fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
            
        except Exception as e:
            logging.error(f"修复相对导入失败: {str(e)}")
            return content
    
    def _convert_to_absolute_import(self, file_path: str, dots: str, module_path: str) -> str:
        """将相对导入转换为绝对导入"""
        try:
            # 对于相对导入，优先使用基于src目录的简单映射
            if dots == '.':
                # 同级导入，如 from .models import xxx
                if 'src' in file_path:
                    return f"src.{module_path}" if module_path else "src"
                else:
                    return module_path if module_path else ""
            elif dots == '..':
                # 上级导入，如 from ..models import xxx
                if 'src' in file_path:
                    # 从src目录开始的绝对导入
                    return f"src.{module_path}" if module_path else "src"
                else:
                    # 对于非src目录的文件，直接使用模块路径
                    return module_path if module_path else ""
            else:
                # 多级上升，统一使用src前缀
                if module_path:
                    return f"src.{module_path}"
                else:
                    return "src"
            
        except Exception as e:
            logging.error(f"转换绝对导入失败: {str(e)}")
            return None
    
    def _post_process_imports(self, content: str, file_path: str) -> str:
        """后处理导入语句，确保正确性"""
        try:
            lines = content.split('\n')
            processed_lines = []
            has_sys_path = False
            import_section_ended = False
            in_multiline_import = False
            multiline_import_buffer = []
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # 检查是否已添加sys.path设置
                if 'sys.path.insert' in line:
                    has_sys_path = True
                
                # 检查是否开始多行导入
                if (line.strip().startswith('from ') or line.strip().startswith('import ')) and '(' in line and ')' not in line:
                    # 开始多行导入，收集所有相关行
                    in_multiline_import = True
                    multiline_import_buffer = [line]
                    i += 1
                    
                    # 继续收集多行import的所有行
                    while i < len(lines) and in_multiline_import:
                        current_line = lines[i]
                        multiline_import_buffer.append(current_line)
                        
                        if ')' in current_line:
                            in_multiline_import = False
                        i += 1
                    
                    # 将完整的多行import添加到处理后的行中
                    processed_lines.extend(multiline_import_buffer)
                    multiline_import_buffer = []
                    continue
                    
                processed_lines.append(line)
                
                # 检查导入区域是否结束（只有当不在多行导入中时才检查）
                if not in_multiline_import and not (line.startswith('import ') or line.startswith('from ') or line.strip() == '' or line.strip().startswith('#')):
                    import_section_ended = True
                
                # 在导入区域结束后添加sys.path设置（如果需要且还没有）
                if not has_sys_path and import_section_ended and 'src' in content:
                    # 在当前位置插入sys.path设置
                    processed_lines.insert(-1, '')  # 空行分隔
                    processed_lines.insert(-1, 'import os')
                    processed_lines.insert(-1, 'import sys')
                    processed_lines.insert(-1, "sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))")
                    processed_lines.insert(-1, '')  # 空行分隔
                    has_sys_path = True
                
                i += 1
            
            return '\n'.join(processed_lines)
            
        except Exception as e:
            logging.error(f"后处理导入失败: {str(e)}")
            return content

    def _determine_file_role(self, file_path: str, structure_analysis: Dict[str, Any]) -> str:
        """确定文件在项目中的角色"""
        try:
            filename = os.path.basename(file_path)
            rel_path = file_path
            
            # 检查是否是入口文件
            recommended_entry = structure_analysis.get('recommended_entry', '')
            if recommended_entry and recommended_entry in rel_path:
                return "主入口文件"
            
            # 检查常见角色
            if filename in ['main.py', 'run.py', 'app.py']:
                return "主入口文件"
            elif filename.startswith('test_'):
                return "测试文件"
            elif filename == '__init__.py':
                return "包初始化文件"
            elif 'config' in filename.lower():
                return "配置文件"
            elif 'util' in filename.lower() or 'helper' in filename.lower():
                return "工具文件"
            else:
                return "普通模块文件"
                
        except Exception as e:
            logging.debug(f"确定文件角色失败: {str(e)}")
            return "普通文件"
    
    def _add_basic_logging(self, content: str) -> str:
        """添加基础日志功能（作为LLM重写的后备方案）"""
        try:
            lines = content.split('\n')
            enhanced_lines = []
            
            # 添加logging导入
            import_added = False
            for i, line in enumerate(lines):
                enhanced_lines.append(line)
                
                # 在第一个import后添加logging
                if not import_added and (line.startswith('import ') or line.startswith('from ')):
                    if i + 1 >= len(lines) or not (lines[i + 1].startswith('import ') or lines[i + 1].startswith('from ')):
                        enhanced_lines.append('import logging')
                        enhanced_lines.append('logging.basicConfig(level=logging.INFO)')
                        import_added = True
                
                # 在函数定义后添加进入日志
                if line.strip().startswith('def ') and ':' in line:
                    func_name = line.split('def ')[1].split('(')[0]
                    indent = len(line) - len(line.lstrip())
                    enhanced_lines.append(' ' * (indent + 4) + f'logging.info(f"[LOG] ENTER {func_name}")')
            
            # 如果没有添加过logging导入，在开头添加
            if not import_added:
                enhanced_lines.insert(0, 'import logging')
                enhanced_lines.insert(1, 'logging.basicConfig(level=logging.INFO)')
            
            return '\n'.join(enhanced_lines)
            
        except Exception as e:
            logging.error(f"添加基础日志失败: {str(e)}")
            return content
    
    def _count_log_points(self, content: str) -> int:
        """统计日志点数量"""
        try:
            return len(re.findall(r'\[LOG\]', content))
        except:
            return 0
    
    def _extract_json_with_retry(self, response: str, file_path: str, max_retries: int = 3) -> Dict[str, Any]:
        """提取JSON内容，包含重试策略"""
        
        for attempt in range(max_retries):
            try:
                # 尝试多种JSON提取模式
                json_patterns = [
                    r'```json\s*(\{.*?\})\s*```',  # 标准代码块
                    r'```\s*(\{.*?\})\s*```',      # 简单代码块
                    r'(\{[^{}]*"rewritten_code"[^{}]*"required_packages"[^{}]*\})',  # 包含关键字段
                    r'(\{.*?"rewritten_code".*?"required_packages".*?\})',  # 更宽松匹配
                ]
                
                json_str = None
                for pattern in json_patterns:
                    json_match = re.search(pattern, response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        break
                
                if json_str:
                    # 清理JSON字符串
                    json_str = json_str.strip()
                    result = json.loads(json_str)
                    
                    # 验证必要字段
                    if 'rewritten_code' in result and 'required_packages' in result:
                        return {
                            'success': True,
                            'code': result['rewritten_code'],
                            'packages': result.get('required_packages', []),
                            'explanation': result.get('explanation', ''),
                            'attempt': attempt + 1
                        }
                
                # JSON提取失败，准备重试
                if attempt < max_retries - 1:
                    logging.warning(f"JSON提取失败，第{attempt+1}次重试: {file_path}")
                    
                    # 重新构造更严格的提示词进行重试
                    retry_prompt = f"""
上一次回复格式不正确。请严格按照以下JSON格式返回，不要添加任何其他内容：

```json
{{
    "rewritten_code": "完整的Python代码（包含import语句）",
    "required_packages": ["package1", "package2"],
    "explanation": "重写说明"
}}
```

要求：
1. required_packages必须是准确的pip安装包名
2. rewritten_code必须是完整可运行的Python代码
3. 必须严格遵循JSON格式

请重写文件: {file_path}
"""
                    response = self._call_llm(retry_prompt)
                
            except json.JSONDecodeError as e:
                logging.warning(f"JSON解析失败: {e}, 尝试{attempt+1}/{max_retries}")
                continue
            except Exception as e:
                logging.warning(f"JSON提取异常: {e}, 尝试{attempt+1}/{max_retries}")
                continue
        
        # 所有重试失败
        logging.error(f"JSON提取完全失败: {file_path}")
        return {
            'success': False,
            'error': 'JSON提取失败',
            'fallback_response': response,
            'attempts': max_retries
        }

    def _analyze_file_dependencies_fallback(self, original_content: str, rewritten_content: str) -> Set[str]:
        """fallback情况下的文件依赖分析（保留作为兜底方案）"""
        try:
            imports = set()
            
            # 简化的import提取
            for match in re.finditer(r'^import\s+([\w.]+)', rewritten_content, re.MULTILINE):
                package = match.group(1).split('.')[0]
                imports.add(package)
            
            for match in re.finditer(r'^from\s+([\w.]+)\s+import', rewritten_content, re.MULTILINE):
                package = match.group(1).split('.')[0]
                imports.add(package)
            
            # 基础的标准库过滤
            standard_libs = {
                'os', 'sys', 'logging', 'json', 'time', 'datetime', 'random',
                'math', 're', 'collections', 'itertools', 'functools', 'typing',
                'pathlib', 'subprocess', 'threading', 'multiprocessing', 'ast',
                'inspect', 'pickle', 'copy', 'csv', 'urllib', 'http', 'socket'
            }
            
            # 基础的包名映射
            package_mapping = {
                'cv2': 'opencv-python',
                'PIL': 'Pillow', 
                'sklearn': 'scikit-learn',
                'psutil': 'psutil'
            }
            
            third_party = set()
            for pkg in imports:
                if pkg not in standard_libs:
                    actual_pkg = package_mapping.get(pkg, pkg)
                    third_party.add(actual_pkg)
            
            return third_party
            
        except Exception as e:
            logging.debug(f"fallback依赖分析失败: {str(e)}")
            return set()
    
    def _extract_dependencies_from_config(self, file_path: str, filename: str):
        """从配置文件中提取依赖"""
        try:
            content = read_file_content(file_path)
            if not content:
                return
            
            if filename == 'requirements.txt':
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # 提取包名（去除版本号）
                        pkg_name = re.split(r'[>=<!=]', line)[0].strip()
                        self.dependencies.add(pkg_name)
            
            elif filename == 'setup.py':
                # 从setup.py中提取install_requires
                install_requires_match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if install_requires_match:
                    requirements = install_requires_match.group(1)
                    for req in re.findall(r'["\']([^"\']+)["\']', requirements):
                        pkg_name = re.split(r'[>=<!=]', req)[0].strip()
                        self.dependencies.add(pkg_name)
            
            # 添加标准依赖
            self.dependencies.update(['logging', 'os', 'sys'])
            
        except Exception as e:
            logging.error(f"提取依赖失败 {file_path}: {str(e)}")
    
    def _generate_requirements_file(self, test_repo_path: str) -> str:
        """生成requirements.txt文件"""
        try:
            requirements_path = os.path.join(test_repo_path, 'requirements.txt')
            
            # 标准库不需要安装，过滤掉
            standard_libs = {
                'os', 'sys', 'logging', 'json', 'time', 'datetime', 'random', 
                'math', 're', 'collections', 'itertools', 'functools', 'typing'
            }
            
            installable_deps = [dep for dep in self.dependencies if dep not in standard_libs]
            
            # 写入requirements文件
            with open(requirements_path, 'w', encoding='utf-8') as f:
                f.write("# Auto-generated requirements for sandbox testing\n")
                f.write("# 沙箱测试自动生成的依赖文件\n\n")
                for dep in sorted(installable_deps):
                    f.write(f"{dep}\n")
            
            logging.info(f"生成requirements文件: {requirements_path}")
            return requirements_path
            
        except Exception as e:
            logging.error(f"生成requirements文件失败: {str(e)}")
            return ""
    
    def _generate_project_summary(self, rewritten_files: List[Dict[str, Any]], structure_analysis: Dict[str, Any]) -> str:
        """生成项目摘要"""
        try:
            python_files = [f for f in rewritten_files if f.get('type') == 'python']
            total_log_points = sum(f.get('log_points', 0) for f in python_files)
            rewritten_count = len([f for f in python_files if f.get('rewritten')])
            
            summary = f"""
代码重写完成摘要：
- 总文件数：{len(rewritten_files)}
- Python文件数：{len(python_files)}
- 成功重写：{rewritten_count}个文件
- 添加日志点：{total_log_points}个
- 生成依赖：{len(self.dependencies)}个
- 推荐入口：{structure_analysis.get('recommended_entry', '未确定')}
"""
            return summary.strip()
            
        except Exception as e:
            logging.error(f"生成项目摘要失败: {str(e)}")
            return "摘要生成失败"

    def _aggregate_dependencies_from_results(self, results: List[Dict[str, Any]]):
        """从重写结果中聚合所有文件的依赖包"""
        try:
            total_deps_count = 0
            for result in results:
                if 'dependencies' in result and result['dependencies']:
                    file_deps = set(result['dependencies'])
                    self.dependencies.update(file_deps)
                    total_deps_count += len(file_deps)
                    
                    # 记录详细的依赖信息用于调试
                    file_path = result.get('original_path', 'unknown')
                    logging.debug(f"文件 {file_path} 的依赖: {file_deps}")
            
            logging.info(f"依赖聚合完成: 收集了 {total_deps_count} 个文件级依赖，去重后总计 {len(self.dependencies)} 个唯一包")
            
        except Exception as e:
            logging.error(f"聚合依赖失败: {str(e)}")
    
 