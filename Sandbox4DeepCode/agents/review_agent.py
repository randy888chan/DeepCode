#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审查代理
功能：分析沙箱完整输出，给出审查和代码迭代建议

"""

import logging
import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent


class ReviewAgent(BaseAgent):
    """审查代理 - 分析沙箱输出并提供代码迭代建议"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理审查请求
        
        Args:
            input_data: 包含structure_analysis, rewrite_result, execution_result的字典
            
        Returns:
            审查结果和代码迭代建议
        """
        try:
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("输入数据无效"))
            
            structure_analysis = input_data['structure_analysis']
            rewrite_result = input_data['rewrite_result']
            execution_result = input_data['execution_result']
            
            # 分析沙箱输出
            output_analysis = self._analyze_sandbox_output(execution_result)
            
            # 分析代码质量
            code_quality_analysis = self._analyze_code_quality(structure_analysis, rewrite_result)
            
            # 分析执行性能
            performance_analysis = self._analyze_execution_performance(execution_result)
            
            # 生成代码迭代建议
            iteration_suggestions = self._generate_iteration_suggestions(
                structure_analysis, rewrite_result, execution_result, output_analysis
            )
            
            # 生成综合审查报告
            comprehensive_review = self._generate_comprehensive_review(
                output_analysis, code_quality_analysis, performance_analysis, iteration_suggestions,
                structure_analysis, execution_result
            )
            
            result = {
                'output_analysis': output_analysis,
                'code_quality_analysis': code_quality_analysis,
                'performance_analysis': performance_analysis,
                'iteration_suggestions': iteration_suggestions,
                'comprehensive_review': comprehensive_review,
                'review_summary': self._generate_review_summary(iteration_suggestions)
            }
            
            return self.format_output(result)
            
        except Exception as e:
            return self.handle_error(e)
    
    def _analyze_sandbox_output(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """分析沙箱输出"""
        try:
            stdout_lines = execution_result.get('stdout', [])
            stderr_lines = execution_result.get('stderr', [])
            
            # 分析日志输出
            log_analysis = self._parse_log_output(stdout_lines)
            
            # 分析错误输出
            error_analysis = self._parse_error_output(stderr_lines)
            
            # 分析函数执行流程
            execution_flow = self._analyze_execution_flow(log_analysis.get('function_calls', []))
            
            return {
                'log_analysis': log_analysis,
                'error_analysis': error_analysis,
                'execution_flow': execution_flow,
                'output_quality': self._assess_output_quality(log_analysis, error_analysis),
                'total_log_entries': len([line for line in stdout_lines if '[LOG]' in line]),
                'execution_successful': execution_result.get('return_code') == 0
            }
            
        except Exception as e:
            logging.error(f"分析沙箱输出失败: {str(e)}")
            return {}
    
    def _analyze_code_quality(self, structure_analysis: Dict[str, Any], rewrite_result: Dict[str, Any]) -> Dict[str, Any]:
        """分析代码质量"""
        try:
            # 统计重写文件信息
            rewritten_files = rewrite_result.get('rewritten_files', [])
            python_files = [f for f in rewritten_files if f.get('type') == 'python']
            successful_rewrites = [f for f in python_files if f.get('rewritten')]
            
            # 计算质量指标
            rewrite_success_rate = len(successful_rewrites) / len(python_files) if python_files else 0
            total_log_points = sum(f.get('log_points', 0) for f in python_files)
            avg_log_points_per_file = total_log_points / len(python_files) if python_files else 0
            
            # 评估项目结构
            project_structure_score = self._evaluate_project_structure(structure_analysis)
            
            return {
                'rewrite_success_rate': rewrite_success_rate,
                'total_files_processed': len(rewritten_files),
                'python_files_count': len(python_files),
                'successful_rewrites': len(successful_rewrites),
                'total_log_points': total_log_points,
                'avg_log_points_per_file': avg_log_points_per_file,
                'project_structure_score': project_structure_score,
                'quality_score': self._calculate_overall_quality_score(rewrite_success_rate, project_structure_score)
            }
            
        except Exception as e:
            logging.error(f"分析代码质量失败: {str(e)}")
            return {}
    
    def _analyze_execution_performance(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """分析执行性能"""
        try:
            execution_time = execution_result.get('execution_time', 0)
            max_memory = execution_result.get('max_memory_mb', 0)
            return_code = execution_result.get('return_code', -1)
            timeout_reached = execution_result.get('timeout_reached', False)
            memory_exceeded = execution_result.get('memory_exceeded', False)
            
            # 性能评级
            performance_rating = self._rate_performance(execution_time, max_memory, return_code)
            
            # 性能问题识别
            performance_issues = []
            if timeout_reached:
                performance_issues.append("执行超时")
            if memory_exceeded:
                performance_issues.append("内存使用超限")
            if return_code != 0:
                performance_issues.append("执行失败")
            if execution_time > 60:
                performance_issues.append("执行时间过长")
            if max_memory > 1000:
                performance_issues.append("内存使用过高")
            
            return {
                'execution_time': execution_time,
                'max_memory_mb': max_memory,
                'return_code': return_code,
                'timeout_reached': timeout_reached,
                'memory_exceeded': memory_exceeded,
                'performance_rating': performance_rating,
                'performance_issues': performance_issues,
                'execution_status': 'success' if return_code == 0 else 'failed'
            }
            
        except Exception as e:
            logging.error(f"分析执行性能失败: {str(e)}")
            return {}
    
    def _generate_iteration_suggestions(self, structure_analysis: Dict[str, Any], 
                                      rewrite_result: Dict[str, Any], execution_result: Dict[str, Any],
                                      output_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成代码迭代建议"""
        try:
            messages = [
                self.create_system_message(
                    "代码迭代顾问",
                    "基于沙箱测试结果分析项目质量，提供具体的代码迭代建议和改进方案，帮助开发者提升代码质量和项目架构。"
                ),
                self.create_user_message(
                    f"""请基于以下沙箱测试结果提供详细的代码迭代建议：

项目结构分析：
- 项目类型：{structure_analysis.get('project_structure', {}).get('project_type', '未知')}
- 总文件数：{structure_analysis.get('project_structure', {}).get('stats', {}).get('total_files', 0)}
- Python文件数：{structure_analysis.get('project_structure', {}).get('stats', {}).get('python_files', 0)}
- 推荐入口：{structure_analysis.get('recommended_entry', '未确定')}

代码重写结果：
- 处理文件数：{rewrite_result.get('total_files', 0)}
- 添加日志点：{rewrite_result.get('log_points_added', 0)}
- 生成依赖数：{len(rewrite_result.get('dependencies', []))}

执行性能：
- 执行状态：{'成功' if execution_result.get('return_code') == 0 else '失败'}
- 执行时间：{execution_result.get('execution_time', 0):.2f}秒
- 内存使用：{execution_result.get('max_memory_mb', 0):.1f}MB
- 超时：{'是' if execution_result.get('timeout_reached') else '否'}

日志分析：
- 日志条目：{output_analysis.get('total_log_entries', 0)}
- 函数调用：{len(output_analysis.get('execution_flow', []))}
- 错误数量：{len(output_analysis.get('error_analysis', {}).get('errors', []))}

请提供以下方面的具体建议：
1. 代码架构优化建议
2. 性能提升方案
3. 代码质量改进措施
4. 测试策略完善
5. 依赖管理优化
6. 错误处理改进
7. 日志记录最佳实践
8. 下一步开发优先级

请以中文回答，提供具体可执行的建议。
""")
            ]
            
            suggestions_response = self.call_llm(messages, max_tokens=6000)
            
            if suggestions_response:
                # 解析建议并结构化
                structured_suggestions = self._structure_iteration_suggestions(suggestions_response)
                return structured_suggestions
            else:
                return self._generate_fallback_suggestions(structure_analysis, rewrite_result, execution_result)
                
        except Exception as e:
            logging.error(f"生成迭代建议失败: {str(e)}")
            return self._generate_fallback_suggestions(structure_analysis, rewrite_result, execution_result)
    
    def _generate_comprehensive_review(self, output_analysis: Dict[str, Any], 
                                     code_quality_analysis: Dict[str, Any],
                                     performance_analysis: Dict[str, Any], 
                                     iteration_suggestions: Dict[str, Any],
                                     structure_analysis: Dict[str, Any] = None,
                                     execution_result: Dict[str, Any] = None) -> str:
        """生成综合审查报告"""
        try:
            # 准备项目结构信息
            project_info = self._extract_project_info(structure_analysis) if structure_analysis else {}
            
            # 准备沙箱输出原文
            sandbox_output = self._extract_sandbox_output(execution_result) if execution_result else {}
            
            messages = [
                self.create_system_message(
                    "项目审查专家",
                    "基于沙箱测试结果生成全面的项目审查报告，包括项目结构分析、沙箱输出分析和具体修改建议。"
                ),
                self.create_user_message(
                    f"""请基于以下信息生成详细的项目审查报告：

## 项目结构信息
{project_info}

## 沙箱输出原文（包含详细错误分析）
{sandbox_output}

## 分析结果
输出分析：
- 日志条目：{output_analysis.get('total_log_entries', 0)}
- 执行成功：{'是' if output_analysis.get('execution_successful') else '否'}
- 输出质量：{output_analysis.get('output_quality', '未评估')}

代码质量分析：
- 重写成功率：{code_quality_analysis.get('rewrite_success_rate', 0):.1%}
- 处理文件数：{code_quality_analysis.get('total_files_processed', 0)}
- 日志点数：{code_quality_analysis.get('total_log_points', 0)}
- 质量评分：{code_quality_analysis.get('quality_score', 0):.1f}

性能分析：
- 执行状态：{performance_analysis.get('execution_status', '未知')}
- 执行时间：{performance_analysis.get('execution_time', 0):.2f}秒
- 性能评级：{performance_analysis.get('performance_rating', '未评级')}
- 性能问题：{', '.join(performance_analysis.get('performance_issues', []))}

请生成包含以下内容的详细审查报告：

## 1. 项目概述
- 项目类型和主要功能
- 技术栈和架构特点
- 代码组织方式

## 2. 沙箱执行分析
- 执行流程和关键步骤
- 成功和失败的部分
- 性能表现评估

## 3. 问题诊断
- 主要错误和异常（必须包含沙箱输出中的详细错误分析）
- 代码质量问题
- 架构和设计问题

## 4. 具体修改建议
- 针对每个具体错误的修复方案（必须基于沙箱输出中的错误位置和类型）
- 代码质量改进建议
- 性能优化建议
- 架构重构建议

## 5. 优先级排序
- 高优先级问题（必须修复）
- 中优先级问题（建议修复）
- 低优先级问题（可选优化）

## 6. 实施计划
- 短期修复计划（1-2周）
- 中期改进计划（1-2月）
- 长期优化计划（3-6月）

**重要要求：**
1. 必须在"问题诊断"部分详细分析沙箱输出中的每个错误，包括错误类型、文件位置、行号、函数名等
2. 必须在"具体修改建议"部分为每个错误提供具体的修复代码或解决方案
3. 确保所有建议都是基于沙箱输出的实际错误信息，而不是泛泛而谈
4. 如果沙箱输出包含"详细错误分析"和"解决方案建议"部分，请充分利用这些信息

请以专业、客观的语气撰写，提供具体可执行的建议，并针对沙箱输出中的具体错误给出明确的修复方案。
""")
            ]
            
            review_report = self.call_llm(messages, max_tokens=6000)
            return review_report or "综合审查报告生成失败"
            
        except Exception as e:
            logging.error(f"生成综合审查报告失败: {str(e)}")
            return "综合审查报告生成失败"
    
    def _extract_project_info(self, structure_analysis: Dict[str, Any]) -> str:
        """提取项目结构信息"""
        try:
            project_structure = structure_analysis.get('project_structure', {})
            project_type = project_structure.get('project_type', '未知项目')
            total_files = project_structure.get('total_files', 0)
            main_files = project_structure.get('main_files', [])
            entry_points = project_structure.get('entry_points', [])
            
            # 构建项目结构树
            structure_tree = self._build_structure_tree(structure_analysis)
            
            info = f"""
### 项目基本信息
- **项目类型**: {project_type}
- **文件总数**: {total_files}
- **主要文件**: {', '.join(main_files[:5])}{'...' if len(main_files) > 5 else ''}
- **入口点**: {', '.join(entry_points)}

### 项目结构
```
{structure_tree}
```

### 技术栈分析
{self._analyze_tech_stack(structure_analysis)}
"""
            return info
            
        except Exception as e:
            logging.error(f"提取项目信息失败: {str(e)}")
            return "项目信息提取失败"
    
    def _extract_sandbox_output(self, execution_result: Dict[str, Any]) -> str:
        """提取沙箱输出原文，包含详细错误分析"""
        try:
            stdout = execution_result.get('stdout', '')
            stderr = execution_result.get('stderr', '')
            return_code = execution_result.get('return_code', -1)
            execution_time = execution_result.get('execution_time', 0)
            
            # 分析错误输出
            if isinstance(stderr, list):
                stderr_lines = stderr
            else:
                stderr_lines = stderr.split('\n') if stderr else []
            error_analysis = self._parse_error_output(stderr_lines)
            detailed_errors = error_analysis.get('detailed_errors', [])
            
            # 构建详细错误报告
            error_report = ""
            if detailed_errors:
                error_report = "\n### 详细错误分析\n"
                for i, error in enumerate(detailed_errors[:10], 1):  # 限制显示前10个错误
                    error_report += f"\n#### 错误 {i}\n"
                    error_report += f"- **错误类型**: {error.get('error_type', 'Unknown')}\n"
                    error_report += f"- **错误消息**: {error.get('error_message', '无详细信息')}\n"
                    if error.get('file_path'):
                        error_report += f"- **文件路径**: {error.get('file_path')}\n"
                    if error.get('line_in_file'):
                        error_report += f"- **行号**: {error.get('line_in_file')}\n"
                    if error.get('function_name'):
                        error_report += f"- **函数名**: {error.get('function_name')}\n"
                    error_report += f"- **原始输出**: `{error.get('raw_line', '')}`\n"
            
            # 构建解决方案建议
            solutions = self._generate_error_solutions(detailed_errors)
            
            output = f"""
### 执行结果
- **返回码**: {return_code}
- **执行时间**: {execution_time:.2f}秒
- **执行状态**: {'成功' if return_code == 0 else '失败'}

### 标准输出 (stdout)
```
{stdout[:2000]}{'...' if len(stdout) > 2000 else ''}
```

### 错误输出 (stderr)
```
{stderr[:2000]}{'...' if len(stderr) > 2000 else ''}
```

{error_report}

### 解决方案建议
{solutions}
"""
            return output
            
        except Exception as e:
            logging.error(f"提取沙箱输出失败: {str(e)}")
            return "沙箱输出提取失败"
    
    def _generate_error_solutions(self, detailed_errors: List[Dict[str, Any]]) -> str:
        """根据详细错误生成解决方案"""
        try:
            if not detailed_errors:
                return "无错误需要解决。"
            
            solutions = []
            for error in detailed_errors:
                error_type = error.get('error_type')
                error_message = error.get('error_message', '')
                file_path = error.get('file_path', '')
                line_in_file = error.get('line_in_file')
                function_name = error.get('function_name', '')
                
                solution = f"\n#### 解决 {error_type} 错误\n"
                solution += f"- **位置**: {file_path}:{line_in_file}" + (f" (函数: {function_name})" if function_name else "") + "\n"
                solution += f"- **问题**: {error_message}\n"
                solution += f"- **解决方案**: "
                
                if error_type == 'NameError':
                    if 'setup_datasets' in error_message:
                        solution += "需要定义 `setup_datasets` 函数或导入相关模块。检查数据加载相关的导入语句。"
                    elif 'is not defined' in error_message:
                        solution += "变量未定义，需要检查变量名拼写或确保在使用前已定义。"
                elif error_type == 'ImportError':
                    solution += "模块导入失败，需要安装缺失的包或检查导入路径。"
                elif error_type == 'SyntaxError':
                    solution += "语法错误，检查代码语法，特别是括号、引号、缩进等。"
                elif error_type == 'IndentationError':
                    solution += "缩进错误，检查代码缩进是否一致。"
                elif error_type == 'AttributeError':
                    solution += "属性错误，检查对象是否有该属性或方法。"
                elif error_type == 'TypeError':
                    solution += "类型错误，检查数据类型是否匹配。"
                elif error_type == 'ValueError':
                    solution += "值错误，检查输入值是否有效。"
                elif error_type == 'KeyError':
                    solution += "键错误，检查字典中是否存在该键。"
                elif error_type == 'IndexError':
                    solution += "索引错误，检查列表或数组索引是否越界。"
                elif error_type == 'FileNotFoundError':
                    solution += "文件不存在，检查文件路径是否正确。"
                elif error_type == 'PermissionError':
                    solution += "权限错误，检查文件或目录的访问权限。"
                elif error_type == 'TimeoutError':
                    solution += "超时错误，检查网络连接或增加超时时间。"
                elif error_type == 'MemoryError':
                    solution += "内存不足，优化内存使用或增加系统内存。"
                elif error_type == 'RecursionError':
                    solution += "递归错误，检查递归深度或改为迭代实现。"
                elif error_type == 'AssertionError':
                    solution += "断言失败，检查断言条件是否正确。"
                elif error_type == 'NotImplementedError':
                    solution += "功能未实现，需要实现该功能或使用替代方案。"
                elif error_type == 'OSError':
                    solution += "操作系统错误，检查系统资源或权限。"
                elif error_type == 'RuntimeError':
                    solution += "运行时错误，检查程序逻辑或异常处理。"
                else:
                    solution += "一般异常，检查代码逻辑和异常处理。"
                
                solution += f"\n- **修复建议**: 在 {file_path} 文件的第 {line_in_file} 行附近进行修复。"
                solutions.append(solution)
            
            return "\n".join(solutions)
            
        except Exception as e:
            logging.error(f"生成错误解决方案失败: {str(e)}")
            return "错误解决方案生成失败"
    
    def _build_structure_tree(self, structure_analysis: Dict[str, Any]) -> str:
        """构建项目结构树"""
        try:
            file_structure = structure_analysis.get('file_structure', {})
            tree_lines = []
            
            def build_tree(data, prefix="", is_last=True):
                if isinstance(data, dict):
                    for i, (key, value) in enumerate(data.items()):
                        is_last_item = i == len(data) - 1
                        current_prefix = "└── " if is_last_item else "├── "
                        tree_lines.append(f"{prefix}{current_prefix}{key}")
                        
                        if isinstance(value, (dict, list)) and value:
                            next_prefix = prefix + ("    " if is_last_item else "│   ")
                            build_tree(value, next_prefix, is_last_item)
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        is_last_item = i == len(data) - 1
                        current_prefix = "└── " if is_last_item else "├── "
                        tree_lines.append(f"{prefix}{current_prefix}{item}")
            
            build_tree(file_structure)
            return "\n".join(tree_lines[:50])  # 限制行数
            
        except Exception as e:
            logging.error(f"构建结构树失败: {str(e)}")
            return "结构树构建失败"
    
    def _analyze_tech_stack(self, structure_analysis: Dict[str, Any]) -> str:
        """分析技术栈"""
        try:
            # 基于文件扩展名和内容分析技术栈
            tech_stack = []
            
            # 检查Python相关
            if structure_analysis.get('project_structure', {}).get('python_files', 0) > 0:
                tech_stack.append("Python")
            
            # 检查配置文件
            config_files = structure_analysis.get('project_structure', {}).get('config_files', [])
            if any('yaml' in f.lower() or 'yml' in f.lower() for f in config_files):
                tech_stack.append("YAML配置")
            if any('json' in f.lower() for f in config_files):
                tech_stack.append("JSON配置")
            
            # 检查依赖文件
            if structure_analysis.get('project_structure', {}).get('requirements_files', []):
                tech_stack.append("pip依赖管理")
            
            return f"**技术栈**: {', '.join(tech_stack)}" if tech_stack else "**技术栈**: 未识别"
                
        except Exception as e:
            logging.error(f"分析技术栈失败: {str(e)}")
            return "**技术栈**: 分析失败"
    
    def _parse_log_output(self, stdout_lines: List[str]) -> Dict[str, Any]:
        """解析日志输出"""
        try:
            log_entries = []
            function_calls = []
            variable_logs = []
            error_logs = []
            
            for line in stdout_lines:
                if '[LOG]' in line:
                    log_entries.append(line.strip())
                    
                    if 'ENTER' in line:
                        function_calls.append(line.strip())
                    elif 'VARIABLE' in line:
                        variable_logs.append(line.strip())
                    elif 'ERROR' in line:
                        error_logs.append(line.strip())
            
            return {
                'total_entries': len(log_entries),
                'function_calls': function_calls,
                'variable_logs': variable_logs,
                'error_logs': error_logs,
                'log_entries': log_entries[:50]  # 限制返回数量
            }
            
        except Exception as e:
            logging.error(f"解析日志输出失败: {str(e)}")
            return {}
    
    def _parse_error_output(self, stderr_lines: List[str]) -> Dict[str, Any]:
        """解析错误输出，精确定位问题位置"""
        try:
            errors = []
            warnings = []
            exceptions = []
            detailed_errors = []
            
            for i, line in enumerate(stderr_lines):
                line_stripped = line.strip()
                line_upper = line_stripped.upper()
                
                # 分类错误类型
                if 'ERROR' in line_upper or 'EXCEPTION' in line_upper:
                    errors.append(line_stripped)
                    if 'Exception' in line_stripped or 'Error' in line_stripped:
                        exceptions.append(line_stripped)
                
                elif 'WARNING' in line_upper:
                    warnings.append(line_stripped)
                
                # 详细错误分析
                error_info = self._extract_error_details(line_stripped, i + 1)
                if error_info:
                    detailed_errors.append(error_info)
            
            return {
                'total_errors': len(errors),
                'total_warnings': len(warnings),
                'total_exceptions': len(exceptions),
                'errors': errors[:10],
                'warnings': warnings[:10],
                'exceptions': exceptions[:5],
                'detailed_errors': detailed_errors,
                'raw_stderr': stderr_lines  # 保留原始输出
            }
            
        except Exception as e:
            logging.error(f"解析错误输出失败: {str(e)}")
            return {}
    
    def _extract_error_details(self, line: str, line_number: int) -> Dict[str, Any]:
        """提取错误详细信息"""
        try:
            error_info = {
                'line_number': line_number,
                'raw_line': line,
                'file_path': None,
                'function_name': None,
                'error_type': None,
                'error_message': None,
                'line_in_file': None
            }
            
            # 解析文件路径和行号 (例如: "File "/path/to/file.py", line 123")
            import re
            
            # 匹配文件路径模式
            file_pattern = r'File\s+"([^"]+)"'
            file_match = re.search(file_pattern, line)
            if file_match:
                error_info['file_path'] = file_match.group(1)
            
            # 匹配行号模式
            line_pattern = r'line\s+(\d+)'
            line_match = re.search(line_pattern, line)
            if line_match:
                error_info['line_in_file'] = int(line_match.group(1))
            
            # 匹配函数名模式 (例如: "in function_name")
            func_pattern = r'in\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            func_match = re.search(func_pattern, line)
            if func_match:
                error_info['function_name'] = func_match.group(1)
            
            # 匹配错误类型和消息
            if 'NameError' in line:
                error_info['error_type'] = 'NameError'
                name_match = re.search(r"name\s+'([^']+)'\s+is\s+not\s+defined", line)
                if name_match:
                    error_info['error_message'] = f"未定义的变量: {name_match.group(1)}"
            elif 'ImportError' in line:
                error_info['error_type'] = 'ImportError'
                import_match = re.search(r"No\s+module\s+named\s+'([^']+)'", line)
                if import_match:
                    error_info['error_message'] = f"缺少模块: {import_match.group(1)}"
            elif 'SyntaxError' in line:
                error_info['error_type'] = 'SyntaxError'
                error_info['error_message'] = "语法错误"
            elif 'IndentationError' in line:
                error_info['error_type'] = 'IndentationError'
                error_info['error_message'] = "缩进错误"
            elif 'AttributeError' in line:
                error_info['error_type'] = 'AttributeError'
                attr_match = re.search(r"'([^']+)'\s+object\s+has\s+no\s+attribute\s+'([^']+)'", line)
                if attr_match:
                    error_info['error_message'] = f"对象 {attr_match.group(1)} 没有属性 {attr_match.group(2)}"
            elif 'TypeError' in line:
                error_info['error_type'] = 'TypeError'
                error_info['error_message'] = "类型错误"
            elif 'ValueError' in line:
                error_info['error_type'] = 'ValueError'
                error_info['error_message'] = "值错误"
            elif 'KeyError' in line:
                error_info['error_type'] = 'KeyError'
                key_match = re.search(r"'([^']+)'", line)
                if key_match:
                    error_info['error_message'] = f"缺少键: {key_match.group(1)}"
            elif 'IndexError' in line:
                error_info['error_type'] = 'IndexError'
                error_info['error_message'] = "索引错误"
            elif 'ZeroDivisionError' in line:
                error_info['error_type'] = 'ZeroDivisionError'
                error_info['error_message'] = "除零错误"
            elif 'FileNotFoundError' in line:
                error_info['error_type'] = 'FileNotFoundError'
                file_match = re.search(r"'([^']+)'", line)
                if file_match:
                    error_info['error_message'] = f"文件不存在: {file_match.group(1)}"
            elif 'PermissionError' in line:
                error_info['error_type'] = 'PermissionError'
                error_info['error_message'] = "权限错误"
            elif 'TimeoutError' in line:
                error_info['error_type'] = 'TimeoutError'
                error_info['error_message'] = "超时错误"
            elif 'MemoryError' in line:
                error_info['error_type'] = 'MemoryError'
                error_info['error_message'] = "内存不足"
            elif 'RecursionError' in line:
                error_info['error_type'] = 'RecursionError'
                error_info['error_message'] = "递归错误"
            elif 'AssertionError' in line:
                error_info['error_type'] = 'AssertionError'
                error_info['error_message'] = "断言失败"
            elif 'NotImplementedError' in line:
                error_info['error_type'] = 'NotImplementedError'
                error_info['error_message'] = "功能未实现"
            elif 'OSError' in line:
                error_info['error_type'] = 'OSError'
                error_info['error_message'] = "操作系统错误"
            elif 'RuntimeError' in line:
                error_info['error_type'] = 'RuntimeError'
                error_info['error_message'] = "运行时错误"
            elif 'Exception' in line:
                error_info['error_type'] = 'Exception'
                error_info['error_message'] = "一般异常"
            
            # 如果没有识别到具体错误类型，但包含错误信息
            if not error_info['error_type'] and ('error' in line.lower() or 'exception' in line.lower()):
                error_info['error_type'] = 'Unknown'
                error_info['error_message'] = line
            
            return error_info if error_info['error_type'] else None
            
        except Exception as e:
            logging.error(f"提取错误详情失败: {str(e)}")
            return None
    
    def _analyze_execution_flow(self, function_calls: List[str]) -> List[str]:
        """分析执行流程"""
        try:
            flow = []
            for call in function_calls[:20]:  # 限制数量
                # 提取函数名
                if 'ENTER' in call:
                    try:
                        func_name = call.split('ENTER')[1].split(':')[0].strip()
                        flow.append(func_name)
                    except:
                        flow.append('unknown_function')
            
            return flow
            
        except Exception as e:
            logging.error(f"分析执行流程失败: {str(e)}")
            return []
    
    def _assess_output_quality(self, log_analysis: Dict[str, Any], error_analysis: Dict[str, Any]) -> str:
        """评估输出质量"""
        try:
            total_logs = log_analysis.get('total_entries', 0)
            total_errors = error_analysis.get('total_errors', 0)
            total_exceptions = error_analysis.get('total_exceptions', 0)
            
            if total_exceptions > 0:
                return "差"
            elif total_errors > 5:
                return "较差"
            elif total_logs < 10:
                return "较少"
            elif total_logs > 50:
                return "丰富"
            else:
                return "正常"
                
        except Exception as e:
            logging.error(f"评估输出质量失败: {str(e)}")
            return "未知"
    
    def _evaluate_project_structure(self, structure_analysis: Dict[str, Any]) -> float:
        """评估项目结构得分"""
        try:
            score = 5.0  # 基础分
            
            project_structure = structure_analysis.get('project_structure', {})
            stats = project_structure.get('stats', {})
            
            # 根据文件数量调整
            python_files = stats.get('python_files', 0)
            if python_files > 0:
                score += 1.0
            if python_files > 5:
                score += 1.0
            
            # 根据项目类型调整
            project_type = project_structure.get('project_type', '')
            if 'Django' in project_type or 'Flask' in project_type:
                score += 1.0
            elif '应用项目' in project_type:
                score += 0.5
            
            # 根据入口文件调整
            if structure_analysis.get('recommended_entry'):
                score += 1.0
            
            return min(10.0, score)
            
        except Exception as e:
            logging.error(f"评估项目结构失败: {str(e)}")
            return 5.0
    
    def _calculate_overall_quality_score(self, rewrite_success_rate: float, project_structure_score: float) -> float:
        """计算整体质量得分"""
        try:
            # 加权平均
            quality_score = (rewrite_success_rate * 10 * 0.6) + (project_structure_score * 0.4)
            return min(10.0, quality_score)
            
        except Exception as e:
            logging.error(f"计算质量得分失败: {str(e)}")
            return 5.0
    
    def _rate_performance(self, execution_time: float, max_memory: float, return_code: int) -> str:
        """评级性能"""
        try:
            if return_code != 0:
                return "失败"
            elif execution_time > 120:
                return "很慢"
            elif execution_time > 60:
                return "较慢"
            elif max_memory > 1000:
                return "内存高"
            elif execution_time < 10 and max_memory < 100:
                return "优秀"
            else:
                return "正常"
                
        except Exception as e:
            logging.error(f"评级性能失败: {str(e)}")
            return "未知"
    
    def _structure_iteration_suggestions(self, suggestions_response: str) -> Dict[str, Any]:
        """结构化迭代建议"""
        try:
            # 解析LLM响应，提取关键建议
            suggestions = {
                'architecture_improvements': [],
                'performance_optimizations': [],
                'code_quality_improvements': [],
                'testing_enhancements': [],
                'dependency_management': [],
                'error_handling_improvements': [],
                'logging_best_practices': [],
                'development_priorities': []
            }
            
            # 简单的关键词提取和分类
            lines = suggestions_response.split('\n')
            current_category = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 识别类别
                if '架构' in line or '结构' in line:
                    current_category = 'architecture_improvements'
                elif '性能' in line:
                    current_category = 'performance_optimizations'
                elif '质量' in line:
                    current_category = 'code_quality_improvements'
                elif '测试' in line:
                    current_category = 'testing_enhancements'
                elif '依赖' in line:
                    current_category = 'dependency_management'
                elif '错误' in line or '异常' in line:
                    current_category = 'error_handling_improvements'
                elif '日志' in line:
                    current_category = 'logging_best_practices'
                elif '优先级' in line or '下一步' in line:
                    current_category = 'development_priorities'
                
                # 添加建议到对应类别
                if current_category and line.startswith(('-', '•', '1.', '2.', '3.', '4.', '5.')):
                    clean_line = line.lstrip('-•123456789. ').strip()
                    if clean_line:
                        suggestions[current_category].append(clean_line)
            
            # 如果没有提取到建议，添加原始响应
            if not any(suggestions.values()):
                suggestions['general_suggestions'] = [suggestions_response[:500]]
            
            return suggestions
            
        except Exception as e:
            logging.error(f"结构化迭代建议失败: {str(e)}")
            return {'general_suggestions': [suggestions_response[:500] if suggestions_response else "建议解析失败"]}
    
    def _generate_fallback_suggestions(self, structure_analysis: Dict[str, Any], 
                                     rewrite_result: Dict[str, Any], 
                                     execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成后备建议"""
        try:
            suggestions = {
                'architecture_improvements': [],
                'performance_optimizations': [],
                'code_quality_improvements': [],
                'testing_enhancements': [],
                'dependency_management': [],
                'error_handling_improvements': [],
                'logging_best_practices': [],
                'development_priorities': []
            }
            
            # 基于执行结果生成基础建议
            if execution_result.get('return_code') != 0:
                suggestions['error_handling_improvements'].append("修复代码执行错误")
                suggestions['development_priorities'].append("优先解决执行失败问题")
            
            if execution_result.get('timeout_reached'):
                suggestions['performance_optimizations'].append("优化算法复杂度，减少执行时间")
                suggestions['development_priorities'].append("解决执行超时问题")
            
            if execution_result.get('max_memory_mb', 0) > 500:
                suggestions['performance_optimizations'].append("优化内存使用，减少内存占用")
            
            # 基于重写结果生成建议
            rewrite_success_rate = rewrite_result.get('log_points_added', 0) / max(rewrite_result.get('total_files', 1), 1)
            if rewrite_success_rate < 0.5:
                suggestions['code_quality_improvements'].append("提高代码重写成功率")
                suggestions['logging_best_practices'].append("增加更多日志记录点")
            
            return suggestions
            
        except Exception as e:
            logging.error(f"生成后备建议失败: {str(e)}")
            return {}
    
    def _generate_review_summary(self, iteration_suggestions: Dict[str, Any]) -> str:
        """生成审查摘要"""
        try:
            total_suggestions = sum(len(v) for v in iteration_suggestions.values() if isinstance(v, list))
            
            key_areas = []
            for category, suggestions in iteration_suggestions.items():
                if isinstance(suggestions, list) and suggestions:
                    if 'architecture' in category:
                        key_areas.append("架构改进")
                    elif 'performance' in category:
                        key_areas.append("性能优化")
                    elif 'quality' in category:
                        key_areas.append("代码质量")
                    elif 'testing' in category:
                        key_areas.append("测试完善")
            
            summary = f"审查完成，共提供{total_suggestions}条建议"
            if key_areas:
                summary += f"，主要涉及：{', '.join(key_areas[:3])}"
            
            return summary
            
        except Exception as e:
            logging.error(f"生成审查摘要失败: {str(e)}")
            return "审查摘要生成失败" 