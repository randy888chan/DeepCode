#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构分析代理
功能：分析repo文件夹结构，识别主文件、入口文件和执行方法

"""

import os
import ast
import logging
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from utils.file_utils import read_file_content, get_project_structure
from utils.colored_logging import log_function_entry, log_function_exit, log_checkpoint, log_detailed


class StructureAnalyzerAgent(BaseAgent):
    """结构分析代理 - 分析repo结构并识别执行方法"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 常见的入口文件名
        self.entry_file_patterns = [
            'main.py', 'run.py', 'app.py', 'start.py', 'server.py',
            '__main__.py', 'manage.py', 'cli.py', 'index.py'
        ]
        # 常见的配置文件
        self.config_file_patterns = [
            'requirements.txt', 'setup.py', 'pyproject.toml', 'setup.cfg',
            'Pipfile', 'poetry.lock', 'conda.yml', 'environment.yml'
        ]
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理结构分析请求
        
        Args:
            input_data: 包含repo_path和帮助文档路径的字典
            
        Returns:
            结构分析结构
        """
        log_function_entry("StructureAnalyzer.process")
        
        try:
            log_checkpoint("验证输入数据")
            if not self.validate_input(input_data):
                log_detailed("ERROR", "输入数据验证失败")
                return self.handle_error(ValueError("输入数据无效"))
            
            repo_path = input_data['repo_path']
            help_files = input_data.get('help_files', [])
            
            log_checkpoint("输入数据验证通过", f"repo_path={repo_path}, help_files={len(help_files)}")
            
            # 1. 提取项目结构
            log_checkpoint("开始提取项目结构")
            project_structure = self._extract_project_structure(repo_path)
            log_checkpoint("项目结构提取完成", f"结果长度: {len(str(project_structure))}")
            
            # 2. 识别入口文件和主文件
            log_checkpoint("开始识别入口文件")
            entry_analysis = self._identify_entry_files(repo_path)
            log_checkpoint("入口文件识别完成", f"找到候选: {entry_analysis.get('total_candidates', 0)}个")
            
            # 3. 分析执行方法
            log_checkpoint("开始分析执行方法")
            execution_methods = self._analyze_execution_methods(repo_path, entry_analysis)
            log_checkpoint("执行方法分析完成", f"方法数: {len(execution_methods.get('methods', []))}")
            
            # 4. 分析帮助文档
            log_checkpoint("开始分析帮助文档")
            help_analysis = self._analyze_help_documents(help_files)
            log_checkpoint("帮助文档分析完成", f"文档数: {help_analysis.get('total_documents', 0)}")
            
            # 5. 生成综合分析
            log_checkpoint("开始生成综合分析（LLM调用）")
            comprehensive_analysis = self._generate_comprehensive_analysis(
                repo_path, project_structure, entry_analysis, 
                execution_methods, help_analysis
            )
            log_checkpoint("综合分析完成", f"分析长度: {len(comprehensive_analysis) if comprehensive_analysis else 0}")
            
            log_checkpoint("构建返回结果")
            result = {
                'project_structure': project_structure,
                'entry_analysis': entry_analysis,
                'execution_methods': execution_methods,
                'help_analysis': help_analysis,
                'comprehensive_analysis': comprehensive_analysis,
                'recommended_entry': entry_analysis.get('primary_entry'),
                'execution_command': execution_methods.get('recommended_command')
            }
            
            log_checkpoint("格式化输出结果")
            formatted_result = self.format_output(result)
            
            log_function_exit("StructureAnalyzer.process", "success")
            return formatted_result
            
        except Exception as e:
            log_detailed("ERROR", f"StructureAnalyzer异常: {str(e)}")
            log_function_exit("StructureAnalyzer.process", "error")
            return self.handle_error(e)
    
    def _extract_project_structure(self, repo_path: str) -> Dict[str, Any]:
        """提取项目结构"""
        try:
            structure = get_project_structure(repo_path, max_depth=5)
            
            # 统计信息
            stats = self._calculate_structure_stats(structure)
            
            # 识别项目类型
            project_type = self._identify_project_type(repo_path, structure)
            
            return {
                'structure': structure,
                'stats': stats,
                'project_type': project_type,
                'root_path': repo_path
            }
            
        except Exception as e:
            logging.error(f"提取项目结构失败: {str(e)}")
            return {}
    
    def _identify_entry_files(self, repo_path: str) -> Dict[str, Any]:
        """识别入口文件"""
        try:
            entry_files = []
            main_function_files = []
            config_files = []
            
            # 搜索入口文件
            for root, dirs, files in os.walk(repo_path):
                # 跳过常见的非代码目录
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_path)
                    
                    # 检查是否是入口文件
                    if file in self.entry_file_patterns:
                        entry_files.append({
                            'file': rel_path,
                            'type': 'named_entry',
                            'confidence': 0.9
                        })
                    
                    # 检查是否包含main函数
                    elif file.endswith('.py'):
                        if self._contains_main_function(file_path):
                            main_function_files.append({
                                'file': rel_path,
                                'type': 'main_function',
                                'confidence': 0.7
                            })
                    
                    # 收集配置文件
                    if file in self.config_file_patterns:
                        config_files.append(rel_path)
            
            # 确定主要入口文件
            primary_entry = self._select_primary_entry(entry_files, main_function_files)
            
            return {
                'entry_files': entry_files,
                'main_function_files': main_function_files,
                'config_files': config_files,
                'primary_entry': primary_entry,
                'total_candidates': len(entry_files) + len(main_function_files)
            }
            
        except Exception as e:
            logging.error(f"识别入口文件失败: {str(e)}")
            return {}
    
    def _contains_main_function(self, file_path: str) -> bool:
        """检查文件是否包含main函数"""
        try:
            content = read_file_content(file_path)
            if not content:
                return False
            
            # 简单文本检查
            if 'if __name__ == "__main__"' in content:
                return True
            
            # AST检查
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.If):
                        if (isinstance(node.test, ast.Compare) and
                            isinstance(node.test.left, ast.Name) and
                            node.test.left.id == '__name__'):
                            return True
            except:
                pass
            
            return False
            
        except Exception as e:
            logging.debug(f"检查main函数失败 {file_path}: {str(e)}")
            return False
    
    def _select_primary_entry(self, entry_files: List[Dict], main_function_files: List[Dict]) -> Optional[str]:
        """选择主要入口文件"""
        try:
            all_candidates = entry_files + main_function_files
            if not all_candidates:
                return None
            
            # 按置信度排序
            all_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            # 优先选择根目录下的文件
            for candidate in all_candidates:
                if '/' not in candidate['file'] and '\\' not in candidate['file']:
                    return candidate['file']
            
            # 如果没有根目录文件，选择置信度最高的
            return all_candidates[0]['file']
            
        except Exception as e:
            logging.error(f"选择主要入口文件失败: {str(e)}")
            return None
    
    def _analyze_execution_methods(self, repo_path: str, entry_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """分析执行方法"""
        try:
            methods = []
            
            primary_entry = entry_analysis.get('primary_entry')
            config_files = entry_analysis.get('config_files', [])
            
            # 基础Python执行
            if primary_entry:
                methods.append({
                    'method': 'python_direct',
                    'command': f'python {primary_entry}',
                    'description': f'直接执行主文件 {primary_entry}',
                    'confidence': 0.9
                })
            
            # 检查setup.py
            if 'setup.py' in config_files:
                methods.append({
                    'method': 'setup_install',
                    'command': 'pip install -e .',
                    'description': '通过setup.py安装并执行',
                    'confidence': 0.8
                })
            
            # 检查是否是Django项目
            if 'manage.py' in [f['file'] for f in entry_analysis.get('entry_files', [])]:
                methods.append({
                    'method': 'django',
                    'command': 'python manage.py runserver',
                    'description': 'Django项目启动服务器',
                    'confidence': 0.9
                })
            
            # 推荐的执行命令
            recommended_command = methods[0]['command'] if methods else None
            
            return {
                'methods': methods,
                'recommended_command': recommended_command,
                'needs_dependencies': len(config_files) > 0,
                'dependency_files': config_files
            }
            
        except Exception as e:
            logging.error(f"分析执行方法失败: {str(e)}")
            return {}
    
    def _analyze_help_documents(self, help_files: List[str]) -> Dict[str, Any]:
        """分析帮助文档"""
        try:
            documents = []
            total_content = ""
            
            for file_path in help_files:
                if os.path.exists(file_path):
                    content = read_file_content(file_path)
                    if content:
                        documents.append({
                            'file': file_path,
                            'length': len(content),
                            'type': self._get_file_type(file_path)
                        })
                        total_content += content + "\n\n"
            
            return {
                'documents': documents,
                'total_documents': len(documents),
                'total_content_length': len(total_content),
                'combined_content': total_content  # 不限制长度，使用完整内容
            }
            
        except Exception as e:
            logging.error(f"分析帮助文档失败: {str(e)}")
            return {}
    
    def _generate_comprehensive_analysis(self, repo_path: str, structure: Dict[str, Any], 
                                       entry_analysis: Dict[str, Any], execution_methods: Dict[str, Any],
                                       help_analysis: Dict[str, Any]) -> str:
        """生成综合分析"""
        try:
            messages = [
                self.create_system_message(
                    "项目结构分析专家",
                    "分析Python项目的文件结构，识别主要入口文件和执行方法，提供详细的项目理解和运行指导。"
                ),
                self.create_user_message(
                    f"""请分析以下Python项目并提供详细的结构分析和执行指导：

项目路径：{repo_path}

项目结构统计：
- 总文件数：{structure.get('stats', {}).get('total_files', 0)}
- 总目录数：{structure.get('stats', {}).get('total_dirs', 0)}
- Python文件数：{structure.get('stats', {}).get('python_files', 0)}
- 项目类型：{structure.get('project_type', '未知')}

入口文件分析：
- 找到入口文件候选：{entry_analysis.get('total_candidates', 0)}个
- 主要入口文件：{entry_analysis.get('primary_entry', '未找到')}
- 配置文件：{', '.join(entry_analysis.get('config_files', []))}

推荐执行方法：
- 推荐命令：{execution_methods.get('recommended_command', '未确定')}
- 需要安装依赖：{'是' if execution_methods.get('needs_dependencies') else '否'}

帮助文档：
- 文档数量：{help_analysis.get('total_documents', 0)}
- 文档内容：{help_analysis.get('combined_content', '无')}

请提供：
1. 项目整体架构分析
2. 主要功能模块识别
3. 入口文件确认和执行流程
4. 依赖关系分析
5. 运行环境要求
6. 可能的执行问题和解决方案
"""
                )
            ]
            
            analysis = self.call_llm(messages)
            return analysis or "LLM分析不可用，请检查配置"
            
        except Exception as e:
            logging.error(f"生成综合分析失败: {str(e)}")
            return f"分析失败: {str(e)}"
    
    def _calculate_structure_stats(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """计算结构统计信息"""
        try:
            stats = {
                'total_files': 0,
                'total_dirs': 0,
                'python_files': 0,
                'max_depth': 0
            }
            
            def count_items(items: Dict, current_depth: int = 0):
                stats['max_depth'] = max(stats['max_depth'], current_depth)
                
                for name, item in items.items():
                    if isinstance(item, dict):
                        if item.get('type') == 'file':
                            stats['total_files'] += 1
                            if name.endswith('.py'):
                                stats['python_files'] += 1
                        elif item.get('type') == 'directory':
                            stats['total_dirs'] += 1
                            if 'children' in item:
                                count_items(item['children'], current_depth + 1)
            
            count_items(structure)
            return stats
            
        except Exception as e:
            logging.error(f"计算结构统计失败: {str(e)}")
            return {}
    
    def _identify_project_type(self, repo_path: str, structure: Dict[str, Any]) -> str:
        """识别项目类型"""
        try:
            # 检查特征文件
            if self._file_exists_in_structure(structure, 'manage.py'):
                return 'Django项目'
            elif self._file_exists_in_structure(structure, 'app.py') or self._file_exists_in_structure(structure, 'wsgi.py'):
                return 'Flask/Web项目'
            elif self._file_exists_in_structure(structure, 'setup.py'):
                return 'Python包项目'
            elif self._file_exists_in_structure(structure, 'requirements.txt'):
                return 'Python应用项目'
            else:
                return '通用Python项目'
                
        except Exception as e:
            logging.error(f"识别项目类型失败: {str(e)}")
            return '未知类型'
    
    def _file_exists_in_structure(self, structure: Dict[str, Any], filename: str) -> bool:
        """检查文件是否存在于结构中"""
        try:
            def search_file(items: Dict) -> bool:
                for name, item in items.items():
                    if name == filename and isinstance(item, dict) and item.get('type') == 'file':
                        return True
                    elif isinstance(item, dict) and 'children' in item:
                        if search_file(item['children']):
                            return True
                return False
            
            return search_file(structure)
            
        except Exception as e:
            logging.debug(f"搜索文件失败: {str(e)}")
            return False
    
    def _get_file_type(self, file_path: str) -> str:
        """获取文件类型"""
        if file_path.endswith('.md'):
            return 'markdown'
        elif file_path.endswith('.txt'):
            return 'text'
        elif file_path.endswith('.py'):
            return 'python'
        else:
            return 'unknown' 