#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沙箱测试系统主入口
功能：按照三Agent架构分析repo，生成测试版本，执行测试，提供审查建议


主要工作流程：
1. Agent1 (结构分析): 提取repo结构并识别执行方法
2. Agent2 (代码重写): 重写代码文件，添加log和测试输入
3. 配置虚拟环境，执行测试repo
4. Agent3 (审查分析): 分析沙箱输出，给出代码迭代建议
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("已加载 .env 文件")
except ImportError:
    print("未安装 python-dotenv，跳过 .env 文件加载")
except Exception as e:
    print(f"加载 .env 文件失败: {e}")

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from agents.structure_analyzer_agent import StructureAnalyzerAgent
from agents.code_rewriter_agent import CodeRewriterAgent
from agents.review_agent import ReviewAgent
from core.executor import SandboxExecutor
from utils.config_utils import load_config
from utils.file_utils import setup_directories
from utils.colored_logging import (
    setup_colored_logging, log_step, log_agent, log_success, log_error, 
    log_warning, print_separator, log_function_entry, log_function_exit, 
    log_checkpoint, log_detailed
)


class SandboxTestSystem:
    """沙箱测试系统主类 - 三Agent架构"""
    
    def __init__(self, config_path: str = "config/sandbox_config.yaml"):
        """
        初始化沙箱测试系统
        
        Args:
            config_path: 配置文件路径
        """
        print("🔧 开始初始化沙箱测试系统...")
        
        print("📋 加载配置文件...")
        self.config = load_config(config_path)
        
        print("📝 设置日志系统...")
        self.setup_logging()
        
        print("📁 设置目录结构...")
        self.setup_directories()
        
        print("🤖 初始化Agent...")
        # 初始化三个核心Agent
        agent_config = self.config.get('agents', {})
        
        print("  - 初始化结构分析代理...")
        self.structure_analyzer = StructureAnalyzerAgent(agent_config.get('structure_analyzer', {}))
        
        print("  - 初始化代码重写代理...")
        self.code_rewriter = CodeRewriterAgent(agent_config.get('code_rewriter', {}))
        
        print("  - 初始化审查代理...")
        self.review_agent = ReviewAgent(agent_config.get('review_agent', {}))
        
        print("⚙️ 初始化执行器...")
        # 初始化执行器
        self.executor = SandboxExecutor(self.config)
        
        print("✅ 沙箱测试系统初始化完成")
        logging.info("沙箱测试系统初始化完成 - 三Agent架构")
    
    def setup_logging(self):
        """设置日志系统"""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        
        # 设置彩色日志系统
        setup_colored_logging(log_level)
        
        # 同时保留文件日志
        file_handler = logging.FileHandler('sandbox_test.log')
        file_handler.setLevel(getattr(logging, log_level))
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)
        
        log_success("彩色日志系统已启动")
    
    def setup_directories(self):
        """设置必要的目录结构"""
        directories = [
            'sandbox/environments',
            'sandbox/isolation',
            'sandbox/monitoring',
            'temp'
        ]
        setup_directories(directories)
    
    def run_complete_analysis(self, repo_path: str, help_files: List[str] = None) -> Dict[str, Any]:
        """
        运行完整的沙箱测试流程
        
        Args:
            repo_path: 代码仓库路径
            help_files: 帮助文档文件路径列表
            
        Returns:
            包含所有结果的字典
        """
        logging.info("开始三Agent沙箱测试流程")
        logging.info(f"目标repo: {repo_path}")
        if help_files:
            logging.info(f"帮助文档: {', '.join(help_files)}")
        
        try:
            # 阶段1: Agent1 - 结构分析
            print_separator("阶段1: 结构分析")
            log_step("开始结构分析 (Agent1)", 1)
            structure_result = self._run_structure_analysis(repo_path, help_files or [])
            
            if structure_result['status'] != 'success':
                log_error(f"结构分析失败: {structure_result.get('error')}")
                return self._create_error_result("结构分析失败", structure_result.get('error'))
            log_success("结构分析完成")
            
            # 阶段2: Agent2 - 代码重写
            print_separator("阶段2: 代码重写")
            log_step("开始代码重写 (Agent2)", 2)
            rewrite_result = self._run_code_rewrite(repo_path, structure_result['result'])
            
            if rewrite_result['status'] != 'success':
                log_error(f"代码重写失败: {rewrite_result.get('error')}")
                return self._create_error_result("代码重写失败", rewrite_result.get('error'))
            log_success("代码重写完成")
            
            # 阶段3: 执行沙箱测试
            print_separator("阶段3: 沙箱执行")
            log_step("开始沙箱执行", 3)
            execution_result = self._run_sandbox_execution(rewrite_result['result'])
            
            if execution_result['status'] != 'success':
                log_warning(f"沙箱执行出现问题: {execution_result.get('error', '未知错误')}")
                # 执行失败不中止流程，继续进行审查
            else:
                log_success("沙箱执行完成")
            
            # 阶段4: Agent3 - 审查分析
            print_separator("阶段4: 审查分析")
            log_step("开始审查分析 (Agent3)", 4)
            review_result = self._run_review_analysis(
                structure_result['result'], 
                rewrite_result['result'], 
                execution_result.get('result', {})
            )
            
            # 整合所有结果
            final_result = {
                'structure_analysis': structure_result['result'],
                'code_rewrite': rewrite_result['result'],
                'sandbox_execution': execution_result.get('result', {}),
                'review_analysis': review_result.get('result', {}),
                'status': 'success',
                'summary': self._generate_final_summary(
                    structure_result['result'],
                    rewrite_result['result'],
                    execution_result.get('result', {}),
                    review_result.get('result', {})
                )
            }
            
            logging.info("=" * 50)
            logging.info("沙箱测试流程完成")
            return final_result
            
        except Exception as e:
            logging.error(f"沙箱测试流程异常: {str(e)}")
            return self._create_error_result("系统异常", str(e))
    
    def _run_structure_analysis(self, repo_path: str, help_files: List[str]) -> Dict[str, Any]:
        """运行结构分析 (Agent1)"""
        log_function_entry("_run_structure_analysis", {"repo_path": repo_path, "help_files_count": len(help_files)})
        
        try:
            log_checkpoint("准备输入数据")
            input_data = {
                'repo_path': repo_path,
                'help_files': help_files
            }
            
            log_checkpoint("验证输入参数", f"项目路径: {repo_path}")
            log_checkpoint("检查帮助文档", f"文档数量: {len(help_files)}")
            
            # 显示帮助文档详情
            for i, help_file in enumerate(help_files):
                log_detailed("INFO", f"帮助文档 {i+1}: {help_file}")
            
            log_checkpoint("开始调用StructureAnalyzerAgent")
            result = self.structure_analyzer.process(input_data)
            log_checkpoint("StructureAnalyzerAgent调用完成", f"状态: {result.get('status')}")
            
            if result['status'] == 'success':
                log_checkpoint("解析分析结果")
                analysis = result['result']
                log_success("结构分析结果:")
                
                # 详细显示分析结果
                project_type = analysis.get('project_structure', {}).get('project_type', '未知')
                total_files = analysis.get('project_structure', {}).get('stats', {}).get('total_files', 0)
                python_files = analysis.get('project_structure', {}).get('stats', {}).get('python_files', 0)
                recommended_entry = analysis.get('recommended_entry', '未确定')
                execution_command = analysis.get('execution_command', '未确定')
                
                log_detailed("INFO", f"项目类型: {project_type}")
                log_detailed("INFO", f"总文件数: {total_files}")
                log_detailed("INFO", f"Python文件: {python_files}")
                log_detailed("INFO", f"推荐入口: {recommended_entry}")
                log_detailed("INFO", f"执行命令: {execution_command}")
                
                log_checkpoint("结构分析成功完成")
            else:
                log_error(f"结构分析失败: {result.get('error', '未知错误')}")
                log_checkpoint("结构分析失败", f"错误: {result.get('error')}")
            
            log_function_exit("_run_structure_analysis", result.get('status'))
            return result
            
        except Exception as e:
            log_error(f"结构分析异常: {str(e)}")
            log_function_exit("_run_structure_analysis", "exception")
            return {'status': 'error', 'error': str(e)}
    
    def _run_code_rewrite(self, repo_path: str, structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """运行代码重写 (Agent2)"""
        try:
            input_data = {
                'repo_path': repo_path,
                'structure_analysis': structure_analysis
            }
            
            result = self.code_rewriter.process(input_data)
            
            if result['status'] == 'success':
                rewrite = result['result']
                logging.info(f"代码重写完成:")
                logging.info(f"  - 测试repo路径: {rewrite.get('test_repo_path', '未知')}")
                logging.info(f"  - 处理文件数: {rewrite.get('total_files', 0)}")
                logging.info(f"  - 添加日志点: {rewrite.get('log_points_added', 0)}")
                logging.info(f"  - 生成依赖数: {len(rewrite.get('dependencies', []))}")
            
            return result
            
        except Exception as e:
            logging.error(f"代码重写异常: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _run_sandbox_execution(self, rewrite_result: Dict[str, Any]) -> Dict[str, Any]:
        """运行沙箱执行"""
        try:
            test_repo_path = rewrite_result.get('test_repo_path')
            if not test_repo_path or not os.path.exists(test_repo_path):
                raise FileNotFoundError(f"测试repo不存在: {test_repo_path}")
            
            # 查找入口文件
            entry_file = self._find_entry_file(test_repo_path)
            if not entry_file:
                raise FileNotFoundError(f"未找到入口文件: {test_repo_path}")
            
            # 查找requirements文件
            requirements_file = os.path.join(test_repo_path, 'requirements.txt')
            if not os.path.exists(requirements_file):
                raise FileNotFoundError(f"requirements.txt不存在: {requirements_file}")
            
            logging.info(f"🚀 开始执行测试: {test_repo_path}")
            logging.info(f"📁 找到入口文件: {entry_file}")
            
            # 使用执行器运行测试
            execution_result = self.executor.run_sandbox(test_repo_path, entry_file, requirements_file)
            
            if execution_result['status'] == 'ok':
                logging.info(f"沙箱执行完成:")
                logging.info(f"  - 执行状态: {'成功' if execution_result.get('returncode') == 0 else '失败'}")
                logging.info(f"  - 返回码: {execution_result.get('returncode', -1)}")
                
                # 处理标准输出
                stdout = execution_result.get('stdout', '')
                stderr = execution_result.get('stderr', '')
                stdout_lines = stdout.split('\n') if stdout else []
                stderr_lines = stderr.split('\n') if stderr else []
                
                logging.info(f"  - 标准输出: {len(stdout_lines)} 行")
                logging.info(f"  - 错误输出: {len(stderr_lines)} 行")
                
                # 统计日志信息
                log_count = len([line for line in stdout_lines if '[LOG]' in line])
                logging.info(f"  - 日志条目: {log_count}")
                
                # 构造兼容的返回格式
                formatted_result = {
                    'execution_result': {
                        'return_code': execution_result.get('returncode', -1),
                        'stdout': stdout_lines,
                        'stderr': stderr_lines,
                        'execution_time': 0,  # executor.py 没有返回执行时间
                        'max_memory_mb': 0    # executor.py 没有返回内存使用
                    }
                }
                
            elif execution_result['status'] == 'error':
                logging.error(f"❌ 代码执行失败")
                stderr_content = execution_result.get('stderr', '')
                stderr_lines = stderr_content.split('\n') if stderr_content else []
                logging.info(f"  - 错误输出: {len(stderr_lines)} 行")
                
                # 显示详细的错误信息
                if stderr_content:
                    logging.error(f"📋 详细错误信息:")
                    for i, line in enumerate(stderr_lines[:10], 1):  # 只显示前10行
                        if line.strip():
                            logging.error(f"  {i:2d}: {line}")
                    if len(stderr_lines) > 10:
                        logging.error(f"  ... 还有 {len(stderr_lines) - 10} 行错误输出")
                
                # 构造兼容的返回格式
                formatted_result = {
                    'execution_result': {
                        'return_code': execution_result.get('returncode', -1),
                        'stdout': [],
                        'stderr': stderr_lines,
                        'execution_time': 0,
                        'max_memory_mb': 0
                    }
                }
            
            return {'status': 'success', 'result': formatted_result}
            
        except Exception as e:
            logging.error(f"沙箱执行异常: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _find_entry_file(self, test_repo_path: str) -> str:
        """查找入口文件"""
        # 常见的入口文件名
        entry_candidates = ['main.py', 'app.py', 'run.py', '__main__.py']
        
        for candidate in entry_candidates:
            entry_path = os.path.join(test_repo_path, candidate)
            if os.path.exists(entry_path):
                return candidate
        
        # 如果没找到，查找包含 if __name__ == '__main__' 的文件
        for root, dirs, files in os.walk(test_repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if "__name__ == '__main__'" in content or '__name__ == "__main__"' in content:
                                # 返回相对于test_repo_path的路径
                                return os.path.relpath(file_path, test_repo_path)
                    except:
                        continue
        
        return None
    
    def _run_review_analysis(self, structure_analysis: Dict[str, Any], 
                           rewrite_result: Dict[str, Any], 
                           execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """运行审查分析 (Agent3)"""
        try:
            # 提取执行结果数据
            exec_data = execution_result.get('execution_result', {})
            
            input_data = {
                'structure_analysis': structure_analysis,
                'rewrite_result': rewrite_result,
                'execution_result': exec_data
            }
            
            result = self.review_agent.process(input_data)
            
            if result['status'] == 'success':
                review = result['result']
                logging.info(f"审查分析完成:")
                logging.info(f"  - 输出质量: {review.get('output_analysis', {}).get('output_quality', '未评估')}")
                logging.info(f"  - 代码质量得分: {review.get('code_quality_analysis', {}).get('quality_score', 0):.1f}")
                logging.info(f"  - 性能评级: {review.get('performance_analysis', {}).get('performance_rating', '未评级')}")
                logging.info(f"  - 审查摘要: {review.get('review_summary', '无')}")
            
            return result
            
        except Exception as e:
            logging.error(f"审查分析异常: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _generate_final_summary(self, structure_analysis: Dict[str, Any],
                              rewrite_result: Dict[str, Any],
                              execution_result: Dict[str, Any],
                              review_result: Dict[str, Any]) -> str:
        """生成最终摘要"""
        try:
            # 基础信息
            project_type = structure_analysis.get('project_structure', {}).get('project_type', '未知项目')
            total_files = rewrite_result.get('total_files', 0)
            log_points = rewrite_result.get('log_points_added', 0)
            
            # 执行信息
            exec_data = execution_result.get('execution_result', {})
            execution_success = exec_data.get('return_code') == 0
            execution_time = exec_data.get('execution_time', 0)
            
            # 审查信息
            quality_score = review_result.get('code_quality_analysis', {}).get('quality_score', 0)
            performance_rating = review_result.get('performance_analysis', {}).get('performance_rating', '未知')
            
            summary = f"""
沙箱测试完成摘要：
📁 项目类型：{project_type}
📊 处理文件：{total_files}个
📝 日志点数：{log_points}个
⚡ 执行状态：{'✅ 成功' if execution_success else '❌ 失败'}
⏱️ 执行时间：{execution_time:.2f}秒
🏆 质量得分：{quality_score:.1f}/10
🚀 性能评级：{performance_rating}
"""
            return summary.strip()
            
        except Exception as e:
            logging.error(f"生成最终摘要失败: {str(e)}")
            return "摘要生成失败"
    
    def _create_error_result(self, error_type: str, error_message: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'status': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'structure_analysis': {},
            'code_rewrite': {},
            'sandbox_execution': {},
            'review_analysis': {}
        }
    

    
    def cleanup(self):
        """清理临时文件"""
        logging.info("清理临时文件")
        # 这里可以添加清理逻辑


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='沙箱测试系统 - 三Agent架构')
    parser.add_argument('--repo', required=True, help='代码仓库路径')
    parser.add_argument('--help-files', nargs='*', help='帮助文档文件路径列表')
    parser.add_argument('--config', default='config/sandbox_config.yaml', help='配置文件路径')
    parser.add_argument('--output', default='output', help='输出目录')
    
    args = parser.parse_args()
    
    # 验证输入路径
    if not os.path.exists(args.repo):
        print(f"❌ 错误：代码仓库路径不存在: {args.repo}")
        sys.exit(1)
    
    # 验证帮助文档
    help_files = []
    if args.help_files:
        for help_file in args.help_files:
            if os.path.exists(help_file):
                help_files.append(help_file)
                print(f"📖 添加帮助文档: {help_file}")
            else:
                print(f"⚠️ 警告：帮助文档不存在: {help_file}")
    
    # 创建沙箱测试系统实例
    try:
        sandbox_system = SandboxTestSystem(args.config)
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        sys.exit(1)
    
    try:
        # 运行完整分析流程
        result = sandbox_system.run_complete_analysis(args.repo, help_files)
        
        # 输出结果
        if result['status'] == 'success':
            print("\n" + "=" * 60)
            print("✅ 沙箱测试完成")
            print(result['summary'])
            
            # 保存结果到文件
            output_file = os.path.join(args.output, 'sandbox_analysis_result.json')
            os.makedirs(args.output, exist_ok=True)
            
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            print(f"📁 完整结果已保存到: {output_file}")
            
            # 保存审查报告
            if result.get('review_analysis', {}).get('comprehensive_review'):
                report_file = os.path.join(args.output, 'review_report.md')
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write("# 沙箱测试审查报告\n\n")
                    f.write(result['review_analysis']['comprehensive_review'])
                print(f"📝 审查报告已保存到: {report_file}")
            
            # 测试repo已直接保存在output目录下
            test_repo_path = result.get('code_rewrite', {}).get('test_repo_path')
            if test_repo_path and os.path.exists(test_repo_path):
                print(f"📁 测试repo已保存到: {test_repo_path}")
                print(f"   - 原repo: {args.repo}")
                print(f"   - 测试repo: {test_repo_path}")
                repo_name = os.path.basename(args.repo.rstrip('/\\'))
                print(f"   - 对比命令: diff -r {args.repo} output/{repo_name}_test_sandbox")
                
        else:
            print(f"\n❌ 测试失败: {result.get('error_type', '未知错误')}")
            print(f"错误信息: {result.get('error_message', '无详细信息')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 系统错误: {str(e)}")
        sys.exit(1)
    finally:
        sandbox_system.cleanup()


if __name__ == "__main__":
    main() 