"""
CLI Module for Paper to Code Agent
CLI版本的论文到代码智能体模块

包含以下组件 / Contains the following components:
- cli_app: CLI应用主程序 / CLI application main program
- cli_interface: CLI界面组件 / CLI interface components
- cli_launcher: CLI启动器 / CLI launcher
"""

__version__ = "1.0.0"
__author__ = "Paper to Code Team - CLI Edition"

from .cli_app import main as cli_main
from .cli_interface import CLIInterface
from .cli_launcher import main as launcher_main

__all__ = [
    "cli_main",
    "CLIInterface", 
    "launcher_main"
] 