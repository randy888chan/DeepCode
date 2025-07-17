"""
Research2Code: A tool for converting research content to executable code
"""

__version__ = "0.1.0"
__author__ = "Zongwei Li"
__email__ = "zongwei9888@gmail.com"
__url__ = "https://github.com/HKUDS/Code-Agent"
__description__ = "A comprehensive tool for analyzing research content and generating executable code implementations"

from .utils import FileProcessor
from .workflows import (
    run_research_analyzer,
    run_resource_processor,
    execute_multi_agent_research_pipeline,
    paper_code_preparation,  # Deprecated, for backward compatibility
    execute_code_implementation
)
from .tools import (
    CodeGenerator,
    PDFDownloader,
    GitHubDownloader
)

__all__ = [
    'FileProcessor',
    'run_research_analyzer',
    'run_resource_processor', 
    'execute_multi_agent_research_pipeline',
    'paper_code_preparation',  # Deprecated, for backward compatibility
    'execute_code_implementation',
    'CodeGenerator',
    'PDFDownloader',
    'GitHubDownloader'
] 