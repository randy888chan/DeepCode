"""
Paper2Code: A tool for converting research papers to executable code
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__url__ = "https://github.com/yourusername/paper2code"
__description__ = "A comprehensive tool for analyzing research papers and generating executable code implementations"

from .utils import FileProcessor
from .workflows import (
    run_paper_analyzer,
    run_paper_downloader,
    paper_code_preparation,
    execute_code_implementation
)
from .tools import (
    CodeGenerator,
    PDFDownloader,
    GitHubDownloader
)

__all__ = [
    'FileProcessor',
    'run_paper_analyzer',
    'run_paper_downloader', 
    'paper_code_preparation',
    'execute_code_implementation',
    'CodeGenerator',
    'PDFDownloader',
    'GitHubDownloader'
] 