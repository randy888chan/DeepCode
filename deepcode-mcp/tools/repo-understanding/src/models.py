"""
数据模型定义
"""
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum

class ChunkType(Enum):
    """代码块类型"""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    IMPORTS = "imports"
    CODE_BLOCK = "code_block"

@dataclass
class CodeChunk:
    """代码片段"""
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_type: ChunkType
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "file_path": self.file_path,
            "content": self.content,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type.value,
            "language": self.language,
            "metadata": self.metadata
        }

@dataclass
class RepositoryInfo:
    """仓库信息"""
    name: str
    path: str
    description: str = ""
    languages: Dict[str, int] = field(default_factory=dict)
    total_files: int = 0
    total_lines: int = 0
    structure: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "languages": self.languages,
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "structure": self.structure
        }

@dataclass
class SearchResult:
    """搜索结果"""
    chunk: CodeChunk
    score: float
    highlights: List[str] = field(default_factory=list)