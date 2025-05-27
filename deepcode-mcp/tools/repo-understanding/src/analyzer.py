"""
代码分析器模块
"""
import os
import ast
from typing import List, Dict, Optional, Any
from pathlib import Path
import tiktoken

from .models import CodeChunk, RepositoryInfo, ChunkType
from .utils import detect_language, read_file_safe

class CodeAnalyzer:
    """代码分析器"""
    
    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.m': 'objc',
        '.mm': 'objcpp'
    }
    
    IGNORE_DIRS = {
        '.git', '.svn', '.hg', 
        'node_modules', '__pycache__', 
        'venv', 'env', '.env',
        'build', 'dist', 'target',
        '.idea', '.vscode', '.vs'
    }
    
    def __init__(self, max_chunk_tokens: int = 500):
        self.max_chunk_tokens = max_chunk_tokens
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    async def analyze_repository(self, repo_path: str) -> RepositoryInfo:
        """分析整个仓库"""
        repo_path = Path(repo_path)
        
        if not repo_path.exists() or not repo_path.is_dir():
            raise ValueError(f"Invalid repository path: {repo_path}")
        
        # 初始化仓库信息
        repo_info = RepositoryInfo(
            name=repo_path.name,
            path=str(repo_path)
        )
        
        # 获取README描述
        repo_info.description = await self._get_repository_description(repo_path)
        
        # 遍历文件
        for root, dirs, files in os.walk(repo_path):
            # 过滤忽略的目录
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]
            
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                if ext in self.SUPPORTED_EXTENSIONS:
                    language = self.SUPPORTED_EXTENSIONS[ext]
                    
                    # 读取文件
                    content = await read_file_safe(file_path)
                    if content:
                        lines = len(content.splitlines())
                        repo_info.total_files += 1
                        repo_info.total_lines += lines
                        repo_info.languages[language] = repo_info.languages.get(language, 0) + lines
        
        # 构建目录结构
        repo_info.structure = await self._build_directory_structure(repo_path)
        
        return repo_info
    
    async def extract_chunks(self, file_path: str) -> List[CodeChunk]:
        """从文件中提取代码块"""
        language = detect_language(file_path)
        if not language:
            return []
        
        content = await read_file_safe(file_path)
        if not content:
            return []
        
        # 根据语言选择提取策略
        if language == 'python':
            return await self._extract_python_chunks(file_path, content)
        else:
            return await self._extract_generic_chunks(file_path, content, language)
    
    async def _extract_python_chunks(self, file_path: str, content: str) -> List[CodeChunk]:
        """提取Python代码块"""
        chunks = []
        
        try:
            tree = ast.parse(content)
            lines = content.splitlines()
            
            # 提取导入语句
            import_nodes = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_nodes.append(node)
            
            if import_nodes:
                # 按行号排序
                import_nodes.sort(key=lambda n: n.lineno)
                start_line = import_nodes[0].lineno - 1
                end_line = import_nodes[-1].end_lineno or import_nodes[-1].lineno
                
                import_lines = lines[start_line:end_line]
                import_content = '\n'.join(import_lines)
                
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=import_content,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=ChunkType.IMPORTS,
                    language='python',
                    metadata={
                        'imports': [self._format_import(node) for node in import_nodes]
                    }
                ))
            
            # 提取类定义
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    chunk = await self._extract_class_chunk(file_path, node, lines)
                    if chunk:
                        chunks.append(chunk)
                elif isinstance(node, ast.FunctionDef):
                    chunk = await self._extract_function_chunk(file_path, node, lines)
                    if chunk:
                        chunks.append(chunk)
            
        except SyntaxError:
            # 语法错误时使用通用提取
            chunks = await self._extract_generic_chunks(file_path, content, 'python')
        
        return chunks
    
    async def _extract_class_chunk(self, file_path: str, node: ast.ClassDef, lines: List[str]) -> Optional[CodeChunk]:
        """提取类定义块"""
        start_line = node.lineno - 1
        end_line = node.end_lineno or node.lineno
        
        class_lines = lines[start_line:end_line]
        class_content = '\n'.join(class_lines)
        
        # 提取方法信息
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append({
                    'name': item.name,
                    'is_async': isinstance(item, ast.AsyncFunctionDef),
                    'args': [arg.arg for arg in item.args.args]
                })
        
        return CodeChunk(
            file_path=file_path,
            content=class_content,
            start_line=start_line,
            end_line=end_line,
            chunk_type=ChunkType.CLASS,
            language='python',
            metadata={
                'name': node.name,
                'methods': methods,
                'decorators': [self._format_decorator(d) for d in node.decorator_list],
                'bases': [self._format_base(b) for b in node.bases]
            }
        )
    
    async def _extract_function_chunk(self, file_path: str, node: ast.FunctionDef, lines: List[str]) -> Optional[CodeChunk]:
        """提取函数定义块"""
        start_line = node.lineno - 1
        end_line = node.end_lineno or node.lineno
        
        func_lines = lines[start_line:end_line]
        func_content = '\n'.join(func_lines)
        
        return CodeChunk(
            file_path=file_path,
            content=func_content,
            start_line=start_line,
            end_line=end_line,
            chunk_type=ChunkType.FUNCTION,
            language='python',
            metadata={
                'name': node.name,
                'args': [arg.arg for arg in node.args.args],
                'is_async': isinstance(node, ast.AsyncFunctionDef),
                'decorators': [self._format_decorator(d) for d in node.decorator_list],
                'returns': self._format_annotation(node.returns) if node.returns else None
            }
        )
    
    async def _extract_generic_chunks(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """通用代码块提取"""
        chunks = []
        lines = content.splitlines()
        
        current_chunk = []
        current_tokens = 0
        start_line = 0
        
        for i, line in enumerate(lines):
            line_tokens = len(self.tokenizer.encode(line))
            
            # 检查是否需要创建新块
            if current_tokens + line_tokens > self.max_chunk_tokens and current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=chunk_content,
                    start_line=start_line,
                    end_line=i,
                    chunk_type=ChunkType.CODE_BLOCK,
                    language=language,
                    metadata={'token_count': current_tokens}
                ))
                
                current_chunk = []
                current_tokens = 0
                start_line = i
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # 添加最后一个块
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(CodeChunk(
                file_path=file_path,
                content=chunk_content,
                start_line=start_line,
                end_line=len(lines),
                chunk_type=ChunkType.CODE_BLOCK,
                language=language,
                metadata={'token_count': current_tokens}
            ))
        
        return chunks
    
    async def _get_repository_description(self, repo_path: Path) -> str:
        """获取仓库描述"""
        for readme_name in ['README.md', 'README.rst', 'README.txt', 'readme.md']:
            readme_path = repo_path / readme_name
            if readme_path.exists():
                content = await read_file_safe(str(readme_path))
                if content:
                    # 提取前几行作为描述
                    lines = content.splitlines()[:5]
                    return ' '.join(lines).strip()[:500]
        return ""
    
    async def _build_directory_structure(self, repo_path: Path) -> Dict[str, Any]:
        """构建目录结构"""
        structure = {"files": [], "directories": {}}
        
        for item in repo_path.iterdir():
            if item.name.startswith('.') or item.name in self.IGNORE_DIRS:
                continue
                
            if item.is_file():
                if item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    structure["files"].append(item.name)
            elif item.is_dir():
                sub_structure = await self._build_directory_structure(item)
                if sub_structure["files"] or sub_structure["directories"]:
                    structure["directories"][item.name] = sub_structure
        
        return structure
    
    def _format_import(self, node) -> str:
        """格式化导入语句"""
        if isinstance(node, ast.Import):
            return ', '.join(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            names = ', '.join(alias.name for alias in node.names)
            return f"from {module} import {names}"
        return ""
    
    def _format_decorator(self, node) -> str:
        """格式化装饰器"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._format_decorator(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return f"{self._format_decorator(node.func)}(...)"
        return ""
    
    def _format_base(self, node) -> str:
        """格式化基类"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._format_base(node.value)}.{node.attr}"
        return ""
    
    def _format_annotation(self, node) -> str:
        """格式化类型注解"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return ""