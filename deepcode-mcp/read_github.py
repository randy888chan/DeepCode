import os
import ast
import git
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from collections import defaultdict
import networkx as nx
import tiktoken
import openai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
import re

@dataclass
class CodeFile:
    """代码文件的数据结构"""
    path: str
    content: str
    language: str
    imports: List[str]
    classes: List[str]
    functions: List[str]
    dependencies: List[str]

@dataclass
class CodeChunk:
    """代码块的数据结构"""
    file_path: str
    start_line: int
    end_line: int
    content: str
    chunk_type: str  # 'class', 'function', 'module', etc.
    name: str
    docstring: Optional[str]

class CodebaseAnalyzer:
    """代码库分析器"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.files = {}
        self.dependency_graph = nx.DiGraph()
        self.language_parsers = {
            '.py': self._parse_python,
            '.js': self._parse_javascript,
            '.java': self._parse_java,
            # 可以添加更多语言
        }
        
    def analyze_repository(self) -> Dict[str, CodeFile]:
        """分析整个仓库"""
        for file_path in self._get_all_files():
            if self._should_analyze(file_path):
                self.files[str(file_path)] = self._analyze_file(file_path)
        
        self._build_dependency_graph()
        return self.files
    
    def _get_all_files(self) -> List[Path]:
        """获取所有需要分析的文件"""
        ignore_patterns = ['.git', '__pycache__', 'node_modules', '.env']
        files = []
        
        for path in self.repo_path.rglob('*'):
            if path.is_file() and not any(pattern in str(path) for pattern in ignore_patterns):
                files.append(path)
        
        return files
    
    def _should_analyze(self, file_path: Path) -> bool:
        """判断是否应该分析该文件"""
        supported_extensions = ['.py', '.js', '.java', '.ts', '.jsx', '.tsx']
        return file_path.suffix in supported_extensions
    
    def _analyze_file(self, file_path: Path) -> CodeFile:
        """分析单个文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        language = file_path.suffix
        parser = self.language_parsers.get(language, self._parse_generic)
        
        return parser(file_path, content)
    
    def _parse_python(self, file_path: Path, content: str) -> CodeFile:
        """解析Python文件"""
        try:
            tree = ast.parse(content)
        except:
            return self._parse_generic(file_path, content)
        
        imports = []
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        return CodeFile(
            path=str(file_path),
            content=content,
            language='python',
            imports=imports,
            classes=classes,
            functions=functions,
            dependencies=imports
        )
    
    def _parse_generic(self, file_path: Path, content: str) -> CodeFile:
        """通用文件解析"""
        return CodeFile(
            path=str(file_path),
            content=content,
            language=file_path.suffix[1:],
            imports=[],
            classes=[],
            functions=[],
            dependencies=[]
        )
    
    def _build_dependency_graph(self):
        """构建依赖关系图"""
        for file_path, code_file in self.files.items():
            self.dependency_graph.add_node(file_path)
            for dep in code_file.dependencies:
                # 简化处理，实际需要更复杂的依赖解析
                dep_file = self._resolve_import(dep, file_path)
                if dep_file and dep_file in self.files:
                    self.dependency_graph.add_edge(file_path, dep_file)
    
    def _resolve_import(self, import_name: str, current_file: str) -> Optional[str]:
        """解析导入路径到实际文件"""
        # 这里需要根据不同语言的导入规则来实现
        # 简化示例
        return None

class CodeChunker:
    """代码分块器"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.encoding_for_model("gpt-4")
    
    def chunk_codebase(self, files: Dict[str, CodeFile]) -> List[CodeChunk]:
        """将代码库分块"""
        chunks = []
        
        for file_path, code_file in files.items():
            # 智能分块：优先按照代码结构分块
            file_chunks = self._smart_chunk_file(code_file)
            chunks.extend(file_chunks)
        
        return chunks
    
    def _smart_chunk_file(self, code_file: CodeFile) -> List[CodeChunk]:
        """智能分块单个文件"""
        chunks = []
        
        if code_file.language == 'python':
            chunks = self._chunk_python_file(code_file)
        else:
            chunks = self._chunk_by_lines(code_file)
        
        return chunks
    
    def _chunk_python_file(self, code_file: CodeFile) -> List[CodeChunk]:
        """Python文件的智能分块"""
        chunks = []
        
        try:
            tree = ast.parse(code_file.content)
            lines = code_file.content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    
                    chunk_content = '\n'.join(lines[start_line:end_line])
                    
                    # 检查token数量
                    tokens = len(self.encoder.encode(chunk_content))
                    if tokens <= self.chunk_size:
                        chunk = CodeChunk(
                            file_path=code_file.path,
                            start_line=start_line,
                            end_line=end_line,
                            content=chunk_content,
                            chunk_type='class' if isinstance(node, ast.ClassDef) else 'function',
                            name=node.name,
                            docstring=ast.get_docstring(node)
                        )
                        chunks.append(chunk)
                    else:
                        # 如果太大，需要进一步分块
                        sub_chunks = self._split_large_chunk(code_file, start_line, end_line)
                        chunks.extend(sub_chunks)
        except:
            # 解析失败时降级到按行分块
            chunks = self._chunk_by_lines(code_file)
        
        return chunks
    
    def _chunk_by_lines(self, code_file: CodeFile) -> List[CodeChunk]:
        """按行分块"""
        chunks = []
        lines = code_file.content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        start_line = 0
        
        for i, line in enumerate(lines):
            line_tokens = len(self.encoder.encode(line))
            
            if current_tokens + line_tokens > self.chunk_size and current_chunk:
                # 创建chunk
                chunk = CodeChunk(
                    file_path=code_file.path,
                    start_line=start_line,
                    end_line=i,
                    content='\n'.join(current_chunk),
                    chunk_type='code_block',
                    name=f"lines_{start_line}_{i}",
                    docstring=None
                )
                chunks.append(chunk)
                
                # 开始新chunk，包含overlap
                overlap_start = max(0, i - self.overlap // 50)  # 假设每行约50个token
                current_chunk = lines[overlap_start:i+1]
                current_tokens = sum(len(self.encoder.encode(line)) for line in current_chunk)
                start_line = overlap_start
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        # 处理最后一个chunk
        if current_chunk:
            chunk = CodeChunk(
                file_path=code_file.path,
                start_line=start_line,
                end_line=len(lines),
                content='\n'.join(current_chunk),
                chunk_type='code_block',
                name=f"lines_{start_line}_{len(lines)}",
                docstring=None
            )
            chunks.append(chunk)
        
        return chunks

class VectorStore:
    """向量存储和检索"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        self.collection = self.client.create_collection(
            name="code_chunks",
            metadata={"hnsw:space": "cosine"}
        )
    
    def index_chunks(self, chunks: List[CodeChunk]):
        """索引代码块"""
        for i, chunk in enumerate(chunks):
            # 创建用于embedding的文本
            embedding_text = self._create_embedding_text(chunk)
            
            # 生成embedding
            embedding = self.embedding_model.encode(embedding_text).tolist()
            
            # 存储到向量数据库
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk.content],
                metadatas=[{
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "name": chunk.name,
                    "docstring": chunk.docstring or ""
                }],
                ids=[f"chunk_{i}"]
            )
    
    def _create_embedding_text(self, chunk: CodeChunk) -> str:
        """创建用于生成embedding的文本"""
        parts = []
        
        # 添加文件路径信息
        parts.append(f"File: {chunk.file_path}")
        
        # 添加chunk类型和名称
        parts.append(f"Type: {chunk.chunk_type}")
        parts.append(f"Name: {chunk.name}")
        
        # 添加文档字符串
        if chunk.docstring:
            parts.append(f"Description: {chunk.docstring}")
        
        # 添加代码的前几行作为摘要
        code_lines = chunk.content.split('\n')[:5]
        parts.append("Code summary: " + " ".join(code_lines))
        
        return "\n".join(parts)
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """搜索相关代码块"""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return self._process_search_results(results)
    
    def _process_search_results(self, results) -> List[Dict]:
        """处理搜索结果"""
        processed_results = []
        
        for i in range(len(results['ids'][0])):
            result = {
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            }
            processed_results.append(result)
        
        return processed_results

class CodeAgent:
    """代码理解Agent主类"""
    
    def __init__(self, repo_path: str, openai_api_key: str):
        self.repo_path = repo_path
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # 初始化各个组件
        self.analyzer = CodebaseAnalyzer(repo_path)
        self.chunker = CodeChunker()
        self.vector_store = VectorStore()
        
        # 存储分析结果
        self.codebase_summary = None
        self.files = None
        self.chunks = None
        
    def initialize(self):
        """初始化代码库分析"""
        print("开始分析代码库...")
        
        # 分析代码库
        self.files = self.analyzer.analyze_repository()
        print(f"发现 {len(self.files)} 个代码文件")
        
        # 生成代码库摘要
        self.codebase_summary = self._generate_codebase_summary()
        
        # 分块
        print("开始分块...")
        self.chunks = self.chunker.chunk_codebase(self.files)
        print(f"生成 {len(self.chunks)} 个代码块")
        
        # 索引
        print("开始建立索引...")
        self.vector_store.index_chunks(self.chunks)
        print("索引建立完成")
    
    def _generate_codebase_summary(self) -> str:
        """生成代码库摘要"""
        summary_parts = []
        
        # 统计信息
        language_stats = defaultdict(int)
        for file in self.files.values():
            language_stats[file.language] += 1
        
        summary_parts.append(f"代码库包含 {len(self.files)} 个文件")
        summary_parts.append("语言分布：")
        for lang, count in language_stats.items():
            summary_parts.append(f"  - {lang}: {count} 个文件")
        
        # 主要模块
        summary_parts.append("\n主要目录结构：")
        dirs = set()
        for file_path in self.files.keys():
            dir_path = os.path.dirname(file_path)
            if dir_path:
                dirs.add(dir_path.split(os.sep)[0])
        
        for dir_name in sorted(dirs)[:10]:  # 只显示前10个
            summary_parts.append(f"  - {dir_name}/")
        
        return "\n".join(summary_parts)
    
    def query(self, user_query: str) -> str:
        """处理用户查询"""
        # 1. 理解用户意图
        intent = self._analyze_query_intent(user_query)
        
        # 2. 检索相关代码
        relevant_chunks = self._retrieve_relevant_code(user_query, intent)
        
        # 3. 构建上下文
        context = self._build_context(relevant_chunks, intent)
        
        # 4. 生成回答
        response = self._generate_response(user_query, context, intent)
        
        return response
    
    def _analyze_query_intent(self, query: str) -> Dict:
        """分析查询意图"""
        prompt = f"""
        分析以下查询的意图：
        查询：{query}
        
        请返回JSON格式：
        {{
            "intent_type": "explain|search|analyze|debug|refactor",
            "scope": "function|class|module|project",
            "specific_targets": [],
            "require_context": true/false
        }}
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是一个代码理解助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        try:
            intent = json.loads(response.choices[0].message['content'])
        except:
            intent = {
                "intent_type": "explain",
                "scope": "project",
                "specific_targets": [],
                "require_context": True
            }
        
        return intent
    
    def _retrieve_relevant_code(self, query: str, intent: Dict) -> List[Dict]:
        """检索相关代码"""
        # 基础向量搜索
        n_results = 10 if intent['scope'] == 'project' else 5
        results = self.vector_store.search(query, n_results=n_results)
        
        # 如果需要更多上下文，添加依赖
        if intent.get('require_context', True):
            expanded_results = self._expand_with_dependencies(results)
            results.extend(expanded_results)
        
        return results
    
    def _expand_with_dependencies(self, initial_results: List[Dict]) -> List[Dict]:
        """扩展搜索结果，包含依赖"""
        expanded = []
        
        for result in initial_results:
            file_path = result['metadata']['file_path']
            
            # 获取该文件的依赖
            if file_path in self.analyzer.dependency_graph:
                deps = list(self.analyzer.dependency_graph.neighbors(file_path))
                
                for dep in deps[:2]:  # 限制数量
                    # 搜索依赖文件中的相关内容
                    dep_results = self.vector_store.search(
                        f"file:{dep}",
                        n_results=2
                    )
                    expanded.extend(dep_results)
        
        return expanded
    
    def _build_context(self, chunks: List[Dict], intent: Dict) -> str:
        """构建LLM的上下文"""
        context_parts = []
        
        # 添加代码库摘要
        context_parts.append("代码库概览：")
        context_parts.append(self.codebase_summary)
        context_parts.append("\n相关代码：")
        
        # 添加相关代码块
        seen_files = set()
        for chunk in chunks:
            file_path = chunk['metadata']['file_path']
            
            # 添加文件信息
            if file_path not in seen_files:
                context_parts.append(f"\n--- 文件: {file_path} ---")
                seen_files.add(file_path)
            
            # 添加代码块信息
            metadata = chunk['metadata']
            context_parts.append(
                f"\n[{metadata['chunk_type']}: {metadata['name']} "
                f"(行 {metadata['start_line']}-{metadata['end_line']})]"
            )
            
            if metadata['docstring']:
                context_parts.append(f"文档: {metadata['docstring']}")
            
            context_parts.append("```")
            context_parts.append(chunk['content'])
            context_parts.append("```")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str, intent: Dict) -> str:
        """生成最终响应"""
        # 根据意图类型选择不同的提示模板
        if intent['intent_type'] == 'explain':
            system_prompt = """
            你是一个专业的代码解释助手。请基于提供的代码上下文，
            详细解释代码的功能、实现原理和设计思路。
            """
        elif intent['intent_type'] == 'analyze':
            system_prompt = """
            你是一个代码分析专家。请分析提供的代码，
            指出其优缺点、潜在问题和改进建议。
            """
        elif intent['intent_type'] == 'debug':
            system_prompt = """
            你是一个调试专家。请帮助分析代码中可能存在的问题，
            并提供调试建议。
            """
        else:
            system_prompt = """
            你是一个代码理解助手。请基于提供的上下文回答用户的问题。
            """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"代码上下文：\n{context}\n\n用户问题：{query}"}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message['content']

# 使用示例
def main():
    # 初始化Code Agent
    agent = CodeAgent(
        repo_path="/path/to/your/github/repo",
        openai_api_key="your-api-key"
    )
    
    # 初始化分析
    agent.initialize()
    
    # 交互式查询
    while True:
        query = input("\n请输入您的问题（输入'quit'退出）: ")
        if query.lower() == 'quit':
            break
        
        response = agent.query(query)
        print("\n回答：")
        print(response)

if __name__ == "__main__":
    main()