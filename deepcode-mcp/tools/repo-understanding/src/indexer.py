"""
仓库索引器模块
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions

from .models import CodeChunk, RepositoryInfo, SearchResult
from .analyzer import CodeAnalyzer
from .utils import get_embedding_function

class RepositoryIndexer:
    """仓库索引器"""
    
    def __init__(self, 
                 collection_name: str,
                 persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.analyzer = CodeAnalyzer()
        
        # 初始化ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = get_embedding_function()
        self.collection = None
        
    async def initialize(self):
        """初始化或获取集合"""
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"created_at": datetime.now().isoformat()}
            )
    
    async def index_repository(self, repo_path: str) -> Dict[str, Any]:
        """索引整个仓库"""
        # 确保集合已初始化
        if not self.collection:
            await self.initialize()
        
        # 清空现有数据
        self.collection.delete(where={})
        
        # 分析仓库
        repo_info = await self.analyzer.analyze_repository(repo_path)
        
        # 索引统计
        indexed_files = 0
        total_chunks = 0
        errors = []
        
        # 遍历所有支持的文件
        for root, dirs, files in os.walk(repo_path):
            # 过滤目录
            dirs[:] = [d for d in dirs if d not in CodeAnalyzer.IGNORE_DIRS]
            
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                if ext in CodeAnalyzer.SUPPORTED_EXTENSIONS:
                    try:
                        # 提取代码块
                        chunks = await self.analyzer.extract_chunks(file_path)
                        
                        if chunks:
                            # 准备批量插入的数据
                            documents = []
                            metadatas = []
                            ids = []
                            
                            for chunk in chunks:
                                # 创建文档
                                doc = self._create_document(chunk, repo_info.name)
                                documents.append(doc)
                                
                                # 元数据
                                metadata = self._create_metadata(chunk, repo_info.name)
                                metadatas.append(metadata)
                                
                                # ID
                                chunk_id = self._create_chunk_id(chunk)
                                ids.append(chunk_id)
                            
                            # 批量添加到集合
                            self.collection.add(
                                documents=documents,
                                metadatas=metadatas,
                                ids=ids
                            )
                            
                            indexed_files += 1
                            total_chunks += len(chunks)
                            
                    except Exception as e:
                        errors.append({
                            'file': file_path,
                            'error': str(e)
                        })
        
        # 保存仓库信息
        await self._save_repository_info(repo_info)
        
        return {
            'status': 'success',
            'repository': repo_info.to_dict(),
            'indexed_files': indexed_files,
            'total_chunks': total_chunks,
            'errors': errors
        }
    
    async def search(self, 
                     query: str, 
                     n_results: int = 5,
                     filter_dict: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """搜索代码"""
        if not self.collection:
            await self.initialize()
        
        # 构建查询
        kwargs = {
            'query_texts': [query],
            'n_results': n_results
        }
        
        if filter_dict:
            kwargs['where'] = filter_dict
        
        # 执行查询
        results = self.collection.query(**kwargs)
        
        # 转换为SearchResult对象
        search_results = []
        for i in range(len(results['ids'][0])):
            # 重构CodeChunk
            metadata = results['metadatas'][0][i]
            chunk = CodeChunk(
                file_path=metadata['file_path'],
                content=self._extract_content_from_document(results['documents'][0][i]),
                start_line=metadata['start_line'],
                end_line=metadata['end_line'],
                chunk_type=ChunkType(metadata['chunk_type']),
                language=metadata['language'],
                metadata=metadata.get('extra_metadata', {})
            )
            
            # 创建搜索结果
            search_result = SearchResult(
                chunk=chunk,
                score=1.0 - results['distances'][0][i] if 'distances' in results else 0.0,
                highlights=[]  # TODO: 实现高亮
            )
            
            search_results.append(search_result)
        
        return search_results
    
    async def get_similar_chunks(self, 
                                chunk_id: str, 
                                n_results: int = 5) -> List[SearchResult]:
        """获取相似的代码块"""
        if not self.collection:
            await self.initialize()
        
        # 获取原始块
        result = self.collection.get(ids=[chunk_id])
        if not result['ids']:
            return []
        
        # 搜索相似块
        doc = result['documents'][0]
        return await self.search(doc, n_results=n_results + 1)
    
    async def delete_collection(self):
        """删除集合"""
        if self.collection:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
    
    def _create_document(self, chunk: CodeChunk, repo_name: str) -> str:
        """创建文档内容"""
        doc = f"Repository: {repo_name}\n"
        doc += f"File: {chunk.file_path}\n"
        doc += f"Language: {chunk.language}\n"
        doc += f"Type: {chunk.chunk_type.value}\n"
        doc += f"Lines: {chunk.start_line}-{chunk.end_line}\n"
        
        # 添加额外的元数据到文档
        if chunk.metadata.get('name'):
            doc += f"Name: {chunk.metadata['name']}\n"
        if chunk.metadata.get('methods'):
            doc += f"Methods: {', '.join(m['name'] for m in chunk.metadata['methods'])}\n"
        
        doc += f"\n{chunk.content}"
        return doc
    
    def _create_metadata(self, chunk: CodeChunk, repo_name: str) -> Dict[str, Any]:
        """创建元数据"""
        metadata = {
            'repo_name': repo_name,
            'file_path': chunk.file_path,
            'language': chunk.language,
            'chunk_type': chunk.chunk_type.value,
            'start_line': chunk.start_line,
            'end_line': chunk.end_line,
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加额外的元数据
        if chunk.metadata:
            metadata['extra_metadata'] = chunk.metadata
        
        return metadata
    
    def _create_chunk_id(self, chunk: CodeChunk) -> str:
        """创建块ID"""
        # 使用文件路径和行号创建唯一ID
        safe_path = chunk.file_path.replace('/', '_').replace('\\', '_')
        return f"{safe_path}_{chunk.start_line}_{chunk.end_line}_{chunk.chunk_type.value}"
    
    def _extract_content_from_document(self, document: str) -> str:
        """从文档中提取内容"""
        # 找到内容开始的位置（在元数据之后）
        lines = document.split('\n')
        content_start = 0
        
        for i, line in enumerate(lines):
            if line.strip() == '':
                content_start = i + 1
                break
        
        return '\n'.join(lines[content_start:])
    
    async def _save_repository_info(self, repo_info: RepositoryInfo):
        """保存仓库信息"""
        # 可以保存到数据库或文件中
        # 这里暂时保存到集合的元数据中
        if self.collection:
            self.collection.modify(
                metadata={
                    "repository_info": repo_info.to_dict(),
                    "updated_at": datetime.now().isoformat()
                }
            )