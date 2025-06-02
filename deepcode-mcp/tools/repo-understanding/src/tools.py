"""
MCP工具定义
"""
import os
from typing import Optional
from fastmcp import FastMCP

from .indexer import RepositoryIndexer
from .analyzer import CodeAnalyzer
from .utils import format_file_tree

# 创建FastMCP实例
mcp = FastMCP("repo-understanding-agent")

# 全局索引器
current_indexer: Optional[RepositoryIndexer] = None

@mcp.tool()
async def index_repository(repo_path: str, collection_name: Optional[str] = None) -> str:
    """
    Index a code repository for understanding and search.
    
    Args:
        repo_path: Path to the repository
        collection_name: Name for the vector database collection (optional)
    
    Returns:
        Status message with indexing statistics
    """
    global current_indexer
    
    # 验证路径
    if not os.path.exists(repo_path):
        return f"❌ Error: Repository path does not exist: {repo_path}"
    
    if not os.path.isdir(repo_path):
        return f"❌ Error: Path is not a directory: {repo_path}"
    
    # 生成collection名称
    if not collection_name:
        repo_name = os.path.basename(os.path.abspath(repo_path))
        from datetime import datetime
        collection_name = f"repo_{repo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建索引器
    current_indexer = RepositoryIndexer(collection_name)
    await current_indexer.initialize()
    
    # 索引仓库
    try:
        result = await current_indexer.index_repository(repo_path)
        
        repo_info = result['repository']
        msg = f"✅ Successfully indexed repository: {repo_info['name']}\n\n"
        msg += f"📊 Statistics:\n"
        msg += f"  • Total files: {repo_info['total_files']}\n"
        msg += f"  • Total lines: {repo_info['total_lines']:,}\n"
        msg += f"  • Indexed files: {result['indexed_files']}\n"
        msg += f"  • Code chunks: {result['total_chunks']}\n\n"
        
        if repo_info['languages']:
            msg += f"📝 Languages:\n"
            for lang, lines in sorted(repo_info['languages'].items(), 
                                    key=lambda x: x[1], reverse=True):
                percentage = (lines / repo_info['total_lines']) * 100
                msg += f"  • {lang}: {lines:,} lines ({percentage:.1f}%)\n"
        
        if result['errors']:
            msg += f"\n⚠️ Errors ({len(result['errors'])} files):\n"
            for err in result['errors'][:5]:  # 只显示前5个错误
                msg += f"  • {err['file']}: {err['error']}\n"
        
        msg += f"\n💾 Collection: {collection_name}"
        
        return msg
        
    except Exception as e:
        return f"❌ Error indexing repository: {str(e)}"

@mcp.tool()
async def search_code(query: str, 
                     max_results: int = 5,
                     language: Optional[str] = None,
                     file_pattern: Optional[str] = None) -> str:
    """
    Search the indexed repository for relevant code.
    
    Args:
        query: Search query (natural language or code snippet)
        max_results: Maximum number of results to return
        language: Filter by programming language (optional)
        file_pattern: Filter by file path pattern (optional)
    
    Returns:
        Relevant code snippets with context
    """
    global current_indexer
    
    if not current_indexer:
        return "❌ Error: No repository indexed. Please run index_repository first."
    
    try:
        # 构建过滤器
        filter_dict = {}
        if language:
            filter_dict['language'] = language
        if file_pattern:
            filter_dict['file_path'] = {'$contains': file_pattern}
        
        # 搜索
        results = await current_indexer.search(
            query, 
            n_results=max_results,
            filter_dict=filter_dict if filter_dict else None
        )
        
        if not results:
            return "No results found for your query."
        
        msg = f"🔍 Found {len(results)} relevant code sections for: {query}\n\n"
        
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            msg += f"### Result {i} (Score: {result.score:.3f})\n"
            msg += f"📄 File: {chunk.file_path}\n"
            msg += f"🔧 Language: {chunk.language}\n"
            msg += f"📍 Lines: {chunk.start_line}-{chunk.end_line}\n"
            msg += f"🏷️ Type: {chunk.chunk_type.value}\n"
            
            # 添加额外的元数据
            if chunk.metadata.get('name'):
                msg += f"📛 Name: {chunk.metadata['name']}\n"
            if chunk.metadata.get('methods'):
                msg += f"🔨 Methods: {', '.join(m['name'] for m in chunk.metadata['methods'])}\n"
            
            # 代码内容（限制显示行数）
            msg += f"\n```{chunk.language}\n"
            lines = chunk.content.split('\n')
            if len(lines) > 30:
                msg += '\n'.join(lines[:30])
                msg += f"\n... ({len(lines) - 30} more lines)\n"
            else:
                msg += chunk.content
            msg += "\n```\n\n"
        
        return msg
        
    except Exception as e:
        return f"❌ Error searching repository: {str(e)}"

@mcp.tool()
async def analyze_structure(repo_path: str, max_depth: int = 3) -> str:
    """
    Analyze the structure and architecture of a repository.
    
    Args:
        repo_path: Path to the repository
        max_depth: Maximum depth for directory tree display
    
    Returns:
        Repository structure analysis
    """
    analyzer = CodeAnalyzer()
    
    try:
        repo_info = await analyzer.analyze_repository(repo_path)
        
        msg = f"📊 Repository Analysis: {repo_info.name}\n"
        msg += f"{'=' * 50}\n\n"
        
        if repo_info.description:
            msg += f"📝 Description:\n{repo_info.description}\n\n"
        
        msg += f"📈 Overview:\n"
        msg += f"  • Total files: {repo_info.total_files}\n"
        msg += f"  • Total lines: {repo_info.total_lines:,}\n"
        msg += f"  • Main language: {max(repo_info.languages.items(), key=lambda x: x[1])[0] if repo_info.languages else 'Unknown'}\n\n"
        
        if repo_info.languages:
            msg += f"🔤 Language Distribution:\n"
            for lang, lines in sorted(repo_info.languages.items(), 
                                    key=lambda x: x[1], reverse=True):
                percentage = (lines / repo_info.total_lines) * 100
                bar_length = int(percentage / 2)
                bar = '█' * bar_length + '░' * (50 - bar_length)
                msg += f"  {lang:12} {bar} {percentage:5.1f}% ({lines:,} lines)\n"
        
        msg += f"\n📁 Directory Structure:\n"
        msg += format_file_tree(repo_info.structure, max_depth=max_depth)
        
        return msg
        
    except Exception as e:
        return f"❌ Error analyzing repository: {str(e)}"

@mcp.tool()
async def explain_code(query: str, context_size: int = 3) -> str:
    """
    Explain code functionality based on the indexed repository.
    
    Args:
        query: What to explain (e.g., "How does authentication work?")
        context_size: Number of relevant code sections to analyze
    
    Returns:
        Explanation based on repository code
    """
    global current_indexer
    
    if not current_indexer:
        return "❌ Error: No repository indexed. Please run index_repository first."
    
    try:
        # 搜索相关代码
        results = await current_indexer.search(query, n_results=context_size)
        
        if not results:
            return "No relevant code found for your query."
        
        msg = f"🤔 Explaining: {query}\n"
        msg += f"{'=' * 50}\n\n"
        msg += f"Based on {len(results)} relevant code sections:\n\n"
        
        # 分析找到的代码
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            msg += f"### Code Section {i}\n"
            msg += f"📄 {chunk.file_path} "
            msg += f"(Lines {chunk.start_line}-{chunk.end_line})\n"
            
            # 根据代码类型提供解释
            if chunk.chunk_type.value == 'class':
                msg += f"🏗️ Class Definition: `{chunk.metadata.get('name', 'Unknown')}`\n"
                if chunk.metadata.get('methods'):
                    msg += f"   Methods: {', '.join(m['name'] for m in chunk.metadata['methods'])}\n"
                if chunk.metadata.get('bases'):
                    msg += f"   Inherits from: {', '.join(chunk.metadata['bases'])}\n"
                    
            elif chunk.chunk_type.value == 'function':
                msg += f"⚡ Function: `{chunk.metadata.get('name', 'Unknown')}`\n"
                if chunk.metadata.get('args'):
                    msg += f"   Parameters: {', '.join(chunk.metadata['args'])}\n"
                if chunk.metadata.get('decorators'):
                    msg += f"   Decorators: {', '.join(chunk.metadata['decorators'])}\n"
                    
            elif chunk.chunk_type.value == 'imports':
                msg += f"📦 Import Section\n"
                if chunk.metadata.get('imports'):
                    msg += f"   Imports: {', '.join(chunk.metadata['imports'][:5])}"
                    if len(chunk.metadata['imports']) > 5:
                        msg += f" ... and {len(chunk.metadata['imports']) - 5} more"
                    msg += "\n"
            
            # 显示代码片段
            msg += f"\n```{chunk.language}\n"
            lines = chunk.content.split('\n')[:10]  # 只显示前10行
            msg += '\n'.join(lines)
            if len(chunk.content.split('\n')) > 10:
                msg += "\n..."
            msg += "\n```\n\n"
        
        msg += "💡 Summary: These code sections are the most relevant to your query. "
        msg += "They show how the repository implements the functionality you're asking about."
        
        return msg
        
    except Exception as e:
        return f"❌ Error explaining code: {str(e)}"

@mcp.tool()
async def find_similar_code(file_path: str, 
                          line_number: int,
                          max_results: int = 5) -> str:
    """
    Find code similar to a specific code section.
    
    Args:
        file_path: Path to the file
        line_number: Line number in the file
        max_results: Maximum number of similar sections to find
    
    Returns:
        Similar code sections
    """
    global current_indexer
    
    if not current_indexer:
        return "❌ Error: No repository indexed. Please run index_repository first."
    
    try:
        # 构建chunk ID来查找
        # 这是一个简化的实现，实际应该查找包含该行的chunk
        results = await current_indexer.search(
            f"file:{file_path} line:{line_number}",
            n_results=max_results
        )
        
        if not results:
            return f"No code found at {file_path}:{line_number}"
        
        msg = f"🔍 Similar code to {file_path}:{line_number}\n\n"
        
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            if i == 1:
                msg += f"### Original Code\n"
            else:
                msg += f"### Similar Code {i-1} (Score: {result.score:.3f})\n"
            
            msg += f"📄 {chunk.file_path} "
            msg += f"(Lines {chunk.start_line}-{chunk.end_line})\n"
            msg += f"🏷️ Type: {chunk.chunk_type.value}\n"
            
            msg += f"\n```{chunk.language}\n"
            msg += chunk.content[:500]  # 限制长度
            if len(chunk.content) > 500:
                msg += "\n..."
            msg += "\n```\n\n"
        
        return msg
        
    except Exception as e:
        return f"❌ Error finding similar code: {str(e)}"