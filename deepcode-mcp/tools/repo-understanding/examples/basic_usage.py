#!/usr/bin/env python3
"""
基本使用示例
"""
import asyncio
import os
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexer import RepositoryIndexer
from src.analyzer import CodeAnalyzer

async def main():
    """主函数"""
    # 1. 分析仓库结构
    print("1. Analyzing repository structure...")
    analyzer = CodeAnalyzer()
    
    # 使用当前项目作为示例
    repo_path = Path(__file__).parent.parent
    repo_info = await analyzer.analyze_repository(str(repo_path))
    
    print(f"Repository: {repo_info.name}")
    print(f"Total files: {repo_info.total_files}")
    print(f"Total lines: {repo_info.total_lines:,}")
    print(f"Languages: {', '.join(repo_info.languages.keys())}")
    print()
    
    # 2. 索引仓库
    print("2. Indexing repository...")
    indexer = RepositoryIndexer("example_repo")
    await indexer.initialize()
    
    result = await indexer.index_repository(str(repo_path))
    print(f"Indexed {result['indexed_files']} files")
    print(f"Created {result['total_chunks']} code chunks")
    print()
    
    # 3. 搜索代码
    print("3. Searching for code...")
    
    # 搜索特定功能
    query = "extract code chunks"
    print(f"Searching for: '{query}'")
    search_results = await indexer.search(query, n_results=3)
    
    for i, result in enumerate(search_results, 1):
        print(f"\nResult {i} (Score: {result.score:.3f}):")
        print(f"File: {result.chunk.file_path}")
        print(f"Type: {result.chunk.chunk_type.value}")
        if result.chunk.metadata.get('name'):
            print(f"Name: {result.chunk.metadata['name']}")
        print(f"Lines: {result.chunk.start_line}-{result.chunk.end_line}")
        
        # 显示代码片段
        lines = result.chunk.content.split('\n')[:5]
        print("Code preview:")
        for line in lines:
            print(f"  {line}")
        if len(result.chunk.content.split('\n')) > 5:
            print("  ...")

if __name__ == "__main__":
    asyncio.run(main())