#!/usr/bin/env python3
"""
Repository Understanding Agent - Main Entry Point
"""
import os
import sys
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.tools import mcp

if __name__ == "__main__":
    print("🚀 Repository Understanding MCP Agent")
    print("📚 This agent can index and understand code repositories\n")
    print("Available tools:")
    print("  • index_repository    - Index a repository for search and analysis")
    print("  • search_code        - Search for code in the indexed repository")
    print("  • analyze_structure  - Analyze repository structure")
    print("  • explain_code       - Explain code functionality")
    print("  • find_similar_code  - Find similar code sections\n")
    
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Using default embeddings.")
        print("   For better results, set your OpenAI API key:\n")
        print("   export OPENAI_API_KEY='your-key-here'\n")
    
    # 运行MCP服务器
    mcp.run()