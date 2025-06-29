#!/usr/bin/env python3
"""
Code Reference Indexer MCP Tool - 代码参考索引器 MCP 工具

专门负责在indexes文件夹中搜索相关的index内容，并整理格式化提供给LLM用于代码实现参考
Specialized in searching relevant index content in indexes folder and formatting it for LLM code implementation reference

核心功能：
1. 搜索indexes文件夹中的所有JSON文件
2. 根据目标文件路径和功能需求匹配相关的参考代码
3. 格式化输出相关的代码示例、函数和概念
4. 提供结构化的参考信息供LLM使用

Core Features:
1. Search all JSON files in indexes folder
2. Match relevant reference code based on target file path and functionality requirements
3. Format output of relevant code examples, functions and concepts
4. Provide structured reference information for LLM use
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

# 导入MCP相关模块
from mcp.server.fastmcp import FastMCP
import mcp.types as types

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastMCP服务器实例
mcp = FastMCP("code-reference-indexer")

# 全局变量：索引缓存
INDEX_CACHE = {}
INDEXES_DIRECTORY = None


@dataclass
class CodeReference:
    """代码参考信息结构"""
    file_path: str
    file_type: str
    main_functions: List[str]
    key_concepts: List[str]
    dependencies: List[str]
    summary: str
    lines_of_code: int
    repo_name: str
    confidence_score: float = 0.0


@dataclass
class RelationshipInfo:
    """关系信息结构"""
    repo_file_path: str
    target_file_path: str
    relationship_type: str
    confidence_score: float
    helpful_aspects: List[str]
    potential_contributions: List[str]
    usage_suggestions: str


def initialize_indexes_directory(indexes_dir: str = None):
    """初始化索引目录"""
    global INDEXES_DIRECTORY
    if indexes_dir is None:
        # 默认查找agent_folders/papers/1/indexes目录
        current_dir = Path.cwd()
        INDEXES_DIRECTORY = current_dir / "agent_folders" / "papers" / "1" / "indexes"
    else:
        INDEXES_DIRECTORY = Path(indexes_dir).resolve()
    
    if not INDEXES_DIRECTORY.exists():
        logger.warning(f"索引目录不存在: {INDEXES_DIRECTORY}")
    else:
        logger.info(f"索引目录初始化: {INDEXES_DIRECTORY}")


def load_index_files() -> Dict[str, Dict]:
    """加载所有索引文件到缓存"""
    global INDEX_CACHE
    
    if INDEXES_DIRECTORY is None:
        initialize_indexes_directory()
    
    if not INDEXES_DIRECTORY.exists():
        return {}
    
    INDEX_CACHE = {}
    
    for index_file in INDEXES_DIRECTORY.glob("*.json"):
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
                INDEX_CACHE[index_file.stem] = index_data
                logger.info(f"加载索引文件: {index_file.name}")
        except Exception as e:
            logger.error(f"加载索引文件失败 {index_file.name}: {e}")
    
    return INDEX_CACHE


def extract_code_references(index_data: Dict) -> List[CodeReference]:
    """从索引数据中提取代码参考信息"""
    references = []
    
    repo_name = index_data.get("repo_name", "Unknown")
    file_summaries = index_data.get("file_summaries", [])
    
    for file_summary in file_summaries:
        reference = CodeReference(
            file_path=file_summary.get("file_path", ""),
            file_type=file_summary.get("file_type", ""),
            main_functions=file_summary.get("main_functions", []),
            key_concepts=file_summary.get("key_concepts", []),
            dependencies=file_summary.get("dependencies", []),
            summary=file_summary.get("summary", ""),
            lines_of_code=file_summary.get("lines_of_code", 0),
            repo_name=repo_name
        )
        references.append(reference)
    
    return references


def extract_relationships(index_data: Dict) -> List[RelationshipInfo]:
    """从索引数据中提取关系信息"""
    relationships = []
    
    relationship_list = index_data.get("relationships", [])
    
    for rel in relationship_list:
        relationship = RelationshipInfo(
            repo_file_path=rel.get("repo_file_path", ""),
            target_file_path=rel.get("target_file_path", ""),
            relationship_type=rel.get("relationship_type", ""),
            confidence_score=rel.get("confidence_score", 0.0),
            helpful_aspects=rel.get("helpful_aspects", []),
            potential_contributions=rel.get("potential_contributions", []),
            usage_suggestions=rel.get("usage_suggestions", "")
        )
        relationships.append(relationship)
    
    return relationships


def calculate_relevance_score(target_file: str, reference: CodeReference, keywords: List[str] = None) -> float:
    """计算参考代码与目标文件的相关性得分"""
    score = 0.0
    
    # 文件名相似性
    target_name = Path(target_file).stem.lower()
    ref_name = Path(reference.file_path).stem.lower()
    
    if target_name in ref_name or ref_name in target_name:
        score += 0.3
    
    # 文件类型匹配
    target_extension = Path(target_file).suffix
    ref_extension = Path(reference.file_path).suffix
    
    if target_extension == ref_extension:
        score += 0.2
    
    # 关键词匹配
    if keywords:
        keyword_matches = 0
        total_searchable_text = (
            " ".join(reference.key_concepts) + " " +
            " ".join(reference.main_functions) + " " +
            reference.summary + " " +
            reference.file_type
        ).lower()
        
        for keyword in keywords:
            if keyword.lower() in total_searchable_text:
                keyword_matches += 1
        
        if keywords:
            score += (keyword_matches / len(keywords)) * 0.5
    
    return min(score, 1.0)


def find_relevant_references(
    target_file: str, 
    keywords: List[str] = None, 
    max_results: int = 10
) -> List[Tuple[CodeReference, float]]:
    """查找与目标文件相关的参考代码"""
    if not INDEX_CACHE:
        load_index_files()
    
    all_references = []
    
    # 从所有索引文件中收集参考信息
    for repo_name, index_data in INDEX_CACHE.items():
        references = extract_code_references(index_data)
        for ref in references:
            relevance_score = calculate_relevance_score(target_file, ref, keywords)
            if relevance_score > 0.1:  # 只保留有一定相关性的结果
                all_references.append((ref, relevance_score))
    
    # 按相关性得分排序
    all_references.sort(key=lambda x: x[1], reverse=True)
    
    return all_references[:max_results]


def find_direct_relationships(target_file: str) -> List[RelationshipInfo]:
    """查找与目标文件的直接关系"""
    if not INDEX_CACHE:
        load_index_files()
    
    relationships = []
    
    # 标准化目标文件路径（移除前缀rice/如果存在）
    normalized_target = target_file.replace("rice/", "").strip("/")
    
    # 从所有索引文件中收集关系信息
    for repo_name, index_data in INDEX_CACHE.items():
        repo_relationships = extract_relationships(index_data)
        for rel in repo_relationships:
            # 标准化关系中的目标文件路径
            normalized_rel_target = rel.target_file_path.replace("rice/", "").strip("/")
            
            # 检查目标文件路径匹配（支持多种匹配方式）
            if (normalized_target == normalized_rel_target or 
                normalized_target in normalized_rel_target or 
                normalized_rel_target in normalized_target or
                target_file in rel.target_file_path or 
                rel.target_file_path in target_file):
                relationships.append(rel)
    
    # 按置信度排序
    relationships.sort(key=lambda x: x.confidence_score, reverse=True)
    
    return relationships


def format_reference_output(
    target_file: str,
    relevant_refs: List[Tuple[CodeReference, float]],
    relationships: List[RelationshipInfo]
) -> str:
    """格式化参考信息输出"""
    output_lines = []
    
    output_lines.append(f"# 代码参考信息 - {target_file}")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # 直接关系信息
    if relationships:
        output_lines.append("## 🎯 直接关系参考 (Direct Relationships)")
        output_lines.append("")
        
        for i, rel in enumerate(relationships[:5], 1):
            output_lines.append(f"### {i}. {rel.repo_file_path}")
            output_lines.append(f"**关系类型**: {rel.relationship_type}")
            output_lines.append(f"**置信度**: {rel.confidence_score:.2f}")
            output_lines.append(f"**有用方面**: {', '.join(rel.helpful_aspects)}")
            output_lines.append(f"**潜在贡献**: {', '.join(rel.potential_contributions)}")
            output_lines.append(f"**使用建议**: {rel.usage_suggestions}")
            output_lines.append("")
    
    # 相关代码参考
    if relevant_refs:
        output_lines.append("## 📚 相关代码参考 (Relevant Code References)")
        output_lines.append("")
        
        for i, (ref, score) in enumerate(relevant_refs[:8], 1):
            output_lines.append(f"### {i}. {ref.file_path} (相关性: {score:.2f})")
            output_lines.append(f"**仓库**: {ref.repo_name}")
            output_lines.append(f"**文件类型**: {ref.file_type}")
            output_lines.append(f"**主要函数**: {', '.join(ref.main_functions[:5])}")
            output_lines.append(f"**关键概念**: {', '.join(ref.key_concepts[:8])}")
            output_lines.append(f"**依赖**: {', '.join(ref.dependencies[:6])}")
            output_lines.append(f"**代码行数**: {ref.lines_of_code}")
            output_lines.append(f"**摘要**: {ref.summary[:300]}...")
            output_lines.append("")
    
    # 实现建议
    output_lines.append("## 💡 实现建议 (Implementation Suggestions)")
    output_lines.append("")
    
    if relevant_refs:
        # 收集所有函数名和概念
        all_functions = set()
        all_concepts = set()
        all_dependencies = set()
        
        for ref, _ in relevant_refs[:5]:
            all_functions.update(ref.main_functions)
            all_concepts.update(ref.key_concepts)
            all_dependencies.update(ref.dependencies)
        
        output_lines.append("**可参考的函数名模式**:")
        for func in sorted(list(all_functions))[:10]:
            output_lines.append(f"- {func}")
        output_lines.append("")
        
        output_lines.append("**重要概念和模式**:")
        for concept in sorted(list(all_concepts))[:15]:
            output_lines.append(f"- {concept}")
        output_lines.append("")
        
        output_lines.append("**可能需要的依赖**:")
        for dep in sorted(list(all_dependencies))[:10]:
            output_lines.append(f"- {dep}")
        output_lines.append("")
    
    output_lines.append("## 🚀 下一步行动 (Next Actions)")
    output_lines.append("1. 分析以上参考代码的设计模式和架构风格")
    output_lines.append("2. 确定需要实现的核心功能和接口")
    output_lines.append("3. 选择合适的依赖库和工具")
    output_lines.append("4. 设计与现有代码风格一致的实现方案")
    output_lines.append("5. 开始编写具体的代码实现")
    
    return "\n".join(output_lines)


# ==================== MCP工具定义 ====================

@mcp.tool()
async def search_reference_code(
    target_file: str,
    keywords: str = "",
    max_results: int = 10
) -> str:
    """
    在索引文件中搜索与目标文件相关的参考代码
    
    Args:
        target_file: 目标文件路径（要实现的文件）
        keywords: 搜索关键词，用逗号分隔
        max_results: 最大返回结果数量
    
    Returns:
        格式化的参考代码信息JSON字符串
    """
    try:
        # 解析关键词
        keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()] if keywords else []
        
        # 查找相关参考代码
        relevant_refs = find_relevant_references(target_file, keyword_list, max_results)
        
        # 查找直接关系
        relationships = find_direct_relationships(target_file)
        
        # 格式化输出
        formatted_output = format_reference_output(target_file, relevant_refs, relationships)
        
        result = {
            "status": "success",
            "target_file": target_file,
            "keywords_used": keyword_list,
            "total_references_found": len(relevant_refs),
            "total_relationships_found": len(relationships),
            "formatted_content": formatted_output,
            "indexes_loaded": list(INDEX_CACHE.keys())
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        result = {
            "status": "error",
            "message": f"搜索参考代码失败: {str(e)}",
            "target_file": target_file
        }
        return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_all_available_references() -> str:
    """
    获取所有可用的参考代码索引信息
    
    Returns:
        所有可用参考代码的概览信息JSON字符串
    """
    try:
        if not INDEX_CACHE:
            load_index_files()
        
        overview = {
            "total_repos": len(INDEX_CACHE),
            "repositories": {}
        }
        
        for repo_name, index_data in INDEX_CACHE.items():
            repo_info = {
                "repo_name": index_data.get("repo_name", repo_name),
                "total_files": index_data.get("total_files", 0),
                "file_types": [],
                "main_concepts": [],
                "total_relationships": len(index_data.get("relationships", []))
            }
            
            # 收集文件类型和概念
            file_summaries = index_data.get("file_summaries", [])
            file_types = set()
            concepts = set()
            
            for file_summary in file_summaries:
                file_types.add(file_summary.get("file_type", "Unknown"))
                concepts.update(file_summary.get("key_concepts", []))
            
            repo_info["file_types"] = sorted(list(file_types))
            repo_info["main_concepts"] = sorted(list(concepts))[:20]  # 限制概念数量
            
            overview["repositories"][repo_name] = repo_info
        
        result = {
            "status": "success",
            "overview": overview,
            "indexes_directory": str(INDEXES_DIRECTORY) if INDEXES_DIRECTORY else "Not set"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        result = {
            "status": "error",
            "message": f"获取参考代码概览失败: {str(e)}"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def set_indexes_directory(indexes_path: str) -> str:
    """
    设置索引文件目录路径
    
    Args:
        indexes_path: 索引文件目录路径
    
    Returns:
        设置结果的JSON字符串
    """
    try:
        global INDEXES_DIRECTORY, INDEX_CACHE
        
        INDEXES_DIRECTORY = Path(indexes_path).resolve()
        
        if not INDEXES_DIRECTORY.exists():
            result = {
                "status": "error",
                "message": f"索引目录不存在: {indexes_path}"
            }
        else:
            # 重新加载索引文件
            INDEX_CACHE = {}
            load_index_files()
            
            result = {
                "status": "success",
                "indexes_directory": str(INDEXES_DIRECTORY),
                "loaded_indexes": list(INDEX_CACHE.keys()),
                "total_loaded": len(INDEX_CACHE)
            }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        result = {
            "status": "error",
            "message": f"设置索引目录失败: {str(e)}"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    # 初始化索引目录
    initialize_indexes_directory()
    
    # 预加载索引文件
    load_index_files()
    
    # 运行MCP服务器
    mcp.run()


if __name__ == "__main__":
    main() 