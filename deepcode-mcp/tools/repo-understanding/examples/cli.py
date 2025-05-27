#!/usr/bin/env python3
"""
CLI for Repository Understanding Agent
"""

import click
import asyncio
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

@click.group()
def cli():
    """Repository Understanding Agent CLI"""
    pass

@cli.command()
@click.argument('repo_path')
@click.option('--name', help='Collection name for the index')
def index(repo_path, name):
    """Index a repository for analysis"""
    async def run():
        server_params = StdioServerParameters(
            command="python",
            args=["main.py"],
            env={**os.environ}
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                args = {"repo_path": repo_path}
                if name:
                    args["collection_name"] = name
                
                result = await session.call_tool("index_repository", arguments=args)
                print(result.content[0].text)
    
    asyncio.run(run())

@cli.command()
@click.argument('query')
@click.option('--max-results', default=5, help='Maximum number of results')
@click.option('--language', help='Filter by programming language')
@click.option('--file-pattern', help='Filter by file path pattern')
def search(query, max_results, language, file_pattern):
    """Search for code in the indexed repository"""
    async def run():
        server_params = StdioServerParameters(
            command="python",
            args=["main.py"],
            env={**os.environ}
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                args = {
                    "query": query,
                    "max_results": max_results
                }
                if language:
                    args["language"] = language
                if file_pattern:
                    args["file_pattern"] = file_pattern
                
                result = await session.call_tool("search_code", arguments=args)
                print(result.content[0].text)
    
    asyncio.run(run())

@cli.command()
@click.argument('repo_path')
@click.option('--max-depth', default=3, help='Maximum depth for directory tree')
def analyze(repo_path, max_depth):
    """Analyze repository structure"""
    async def run():
        server_params = StdioServerParameters(
            command="python",
            args=["main.py"],
            env={**os.environ}
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "analyze_structure",
                    arguments={
                        "repo_path": repo_path,
                        "max_depth": max_depth
                    }
                )
                print(result.content[0].text)
    
    asyncio.run(run())

@cli.command()
@click.argument('query')
@click.option('--context-size', default=3, help='Number of code sections to analyze')
def explain(query, context_size):
    """Explain code functionality"""
    async def run():
        server_params = StdioServerParameters(
            command="python",
            args=["main.py"],
            env={**os.environ}
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "explain_code",
                    arguments={
                        "query": query,
                        "context_size": context_size
                    }
                )
                print(result.content[0].text)
    
    asyncio.run(run())

@cli.command()
@click.argument('file_path')
@click.argument('line_number', type=int)
@click.option('--max-results', default=5, help='Maximum similar sections to find')
def similar(file_path, line_number, max_results):
    """Find similar code sections"""
    async def run():
        server_params = StdioServerParameters(
            command="python",
            args=["main.py"],
            env={**os.environ}
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "find_similar_code",
                    arguments={
                        "file_path": file_path,
                        "line_number": line_number,
                        "max_results": max_results
                    }
                )
                print(result.content[0].text)
    
    asyncio.run(run())

@cli.command()
def list_tools():
    """List all available tools"""
    async def run():
        server_params = StdioServerParameters(
            command="python",
            args=["main.py"],
            env={**os.environ}
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                tools = await session.list_tools()
                print("\n📋 Available tools:\n")
                for tool in tools:
                    print(f"  • {tool.name}")
                    print(f"    {tool.description}")
                    if hasattr(tool, 'input_schema') and tool.input_schema.get('properties'):
                        print(f"    Parameters:")
                        for param, details in tool.input_schema['properties'].items():
                            required = param in tool.input_schema.get('required', [])
                            req_mark = '*' if required else ''
                            print(f"      - {param}{req_mark}: {details.get('description', 'No description')}")
                    print()
    
    asyncio.run(run())

if __name__ == '__main__':
    cli()