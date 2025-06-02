# github_mcp_server.py
import asyncio
import json
from typing import Dict, List, Any
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    CallToolRequest,
    ListResourcesRequest,
    ListToolsRequest,
    ReadResourceRequest,
)

class GitHubMCPServer:
    def __init__(self, github_token: str = None):
        self.github_token = github_token
        self.server = Server("github-code-reader")
        self.setup_handlers()
    
    def setup_handlers(self):
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            return [
                Resource(
                    uri="github://repository",
                    name="GitHub Repository Browser",
                    mimeType="application/json"
                )
            ]
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="read_github_file",
                    description="Read a specific file from a GitHub repository",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "Repository owner"},
                            "repo": {"type": "string", "description": "Repository name"},
                            "path": {"type": "string", "description": "File path"},
                            "branch": {"type": "string", "description": "Branch name", "default": "main"}
                        },
                        "required": ["owner", "repo", "path"]
                    }
                ),
                Tool(
                    name="list_repository_files",
                    description="List files in a GitHub repository directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "Repository owner"},
                            "repo": {"type": "string", "description": "Repository name"},
                            "path": {"type": "string", "description": "Directory path", "default": ""},
                            "branch": {"type": "string", "description": "Branch name", "default": "main"}
                        },
                        "required": ["owner", "repo"]
                    }
                ),
                Tool(
                    name="search_code",
                    description="Search for code in a GitHub repository",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "Repository owner"},
                            "repo": {"type": "string", "description": "Repository name"},
                            "query": {"type": "string", "description": "Search query"},
                            "language": {"type": "string", "description": "Programming language filter"}
                        },
                        "required": ["owner", "repo", "query"]
                    }
                ),
                Tool(
                    name="analyze_repository_structure",
                    description="Analyze the overall structure of a repository",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "Repository owner"},
                            "repo": {"type": "string", "description": "Repository name"},
                            "branch": {"type": "string", "description": "Branch name", "default": "main"}
                        },
                        "required": ["owner", "repo"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name == "read_github_file":
                return await self.read_github_file(**arguments)
            elif name == "list_repository_files":
                return await self.list_repository_files(**arguments)
            elif name == "search_code":
                return await self.search_code(**arguments)
            elif name == "analyze_repository_structure":
                return await self.analyze_repository_structure(**arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
    
    async def read_github_file(self, owner: str, repo: str, path: str, branch: str = "main") -> List[TextContent]:
        """Read a specific file from GitHub"""
        headers = {}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": branch}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("type") == "file":
                    import base64
                    content = base64.b64decode(data["content"]).decode("utf-8")
                    return [TextContent(
                        type="text",
                        text=f"File: {path}\n\n{content}"
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"Error: {path} is not a file"
                    )]
            else:
                return [TextContent(
                    type="text",
                    text=f"Error reading file: {response.status_code} - {response.text}"
                )]
    
    async def list_repository_files(self, owner: str, repo: str, path: str = "", branch: str = "main") -> List[TextContent]:
        """List files in a repository directory"""
        headers = {}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": branch}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                files_info = []
                
                for item in data:
                    files_info.append({
                        "name": item["name"],
                        "type": item["type"],
                        "path": item["path"],
                        "size": item.get("size", 0)
                    })
                
                return [TextContent(
                    type="text",
                    text=f"Repository structure for {path}:\n\n" + 
                         json.dumps(files_info, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"Error listing files: {response.status_code} - {response.text}"
                )]
    
    async def search_code(self, owner: str, repo: str, query: str, language: str = None) -> List[TextContent]:
        """Search for code in a repository"""
        headers = {}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        search_query = f"{query} repo:{owner}/{repo}"
        if language:
            search_query += f" language:{language}"
        
        url = "https://api.github.com/search/code"
        params = {"q": search_query}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get("items", []):
                    results.append({
                        "path": item["path"],
                        "name": item["name"],
                        "html_url": item["html_url"]
                    })
                
                return [TextContent(
                    type="text",
                    text=f"Code search results:\n\n" + 
                         json.dumps(results, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"Error searching code: {response.status_code} - {response.text}"
                )]
    
    async def analyze_repository_structure(self, owner: str, repo: str, branch: str = "main") -> List[TextContent]:
        """Analyze repository structure recursively"""
        async def get_tree(path="", depth=0, max_depth=3):
            if depth > max_depth:
                return []
            
            headers = {}
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"
            
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
            params = {"ref": branch}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=params)
                
                if response.status_code != 200:
                    return []
                
                data = response.json()
                structure = []
                
                for item in data:
                    item_info = {
                        "name": item["name"],
                        "type": item["type"],
                        "path": item["path"]
                    }
                    
                    if item["type"] == "dir" and depth < max_depth:
                        item_info["children"] = await get_tree(item["path"], depth + 1, max_depth)
                    
                    structure.append(item_info)
                
                return structure
        
        structure = await get_tree()
        return [TextContent(
            type="text",
            text=f"Repository structure analysis:\n\n" + 
                 json.dumps(structure, indent=2)
        )]

async def main():
    import os
    github_token = os.getenv("GITHUB_TOKEN")
    server = GitHubMCPServer(github_token)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            server.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())