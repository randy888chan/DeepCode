# github_mcp_server.py
import asyncio
import json
import base64
import os
from typing import Dict, List, Any, Optional, Set
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

class GitHubRepositoryReader:
    """Complete GitHub Repository Reader for MCP Server"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token
        self.server = Server("github-repo-reader")
        
        # Supported code file extensions
        self.code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.cs', 
            '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.r', '.m',
            '.h', '.hpp', '.cc', '.cxx', '.html', '.css', '.scss', '.sass',
            '.vue', '.svelte', '.sql', '.sh', '.bash', '.ps1', '.yaml', '.yml',
            '.json', '.xml', '.toml', '.ini', '.cfg', '.conf', '.md', '.rst',
            '.dockerfile', '.gradle', '.maven', '.cmake', '.makefile'
        }
        
        # Files to always include
        self.special_files = {
            'readme.md', 'license', 'dockerfile', 'makefile', 'cmakelists.txt',
            'package.json', 'requirements.txt', 'setup.py', 'cargo.toml',
            'go.mod', 'pom.xml', 'build.gradle', '.gitignore', '.env.example'
        }
        
        self.setup_handlers()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for GitHub API requests"""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MCP-GitHub-Reader/1.0"
        }
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        return headers
    
    def _is_code_file(self, filename: str) -> bool:
        """Check if a file should be included based on extension or name"""
        filename_lower = filename.lower()
        
        # Check special files
        if filename_lower in self.special_files:
            return True
        
        # Check file extension
        for ext in self.code_extensions:
            if filename_lower.endswith(ext):
                return True
        
        # Check files without extension that might be code
        if '.' not in filename and filename_lower in ['dockerfile', 'makefile', 'rakefile']:
            return True
        
        return False
    
    async def _get_file_content(self, url: str) -> Optional[str]:
        """Get file content from GitHub API"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self._get_headers())
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("type") == "file" and data.get("content"):
                        try:
                            content = base64.b64decode(data["content"]).decode("utf-8")
                            return content
                        except UnicodeDecodeError:
                            # Skip binary files
                            return None
                return None
        except Exception as e:
            print(f"Error reading file {url}: {e}")
            return None
    
    async def _get_directory_contents(self, owner: str, repo: str, path: str = "", branch: str = "main") -> List[Dict]:
        """Get contents of a directory"""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": branch}
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self._get_headers(), params=params)
                
                if response.status_code == 200:
                    return response.json()
                return []
        except Exception as e:
            print(f"Error reading directory {path}: {e}")
            return []
    
    async def _read_all_files_recursive(self, owner: str, repo: str, path: str = "", 
                                      branch: str = "main", max_depth: int = 10, 
                                      current_depth: int = 0) -> Dict[str, str]:
        """Recursively read all code files in a repository"""
        if current_depth > max_depth:
            return {}
        
        all_files = {}
        contents = await self._get_directory_contents(owner, repo, path, branch)
        
        for item in contents:
            item_path = item["path"]
            item_name = item["name"]
            item_type = item["type"]
            
            if item_type == "file" and self._is_code_file(item_name):
                # Read file content
                print(f"Reading file: {item_path}")
                content = await self._get_file_content(item["url"])
                if content is not None:
                    all_files[item_path] = content
            
            elif item_type == "dir":
                # Skip common non-code directories
                skip_dirs = {
                    'node_modules', '.git', '__pycache__', '.pytest_cache',
                    'venv', 'env', '.env', 'build', 'dist', 'target',
                    '.idea', '.vscode', 'logs', 'tmp', 'temp'
                }
                
                if item_name.lower() not in skip_dirs:
                    print(f"Entering directory: {item_path}")
                    subdir_files = await self._read_all_files_recursive(
                        owner, repo, item_path, branch, max_depth, current_depth + 1
                    )
                    all_files.update(subdir_files)
        
        return all_files
    
    def setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            return [
                Resource(
                    uri="github://repository",
                    name="GitHub Repository Complete Reader",
                    mimeType="application/json"
                )
            ]
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="read_entire_repository",
                    description="Read all code files from a GitHub repository",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "Repository owner"},
                            "repo": {"type": "string", "description": "Repository name"},
                            "branch": {"type": "string", "description": "Branch name", "default": "main"},
                            "max_depth": {"type": "integer", "description": "Maximum directory depth", "default": 10},
                            "include_patterns": {"type": "array", "items": {"type": "string"}, "description": "File patterns to include"},
                            "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "File patterns to exclude"}
                        },
                        "required": ["owner", "repo"]
                    }
                ),
                Tool(
                    name="get_repository_summary",
                    description="Get a summary of repository structure and file types",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "Repository owner"},
                            "repo": {"type": "string", "description": "Repository name"},
                            "branch": {"type": "string", "description": "Branch name", "default": "main"}
                        },
                        "required": ["owner", "repo"]
                    }
                ),
                Tool(
                    name="read_files_by_extension",
                    description="Read all files with specific extensions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "Repository owner"},
                            "repo": {"type": "string", "description": "Repository name"},
                            "extensions": {"type": "array", "items": {"type": "string"}, "description": "File extensions to read (e.g., ['.py', '.js'])"},
                            "branch": {"type": "string", "description": "Branch name", "default": "main"}
                        },
                        "required": ["owner", "repo", "extensions"]
                    }
                ),
                Tool(
                    name="create_codebase_index",
                    description="Create an index/map of the entire codebase",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "Repository owner"},
                            "repo": {"type": "string", "description": "Repository name"},
                            "branch": {"type": "string", "description": "Branch name", "default": "main"},
                            "include_content_preview": {"type": "boolean", "description": "Include content preview", "default": True}
                        },
                        "required": ["owner", "repo"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name == "read_entire_repository":
                return await self.read_entire_repository(**arguments)
            elif name == "get_repository_summary":
                return await self.get_repository_summary(**arguments)
            elif name == "read_files_by_extension":
                return await self.read_files_by_extension(**arguments)
            elif name == "create_codebase_index":
                return await self.create_codebase_index(**arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
    
    async def read_entire_repository(self, owner: str, repo: str, branch: str = "main", 
                                   max_depth: int = 10, include_patterns: List[str] = None,
                                   exclude_patterns: List[str] = None) -> List[TextContent]:
        """Read all code files from a repository"""
        print(f"Starting to read entire repository: {owner}/{repo}")
        
        try:
            all_files = await self._read_all_files_recursive(owner, repo, "", branch, max_depth)
            
            # Apply include/exclude patterns if provided
            if include_patterns or exclude_patterns:
                filtered_files = {}
                for file_path, content in all_files.items():
                    include = True
                    
                    if include_patterns:
                        include = any(pattern in file_path for pattern in include_patterns)
                    
                    if exclude_patterns and include:
                        include = not any(pattern in file_path for pattern in exclude_patterns)
                    
                    if include:
                        filtered_files[file_path] = content
                
                all_files = filtered_files
            
            # Format the output
            result = {
                "repository": f"{owner}/{repo}",
                "branch": branch,
                "total_files": len(all_files),
                "files": {}
            }
            
            for file_path, content in all_files.items():
                result["files"][file_path] = {
                    "content": content,
                    "lines": len(content.splitlines()),
                    "size": len(content)
                }
            
            return [TextContent(
                type="text",
                text=f"Complete repository content:\n\n{json.dumps(result, indent=2, ensure_ascii=False)}"
            )]
        
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error reading repository: {str(e)}"
            )]
    
    async def get_repository_summary(self, owner: str, repo: str, branch: str = "main") -> List[TextContent]:
        """Get repository structure summary"""
        try:
            all_files = await self._read_all_files_recursive(owner, repo, "", branch)
            
            # Analyze file types
            file_types = {}
            total_lines = 0
            total_size = 0
            
            for file_path, content in all_files.items():
                ext = os.path.splitext(file_path)[1].lower() or 'no_extension'
                if ext not in file_types:
                    file_types[ext] = {"count": 0, "lines": 0, "size": 0}
                
                lines = len(content.splitlines())
                size = len(content)
                
                file_types[ext]["count"] += 1
                file_types[ext]["lines"] += lines
                file_types[ext]["size"] += size
                
                total_lines += lines
                total_size += size
            
            summary = {
                "repository": f"{owner}/{repo}",
                "branch": branch,
                "summary": {
                    "total_files": len(all_files),
                    "total_lines": total_lines,
                    "total_size_bytes": total_size,
                    "file_types": file_types
                },
                "directory_structure": list(set(os.path.dirname(path) for path in all_files.keys() if os.path.dirname(path)))
            }
            
            return [TextContent(
                type="text",
                text=f"Repository summary:\n\n{json.dumps(summary, indent=2)}"
            )]
        
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error getting repository summary: {str(e)}"
            )]
    
    async def read_files_by_extension(self, owner: str, repo: str, extensions: List[str], 
                                    branch: str = "main") -> List[TextContent]:
        """Read files with specific extensions"""
        try:
            all_files = await self._read_all_files_recursive(owner, repo, "", branch)
            
            # Filter by extensions
            filtered_files = {}
            for file_path, content in all_files.items():
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in extensions:
                    filtered_files[file_path] = content
            
            result = {
                "repository": f"{owner}/{repo}",
                "branch": branch,
                "extensions": extensions,
                "matched_files": len(filtered_files),
                "files": {path: {"content": content, "lines": len(content.splitlines())} 
                         for path, content in filtered_files.items()}
            }
            
            return [TextContent(
                type="text",
                text=f"Files by extension:\n\n{json.dumps(result, indent=2, ensure_ascii=False)}"
            )]
        
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error reading files by extension: {str(e)}"
            )]
    
    async def create_codebase_index(self, owner: str, repo: str, branch: str = "main", 
                                  include_content_preview: bool = True) -> List[TextContent]:
        """Create a searchable index of the codebase"""
        try:
            all_files = await self._read_all_files_recursive(owner, repo, "", branch)
            
            index = {
                "repository": f"{owner}/{repo}",
                "branch": branch,
                "index_created_at": "timestamp_placeholder",
                "files": {}
            }
            
            for file_path, content in all_files.items():
                lines = content.splitlines()
                preview = lines[:5] if include_content_preview and lines else []
                
                # Extract some metadata
                file_info = {
                    "path": file_path,
                    "extension": os.path.splitext(file_path)[1],
                    "lines_count": len(lines),
                    "size_bytes": len(content),
                    "preview": preview if include_content_preview else None
                }
                
                # Try to extract functions/classes for code files
                if file_path.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c')):
                    file_info["code_elements"] = self._extract_code_elements(content, file_path)
                
                index["files"][file_path] = file_info
            
            return [TextContent(
                type="text",
                text=f"Codebase index:\n\n{json.dumps(index, indent=2, ensure_ascii=False)}"
            )]
        
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error creating codebase index: {str(e)}"
            )]
    
    def _extract_code_elements(self, content: str, file_path: str) -> List[Dict[str, str]]:
        """Extract functions, classes, etc. from code content"""
        elements = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Python
            if file_path.endswith('.py'):
                if line_stripped.startswith('def ') or line_stripped.startswith('class '):
                    elements.append({
                        "type": "function" if line_stripped.startswith('def ') else "class",
                        "name": line_stripped.split('(')[0].split(':')[0].replace('def ', '').replace('class ', ''),
                        "line": i + 1
                    })
            
            # JavaScript/TypeScript
            elif file_path.endswith(('.js', '.ts')):
                if 'function ' in line_stripped or '=>' in line_stripped:
                    elements.append({
                        "type": "function",
                        "name": line_stripped.split('(')[0].split(' ')[-1] if 'function ' in line_stripped else "arrow_function",
                        "line": i + 1
                    })
        
        return elements[:20]  # Limit to first 20 elements

async def main():
    """Main function to run the MCP server"""
    github_token = os.getenv("GITHUB_TOKEN")
    reader = GitHubRepositoryReader(github_token)
    
    async with stdio_server() as (read_stream, write_stream):
        await reader.server.run(
            read_stream,
            write_stream,
            reader.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())