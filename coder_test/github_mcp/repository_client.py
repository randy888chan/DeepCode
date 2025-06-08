# repository_client.py
import asyncio
import json
import os
from dotenv import load_dotenv
from mcp.client import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class RepositoryCodeReader:
    """Client to interact with the GitHub repository MCP server"""
    
    def __init__(self):
        self.session = None
    
    async def connect(self):
        """Connect to the MCP server"""
        server_params = StdioServerParameters(
            command="python",
            args=["github_mcp_server.py"],
            env={"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")}
        )
        
        self.session = await stdio_client(server_params)
        print("✅ Connected to GitHub repository MCP server")
    
    async def read_entire_repo(self, owner: str, repo: str, branch: str = "main", 
                              max_depth: int = 10) -> dict:
        """Read all code files from a repository"""
        if not self.session:
            await self.connect()
        
        print(f"📚 Reading entire repository: {owner}/{repo}")
        
        result = await self.session.call_tool(
            "read_entire_repository",
            {
                "owner": owner,
                "repo": repo,
                "branch": branch,
                "max_depth": max_depth
            }
        )
        
        # Parse the JSON response
        content = result[0].text
        repo_data = json.loads(content.split("Complete repository content:\n\n")[1])
        return repo_data
    
    async def get_repo_summary(self, owner: str, repo: str, branch: str = "main") -> dict:
        """Get repository summary"""
        if not self.session:
            await self.connect()
        
        print(f"📊 Getting repository summary: {owner}/{repo}")
        
        result = await self.session.call_tool(
            "get_repository_summary",
            {
                "owner": owner,
                "repo": repo,
                "branch": branch
            }
        )
        
        content = result[0].text
        summary_data = json.loads(content.split("Repository summary:\n\n")[1])
        return summary_data
    
    async def read_by_extension(self, owner: str, repo: str, extensions: list, 
                               branch: str = "main") -> dict:
        """Read files with specific extensions"""
        if not self.session:
            await self.connect()
        
        print(f"🔍 Reading files with extensions {extensions}: {owner}/{repo}")
        
        result = await self.session.call_tool(
            "read_files_by_extension",
            {
                "owner": owner,
                "repo": repo,
                "extensions": extensions,
                "branch": branch
            }
        )
        
        content = result[0].text
        files_data = json.loads(content.split("Files by extension:\n\n")[1])
        return files_data
    
    async def create_index(self, owner: str, repo: str, branch: str = "main", 
                          include_preview: bool = True) -> dict:
        """Create codebase index"""
        if not self.session:
            await self.connect()
        
        print(f"📇 Creating codebase index: {owner}/{repo}")
        
        result = await self.session.call_tool(
            "create_codebase_index",
            {
                "owner": owner,
                "repo": repo,
                "branch": branch,
                "include_content_preview": include_preview
            }
        )
        
        content = result[0].text
        index_data = json.loads(content.split("Codebase index:\n\n")[1])
        return index_data

# Example usage and testing
async def example_usage():
    """Example of how to use the repository reader"""
    
    load_dotenv()
    
    reader = RepositoryCodeReader()
    
    # Example 1: Get repository summary
    print("=" * 60)
    print("Example 1: Repository Summary")
    print("=" * 60)
    
    summary = await reader.get_repo_summary("octocat", "Hello-World")
    print(f"📊 Repository has {summary['summary']['total_files']} files")
    print(f"📈 Total lines of code: {summary['summary']['total_lines']}")
    print("File types:")
    for ext, info in summary['summary']['file_types'].items():
        print(f"  {ext}: {info['count']} files, {info['lines']} lines")
    
    print("\n" + "=" * 60)
    print("Example 2: Read Python Files Only")
    print("=" * 60)
    
    # Example 2: Read only Python files
    python_files = await reader.read_by_extension("octocat", "Hello-World", [".py"])
    print(f"🐍 Found {python_files['matched_files']} Python files")
    
    for file_path, file_info in python_files['files'].items():
        print(f"📄 {file_path}: {file_info['lines']} lines")
        print(f"   Preview: {file_info['content'][:100]}...")
    
    print("\n" + "=" * 60)
    print("Example 3: Complete Repository Read")
    print("=" * 60)
    
    # Example 3: Read entire small repository
    complete_repo = await reader.read_entire_repo("octocat", "Hello-World", max_depth=5)
    print(f"📚 Complete repository: {complete_repo['total_files']} files read")
    
    # Show first few files
    for i, (file_path, file_info) in enumerate(complete_repo['files'].items()):
        if i >= 3:  # Show only first 3 files
            break
        print(f"\n📄 File: {file_path}")
        print(f"   Lines: {file_info['lines']}")
        print(f"   Size: {file_info['size']} bytes")
        print(f"   Content preview:")
        print(f"   {file_info['content'][:200]}...")
    
    print("\n" + "=" * 60)
    print("Example 4: Create Codebase Index")
    print("=" * 60)
    
    # Example 4: Create searchable index
    index = await reader.create_index("octocat", "Hello-World")
    print(f"📇 Created index for {len(index['files'])} files")
    
    for file_path, file_info in list(index['files'].items())[:3]:
        print(f"\n📄 {file_path}")
        print(f"   Extension: {file_info['extension']}")
        print(f"   Lines: {file_info['lines_count']}")
        if file_info.get('code_elements'):
            print(f"   Code elements: {len(file_info['code_elements'])}")
        if file_info.get('preview'):
            print(f"   Preview: {file_info['preview'][:2]}")

if __name__ == "__main__":
    asyncio.run(example_usage())