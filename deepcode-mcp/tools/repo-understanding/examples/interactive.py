#!/usr/bin/env python3
"""
Interactive mode for Repository Understanding Agent
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
import readline  # For better input handling
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

class InteractiveSession:
    """Interactive session manager"""
    
    def __init__(self):
        self.session = None
        self.indexed_repo = None
        self.history = []
        
    async def connect(self):
        """Connect to the MCP server"""
        server_params = StdioServerParameters(
            command="python",
            args=["main.py"],
            env={**os.environ}
        )
        
        self.read, self.write = await stdio_client(server_params).__aenter__()
        self.session = await ClientSession(self.read, self.write).__aenter__()
        await self.session.initialize()
        
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.session:
            await self.session.__aexit__(None, None, None)
            await self.write.close()
            
    def print_welcome(self):
        """Print welcome message"""
        print("\n" + "="*60)
        print("🚀 Repository Understanding Agent - Interactive Mode")
        print("="*60)
        print("\nCommands:")
        print("  help                    - Show this help message")
        print("  index <path>           - Index a repository")
        print("  search <query>         - Search for code")
        print("  analyze <path>         - Analyze repository structure")
        print("  explain <query>        - Explain code functionality")
        print("  similar <file> <line>  - Find similar code")
        print("  status                 - Show current status")
        print("  history                - Show command history")
        print("  clear                  - Clear screen")
        print("  quit/exit              - Exit interactive mode")
        print("\nTips:")
        print("  - Use TAB for path completion")
        print("  - Use UP/DOWN arrows for command history")
        print("="*60 + "\n")
        
    def print_status(self):
        """Print current status"""
        print("\n📊 Current Status:")
        print(f"  • Server: {'Connected' if self.session else 'Disconnected'}")
        print(f"  • Indexed Repository: {self.indexed_repo or 'None'}")
        print(f"  • Commands in history: {len(self.history)}")
        print()
        
    async def handle_index(self, args: list):
        """Handle index command"""
        if len(args) < 1:
            print("❌ Usage: index <repository_path> [collection_name]")
            return
            
        repo_path = args[0]
        collection_name = args[1] if len(args) > 1 else None
        
        print(f"🔄 Indexing repository: {repo_path}")
        
        try:
            result = await self.session.call_tool(
                "index_repository",
                arguments={
                    "repo_path": repo_path,
                    "collection_name": collection_name
                }
            )
            print(result.content[0].text)
            self.indexed_repo = repo_path
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
    async def handle_search(self, args: list):
        """Handle search command"""
        if not self.indexed_repo:
            print("❌ No repository indexed. Please run 'index' first.")
            return
            
        if len(args) < 1:
            print("❌ Usage: search <query> [max_results] [language] [file_pattern]")
            return
            
        query = args[0]
        max_results = int(args[1]) if len(args) > 1 else 5
        language = args[2] if len(args) > 2 else None
        file_pattern = args[3] if len(args) > 3 else None
        
        print(f"🔍 Searching for: {query}")
        
        try:
            arguments = {
                "query": query,
                "max_results": max_results
            }
            if language:
                arguments["language"] = language
            if file_pattern:
                arguments["file_pattern"] = file_pattern
                
            result = await self.session.call_tool("search_code", arguments=arguments)
            print(result.content[0].text)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
    async def handle_analyze(self, args: list):
        """Handle analyze command"""
        if len(args) < 1:
            print("❌ Usage: analyze <repository_path> [max_depth]")
            return
            
        repo_path = args[0]
        max_depth = int(args[1]) if len(args) > 1 else 3
        
        print(f"📊 Analyzing repository: {repo_path}")
        
        try:
            result = await self.session.call_tool(
                "analyze_structure",
                arguments={
                    "repo_path": repo_path,
                    "max_depth": max_depth
                }
            )
            print(result.content[0].text)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
    async def handle_explain(self, args: list):
        """Handle explain command"""
        if not self.indexed_repo:
            print("❌ No repository indexed. Please run 'index' first.")
            return
            
        if len(args) < 1:
            print("❌ Usage: explain <query> [context_size]")
            return
            
        query = " ".join(args[:-1] if len(args) > 1 and args[-1].isdigit() else args)
        context_size = int(args[-1]) if len(args) > 1 and args[-1].isdigit() else 3
        
        print(f"🤔 Explaining: {query}")
        
        try:
            result = await self.session.call_tool(
                "explain_code",
                arguments={
                    "query": query,
                    "context_size": context_size
                }
            )
            print(result.content[0].text)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
    async def handle_similar(self, args: list):
        """Handle similar command"""
        if not self.indexed_repo:
            print("❌ No repository indexed. Please run 'index' first.")
            return
            
        if len(args) < 2:
            print("❌ Usage: similar <file_path> <line_number> [max_results]")
            return
            
        file_path = args[0]
        try:
            line_number = int(args[1])
            max_results = int(args[2]) if len(args) > 2 else 5
        except ValueError:
            print("❌ Line number must be an integer")
            return
            
        print(f"🔍 Finding similar code to {file_path}:{line_number}")
        
        try:
            result = await self.session.call_tool(
                "find_similar_code",
                arguments={
                    "file_path": file_path,
                    "line_number": line_number,
                    "max_results": max_results
                }
            )
            print(result.content[0].text)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
    def show_history(self):
        """Show command history"""
        print("\n📜 Command History:")
        if not self.history:
            print("  (empty)")
        else:
            for i, cmd in enumerate(self.history[-20:], 1):  # Show last 20 commands
                print(f"  {i:3d}. {cmd}")
        print()
        
    async def run(self):
        """Run the interactive session"""
        await self.connect()
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                command = input("\n> ").strip()
                
                if not command:
                    continue
                    
                # Add to history
                self.history.append(command)
                
                # Parse command
                parts = command.split()
                cmd = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                # Handle commands
                if cmd in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                    
                elif cmd == 'help':
                    self.print_welcome()
                    
                elif cmd == 'status':
                    self.print_status()
                    
                elif cmd == 'history':
                    self.show_history()
                    
                elif cmd == 'clear':
                    os.system('clear' if os.name != 'nt' else 'cls')
                    
                elif cmd == 'index':
                    await self.handle_index(args)
                    
                elif cmd == 'search':
                    await self.handle_search(args)
                    
                elif cmd == 'analyze':
                    await self.handle_analyze(args)
                    
                elif cmd == 'explain':
                    await self.handle_explain(args)
                    
                elif cmd == 'similar':
                    await self.handle_similar(args)
                    
                else:
                    print(f"❌ Unknown command: {cmd}")
                    print("   Type 'help' for available commands")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Use 'quit' or 'exit' to leave interactive mode")
                continue
                
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue
                
        await self.disconnect()

async def main():
    """Main entry point"""
    session = InteractiveSession()
    try:
        await session.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")