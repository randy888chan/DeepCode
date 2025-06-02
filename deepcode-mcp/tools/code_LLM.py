# code_agent.py
import asyncio
import json
from typing import Dict, List, Any
from dataclasses import dataclass
import openai
from mcp.client import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

@dataclass
class CodeGenerationRequest:
    description: str
    language: str
    context_files: List[str] = None
    reference_repo: str = None
    style_guide: str = None

class CodeAgent:
    def __init__(self, openai_api_key: str, github_token: str = None):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.github_token = github_token
        self.mcp_session = None
    
    async def start_mcp_server(self):
        """Start the MCP server connection"""
        server_params = StdioServerParameters(
            command="python",
            args=["github_mcp_server.py"],
            env={"GITHUB_TOKEN": self.github_token} if self.github_token else None
        )
        
        self.mcp_session = await stdio_client(server_params)
    
    async def analyze_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """Analyze a repository to understand its structure and patterns"""
        if not self.mcp_session:
            await self.start_mcp_server()
        
        # Get repository structure
        structure_result = await self.mcp_session.call_tool(
            "analyze_repository_structure",
            {"owner": owner, "repo": repo}
        )
        
        # Read key files (README, package.json, requirements.txt, etc.)
        key_files = ["README.md", "package.json", "requirements.txt", "setup.py", "Cargo.toml"]
        context_info = {
            "structure": structure_result,
            "key_files": {}
        }
        
        for file_path in key_files:
            try:
                file_result = await self.mcp_session.call_tool(
                    "read_github_file",
                    {"owner": owner, "repo": repo, "path": file_path}
                )
                context_info["key_files"][file_path] = file_result
            except:
                continue
        
        return context_info
    
    async def get_code_examples(self, owner: str, repo: str, language: str, pattern: str) -> List[str]:
        """Get code examples from repository based on patterns"""
        if not self.mcp_session:
            await self.start_mcp_server()
        
        search_result = await self.mcp_session.call_tool(
            "search_code",
            {
                "owner": owner,
                "repo": repo,
                "query": pattern,
                "language": language
            }
        )
        
        # Read the actual files found in search
        examples = []
        search_data = json.loads(search_result[0].text.split("Code search results:\n\n")[1])
        
        for item in search_data[:5]:  # Limit to 5 examples
            try:
                file_content = await self.mcp_session.call_tool(
                    "read_github_file",
                    {
                        "owner": owner,
                        "repo": repo,
                        "path": item["path"]
                    }
                )
                examples.append({
                    "path": item["path"],
                    "content": file_content[0].text
                })
            except:
                continue
        
        return examples
    
    async def generate_code(self, request: CodeGenerationRequest) -> str:
        """Generate code based on the request and repository context"""
        context_parts = []
        
        # If reference repository is provided, analyze it
        if request.reference_repo:
            owner, repo = request.reference_repo.split("/")
            repo_context = await self.analyze_repository(owner, repo)
            context_parts.append(f"Repository context:\n{json.dumps(repo_context, indent=2)}")
            
            # Get relevant code examples
            examples = await self.get_code_examples(owner, repo, request.language, request.description)
            if examples:
                context_parts.append("Code examples from repository:")
                for example in examples:
                    context_parts.append(f"File: {example['path']}\n{example['content']}")
        
        # Read specific context files if provided
        if request.context_files and request.reference_repo:
            owner, repo = request.reference_repo.split("/")
            for file_path in request.context_files:
                try:
                    file_content = await self.mcp_session.call_tool(
                        "read_github_file",
                        {"owner": owner, "repo": repo, "path": file_path}
                    )
                    context_parts.append(f"Context file: {file_path}\n{file_content[0].text}")
                except:
                    continue
        
        # Build the prompt
        prompt = f"""
You are an expert code generator. Generate high-quality {request.language} code based on the following requirements:

Requirements: {request.description}

Context Information:
{chr(10).join(context_parts)}

Style Guide: {request.style_guide or "Follow best practices for " + request.language}

Generate clean, well-documented, and production-ready code that follows the patterns and style observed in the context.
Include appropriate comments and documentation.
"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert software developer who writes clean, efficient, and well-documented code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    async def refine_code(self, code: str, feedback: str, context: str = "") -> str:
        """Refine generated code based on feedback"""
        prompt = f"""
Please refine the following code based on the feedback provided:

Original Code:
{code}

Feedback:
{feedback}

Additional Context:
{context}

Provide the improved version of the code.
"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert code reviewer and improver."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content

# Usage example
async def main():
    import os
    
    # Initialize the code agent
    agent = CodeAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        github_token=os.getenv("GITHUB_TOKEN")
    )
    
    # Example: Generate a REST API endpoint based on a reference repository
    request = CodeGenerationRequest(
        description="Create a REST API endpoint for user authentication with JWT tokens",
        language="python",
        reference_repo="fastapi/fastapi",  # Use FastAPI repo as reference
        context_files=["examples/security/security_first_steps.py"],
        style_guide="Follow FastAPI conventions and best practices"
    )
    
    generated_code = await agent.generate_code(request)
    print("Generated Code:")
    print(generated_code)
    
    # Refine the code if needed
    refined_code = await agent.refine_code(
        generated_code,
        "Add input validation and better error handling",
        "This is for a production API"
    )
    print("\nRefined Code:")
    print(refined_code)

if __name__ == "__main__":
    asyncio.run(main())