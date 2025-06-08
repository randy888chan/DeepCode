# Example usage patterns
async def example_usage():
    agent = CodeAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        github_token=os.getenv("GITHUB_TOKEN")
    )
    
    # Generate a React component based on Material-UI patterns
    react_request = CodeGenerationRequest(
        description="Create a responsive data table component with sorting and filtering",
        language="typescript",
        reference_repo="mui/material-ui",
        context_files=["docs/data/material/components/table/table.md"],
        style_guide="Use Material-UI components and TypeScript best practices"
    )
    
    # Generate a Python microservice based on FastAPI patterns
    python_request = CodeGenerationRequest(
        description="Create a microservice for file upload and processing",
        language="python",
        reference_repo="tiangolo/fastapi",
        context_files=["docs_src/request_files/tutorial001.py"],
        style_guide="Follow FastAPI and async/await patterns"
    )
    
    # Generate code
    react_code = await agent.generate_code(react_request)
    python_code = await agent.generate_code(python_request)