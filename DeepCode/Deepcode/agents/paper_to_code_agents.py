import asyncio
from mcp_agent.core.fastagent import FastAgent
from Deepcode.prompts.code_prompts import (
    CODE_GENERATION_AGENT,
    CODE_UNDERSTANDING_AGENT,
    IMPLEMENTATION_AGENT,
    DOCUMENTATION_GENERATOR_AGENT,
)
from pathlib import Path

# ==================== Paper to Code Workflow Prompts ====================

PAPER_INPUT_ANALYZER_PROMPT = """You are an expert at analyzing user input for paper-to-code replication tasks.
Your task is to analyze the user's input and determine:
1. The type of input (PDF file, directory, or URL)
2. The paper's title and authors
3. The specific code sections that need to be replicated
4. Any additional requirements or constraints

Output your analysis in a structured format:
{
    "input_type": "file|directory|url",
    "paper_info": {
        "title": "paper title",
        "authors": ["author1", "author2"],
        "year": "publication year"
    },
    "code_sections": [
        {
            "section_name": "section name",
            "description": "brief description",
            "priority": "high|medium|low"
        }
    ],
    "requirements": [
        "requirement1",
        "requirement2"
    ]
}"""

PAPER_DOWNLOADER_PROMPT = """You are an expert at downloading and managing academic papers.
Your task is to:
1. Download papers from various sources (arXiv, PDF files, etc.)
2. Organize them in the specified directory
3. Verify the download was successful
4. Extract metadata from the paper

Use the following tools:
- For arXiv papers: Use the download_arxiv_pdf.py to download the paper
- For local files: Copy to the papers directory
- For URLs: Download and save to the papers directory

Output your actions in a structured format:
{
    "status": "success|failure",
    "paper_path": "path/to/paper.pdf",
    "metadata": {
        "title": "paper title",
        "authors": ["author1", "author2"],
        "year": "publication year"
    }
}"""

PAPER_CONTENT_ANALYZER_PROMPT = """You are an expert at analyzing academic papers for code replication.
Your task is to:
1. Read and understand the paper's content
2. Identify code-related sections
3. Extract algorithms, formulas, and implementation details
4. Create a structured representation of the code to be implemented

Output your analysis in a structured format:
{
    "paper_summary": "brief summary of the paper",
    "code_components": [
        {
            "name": "component name",
            "description": "detailed description",
            "dependencies": ["dependency1", "dependency2"],
            "implementation_steps": [
                "step1",
                "step2"
            ]
        }
    ],
    "technical_requirements": [
        "requirement1",
        "requirement2"
    ]
}"""

CODE_REPLICATION_PROMPT = """You are an expert at replicating code from academic papers.
Your task is to:
1. Implement the code based on the paper's description
2. Follow best practices and coding standards
3. Add necessary documentation and comments
4. Ensure the implementation matches the paper's methodology

Use the following guidelines:
- Break down complex algorithms into manageable components
- Add detailed comments explaining the implementation
- Include error handling and input validation
- Follow the paper's methodology precisely

Output your implementation in a structured format:
{
    "implementation_status": "success|partial|failure",
    "components": [
        {
            "name": "component name",
            "code": "implemented code",
            "documentation": "documentation",
            "tests": "test cases"
        }
    ],
    "dependencies": [
        "dependency1",
        "dependency2"
    ]
}"""

CODE_VERIFICATION_PROMPT = """You are an expert at verifying code replication from academic papers.
Your task is to:
1. Verify the implemented code matches the paper's description
2. Test the code with example inputs
3. Compare results with paper's reported results
4. Identify and fix any discrepancies

Use the following verification steps:
- Compare algorithm steps with paper's description
- Test with paper's example inputs
- Verify output matches paper's results
- Check edge cases and error handling

Output your verification in a structured format:
{
    "verification_status": "success|partial|failure",
    "test_results": [
        {
            "test_case": "test description",
            "expected": "expected result",
            "actual": "actual result",
            "status": "pass|fail"
        }
    ],
    "discrepancies": [
        {
            "description": "discrepancy description",
            "severity": "high|medium|low",
            "suggestion": "fix suggestion"
        }
    ]
}"""

# ==================== Paper to Code Workflow Agents ====================

agents = FastAgent(name="PaperToCode")

@agents.agent(
    name="PaperInputAnalyzerAgent",
    model="sonnet",
    instruction=PAPER_INPUT_ANALYZER_PROMPT,
    servers=["filesystem"]
)

@agents.agent(
    name="PaperDownloaderAgent",
    model="sonnet",
    instruction=PAPER_DOWNLOADER_PROMPT,
    servers=["filesystem","interpreter"]
)

@agents.agent(
    name="PaperContentAnalyzerAgent",
    model="sonnet",
    instruction=PAPER_CONTENT_ANALYZER_PROMPT,
    servers=["interpreter", "filesystem", "brave"]
)

@agents.agent(
    name="CodeReplicationAgent",
    model="sonnet",
    instruction=CODE_REPLICATION_PROMPT,
    servers=["interpreter", "filesystem"]
)

@agents.agent(
    name="CodeVerificationAgent",
    model="sonnet",
    instruction=CODE_VERIFICATION_PROMPT,
    servers=["interpreter", "filesystem"]
)

# ==================== Workflow Definitions ====================

@agents.chain(
    name="PaperToCodeDownloadFlow",
    sequence=[
        "PaperInputAnalyzerAgent",
        "PaperDownloaderAgent",
    ],
    instruction="A comprehensive workflow for downloading academic papers",
    cumulative=False
)

# @agents.chain(
#     name="PaperToCodeWorkflow",
#     sequence=[
#         "PaperInputAnalyzerAgent",
#         "PaperDownloaderAgent",
#         "PaperContentAnalyzerAgent",
#         "CodeReplicationAgent",
#         "CodeVerificationAgent",
#         "DocumentationGeneratorAgent"
#     ],
#     instruction="A comprehensive workflow for replicating code from academic papers",
#     cumulative=True
# )

async def main() -> None:
    async with agents.run() as agent:
        await agent.prompt("PaperToCodeDownloadFlow")

if __name__ == "__main__":
    asyncio.run(main()) 