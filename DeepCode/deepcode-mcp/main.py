import asyncio
import os
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import PyPDF2

from mcp_agent.app import MCPApp
from mcp_agent.workflows.swarm.swarm import DoneAgent, SwarmAgent
from mcp_agent.workflows.swarm.swarm_anthropic import AnthropicSwarm
from mcp_agent.human_input.handler import console_input_callback

# Initialize the MCP application
app = MCPApp(name="paper_to_code", human_input_callback=console_input_callback)

# Tools
def read_pdf_metadata(file_path: Path) -> dict:
    """Read PDF metadata with proper encoding handling."""
    try:
        print(f"\nAttempting to read PDF metadata from: {file_path}")
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            info = pdf_reader.metadata
            first_page = pdf_reader.pages[0]
            text = first_page.extract_text()
            lines = text.split('\n')[:10]
            
            title = None
            authors = []
            
            if info:
                title = info.get('/Title', '').strip().replace('\x00', '')
                author = info.get('/Author', '').strip().replace('\x00', '')
                if author:
                    authors = [author]
            
            if not title and lines:
                title = lines[0].strip()
            
            if not authors and len(lines) > 1:
                for line in lines[1:3]:
                    if 'author' in line.lower() or 'by' in line.lower():
                        authors = [line.strip()]
                        break
            
            return {
                "title": title if title else 'Unknown Title',
                "authors": authors if authors else ['Unknown Author'],
                "year": info.get('/CreationDate', '')[:4] if info else 'Unknown Year',
                "first_lines": lines
            }
            
    except Exception as e:
        print(f"\nError reading PDF: {str(e)}")
        return {
            "title": "Error reading PDF",
            "authors": ["Unknown"],
            "year": "Unknown",
            "first_lines": []
        }

def transfer_to_paper_downloader():
    """Transfer to paper downloader agent"""
    return paper_downloader

def transfer_to_paper_analyzer():
    """Transfer to paper content analyzer agent"""
    return paper_analyzer

def transfer_to_code_replicator():
    """Transfer to code replication agent"""
    return code_replicator

def transfer_to_code_verifier():
    """Transfer to code verification agent"""
    return code_verifier

def transfer_to_triage():
    """Transfer back to triage agent"""
    return triage_agent

def task_completed():
    """Mark the task as completed"""
    return DoneAgent()

# Agent Instructions
def triage_instructions(context_variables):
    paper_context = context_variables.get("paper_context", "None")
    return f"""You are the Triage Agent for the Paper to Code workflow. Your responsibilities are:

1. Analyze the user's input to determine the type of request:
   - Paper download request
   - Paper analysis request
   - Code replication request
   - Code verification request
2. Extract key information from the request
3. Transfer the request to the appropriate agent

When determining the request type, consider:
- Paper download requests typically include URLs or paper identifiers
- Paper analysis requests focus on understanding paper content
- Code replication requests involve implementing algorithms
- Code verification requests involve testing and validation

Paper context: {paper_context}"""

def paper_downloader_instructions(context_variables):
    paper_context = context_variables.get("paper_context", "None")
    return f"""You are the Paper Downloader Agent. Your responsibilities are:

1. Download academic papers from various sources
2. Save papers locally
3. Extract and validate paper metadata
4. Transfer to Paper Analyzer when download is complete

Paper context: {paper_context}"""

def paper_analyzer_instructions(context_variables):
    paper_context = context_variables.get("paper_context", "None")
    return f"""You are the Paper Content Analyzer Agent. Your responsibilities are:

1. Analyze paper content and structure
2. Extract key algorithms and methods
3. Identify implementation requirements
4. Transfer to Code Replicator when analysis is complete

Paper context: {paper_context}"""

def code_replicator_instructions(context_variables):
    paper_context = context_variables.get("paper_context", "None")
    return f"""You are the Code Replication Agent. Your responsibilities are:

1. Implement algorithms from the paper
2. Create necessary code structure
3. Ensure code quality and documentation
4. Transfer to Code Verifier when implementation is complete

Paper context: {paper_context}"""

def code_verifier_instructions(context_variables):
    paper_context = context_variables.get("paper_context", "None")
    return f"""You are the Code Verification Agent. Your responsibilities are:

1. Test implemented code
2. Validate against paper specifications
3. Ensure correctness and performance
4. Mark task as completed when verification is done

Paper context: {paper_context}"""

# Create Agents
triage_agent = SwarmAgent(
    name="Triage Agent",
    instruction=triage_instructions,
    functions=[
        transfer_to_paper_downloader,
        transfer_to_paper_analyzer,
        transfer_to_code_replicator,
        transfer_to_code_verifier
    ],
    server_names=["filesystem", "brave"],
    human_input_callback=console_input_callback,
)

paper_downloader = SwarmAgent(
    name="Paper Downloader",
    instruction=paper_downloader_instructions,
    functions=[transfer_to_paper_analyzer, transfer_to_triage],
    server_names=["filesystem", "interpreter"],
    human_input_callback=console_input_callback,
)

paper_analyzer = SwarmAgent(
    name="Paper Analyzer",
    instruction=paper_analyzer_instructions,
    functions=[transfer_to_code_replicator, transfer_to_triage],
    server_names=["interpreter", "filesystem", "brave"],
    human_input_callback=console_input_callback,
)

code_replicator = SwarmAgent(
    name="Code Replicator",
    instruction=code_replicator_instructions,
    functions=[transfer_to_code_verifier, transfer_to_triage],
    server_names=["interpreter", "filesystem"],
    human_input_callback=console_input_callback,
)

code_verifier = SwarmAgent(
    name="Code Verifier",
    instruction=code_verifier_instructions,
    functions=[task_completed, transfer_to_triage],
    server_names=["interpreter", "filesystem"],
    human_input_callback=console_input_callback,
)

async def main():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        # Add current directory to filesystem server
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        # Get user input
        print("\n=== Paper to Code Replication Tool ===")
        print("Press 'F' to upload a file, or enter your text directly:")
        
        user_input = input().strip()
        file_path = None
        additional_input = None
        metadata = None

        if user_input.upper() == 'F':
            root = tk.Tk()
            root.withdraw()
            
            print("\nPlease select a paper file (PDF, directory, or URL)...")
            file_path = filedialog.askopenfilename(
                title="Select Paper File",
                filetypes=[
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ]
            )

            if not file_path:
                print("No file selected. Exiting...")
                return

            file_path = Path(file_path)
            if not file_path.exists():
                print(f"Error: File {file_path} does not exist")
                return
            
            file_path = file_path.absolute()
            print(f"\nSelected file: {file_path}")
            
            if file_path.suffix.lower() == '.pdf':
                metadata = read_pdf_metadata(file_path)
            
            print("\nPlease enter your requirements:")
            additional_input = input().strip()
        else:
            additional_input = user_input

        # Prepare context variables
        context_variables = {
            "paper_context": {
                "paper_path": str(file_path) if file_path else None,
                "additional_input": additional_input if additional_input else None,
                "metadata": metadata if metadata else None
            }
        }

        # Initialize swarm with triage agent
        triage_agent.instruction = triage_agent.instruction(context_variables)
        swarm = AnthropicSwarm(agent=triage_agent, context_variables=context_variables)

        # Process the request
        result = await swarm.generate_str(additional_input)
        logger.info(f"Result: {result}")

        # Cleanup
        await triage_agent.shutdown()

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Total run time: {end - start:.2f}s") 