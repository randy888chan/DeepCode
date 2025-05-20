import asyncio
import argparse
from mcp_agent.core.fastagent import FastAgent
from Deepcode.prompts.code_prompts import (
    CODE_GENERATION_AGENT,
    CODE_UNDERSTANDING_AGENT,
    IMPLEMENTATION_AGENT,
    DOCUMENTATION_GENERATOR_AGENT,
    PAPER_INPUT_ANALYZER_PROMPT,
    PAPER_DOWNLOADER_PROMPT,
    PAPER_CONTENT_ANALYZER_PROMPT,
    CODE_REPLICATION_PROMPT,
    CODE_VERIFICATION_PROMPT,
)
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import sys
import PyPDF2
import io

# ==================== Paper to Code Workflow Agents ====================

agents = FastAgent(name="PaperToCode")

@agents.agent(
    name="PaperInputAnalyzerAgent",
    model="sonnet",
    instruction=PAPER_INPUT_ANALYZER_PROMPT,
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

async def read_pdf_metadata(file_path: Path) -> dict:
    """Read PDF metadata with proper encoding handling."""
    try:
        print(f"\nAttempting to read PDF metadata from: {file_path}")
        with open(file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get document info
            info = pdf_reader.metadata
            print(f"\nRaw PDF metadata: {info}")
            
            # Extract text from first page
            first_page = pdf_reader.pages[0]
            text = first_page.extract_text()
            print(f"\nFirst page text preview: {text[:200]}...")
            
            # Get first 10 lines
            lines = text.split('\n')[:10]
            print(f"\nFirst 10 lines:")
            for i, line in enumerate(lines):
                print(f"Line {i+1}: {line}")
            
            # Try to extract title and author from first page text if metadata is missing
            title = None
            authors = []
            
            if info:
                print("\nProcessing metadata...")
                # Try to get title from metadata
                title = info.get('/Title', '')
                print(f"Raw title from metadata: {title}")
                if title:
                    # Clean up title (remove special characters and extra spaces)
                    title = title.strip().replace('\x00', '')
                    print(f"Cleaned title: {title}")
                
                # Try to get author from metadata
                author = info.get('/Author', '')
                print(f"Raw author from metadata: {author}")
                if author:
                    # Clean up author (remove special characters and extra spaces)
                    author = author.strip().replace('\x00', '')
                    authors = [author]
                    print(f"Cleaned author: {author}")
            
            # If title is not found in metadata, try to get it from first line
            if not title and lines:
                print("\nTrying to extract title from first line...")
                title = lines[0].strip()
                print(f"Title from first line: {title}")
            
            # If authors are not found in metadata, try to find them in first few lines
            if not authors and len(lines) > 1:
                print("\nTrying to extract authors from first few lines...")
                for line in lines[1:3]:  # Check second and third lines
                    print(f"Checking line: {line}")
                    if 'author' in line.lower() or 'by' in line.lower():
                        authors = [line.strip()]
                        print(f"Found author: {authors[0]}")
                        break
            
            result = {
                "title": title if title else 'Unknown Title',
                "authors": authors if authors else ['Unknown Author'],
                "year": info.get('/CreationDate', '')[:4] if info else 'Unknown Year',
                "first_lines": lines
            }
            print("\nFinal extracted metadata:")
            print(f"Title: {result['title']}")
            print(f"Authors: {result['authors']}")
            print(f"Year: {result['year']}")
            return result
            
    except Exception as e:
        print(f"\nError reading PDF: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "title": "Error reading PDF",
            "authors": ["Unknown"],
            "year": "Unknown",
            "first_lines": []
        }

async def main() -> None:
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Paper to Code Replication Tool')
    args = parser.parse_args()

    print("\n=== Paper to Code Replication Tool ===")
    print("Press 'F' to upload a file, or enter your text directly:")
    
    # Get user input
    user_input = input().strip()
    file_path = None
    additional_input = None
    metadata = None

    if user_input.upper() == 'F':
        # Initialize tkinter root window (but keep it hidden)
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # Open file dialog
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

        # Convert to Path object and validate
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist")
            return
        
        # Convert to absolute path
        file_path = file_path.absolute()
        print(f"\nSelected file: {file_path}")
        
        # Read PDF metadata if it's a PDF file
        if file_path.suffix.lower() == '.pdf':
            metadata = await read_pdf_metadata(file_path)
        
        # Get additional requirements
        print("\nPlease enter your requirements:")
        additional_input = input().strip()
    else:
        # If user entered text directly, use it as input
        additional_input = user_input

    # Prepare the input data in a clean format
    input_data = {
        "paper_path": str(file_path) if file_path else None,
        "additional_input": additional_input if additional_input else None,
        "metadata": metadata if metadata else None
    }
    
    print("\nProcessing your request...")
    
    # Format the prompt as a clean string
    prompt_text = f"""
Input Data:
Paper Path: {input_data['paper_path'] if input_data['paper_path'] else 'None'}
Title: {input_data['metadata']['title'] if input_data['metadata'] else 'Unknown'}
Authors: {', '.join(input_data['metadata']['authors']) if input_data['metadata'] else 'Unknown'}
Year: {input_data['metadata']['year'] if input_data['metadata'] else 'Unknown'}
Additional Input: {input_data['additional_input'] if input_data['additional_input'] else 'None'}
"""
    
    # Start the agent workflow
    async with agents.run() as agent:
        await agent.prompt("PaperToCodeDownloadFlow", prompt_text)

if __name__ == "__main__":
    asyncio.run(main()) 