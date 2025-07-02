#!/usr/bin/env python3
"""
Simple script to process paper.md and add VLM-generated captions
"""

import os
from image_caption_generator import create_vision_model_func, process_markdown_file

def process_paper_with_captions():
    """
    Process paper.md file to add VLM-generated captions to all images
    """
    
    # Configuration - you can modify these as needed
    API_KEY = os.getenv("OPENAI_API_KEY")  # Set this environment variable
    BASE_URL = os.getenv("OPENAI_BASE_URL")  # Optional: custom base URL
    
    input_file = "paper.md"
    output_file = "paper_with_captions.md"
    
    # Custom prompt for academic paper figures
    custom_prompt = """
    Please analyze this academic figure and provide a comprehensive caption. 
    The caption should:
    1. Clearly describe what is shown in the figure (e.g., algorithm workflow, experimental results, system architecture)
    2. Explain key components, data trends, or relationships visible in the image
    3. Mention any important labels, legends, or annotations
    4. Use precise academic language appropriate for a research paper
    5. Keep it concise but informative (2-4 sentences)
    
    Return only the caption text without additional formatting.
    """
    
    print("=== Processing paper.md with VLM-generated captions ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    if not API_KEY:
        print("Warning: OPENAI_API_KEY environment variable not set!")
        print("You can set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Process the file
    try:
        result = process_markdown_file(
            input_file=input_file,
            output_file=output_file,
            api_key=API_KEY,
            base_url=BASE_URL,
            custom_prompt=custom_prompt,
            dry_run=False
        )
        
        if result:
            print("\n=== Processing completed successfully! ===")
            print(f"Check the output file: {output_file}")
        else:
            print("Error: Processing failed")
            
    except Exception as e:
        print(f"Error during processing: {e}")


def dry_run_paper():
    """
    Run a dry run to see what images would be processed
    """
    input_file = "paper.md"
    
    print("=== Dry run: Checking what images would be processed ===")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    try:
        process_markdown_file(
            input_file=input_file,
            dry_run=True
        )
    except Exception as e:
        print(f"Error during dry run: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        dry_run_paper()
    else:
        process_paper_with_captions() 