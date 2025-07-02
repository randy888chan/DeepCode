#!/usr/bin/env python3
"""
Image Caption Generator for Markdown Files
Analyzes images in markdown files using VLM and adds captions
"""

import re
import os
import openai
import base64
import argparse
from typing import List, Tuple, Optional
from pathlib import Path

def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List = None,
    messages: List = None,
    api_key: str = None,
    base_url: str = None,
    **kwargs
):
    """
    Placeholder function for OpenAI API call with caching
    You need to implement this function or import it from your existing codebase
    """
    
    if api_key:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = openai.OpenAI()
    
    if messages:
        response = client.chat.completions.create(
            model=model,
            messages=[msg for msg in messages if msg is not None],
            **kwargs
        )
        return response.choices[0].message.content
    else:
        # Fallback implementation
        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            final_messages.extend(history_messages)
        final_messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=final_messages,
            **kwargs
        )
        return response.choices[0].message.content


def create_vision_model_func(api_key: str = None, base_url: str = None):
    """
    Create the vision model function using the provided structure
    """
    def vision_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List = None,
        image_data: Optional[str] = None,
        **kwargs
    ) -> str:
        if history_messages is None:
            history_messages = []
            
        return openai_complete_if_cache(
            "gpt-4o",
            "",
            system_prompt=system_prompt,
            history_messages=history_messages,
            messages=[
                {"role": "system", "content": system_prompt}
                if system_prompt
                else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                }
                if image_data
                else {"role": "user", "content": prompt},
            ],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    
    return vision_model_func


def encode_image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None


def find_image_references(markdown_content: str) -> List[Tuple[str, str, int, int]]:
    """
    Find all image references in markdown content
    Returns list of tuples: (full_match, image_path, start_pos, end_pos)
    """
    # Pattern to match ![alt_text](image_path) or ![](image_path)
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    matches = []
    
    for match in re.finditer(pattern, markdown_content):
        full_match = match.group(0)
        alt_text = match.group(1)
        image_path = match.group(2)
        start_pos = match.start()
        end_pos = match.end()
        
        matches.append((full_match, image_path, start_pos, end_pos))
    
    return matches


def generate_image_caption(
    image_path: str, 
    vision_model_func,
    custom_prompt: str = None
) -> str:
    """
    Generate caption for an image using VLM
    """
    # Encode image to base64
    image_data = encode_image_to_base64(image_path)
    if not image_data:
        return "Error: Could not read image file"
    
    # Default prompt for academic paper figures
    default_prompt = """
    Please provide a detailed, technical caption for this academic figure. 
    The caption should:
    1. Describe what is shown in the image objectively
    2. Explain the key elements, data, or results displayed
    3. Note any important patterns, trends, or relationships
    4. Use appropriate academic language
    5. Be concise but comprehensive
    
    Format the response as a single paragraph suitable for an academic paper caption.
    """
    
    prompt = custom_prompt if custom_prompt else default_prompt
    
    system_prompt = """
    You are an expert at writing academic figure captions. 
    Provide clear, technical, and informative captions suitable for research papers.
    """
    
    try:
        caption = vision_model_func(
            prompt=prompt,
            system_prompt=system_prompt,
            image_data=image_data
        )
        return caption.strip()
    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return f"Error generating caption: {str(e)}"


def process_markdown_file(
    input_file: str,
    output_file: str = None,
    vision_model_func = None,
    api_key: str = None,
    base_url: str = None,
    custom_prompt: str = None,
    dry_run: bool = False
) -> str:
    """
    Process markdown file to add captions to images
    """
    if vision_model_func is None:
        vision_model_func = create_vision_model_func(api_key, base_url)
    
    # Read the markdown file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        return None
    
    # Find all image references
    image_refs = find_image_references(content)
    
    if not image_refs:
        print("No image references found in the markdown file")
        return content
    
    print(f"Found {len(image_refs)} image references")
    
    # Process images in reverse order to maintain string indices
    new_content = content
    processed_count = 0
    
    for full_match, image_path, start_pos, end_pos in reversed(image_refs):
        print(f"Processing image: {image_path}")
        
        # Resolve relative path
        if not os.path.isabs(image_path):
            base_dir = os.path.dirname(input_file)
            full_image_path = os.path.join(base_dir, image_path)
        else:
            full_image_path = image_path
        
        # Check if image file exists
        if not os.path.exists(full_image_path):
            print(f"Warning: Image file not found: {full_image_path}")
            continue
        
        if dry_run:
            print(f"  Would process: {full_image_path}")
            continue
        
        # Generate caption
        caption = generate_image_caption(full_image_path, vision_model_func, custom_prompt)
        
        # Format the caption
        formatted_caption = f"\n\n*Figure Caption: {caption}*"
        
        # Insert caption after the image reference
        new_content = (
            new_content[:end_pos] + 
            formatted_caption + 
            new_content[end_pos:]
        )
        
        processed_count += 1
        print(f"  Added caption for {image_path}")
    
    if dry_run:
        print(f"Dry run completed. Would process {len(image_refs)} images.")
        return content
    
    # Write output file
    if output_file is None:
        output_file = input_file.replace('.md', '_with_captions.md')
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Successfully wrote output to {output_file}")
        print(f"Processed {processed_count} images")
    except Exception as e:
        print(f"Error writing output file {output_file}: {e}")
        return None
    
    return new_content


def main():
    parser = argparse.ArgumentParser(description="Add VLM-generated captions to images in markdown files")
    parser.add_argument("input_file", help="Input markdown file path")
    parser.add_argument("-o", "--output", help="Output file path (default: input_file_with_captions.md)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--base-url", help="OpenAI API base URL")
    parser.add_argument("--prompt", help="Custom prompt for image description")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without making changes")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found")
        return 1
    
    # Process the file
    result = process_markdown_file(
        input_file=args.input_file,
        output_file=args.output,
        api_key=args.api_key,
        base_url=args.base_url,
        custom_prompt=args.prompt,
        dry_run=args.dry_run
    )
    
    if result is None:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 