"""
File processing utilities for handling paper files and related operations.
"""

import json
import os
import re
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

class FileProcessor:
    """
    A class to handle file processing operations including path extraction and file reading.
    """
    
    @staticmethod
    def extract_file_path(file_info: Union[str, Dict]) -> Optional[str]:
        """
        Extract file path from the input information.
        
        Args:
            file_info: Either a JSON string or a dictionary containing file information
            
        Returns:
            Optional[str]: The extracted file path or None if not found
        """
        try:
            # Convert string to dict if necessary
            if isinstance(file_info, str):
                info_dict = json.loads(file_info)
            else:
                info_dict = file_info
                
            # Extract paper path
            paper_path = info_dict.get('paper_path')
            if not paper_path:
                return None
                
            # Convert to absolute path if relative
            if not os.path.isabs(paper_path):
                paper_path = os.path.abspath(paper_path)
                
            return paper_path
            
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            raise ValueError(f"Invalid file information format: {str(e)}")
    
    @staticmethod
    def parse_markdown_sections(content: str) -> List[Dict[str, Union[str, int, List]]]:
        """
        Parse markdown content and organize it by sections based on headers.
        
        Args:
            content: The markdown content to parse
            
        Returns:
            List[Dict]: A list of sections, each containing:
                - level: The header level (1-6)
                - title: The section title
                - content: The section content
                - subsections: List of subsections
        """
        # Split content into lines
        lines = content.split('\n')
        sections = []
        current_section = None
        current_content = []
        
        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                # If we were building a section, save its content
                if current_section is not None:
                    current_section['content'] = '\n'.join(current_content).strip()
                    sections.append(current_section)
                
                # Start a new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = {
                    'level': level,
                    'title': title,
                    'content': '',
                    'subsections': []
                }
                current_content = []
            elif current_section is not None:
                current_content.append(line)
            
        # Don't forget to save the last section
        if current_section is not None:
            current_section['content'] = '\n'.join(current_content).strip()
            sections.append(current_section)
        
        return FileProcessor._organize_sections(sections)
    
    @staticmethod
    def _organize_sections(sections: List[Dict]) -> List[Dict]:
        """
        Organize sections into a hierarchical structure based on their levels.
        
        Args:
            sections: List of sections with their levels
            
        Returns:
            List[Dict]: Organized hierarchical structure of sections
        """
        result = []
        section_stack = []
        
        for section in sections:
            while section_stack and section_stack[-1]['level'] >= section['level']:
                section_stack.pop()
                
            if section_stack:
                section_stack[-1]['subsections'].append(section)
            else:
                result.append(section)
                
            section_stack.append(section)
            
        return result
            
    @staticmethod
    async def read_file_content(file_path: str) -> str:
        """
        Read the content of a file asynchronously.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            str: The content of the file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If there's an error reading the file
        """
        try:
            # Ensure the file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Read file content
            # Note: Using async with would be better for large files
            # but for simplicity and compatibility, using regular file reading
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            return content
            
        except Exception as e:
            raise IOError(f"Error reading file {file_path}: {str(e)}")
            
    @staticmethod
    def format_section_content(section: Dict) -> str:
        """
        Format a section's content with standardized spacing and structure.
        
        Args:
            section: Dictionary containing section information
            
        Returns:
            str: Formatted section content
        """
        # Start with section title
        formatted = f"\n{'#' * section['level']} {section['title']}\n"
        
        # Add section content if it exists
        if section['content']:
            formatted += f"\n{section['content'].strip()}\n"
        
        # Process subsections
        if section['subsections']:
            # Add a separator before subsections if there's content
            if section['content']:
                formatted += "\n---\n"
            
            # Process each subsection
            for subsection in section['subsections']:
                formatted += FileProcessor.format_section_content(subsection)
        
        # Add section separator
        formatted += "\n" + "=" * 80 + "\n"
        
        return formatted

    @staticmethod
    def standardize_output(sections: List[Dict]) -> str:
        """
        Convert structured sections into a standardized string format.
        
        Args:
            sections: List of section dictionaries
            
        Returns:
            str: Standardized string output
        """
        output = []
        
        # Process each top-level section
        for section in sections:
            output.append(FileProcessor.format_section_content(section))
        
        # Join all sections with clear separation
        return "\n".join(output)

    @classmethod
    async def process_file_input(cls, file_input: Union[str, Dict]) -> Dict:
        """
        Process file input information and return the structured content.
        
        Args:
            file_input: File input information (JSON string or dict)
            
        Returns:
            Dict: The structured content with sections and standardized text
        """
        # Extract file path
        file_path = cls.extract_file_path(file_input)
        if not file_path:
            raise ValueError("No valid file path found in input")
            
        # Read file content
        content = await cls.read_file_content(file_path)
        
        # Parse and structure the content
        structured_content = cls.parse_markdown_sections(content)
        
        # Generate standardized text output
        standardized_text = cls.standardize_output(structured_content)
        
        return {
            'file_path': file_path,
            'sections': structured_content,
            'standardized_text': standardized_text
        } 