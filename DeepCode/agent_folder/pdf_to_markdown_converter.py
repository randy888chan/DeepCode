#!/usr/bin/env python3
"""
PDF to Markdown Converter

A comprehensive tool for converting PDF files to Markdown format.

Dependencies:
- PyPDF2
- python-docx (optional)

Install dependencies:
pip install PyPDF2
"""

import os
import re
import logging
from typing import Optional, List, Union

try:
    import PyPDF2
except ImportError:
    print("Please install PyPDF2: pip install PyPDF2")
    raise

class PDFToMarkdownConverter:
    def __init__(self, log_level=logging.INFO):
        """
        Initialize the PDF to Markdown converter.
        
        Args:
            log_level (int): Logging level (default: logging.INFO)
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def basic_pdf_to_markdown(
        self, 
        pdf_path: str, 
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Basic PDF to Markdown conversion.
        
        Args:
            pdf_path (str): Path to the input PDF file
            output_path (str, optional): Path to save the output Markdown file
        
        Returns:
            str: Path to the generated Markdown file or None if conversion fails
        """
        try:
            # Validate input file
            self._validate_pdf_file(pdf_path)
            
            # Determine output path
            if output_path is None:
                output_path = os.path.splitext(pdf_path)[0] + '.md'
            
            # Extract text from PDF
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                markdown_content = []
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    markdown_content.append(page_text)
                
                # Combine pages
                full_markdown = '\n\n'.join(markdown_content)
            
            # Write to Markdown file
            with open(output_path, 'w', encoding='utf-8') as md_file:
                md_file.write(full_markdown)
            
            self.logger.info(f"Basic Markdown file created: {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"Error in basic PDF to Markdown conversion: {e}")
            return None

    def advanced_pdf_to_markdown(
        self, 
        pdf_path: str, 
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Advanced PDF to Markdown conversion with formatting.
        
        Args:
            pdf_path (str): Path to the input PDF file
            output_path (str, optional): Path to save the output Markdown file
        
        Returns:
            str: Path to the generated Markdown file or None if conversion fails
        """
        try:
            # Validate input file
            self._validate_pdf_file(pdf_path)
            
            # Determine output path
            if output_path is None:
                output_path = os.path.splitext(pdf_path)[0] + '.md'
            
            # Extract and format text from PDF
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                markdown_content = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    # Extract text
                    page_text = page.extract_text()
                    
                    # Add page numbers as headers
                    page_header = f"\n## Page {page_num}\n"
                    
                    # Identify potential headers and list items
                    formatted_text = self._format_markdown_text(page_text)
                    
                    # Combine page header and formatted text
                    markdown_page = page_header + formatted_text
                    markdown_content.append(markdown_page)
                
                # Join pages
                full_markdown = '\n\n'.join(markdown_content)
            
            # Write to Markdown file
            with open(output_path, 'w', encoding='utf-8') as md_file:
                md_file.write(full_markdown)
            
            self.logger.info(f"Advanced Markdown file created: {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"Error in advanced PDF to Markdown conversion: {e}")
            return None

    def _validate_pdf_file(self, pdf_path: str) -> None:
        """
        Validate the input PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a PDF
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("Input file must be a PDF")

    def _format_markdown_text(self, text: str) -> str:
        """
        Apply basic Markdown formatting to the text.
        
        Args:
            text (str): Input text to format
        
        Returns:
            str: Formatted Markdown text
        """
        # Convert potential headers (lines starting with capital letters)
        text = re.sub(r'^([A-Z][^\n]{10,})$', r'### \1', text, flags=re.MULTILINE)
        
        # Identify potential list items
        text = re.sub(r'^(\s*[-•*])', r'- ', text, flags=re.MULTILINE)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text

    def batch_convert(
        self, 
        input_dir: str, 
        output_dir: Optional[str] = None, 
        conversion_method: str = 'advanced'
    ) -> List[str]:
        """
        Batch convert PDF files in a directory to Markdown.
        
        Args:
            input_dir (str): Directory containing PDF files
            output_dir (str, optional): Directory to save Markdown files
            conversion_method (str): 'basic' or 'advanced' conversion method
        
        Returns:
            List of converted Markdown file paths
        """
        # Validate input directory
        if not os.path.isdir(input_dir):
            raise ValueError(f"Invalid input directory: {input_dir}")
        
        # Create output directory if not specified
        if output_dir is None:
            output_dir = os.path.join(input_dir, 'markdown_output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Find PDF files
        pdf_files = [
            f for f in os.listdir(input_dir) 
            if f.lower().endswith('.pdf')
        ]
        
        converted_files = []
        
        # Convert each PDF file
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            md_filename = os.path.splitext(pdf_file)[0] + '.md'
            md_path = os.path.join(output_dir, md_filename)
            
            try:
                if conversion_method == 'basic':
                    result = self.basic_pdf_to_markdown(pdf_path, md_path)
                else:
                    result = self.advanced_pdf_to_markdown(pdf_path, md_path)
                
                if result:
                    converted_files.append(result)
            except Exception as e:
                self.logger.error(f"Failed to convert {pdf_file}: {e}")
        
        self.logger.info(f"Batch conversion complete. Converted {len(converted_files)} files.")
        return converted_files

def main():
    """
    Main function to demonstrate PDF to Markdown conversion.
    """
    converter = PDFToMarkdownConverter()
    
    # Example usage instructions
    print("PDF to Markdown Converter")
    print("------------------------")
    print("Usage examples:")
    print("1. Convert a single PDF (basic method):")
    print("   converter.basic_pdf_to_markdown('document.pdf')")
    print("\n2. Convert a single PDF (advanced method):")
    print("   converter.advanced_pdf_to_markdown('document.pdf', 'output.md')")
    print("\n3. Batch convert PDFs in a directory:")
    print("   converter.batch_convert('/path/to/pdf/directory')")

if __name__ == '__main__':
    main()