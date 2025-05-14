# PDF to Markdown Converter

## Overview
This Python script provides a comprehensive tool for converting PDF files to Markdown format. It offers two conversion methods: basic and advanced.

## Features
- Basic text extraction from PDFs
- Advanced text extraction with formatting
- Batch conversion of multiple PDFs
- Logging support
- Error handling

## Prerequisites
- Python 3.7+
- PyPDF2 library

## Installation
```bash
pip install PyPDF2
```

## Usage

### Basic Conversion
```python
from pdf_to_markdown_converter import PDFToMarkdownConverter

# Create converter instance
converter = PDFToMarkdownConverter()

# Convert a single PDF using basic method
converter.basic_pdf_to_markdown('document.pdf')
```

### Advanced Conversion
```python
# Convert a single PDF using advanced method
converter.advanced_pdf_to_markdown('document.pdf', 'output.md')
```

### Batch Conversion
```python
# Convert all PDFs in a directory
converter.batch_convert('/path/to/pdf/directory')
```

## Conversion Methods
1. **Basic Method**: Simple text extraction
2. **Advanced Method**: 
   - Adds page numbers as headers
   - Attempts to format headers
   - Identifies potential list items

## Limitations
- May not perfectly preserve complex formatting
- Limited support for images and tables
- Works best with text-heavy PDFs

## Troubleshooting
- Ensure PDFs are text-based
- Check file permissions
- Verify PyPDF2 is correctly installed

## Contributing
Contributions are welcome! Please submit pull requests or open issues on the repository.

## License
[Insert your license information here]