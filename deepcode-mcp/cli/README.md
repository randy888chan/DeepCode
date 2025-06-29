# Paper to Code CLI

🧬 **Command Line Interface for AI Research Engine**

This is the CLI version of Paper to Code, providing the same powerful functionality as the web interface but optimized for terminal users.

## ✨ Features

- **Multi-Agent Research Pipeline**: Advanced AI agents work together to analyze and reproduce research papers
- **Intelligent Document Processing**: Support for PDF, DOCX, PPTX, HTML, and TXT files
- **URL Processing**: Direct support for arXiv, IEEE, ACM, and other academic platforms
- **Code Generation**: Automatic code implementation based on paper analysis
- **Progress Tracking**: Real-time terminal-based progress indicators
- **Interactive Interface**: Professional CLI with colored output and progress bars

## 🚀 Quick Start

### Using the CLI Launcher

```bash
# From project root directory
python cli/paper_to_code_cli.py
```

### Direct CLI App

```bash
# From project root directory  
python cli/cli_app.py
```

### Using the Legacy Main (if you prefer)

```bash
# From project root directory
python main.py
```

## 📋 Usage

1. **Launch the CLI**: Run the launcher script
2. **Choose Input Method**: 
   - `U` - Enter a URL (arXiv, IEEE, ACM, etc.)
   - `F` - Upload a file (PDF, DOCX, PPTX, HTML, TXT)
   - `Q` - Quit the application
3. **Monitor Progress**: Watch the real-time processing stages
4. **Review Results**: View analysis, download, and implementation results
5. **Process More Papers**: Continue with additional papers or exit

## 🔧 Processing Pipeline

The CLI follows the same multi-stage pipeline as the web version:

1. **🚀 Initialize** - Setting up AI engine
2. **📊 Analyze** - Analyzing paper content  
3. **📥 Download** - Processing document
4. **🔍 References** - Analyzing references
5. **📋 Plan** - Generating code plan
6. **📦 Repos** - Downloading repositories
7. **🗂️ Index** - Building code index
8. **⚙️ Implement** - Implementing code

## 💻 Interface Features

- **Enhanced ASCII Art**: Beautiful terminal logos and banners
- **Color-Coded Output**: Different colors for different types of information
- **Progress Indicators**: Animated progress bars and spinners
- **Error Handling**: Comprehensive error messages with troubleshooting tips
- **File Validation**: Smart file type detection and validation
- **URL Validation**: Academic platform recognition and validation

## 🔗 Supported Platforms

### Academic URLs
- arXiv (arxiv.org)
- IEEE Xplore (ieeexplore.ieee.org)
- ACM Digital Library (dl.acm.org)
- SpringerLink (link.springer.com)
- Nature (nature.com)
- Science (science.org)
- Google Scholar
- ResearchGate
- Semantic Scholar

### File Formats
- PDF Documents
- Word Documents (.docx, .doc)
- PowerPoint Presentations (.pptx, .ppt)
- HTML Files
- Text Files (.txt, .md)

## 🛠️ Technical Details

### Dependencies
- Python 3.8+
- MCP Agent Framework
- Asyncio for concurrent processing
- PyYAML for configuration
- Tkinter for file dialogs (optional)

### Architecture
- **CLI Interface** (`cli_interface.py`): Terminal UI components
- **CLI App** (`cli_app.py`): Main application logic
- **CLI Launcher** (`paper_to_code_cli.py`): Entry point and dependency checking

## 🆚 CLI vs Web UI

| Feature | CLI Version | Web UI Version |
|---------|-------------|----------------|
| **Interface** | Terminal-based | Browser-based |
| **Performance** | Faster startup | Slower startup |
| **Resource Usage** | Lower memory | Higher memory |
| **Accessibility** | SSH-friendly | GUI required |
| **Functionality** | 100% same | 100% same |
| **Progress Tracking** | Text-based | Visual |

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **File Not Found**: Check file paths and permissions
3. **tkinter Issues**: File dialogs will fall back to manual input
4. **Network Issues**: Check internet connection for URL processing

### Getting Help

- Check the main project documentation
- Review error messages carefully
- Use the `--help` flag for command information
- Submit issues on GitHub if problems persist

## 📚 Examples

### Processing an arXiv Paper
```bash
# Start CLI
python cli/paper_to_code_cli.py

# Choose 'U' for URL
# Enter: https://arxiv.org/abs/2301.07041
```

### Processing a Local PDF
```bash
# Start CLI  
python cli/paper_to_code_cli.py

# Choose 'F' for file
# Select your PDF file via dialog or enter path manually
```

## 🔮 Advanced Usage

The CLI maintains full compatibility with all MCP agents and workflows used in the web version, including:

- Custom agent configurations
- Workflow modifications
- Output customization
- Integration with external tools

---

**Note**: This CLI version provides identical functionality to the web interface while offering better performance and accessibility for terminal users. 