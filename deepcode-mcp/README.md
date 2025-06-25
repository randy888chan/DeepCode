# Paper to Code - AI Research Engine

🧬 **Next-Generation AI Research Automation Platform**

Transform research papers into working code automatically using advanced AI agents and multi-modal processing.

## 🚀 Quick Start

### Web UI Version (Streamlit)
```bash
python paper_to_code.py
```

### CLI Version (Command Line)
```bash
python cli/paper_to_code_cli.py
```

## ✨ Features

- **Multi-Agent Research Pipeline**: Advanced AI agents work together to analyze and reproduce research papers
- **Intelligent Document Processing**: Support for PDF, DOCX, PPTX, HTML, and TXT files  
- **URL Processing**: Direct support for arXiv, IEEE, ACM, and other academic platforms
- **Code Generation**: Automatic code implementation based on paper analysis
- **Repository Management**: Smart GitHub repository downloading and indexing
- **Multiple Interfaces**: Both Web UI and CLI versions available

## 📋 Usage Options

### 1. Web Interface (Recommended for Beginners)
- Beautiful visual interface
- Drag-and-drop file uploads
- Real-time progress tracking
- Interactive results display

```bash
python paper_to_code.py
# Opens web browser at http://localhost:8501
```

### 2. Command Line Interface (Recommended for Developers)
- Fast terminal-based interface  
- SSH-friendly for remote servers
- Lower resource usage
- Professional terminal experience

```bash
python cli/paper_to_code_cli.py
# Interactive CLI with menus and progress tracking
```

### 3. Legacy CLI (Backward Compatibility)
```bash
python main.py
# Original CLI implementation
```

## 🔧 Processing Pipeline

Both interfaces use the same powerful multi-stage pipeline:

1. **🚀 Initialize** - Setting up AI engine
2. **📊 Analyze** - Analyzing paper content  
3. **📥 Download** - Processing document
4. **🔍 References** - Analyzing references
5. **📋 Plan** - Generating code plan
6. **📦 Repos** - Downloading repositories
7. **🗂️ Index** - Building code index
8. **⚙️ Implement** - Implementing code

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd deepcode-mcp

# Install dependencies
pip install -r requirements.txt

# For web UI
pip install streamlit>=1.28.0

# For CLI (tkinter for file dialogs - optional)
# tkinter is usually included with Python
```

## 📚 Documentation

- **Main Documentation**: Check the `docs/` folder
- **CLI Documentation**: See `cli/README.md`
- **UI Documentation**: See `ui/` folder
- **Architecture**: Check `rice_architecture/` folder

## 🆚 Interface Comparison

| Feature | Web UI | CLI |
|---------|--------|-----|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Performance** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Visual Appeal** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Remote Access** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Resource Usage** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Functionality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🔗 Supported Platforms

### Academic URLs
- arXiv (arxiv.org)
- IEEE Xplore (ieeexplore.ieee.org)
- ACM Digital Library (dl.acm.org)
- SpringerLink (link.springer.com)
- Nature (nature.com)
- Science (science.org)
- Google Scholar, ResearchGate, Semantic Scholar

### File Formats
- PDF Documents
- Word Documents (.docx, .doc)
- PowerPoint Presentations (.pptx, .ppt)
- HTML Files
- Text Files (.txt, .md)

## 🚀 Examples

### Processing an arXiv Paper (Web UI)
1. Run `python paper_to_code.py`
2. Open http://localhost:8501
3. Enter URL: `https://arxiv.org/abs/2301.07041`
4. Click "Start Processing"

### Processing a Local PDF (CLI)
1. Run `python cli/paper_to_code_cli.py`
2. Choose `F` for file upload
3. Select your PDF file
4. Watch the processing pipeline

## 🧬 Technical Architecture

- **MCP Agent Framework**: For multi-agent orchestration
- **Docling**: For document processing
- **LLM Integration**: Support for OpenAI and Anthropic models
- **Async Processing**: For concurrent operations
- **Modular Design**: Easy to extend and customize

## 🛡️ Error Handling

Both interfaces include comprehensive error handling:
- Dependency checking
- File validation
- Network error recovery
- Graceful degradation
- Detailed error messages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test both web and CLI interfaces
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- **Issues**: Submit on GitHub
- **Documentation**: Check the docs/ folder
- **CLI Help**: Run with `--help` flag
- **Community**: Join our discussions

---

**Choose your preferred interface and start transforming research papers into working code! 🧬✨** 