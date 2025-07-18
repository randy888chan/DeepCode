# DeepCode CLI - Open-Source Code Agent

🧬 **Data Intelligence Lab @ HKU** • ⚡ **Revolutionizing Research Reproducibility**

DeepCode CLI is a command-line interface for the DeepCode multi-agent system that transforms research papers and user requirements into working code through intelligent AI orchestration.

## ✨ Key Features

### 🎯 **Multiple Input Methods**
- **📁 File Processing**: Upload PDF, DOCX, PPTX, HTML, or TXT files
- **🌐 URL Processing**: Process research papers from academic URLs (arXiv, IEEE, ACM, etc.)
- **💬 Chat Input**: ⭐ **NEW!** Describe coding requirements in natural language

### 🤖 **AI-Powered Processing Modes**
- **🧠 Comprehensive Mode**: Full intelligence analysis with codebase indexing
- **⚡ Optimized Mode**: Fast processing without indexing for quicker results
- **💬 Chat Planning Mode**: ⭐ **NEW!** Direct requirements-to-code pipeline

### 🔄 **Intelligent Workflows**
- Multi-agent collaborative architecture
- Real-time progress tracking
- Automated workspace setup
- Code generation and validation

## 🚀 Quick Start

### Interactive Mode
```bash
python cli/main_cli.py
```

### Direct Processing
```bash
# Process a research paper file
python cli/main_cli.py --file paper.pdf

# Process from URL
python cli/main_cli.py --url "https://arxiv.org/abs/..."

# 💬 NEW: Process coding requirements via chat
python cli/main_cli.py --chat "Build a web application with user authentication and data visualization dashboard"

# Use optimized mode for faster processing
python cli/main_cli.py --optimized
```

## 💬 Chat Input Feature (NEW!)

The Chat Input feature allows you to describe your coding requirements in natural language, and DeepCode will automatically generate a comprehensive implementation plan and working code.

### Usage Examples

**Academic Research:**
```bash
python cli/main_cli.py --chat "I need to implement a reinforcement learning algorithm for robotic control with deep neural networks"
```

**Engineering Projects:**
```bash
python cli/main_cli.py --chat "Develop a web application for project management with user authentication, task tracking, and real-time collaboration features"
```

**Mixed Projects:**
```bash
python cli/main_cli.py --chat "Implement a machine learning model with a web interface for real-time predictions and data visualization"
```

### Interactive Chat Mode

In interactive mode, select option **[T] Chat Input** to access the enhanced chat interface:

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                MAIN MENU                                      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  🌐 [U] Process URL       │  📁 [F] Upload File    │  💬 [T] Chat Input    ║
║  ⚙️  [C] Configure        │  📊 [H] History        │  ❌ [Q] Quit         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### Chat Workflow

1. **🚀 Initialize**: Setup chat-based planning engine
2. **💬 Planning**: AI analyzes your requirements and generates implementation plan
3. **🏗️ Setup**: Creates optimized workspace structure
4. **📝 Save Plan**: Saves detailed implementation plan
5. **⚙️ Implement**: Generates complete, working code

## 📋 Command-Line Options

```
Usage: main_cli.py [-h] [--file FILE] [--url URL] [--chat CHAT] [--optimized] [--verbose]

Options:
  -h, --help       Show help message and exit
  --file, -f FILE  Process a specific file (PDF, DOCX, TXT, etc.)
  --url, -u URL    Process a research paper from URL
  --chat, -t CHAT  Process coding requirements via chat input
  --optimized, -o  Use optimized mode (skip indexing for faster processing)
  --verbose, -v    Enable verbose output
```

## 🎯 Pipeline Modes Comparison

| Mode | Description | Speed | Features | Best For |
|------|-------------|-------|----------|----------|
| **💬 Chat Planning** | Requirements → Code | ⚡⚡⚡ Fastest | AI Planning, Direct Implementation | Custom coding projects |
| **⚡ Optimized** | Fast paper processing | ⚡⚡ Fast | Paper analysis, Code generation | Quick prototypes |
| **🧠 Comprehensive** | Full intelligence analysis | ⚡ Thorough | All features, Codebase indexing | Research reproduction |

## 🛠️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-org/deepcode-mcp.git
cd deepcode-mcp
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys** (optional)
```bash
cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml
# Edit the file with your API keys
```

4. **Run CLI**
```bash
python cli/main_cli.py
```

## 🔧 Configuration

### Pipeline Mode Settings
- **Comprehensive Mode**: Enables all features including codebase indexing
- **Optimized Mode**: Skips indexing for faster processing
- **Chat Mode**: Automatically selected when using chat input

### API Configuration
Configure your preferred LLM provider in `mcp_agent.secrets.yaml`:
- Anthropic Claude (recommended)
- OpenAI GPT (fallback)

## 📊 Example Output

### Chat Mode Results
```
🤖 PIPELINE MODE: 💬 Chat Planning Mode

🔄 COMPLETED WORKFLOW STAGES:
  ✅ 🚀 Engine Initialization
  ✅ 💬 Requirements Analysis
  ✅ 🏗️ Workspace Setup
  ✅ 📝 Implementation Plan Generation
  ✅ ⚙️ Code Implementation

📁 Generated Code Directory: /path/to/generated/code
💬 Generated from user requirements via chat interface
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## 📄 License

MIT License - see LICENSE file for details.

## 🙋 Support

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: See full documentation at [link]
- **Community**: Join our research community

---

🧬 **Data Intelligence Lab @ HKU** • Building the future of AI-powered development 