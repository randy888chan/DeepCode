# DeepCode CLI - Open-Source Code Agent

**DeepCode** is an open-source code agent developed by the Data Intelligence Lab @ HKU. This enhanced command line interface enables revolutionizing research reproducibility through collaborative multi-agent architecture.

## 🚀 Features

### Core Capabilities
- **🤖 Automated Code Reproduction**: Multi-agent coordination for faithful research reproduction
- **⚡ Dual Pipeline Modes**: Comprehensive (full analysis) and Optimized (fast processing)
- **🧠 Collaborative Multi-Agent**: Specialized agents working together for accuracy and completeness
- **📊 Real-time Progress Tracking**: Enhanced progress display with pipeline status monitoring
- **🌱 Open-Source & Extensible**: Built for the community with customizable design

### Our Mission

We believe that, in the near future, codebases can be reproduced—or even created from scratch—simply by describing them in natural language. DeepCode is making this vision a reality through:

1. **Research Analysis Agent**: Intelligent content processing and extraction
2. **Workspace Infrastructure Agent**: Automated environment synthesis
3. **Code Architecture Agent**: AI-driven design and planning
4. **Reference Intelligence Agent**: Automated knowledge discovery (Comprehensive mode)
5. **Repository Acquisition Agent**: Intelligent code repository management (Comprehensive mode)
6. **Codebase Intelligence Agent**: Advanced relationship analysis (Comprehensive mode)
7. **Code Implementation Agent**: AI-powered code synthesis

## 📋 Usage

### Enhanced CLI Interface

```bash
# Interactive mode with enhanced agent orchestration
python main_cli.py

# Direct file processing
python main_cli.py --file paper.pdf

# Direct URL processing  
python main_cli.py --url https://arxiv.org/abs/2301.12345

# Optimized mode (faster processing)
python main_cli.py --optimized

# Comprehensive mode (full intelligence analysis)
python main_cli.py --file paper.pdf
```

### Interactive Menu Options

- **[U] Process URL**: Enter research paper URL with intelligent analysis
- **[F] Upload File**: Select and upload file for agent orchestration
- **[C] Configure**: Access configuration menu for pipeline modes
- **[H] History**: View processing history with pipeline mode tracking
- **[Q] Quit**: Exit application with cleanup

### Pipeline Modes

#### 🧠 Comprehensive Mode (Default)
- Full research reproducibility analysis with all agents
- Research Analysis + Resource Processing
- Reference Intelligence Discovery
- Automated Repository Acquisition
- Codebase Intelligence Orchestration
- Intelligent Code Implementation Synthesis

#### ⚡ Optimized Mode
- Fast processing with core agents only
- Research Analysis + Resource Processing
- Code Architecture Synthesis
- Intelligent Code Implementation Synthesis
- Skips: Reference Intelligence, Repository Acquisition, Codebase Intelligence

## 🏗️ Architecture

### Enhanced Components

#### CLI Workflows
- `cli/workflows/cli_workflow_adapter.py`: CLI-optimized workflow adapter
- `cli/workflows/__init__.py`: Workflow module initialization

#### Core Files
- `cli/main_cli.py`: Enhanced CLI launcher with argument parsing
- `cli/cli_app.py`: Main application with agent orchestration integration
- `cli/cli_interface.py`: Enhanced UI with configuration support

#### Integration
- **Agent Orchestration Engine**: `workflows/agent_orchestration_engine.py`
- **MCP Framework**: Model Control Protocol for agent communication
- **Collaborative Architecture**: Multi-agent task distribution and monitoring
- **Data Intelligence Lab @ HKU**: Open-source research initiative

## 🔧 Configuration

### Pipeline Configuration
Access via `[C] Configure` in the main menu:

- **Toggle Pipeline Mode**: Switch between Comprehensive and Optimized
- **View Current Settings**: Real-time configuration display
- **Mode Descriptions**: Detailed explanation of each mode's capabilities

### Command Line Options
```bash
# Available options
--file, -f        Process specific file
--url, -u         Process research paper from URL  
--optimized, -o   Use optimized mode (skip indexing)
--verbose, -v     Enable verbose output
--help, -h        Show help message
```

## 📊 Processing Stages

### Comprehensive Mode (8 stages)
1. **🚀 Initialize**: Agent orchestration engine setup
2. **📊 Analyze**: Research content analysis
3. **📥 Download**: Document processing
4. **📋 Plan**: Code architecture synthesis
5. **🔍 References**: Reference intelligence discovery
6. **📦 Repos**: Repository acquisition automation
7. **🗂️ Index**: Codebase intelligence orchestration
8. **⚙️ Implement**: Code implementation synthesis

### Optimized Mode (5 stages)
1. **🚀 Initialize**: Agent orchestration engine setup
2. **📊 Analyze**: Research content analysis
3. **📥 Download**: Document processing
4. **📋 Plan**: Code architecture synthesis
5. **⚙️ Implement**: Code implementation synthesis

## 🛠️ Installation & Setup

### Requirements
- Python 3.8+
- Enhanced MCP agent framework
- Agent orchestration engine dependencies

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run enhanced CLI
python cli/main_cli.py
```

## 📈 Advanced Features

### Enhanced Error Handling
- **Agent-specific error recovery**: Individual agent failure handling
- **Pipeline resilience**: Continue processing despite partial failures
- **Intelligent retry mechanisms**: Automatic recovery strategies

### Performance Optimization
- **Adaptive mode selection**: Automatic recommendations based on input
- **Resource monitoring**: Real-time system resource tracking
- **Cache management**: Intelligent caching for repeated operations

### Monitoring & Logging
- **Agent activity tracking**: Detailed agent coordination logs
- **Pipeline metrics**: Processing time and success rate analytics
- **Configuration persistence**: Settings saved across sessions

## 🎯 Examples

### Comprehensive Analysis
```bash
# Full intelligence analysis
python main_cli.py --file research_paper.pdf
# Uses all 8 stages with complete agent orchestration
```

### Fast Processing
```bash
# Optimized for speed
python main_cli.py --optimized --file paper.pdf
# Uses 5 core stages, skips indexing-related agents
```

### URL Processing with Configuration
```bash
# Interactive with custom configuration
python main_cli.py
# Select [C] to configure, then [U] to process URL
```

## 🔍 Troubleshooting

### Agent Orchestration Issues
- **Agent initialization failures**: Check MCP server connectivity
- **Pipeline coordination errors**: Verify agent dependencies
- **Resource conflicts**: Monitor system resource usage

### Performance Optimization
- **Slow processing**: Try optimized mode for faster results
- **Memory issues**: Use optimized mode or restart application
- **Network timeouts**: Check connectivity and retry

### Configuration Problems
- **Settings not saved**: Ensure write permissions
- **Mode switching fails**: Restart application if persistent
- **Display issues**: Check terminal compatibility

## 🆕 Upgrade Notes

### Enhanced CLI vs Legacy
- **New**: Intelligent agent orchestration engine integration
- **New**: Configurable pipeline modes (Comprehensive/Optimized)
- **New**: Enhanced progress tracking with 8-stage pipeline
- **New**: CLI workflow adapter for optimized command-line usage
- **Improved**: Error handling with agent-specific recovery
- **Improved**: Performance with adaptive processing modes

### Migration from Legacy CLI
The enhanced CLI maintains full backward compatibility while adding:
- Advanced agent orchestration capabilities
- Configurable processing modes
- Enhanced progress tracking
- Improved error handling and recovery 