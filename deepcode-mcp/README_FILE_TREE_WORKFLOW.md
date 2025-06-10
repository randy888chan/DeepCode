# 通用论文代码复现Agent - 文件树创建功能

## 概述

本项目实现了一个通用的论文代码复现Agent，专门用于从论文实现计划中自动提取文件结构信息并创建完整的项目文件树。这是论文代码复现流程的第一步，为后续的代码实现奠定基础。

## 功能特点

### 🎯 核心功能
- **智能文件树提取**: 从论文实现计划中自动识别和提取项目文件结构
- **自动文件创建**: 根据提取的结构在指定目录创建完整的文件树
- **多种提取模式**: 支持LLM智能提取和文本直接解析两种模式
- **灵活配置**: 支持多种LLM后端（OpenAI GPT-4、Anthropic Claude）

### 🔧 技术特点
- **模块化设计**: 清晰的模块结构，易于扩展和维护
- **错误处理**: 完善的错误处理和恢复机制
- **路径解析**: 智能的文件路径解析和目录层级处理
- **跨平台支持**: 支持Windows、Linux、macOS等多种操作系统

## 项目结构

```
├── workflows/
│   └── code_implementation_workflow.py   # 主工作流实现
├── tools/
│   ├── file_tree_creator.py              # MCP工具实现
│   └── simple_file_creator.py            # 简化文件创建工具
├── prompts/
│   └── code_prompts.py                   # LLM提示词定义
├── test_final_workflow.py                # 完整功能测试
└── README_FILE_TREE_WORKFLOW.md          # 本文档
```

## 使用方法

### 1. 基本使用

```python
from workflows.code_implementation_workflow import CodeImplementationWorkflow

# 创建工作流实例
workflow = CodeImplementationWorkflow()

# 运行文件树创建
result = await workflow.run_file_tree_creation(
    plan_file_path="path/to/your/initial_plan.txt",
    target_directory="path/to/output/directory",
    use_llm_for_extraction=False  # 使用文本解析模式
)
```

### 2. 便捷函数

```python
from workflows.code_implementation_workflow import create_project_structure

# 一键创建项目结构
result = await create_project_structure(
    plan_file_path="path/to/your/initial_plan.txt",
    target_directory="path/to/output/directory"
)
```

### 3. 命令行使用

```bash
# 直接运行主工作流
python workflows/code_implementation_workflow.py

# 运行完整测试
python test_final_workflow.py
```

## 配置要求

### 1. API配置

在`mcp_agent.secrets.yaml`中配置LLM API：

```yaml
openai:
  api_key: "your-openai-api-key"

anthropic:
  api_key: "your-anthropic-api-key"
```

### 2. 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：
- `openai>=1.0.0`
- `anthropic>=0.7.0` 
- `PyYAML>=6.0`
- `pathlib`

## 实现计划格式

工作流支持以下格式的实现计划：

```
Implementation Plan

---

1. Project Overview
...

4. Code Organization (File Tree)

project/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── gcn.py        # GCN encoder
│   │   ├── diffusion.py  # forward/reverse processes
│   │   ├── denoiser.py   # denoising MLP
│   │   └── fusion.py     # fusion combiner
│   ├── models/           # model wrapper classes
│   │   └── recdiff.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data.py       # loading & preprocessing
│   │   ├── predictor.py  # scoring functions
│   │   ├── loss.py       # loss functions
│   │   ├── metrics.py    # NDCG, Recall etc.
│   │   └── sched.py      # beta/alpha schedule utils
│   └── configs/
│       └── default.yaml  # hyperparameters, paths
├── tests/
│   ├── test_gcn.py
│   ├── test_diffusion.py
│   ├── test_denoiser.py
│   ├── test_loss.py
│   └── test_pipeline.py
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── README.md
├── experiments/
│   ├── run_experiment.py
│   └── notebooks/
│       └── analysis.ipynb
├── requirements.txt
└── setup.py

---
```

## 输出示例

成功运行后，会在目标目录下创建`generate_code`文件夹，包含完整的项目结构：

```
generate_code/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── gcn.py
│   │   ├── diffusion.py
│   │   ├── denoiser.py
│   │   └── fusion.py
│   ├── models/
│   │   └── recdiff.py
│   └── utils/
│       ├── __init__.py
│       ├── data.py
│       ├── predictor.py
│       ├── loss.py
│       ├── metrics.py
│       └── sched.py
├── tests/
│   ├── test_gcn.py
│   ├── test_diffusion.py
│   ├── test_denoiser.py
│   ├── test_loss.py
│   └── test_pipeline.py
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── README.md
├── requirements.txt
└── setup.py
```

## 测试验证

### 运行测试

```bash
# 完整功能测试
python test_final_workflow.py

# 简化工具测试
python tools/simple_file_creator.py

# 工作流单元测试
python workflows/code_implementation_workflow.py
```

### 预期输出

成功运行后应看到类似输出：

```
============================================================
   通用论文代码复现Agent - 文件树创建测试
============================================================

📄 输入计划文件: agent_folders\papers\paper_3\initial_plan.txt

🚀 开始运行文件树创建工作流...
开始处理计划文件: agent_folders\papers\paper_3\initial_plan.txt
目标目录: agent_folders\papers\paper_3
步骤1: 提取文件树结构...
文件树提取完成
步骤2: 解析文件列表...
解析文件列表: 32 个文件
步骤3: 创建文件结构...
文件结构创建完成

============================================================
   工作流执行结果
============================================================
📊 执行状态: success
📁 目标目录: agent_folders\papers\paper_3
📝 创建文件数: 32
📋 计划文件: agent_folders\papers\paper_3\initial_plan.txt

✅ 文件树创建成功！
```

## 扩展性

### 1. 支持新的文件格式
可以通过修改`_extract_file_tree_from_text`方法来支持不同的计划文件格式。

### 2. 添加新的LLM后端
在`_setup_llm_clients`方法中添加新的LLM客户端支持。

### 3. 自定义文件创建逻辑
通过扩展`simple_file_creator.py`来支持更复杂的文件创建需求。

## 故障排除

### 常见问题

1. **导入错误**: 确保Python路径正确设置
2. **API超时**: 检查网络连接和API密钥
3. **文件权限**: 确保有目标目录的写入权限
4. **路径问题**: 使用绝对路径或确保相对路径正确

### 调试技巧

- 启用详细日志输出
- 逐步运行各个阶段
- 检查中间结果文件
- 验证API配置正确性

## 贡献指南

欢迎提交问题报告和改进建议！在提交PR前请确保：

1. 代码符合项目规范
2. 添加适当的测试用例
3. 更新相关文档
4. 通过所有现有测试

## 许可证

本项目采用MIT许可证。详见LICENSE文件。 