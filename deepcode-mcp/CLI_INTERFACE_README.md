# Paper-to-Code Engine - Professional CLI Interface

## 🚀 概述

我们为Paper-to-Code Engine设计了一个全新的专业级命令行界面，提供美观、直观和功能丰富的用户体验。

## ✨ 新功能亮点

### 🎨 视觉设计
- **渐变色ASCII Logo**: 使用多色渐变的专业ASCII艺术logo
- **色彩编码状态**: 不同类型的消息使用不同颜色和图标
- **优雅的边框设计**: 统一的Unicode字符边框风格
- **实时进度显示**: 动态进度条和旋转加载动画

### 📁 智能文件上传
- **GUI文件选择器**: 使用Tkinter提供现代化的文件对话框
- **多格式支持**: 支持PDF、Word、PowerPoint、HTML、文本等格式
- **文件信息预览**: 显示文件名、路径、大小和类型
- **路径智能截断**: 自动处理长文件路径的显示

### 🌐 增强URL处理
- **URL示例展示**: 提供常见学术网站的URL格式示例
- **智能验证**: 基本的URL格式验证和域名识别
- **支持特殊格式**: 识别arXiv的@开头URL格式

### 🔄 交互体验
- **直观菜单**: 清晰的选项布局和快捷键提示
- **状态反馈**: 实时显示操作进度和结果
- **错误处理**: 美观的错误信息展示
- **优雅退出**: 个性化的告别消息

## 📂 文件结构

```
├── main.py                 # 主程序入口（已重构）
├── utils/
│   └── cli_interface.py   # CLI界面模块
└── tools/
    └── pdf_downloader.py  # 增强的文档下载器
```

## 🎯 核心组件

### CLIInterface 类
专业的CLI界面管理器，包含：

- `print_logo()` - 显示渐变色logo
- `print_welcome_banner()` - 显示欢迎横幅
- `create_menu()` - 创建交互式菜单
- `upload_file_gui()` - GUI文件选择器
- `get_url_input()` - URL输入界面
- `show_progress_bar()` - 进度条动画
- `show_spinner()` - 旋转加载动画
- `print_status()` - 状态消息显示
- `print_error_box()` - 错误信息框
- `print_goodbye()` - 告别消息

### Colors 类
完整的ANSI颜色定义：
- 基础颜色：红、绿、蓝、黄、青、紫
- 样式：粗体、下划线、高亮
- 状态颜色：成功、警告、错误、信息

## 🚀 使用方法

### 启动程序
```bash
python main.py
```

### 操作选项
- **[U]** - 输入研究论文URL
- **[F]** - 上传本地文件
- **[Q]** - 退出程序

### 支持的输入格式

#### URL格式
```
✅ arXiv: https://arxiv.org/pdf/2403.00813
✅ arXiv: @https://arxiv.org/pdf/2403.00813
✅ IEEE: https://ieeexplore.ieee.org/document/...
✅ ACM: https://dl.acm.org/doi/...
✅ 直接PDF: https://example.com/paper.pdf
```

#### 文件格式
```
✅ PDF文件 (.pdf)
✅ Word文档 (.docx, .doc)
✅ PowerPoint (.pptx, .ppt)
✅ HTML文件 (.html, .htm)
✅ 文本文件 (.txt, .md)
```

## 🎨 界面预览

### Logo展示
```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  ██████╗  █████╗ ██████╗ ███████╗██████╗     ████████╗ ██████╗                ║
║  ██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗    ╚══██╔══╝██╔═══██╗               ║
║  ██████╔╝███████║██████╔╝█████╗  ██████╔╝       ██║   ██║   ██║               ║
║  ██╔═══╝ ██╔══██║██╔═══╝ ██╔══╝  ██╔══██╗       ██║   ██║   ██║               ║
║  ██║     ██║  ██║██║     ███████╗██║  ██║       ██║   ╚██████╔╝               ║
║  ╚═╝     ╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝       ╚═╝    ╚═════╝                ║
║                                                                               ║
║  🚀 AI-Powered Research Paper to Code Generation Engine 🚀                   ║
║                                                                               ║
║  ✨ Features:                                                                ║
║     • Intelligent PDF Analysis & Code Extraction                             ║
║     • Advanced Document Processing with Docling                             ║
║     • Multi-format Support (PDF, DOCX, PPTX, HTML)                         ║
║     • Smart File Upload Interface                                          ║
║     • Automated GitHub Repository Management                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 主菜单
```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                MAIN MENU                                      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  🌐 [U] Process URL       │  📁 [F] Upload File    │  ❌ [Q] Quit           ║
║                                                                               ║
║  📝 Enter a research paper URL (arXiv, IEEE, ACM, etc.)                      ║
║     or upload a PDF/DOC file for intelligent analysis                        ║
║                                                                               ║
║  💡 Tip: Press 'F' to open file browser or 'U' to enter URL manually        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 状态指示器
```
✅ 成功操作
❌ 错误信息
⚠️ 警告提示
ℹ️ 普通信息
⏳ 处理中
📁 文件操作
📥 下载操作
🔍 分析操作
```

## 🔧 技术特性

### 模块化设计
- CLI界面完全分离到独立模块
- 主程序逻辑更加简洁清晰
- 易于维护和扩展

### 跨平台兼容
- Windows、macOS、Linux全平台支持
- 自动检测操作系统并适配命令
- 统一的字符编码处理

### 用户体验优化
- 智能输入验证
- 友好的错误提示
- 流畅的操作流程
- 专业的视觉设计

## 🛠️ 自定义配置

### 修改颜色主题
编辑 `utils/cli_interface.py` 中的 `Colors` 类：
```python
class Colors:
    # 自定义你的颜色方案
    PRIMARY = '\033[36m'    # 主色调
    SUCCESS = '\033[92m'    # 成功色
    ERROR = '\033[91m'      # 错误色
    WARNING = '\033[93m'    # 警告色
```

### 自定义Logo
修改 `CLIInterface.print_logo()` 方法来使用你的自定义logo。

### 添加新的状态类型
在 `print_status()` 方法中添加新的状态样式：
```python
status_styles = {
    "custom": f"{Colors.PURPLE}🎯",
    # 添加更多自定义状态
}
```

## 📝 更新日志

### v2.0.0 - 专业CLI界面
- ✨ 全新的渐变色ASCII logo设计
- 🎨 完整的颜色主题系统
- 📁 GUI文件选择器集成
- 🔄 动态进度条和加载动画
- 📊 增强的状态反馈系统
- 🌐 智能URL验证和示例
- 💡 用户友好的提示和帮助
- 🚀 模块化架构重构

## 🤝 贡献指南

欢迎提交建议和改进：
1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

MIT License - 详见 LICENSE 文件

---

**Paper-to-Code Engine** - 让研究论文到代码的转换变得简单而优雅！ ✨ 