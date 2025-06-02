# 增强的PDF转换功能

本项目基于 `pdf2md.py` 的实现方式，完善了 `pdf_downloader.py` 中的文档转换功能，现在支持图片提取和处理。

## 新增功能

### 1. 图片提取支持
- 自动从PDF、Word、PowerPoint等文档中提取图片
- 将图片保存到 `images/` 子目录
- 自动更新Markdown中的图片链接为本地路径
- 支持多种图片格式：PNG, JPG, JPEG, GIF, BMP, WebP

### 2. URL直接转换
- 支持直接从URL转换文档，无需先下载
- 适用于arXiv论文、在线文档等

### 3. 增强的错误处理
- 更好的错误信息和警告
- 优雅的降级处理（docling不可用时使用PyPDF2）

## 主要改进

### DoclingConverter类增强

```python
class DoclingConverter:
    def extract_images(self, doc, output_dir: str) -> Dict[str, str]:
        """提取文档中的图片并保存到本地"""
        
    def process_markdown_with_images(self, markdown_content: str, image_map: Dict[str, str]) -> str:
        """处理Markdown内容，替换图片占位符为实际的图片路径"""
        
    def convert_to_markdown(self, input_file: str, output_file: Optional[str] = None, extract_images: bool = True):
        """将文档转换为Markdown格式，支持图片提取"""
```

### 新增工具函数

1. **convert_url_to_markdown** - 直接从URL转换文档
2. **增强的convert_document_to_markdown** - 支持图片提取参数

## 使用示例

### 1. 转换本地PDF文件（带图片提取）

```python
# 使用MCP工具
result = await convert_document_to_markdown(
    file_path="document.pdf",
    output_path="output.md",
    extract_images=True
)
```

### 2. 直接从URL转换文档

```python
# 转换arXiv论文
result = await convert_url_to_markdown(
    url="https://arxiv.org/pdf/2406.01629.pdf",
    output_path="paper.md",
    extract_images=True
)
```

### 3. 下载并转换（自动图片提取）

```python
# 自然语言指令
result = await download_files("Download https://arxiv.org/pdf/2406.01629.pdf to papers/")
```

## 输出结构

转换后的文件结构：
```
output_directory/
├── document.md          # 转换后的Markdown文件
└── images/              # 提取的图片目录
    ├── image_1.png
    ├── image_2.jpg
    └── ...
```

Markdown文件中的图片引用：
```markdown
# 文档标题

这是一些文本内容。

![Image](images/image_1.png)

更多内容...

![Image](images/image_2.jpg)
```

## 依赖要求

```bash
# 核心依赖
pip install docling
pip install aiohttp
pip install aiofiles

# 可选依赖（用于降级处理）
pip install PyPDF2
```

## 配置选项

### PDF处理选项
```python
pdf_pipeline_options = PdfPipelineOptions()
pdf_pipeline_options.do_ocr = False  # OCR识别
pdf_pipeline_options.do_table_structure = False  # 表格结构识别
```

### 图片提取选项
- `extract_images=True` - 启用图片提取（默认）
- `extract_images=False` - 禁用图片提取，仅提取文本

## 错误处理

1. **docling不可用** - 自动降级到PyPDF2（仅文本提取）
2. **图片提取失败** - 继续转换文本，显示警告信息
3. **URL访问失败** - 提供详细的错误信息

## 性能优化

- 异步处理，支持并发下载和转换
- 智能文件名推断
- 增量图片提取（避免重复处理）

## 测试

运行测试脚本：
```bash
python test_pdf_converter.py
```

测试内容包括：
- 依赖检查
- 本地文件转换
- URL文档转换
- 图片提取功能

## 与pdf2md.py的对比

| 功能 | pdf2md.py | 增强版pdf_downloader.py |
|------|-----------|------------------------|
| 图片提取 | ✓ | ✓ |
| URL支持 | ✓ | ✓ |
| 批量下载 | ✗ | ✓ |
| 自然语言指令 | ✗ | ✓ |
| 多格式支持 | PDF only | PDF, DOCX, PPTX, HTML |
| 异步处理 | ✗ | ✓ |
| MCP集成 | ✗ | ✓ |
| 错误处理 | 基础 | 增强 |

## 注意事项

1. 图片提取需要docling包，确保正确安装
2. 大文件转换可能需要较长时间
3. 网络文档转换依赖网络连接稳定性
4. 某些PDF可能包含受保护的图片，无法提取 