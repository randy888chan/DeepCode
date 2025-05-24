import sys
import os
from urllib.parse import urlparse

def is_url(path):
    try:
        result = urlparse(path)
        return result.scheme in ("http", "https")
    except Exception:
        return False

def pdf_to_md_with_images(input_path, output_path):
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        print("You need to install docling: pip install docling")
        sys.exit(1)

    converter = DocumentConverter()
    # 支持本地文件和URL
    if not is_url(input_path):
        if not os.path.isfile(input_path):
            print(f"Error: {input_path} is not a valid file.")
            sys.exit(1)
        if not input_path.lower().endswith('.pdf'):
            print(f"Error: {input_path} is not a PDF file.")
            sys.exit(1)
    else:
        if not input_path.lower().endswith('.pdf'):
            print(f"Error: {input_path} is not a PDF url.")
            sys.exit(1)
    result = converter.convert(input_path)
    doc = result.document

    # 1. 提取图片
    images_dir = os.path.join(os.path.dirname(output_path), "images")
    os.makedirs(images_dir, exist_ok=True)
    image_map = {}  # docling图片id -> 本地文件名

    for idx, img in enumerate(getattr(doc, 'images', [])):
        ext = getattr(img, 'format', None) or 'png'
        filename = f"image_{idx+1}.{ext}"
        filepath = os.path.join(images_dir, filename)
        with open(filepath, "wb") as f:
            f.write(img.data)
        image_map[getattr(img, 'id', str(idx+1))] = os.path.relpath(filepath, os.path.dirname(output_path))

    # 2. 导出Markdown并替换图片占位符
    md_content = doc.export_to_markdown()
    import re
    def replace_img(match):
        img_id = match.group(1)
        if img_id in image_map:
            return f"![Image]({image_map[img_id]})"
        else:
            return match.group(0)
    md_content = re.sub(r'!\[Image\]\(docling://image/([^)]+)\)', replace_img, md_content)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Converted {input_path} to {output_path} (images saved to {images_dir})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_pdf_path_or_url> <output_md_path>")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    # input_path = "https://arxiv.org/pdf/2406.01629.pdf"
    # output_path = "./papers/1/2406.01629.md"
    pdf_to_md_with_images(input_path, output_path) 