import os
import fnmatch
from typing import List, Dict

# 定义文件类型
DOCUMENT_EXTENSIONS = {'.md', '.txt', '.pdf', '.rst', '.docx'}
CODE_EXTENSIONS = {'.py', '.js', '.java', '.c', '.cpp', '.h', '.go', '.rs'}


def get_file_type(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    if ext in DOCUMENT_EXTENSIONS:
        return 'document'
    elif ext in CODE_EXTENSIONS:
        return 'code'
    else:
        return 'other'


def should_ignore(path: str, ignore_patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in ignore_patterns)


def build_file_tree(root_dir: str, ignore_patterns: List[str] = []) -> Dict:
    file_tree = {
        'name': os.path.basename(root_dir),
        'type': 'directory',
        'children': []
    }

    for root, dirs, files in os.walk(root_dir):
        relative_root = os.path.relpath(root, root_dir)

        path_parts = relative_root.split(os.sep) if relative_root != '.' else []
        subtree = file_tree

        # 判断是否忽略当前目录
        if relative_root != '.' and should_ignore(relative_root, ignore_patterns):
            dirs[:] = []  # Skip subdirectories
            continue

        for part in path_parts:
            found = False
            for child in subtree['children']:
                if child['type'] == 'directory' and child['name'] == part:
                    subtree = child
                    found = True
                    break
            if not found:
                new_dir = {'name': part, 'type': 'directory', 'children': []}
                subtree['children'].append(new_dir)
                subtree = new_dir

        files_to_add = []
        for file in files:
            relative_file_path = os.path.normpath(os.path.join(relative_root, file))
            if should_ignore(relative_file_path, ignore_patterns):
                continue

            file_type = get_file_type(file)
            files_to_add.append({
                'name': file,
                'type': file_type,
                'path': os.path.join(root, file)
            })

        subtree['children'].extend(files_to_add)

    return file_tree


# 示例用法
if __name__ == "__main__":
    repo_path = './'  # 修改为你本地的repo路径
    ignore_list = ['*.log', '*/__pycache__/*', '.*', 'LICENSE', "assets"]  # 示例忽略列表，可自行修改
    # ignore_list = []
    tree = build_file_tree(repo_path, ignore_list)

    import json
    print(json.dumps(tree, indent=2, ensure_ascii=False))