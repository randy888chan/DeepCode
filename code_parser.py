from tree_sitter import Language, Parser
from collections import defaultdict
import os
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())

parser = Parser(PY_LANGUAGE)

class TSFunctionDependencyParser:
    def __init__(self, source_code):
        self.source_code = source_code.encode()
        self.tree = parser.parse(self.source_code)
        self.dependencies = defaultdict(set)
        self.nodes = set()
        self.imports = set()  # 用于存储import语句
        self.global_definitions = set()  # 用于存储全局定义

    def parse(self):
        self._visit_node(self.tree.root_node)
        # 打包import和全局定义为单独的节点
        self.insert_node("Imports", "\n".join(self.imports))
        self.insert_node("Global Definitions", "\n".join(self.global_definitions))
        return self.dependencies

    def _visit_node(self, node, current_class=None, current_function=None):
        # 处理import语句
        if node.type == 'import_statement':
            import_text = node.text.decode()
            self.imports.add(import_text)
            print(f"Import found: {import_text}")

        # 处理全局变量定义
        elif node.type == 'assignment' and not current_function:
            assignment_text = node.text.decode()
            self.global_definitions.add(assignment_text)
            print(f"Global variable assignment: {assignment_text}")

        # 解析类定义
        elif node.type == 'class_definition':
            class_name = node.child_by_field_name('name').text.decode()
            self.insert_node(class_name, f"Class: {class_name}")
            for child in node.children:
                self._visit_node(child, current_class=class_name)

        # 解析函数定义
        elif node.type == 'function_definition':
            func_name = node.child_by_field_name('name').text.decode()
            full_func_name = f"{current_class}.{func_name}" if current_class else func_name
            func_content = self._get_source_segment(node)
            self.insert_node(full_func_name, func_content)

            for child in node.children:
                self._visit_node(child, current_class, current_function=full_func_name)

        # 解析函数调用
        elif node.type == 'call' and current_function:
            called_function = node.child_by_field_name('function')
            if called_function:
                called_name = called_function.text.decode()
                self.dependencies[current_function].add(called_name)
                self.insert_relation(current_function, called_name)

        else:
            for child in node.children:
                self._visit_node(child, current_class, current_function)

    def _get_source_segment(self, node):
        return self.source_code[node.start_byte:node.end_byte].decode()

    def insert_node(self, function_name, content):
        if function_name not in self.nodes:
            self.nodes.add(function_name)
            print(f"Inserting node: {function_name}")
            # print(f"Content: {content}")

    def insert_relation(self, from_function, to_function):
        if to_function in self.nodes:
            print(f"Inserting relation from {from_function} to {to_function}")

def parse_python_dependencies(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()

    ts_parser = TSFunctionDependencyParser(source_code)
    dependencies = ts_parser.parse()

    return dependencies

# 示例调用
if __name__ == "__main__":
    dependencies = parse_python_dependencies('code_parser.py')
    print(dependencies)
