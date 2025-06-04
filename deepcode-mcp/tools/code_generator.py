"""
Code Generator MCP Server

This server provides tools for generating and managing code files during the paper-to-code implementation process.
Uses FastMCP for standardized MCP server implementation.
"""

import os
import json
import logging
from typing import Dict, List, Any
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("code-generator")

def _dict_to_yaml(data: Dict, indent: int = 0) -> str:
    """Simple YAML serializer for basic data structures."""
    yaml_str = ""
    indent_str = "  " * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            yaml_str += f"{indent_str}{key}:\n"
            yaml_str += _dict_to_yaml(value, indent + 1)
        elif isinstance(value, list):
            yaml_str += f"{indent_str}{key}:\n"
            for item in value:
                yaml_str += f"{indent_str}  - {item}\n"
        else:
            yaml_str += f"{indent_str}{key}: {value}\n"
    
    return yaml_str

@mcp.tool()
async def create_project_structure(base_path: str, structure: dict) -> str:
    """Create a complete project directory structure with skeleton files.
    
    Args:
        base_path: Base directory path for the project
        structure: Project structure as nested dictionary
    """
    try:
        created_items = []
        
        def create_structure_recursive(current_path: str, struct: Dict):
            """Recursively create directories and files."""
            for name, content in struct.items():
                item_path = os.path.join(current_path, name)
                
                if isinstance(content, dict):
                    # It's a directory
                    os.makedirs(item_path, exist_ok=True)
                    created_items.append({"type": "directory", "path": item_path})
                    create_structure_recursive(item_path, content)
                elif isinstance(content, str):
                    # It's a file
                    os.makedirs(os.path.dirname(item_path), exist_ok=True)
                    with open(item_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    created_items.append({"type": "file", "path": item_path})
        
        # Create base directory
        os.makedirs(base_path, exist_ok=True)
        created_items.append({"type": "directory", "path": base_path})
        
        # Create structure
        create_structure_recursive(base_path, structure)
        
        result = {
            "status": "success",
            "message": f"Created project structure at {base_path}",
            "created_items": created_items,
            "total_directories": len([i for i in created_items if i["type"] == "directory"]),
            "total_files": len([i for i in created_items if i["type"] == "file"])
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error creating project structure: {e}")
        error_result = {
            "status": "error",
            "message": f"Failed to create project structure: {str(e)}"
        }
        return json.dumps(error_result, indent=2)

@mcp.tool()
async def generate_python_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """Generate a Python file with proper structure, imports, and docstrings.
    
    Args:
        file_path: Path where the file should be created
        content: Python code content for the file
        overwrite: Whether to overwrite existing file (default: False)
    """
    try:
        # Check if file exists
        if os.path.exists(file_path) and not overwrite:
            error_result = {
                "status": "error",
                "message": f"File already exists: {file_path}. Set overwrite=True to replace."
            }
            return json.dumps(error_result, indent=2)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Validate syntax
        syntax_check = await validate_python_syntax(file_path)
        syntax_data = json.loads(syntax_check)
        
        result = {
            "status": "success",
            "message": f"Generated Python file: {file_path}",
            "file_path": file_path,
            "size": os.path.getsize(file_path),
            "syntax_valid": syntax_data.get("valid", False),
            "syntax_errors": syntax_data.get("errors", [])
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error generating Python file: {e}")
        error_result = {
            "status": "error",
            "message": f"Failed to generate Python file: {str(e)}"
        }
        return json.dumps(error_result, indent=2)

@mcp.tool()
async def generate_requirements_file(project_path: str, dependencies: List[str]) -> str:
    """Generate a requirements.txt file with dependencies.
    
    Args:
        project_path: Project directory path
        dependencies: List of dependencies with versions
    """
    try:
        req_path = os.path.join(project_path, "requirements.txt")
        
        # Create directory if needed
        os.makedirs(project_path, exist_ok=True)
        
        # Write requirements
        with open(req_path, 'w', encoding='utf-8') as f:
            for dep in dependencies:
                f.write(f"{dep}\n")
        
        result = {
            "status": "success",
            "message": f"Generated requirements.txt at {req_path}",
            "file_path": req_path,
            "dependencies_count": len(dependencies),
            "dependencies": dependencies
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error generating requirements file: {e}")
        error_result = {
            "status": "error",
            "message": f"Failed to generate requirements file: {str(e)}"
        }
        return json.dumps(error_result, indent=2)

@mcp.tool()
async def generate_config_file(file_path: str, config_data: dict, format_type: str) -> str:
    """Generate configuration files (YAML/JSON).
    
    Args:
        file_path: Path for the config file
        config_data: Configuration data as dictionary
        format_type: Config file format ('yaml' or 'json')
    """
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if format_type == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
        elif format_type == "yaml":
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(_dict_to_yaml(config_data))
        else:
            error_result = {
                "status": "error",
                "message": f"Unsupported format: {format_type}"
            }
            return json.dumps(error_result, indent=2)
        
        result = {
            "status": "success",
            "message": f"Generated {format_type.upper()} config file: {file_path}",
            "file_path": file_path,
            "format": format_type
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error generating config file: {e}")
        error_result = {
            "status": "error",
            "message": f"Failed to generate config file: {str(e)}"
        }
        return json.dumps(error_result, indent=2)

@mcp.tool()
async def validate_python_syntax(file_path: str) -> str:
    """Validate Python file syntax and return any errors.
    
    Args:
        file_path: Path to the Python file to validate
    """
    try:
        if not os.path.exists(file_path):
            result = {
                "valid": False,
                "errors": [f"File not found: {file_path}"],
                "message": "File not found"
            }
            return json.dumps(result, indent=2)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        try:
            compile(code, file_path, 'exec')
            result = {
                "valid": True,
                "errors": [],
                "message": "Python syntax is valid"
            }
            return json.dumps(result, indent=2)
        except SyntaxError as e:
            result = {
                "valid": False,
                "errors": [{
                    "line": e.lineno,
                    "offset": e.offset,
                    "message": e.msg,
                    "text": e.text
                }],
                "message": "Python syntax errors found"
            }
            return json.dumps(result, indent=2)
            
    except Exception as e:
        logger.error(f"Error validating Python syntax: {e}")
        result = {
            "valid": False,
            "errors": [str(e)],
            "message": "Error during validation"
        }
        return json.dumps(result, indent=2)

@mcp.tool()
async def create_directory(directory_path: str) -> str:
    """Create a directory and any necessary parent directories.
    
    Args:
        directory_path: Path of the directory to create
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        
        result = {
            "status": "success",
            "message": f"Created directory: {directory_path}",
            "path": directory_path,
            "exists": os.path.exists(directory_path)
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error creating directory: {e}")
        error_result = {
            "status": "error",
            "message": f"Failed to create directory: {str(e)}"
        }
        return json.dumps(error_result, indent=2)

@mcp.tool()
async def write_file(file_path: str, content: str, encoding: str = "utf-8") -> str:
    """Write content to a file.
    
    Args:
        file_path: Path where the file should be created/written
        content: Content to write to the file
        encoding: File encoding (default: utf-8)
    """
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write file
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        result = {
            "status": "success",
            "message": f"Successfully wrote file: {file_path}",
            "file_path": file_path,
            "size": os.path.getsize(file_path),
            "encoding": encoding
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error writing file: {e}")
        error_result = {
            "status": "error",
            "message": f"Failed to write file: {str(e)}"
        }
        return json.dumps(error_result, indent=2)

if __name__ == "__main__":
    # Initialize and run the server using FastMCP
    mcp.run(transport='stdio') 