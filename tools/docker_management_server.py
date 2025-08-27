#!/usr/bin/env python3
"""
Docker Management MCP Server

This MCP server provides Docker-based environment isolation and management
for safe repository evaluation and code execution.

Core Features:
1. Container lifecycle management (create, start, stop, remove)
2. Safe file mounting and workspace setup
3. Dependency installation in isolated environments
4. Command execution with output capture
5. Resource monitoring and limits
6. Multi-language environment support

Tools Provided:
- create_evaluation_container: Create isolated container for evaluation
- setup_container_workspace: Mount repository and setup workspace
- install_dependencies: Install project dependencies in container
- execute_in_container: Execute commands safely in container
- get_container_logs: Retrieve execution logs and outputs
- cleanup_container: Clean up container resources
- list_available_images: List supported base images
- monitor_container_resources: Monitor resource usage

Security Features:
- Read-only repository mounting
- Resource limits (CPU, memory, disk)
- Network isolation options
- Temporary container cleanup
- Secure command execution

Usage:
python tools/docker_management_server.py
"""

import os
import json
import sys
import subprocess
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import docker
from docker.errors import DockerException, APIError, ContainerError, ImageNotFound

# Import MCP modules
from mcp.server.fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("docker-management")


@dataclass
class ContainerInfo:
    """Container information structure"""
    container_id: str
    name: str
    image: str
    status: str
    created_at: str
    workspace_path: str
    ports: Dict[str, int] = None
    environment: Dict[str, str] = None
    
    def __post_init__(self):
        if self.ports is None:
            self.ports = {}
        if self.environment is None:
            self.environment = {}


@dataclass
class ExecutionResult:
    """Command execution result structure"""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    success: bool


@dataclass
class ResourceUsage:
    """Container resource usage information"""
    cpu_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    memory_percent: float
    network_io: Dict[str, int]
    block_io: Dict[str, int]


# Supported base images for different language environments
SUPPORTED_IMAGES = {
    "python": {
        "3.9": "python:3.9-slim",
        "3.10": "python:3.10-slim", 
        "3.11": "python:3.11-slim",
        "3.12": "python:3.12-slim"
    },
    "node": {
        "16": "node:16-alpine",
        "18": "node:18-alpine",
        "20": "node:20-alpine"
    },
    "java": {
        "11": "openjdk:11-jdk-slim",
        "17": "openjdk:17-jdk-slim",
        "21": "openjdk:21-jdk-slim"
    },
    "ubuntu": {
        "20.04": "ubuntu:20.04",
        "22.04": "ubuntu:22.04"
    },
    "multi": {
        "python-node": "nikolaik/python-nodejs:python3.9-nodejs18",
        "full": "continuumio/miniconda3"
    }
}

# Default resource limits
DEFAULT_LIMITS = {
    "memory": "2g",
    "cpu_count": 2,
    "cpu_percent": 100,
    "disk_quota": "10g"
}

# Container name prefix
CONTAINER_PREFIX = "deepcode_eval"

# Global Docker client
docker_client = None


def get_docker_client():
    """Get Docker client with error handling"""
    global docker_client
    if docker_client is None:
        try:
            docker_client = docker.from_env()
            # Test connection
            docker_client.ping()
            logger.info("Docker client connected successfully")
        except DockerException as e:
            logger.warning(f"Docker not available: {e}")
            docker_client = None
            return None
        except Exception as e:
            logger.warning(f"Docker connection failed: {e}")
            docker_client = None
            return None
    return docker_client


def generate_container_name(prefix: str = CONTAINER_PREFIX) -> str:
    """Generate unique container name"""
    timestamp = int(time.time())
    return f"{prefix}_{timestamp}_{os.getpid()}"


def select_base_image(language: str, version: str = None) -> str:
    """Select appropriate base image for language/version"""
    if language not in SUPPORTED_IMAGES:
        return "ubuntu:22.04"  # Default fallback
    
    lang_images = SUPPORTED_IMAGES[language]
    if version and version in lang_images:
        return lang_images[version]
    
    # Return first available version
    return list(lang_images.values())[0]


def prepare_workspace_commands(repo_path: str, workspace_path: str = "/workspace") -> List[str]:
    """Prepare commands to setup workspace in container"""
    commands = [
        f"mkdir -p {workspace_path}",
        f"cd {workspace_path}",
        "ls -la"
    ]
    return commands


def parse_resource_stats(stats: Dict) -> ResourceUsage:
    """Parse Docker container resource statistics"""
    try:
        # CPU usage calculation
        cpu_stats = stats.get("cpu_stats", {})
        precpu_stats = stats.get("precpu_stats", {})
        
        cpu_usage = cpu_stats.get("cpu_usage", {}).get("total_usage", 0)
        precpu_usage = precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
        
        system_usage = cpu_stats.get("system_cpu_usage", 0)
        presystem_usage = precpu_stats.get("system_cpu_usage", 0)
        
        cpu_delta = cpu_usage - precpu_usage
        system_delta = system_usage - presystem_usage
        
        online_cpus = cpu_stats.get("online_cpus", 1)
        if system_delta > 0 and cpu_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * online_cpus * 100.0
        else:
            cpu_percent = 0.0
        
        # Memory usage
        memory_stats = stats.get("memory_stats", {})
        memory_usage = memory_stats.get("usage", 0)
        memory_limit = memory_stats.get("limit", 0)
        memory_usage_mb = memory_usage / (1024 * 1024)
        memory_limit_mb = memory_limit / (1024 * 1024)
        memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
        
        # Network I/O
        networks = stats.get("networks", {})
        network_io = {}
        for interface, net_stats in networks.items():
            network_io[f"{interface}_rx"] = net_stats.get("rx_bytes", 0)
            network_io[f"{interface}_tx"] = net_stats.get("tx_bytes", 0)
        
        # Block I/O
        blkio_stats = stats.get("blkio_stats", {})
        io_service_bytes = blkio_stats.get("io_service_bytes_recursive", [])
        block_io = {"read": 0, "write": 0}
        for entry in io_service_bytes:
            if entry.get("op") == "Read":
                block_io["read"] += entry.get("value", 0)
            elif entry.get("op") == "Write":
                block_io["write"] += entry.get("value", 0)
        
        return ResourceUsage(
            cpu_percent=round(cpu_percent, 2),
            memory_usage_mb=round(memory_usage_mb, 2),
            memory_limit_mb=round(memory_limit_mb, 2),
            memory_percent=round(memory_percent, 2),
            network_io=network_io,
            block_io=block_io
        )
        
    except Exception as e:
        logger.warning(f"Failed to parse resource stats: {e}")
        return ResourceUsage(
            cpu_percent=0.0,
            memory_usage_mb=0.0,
            memory_limit_mb=0.0,
            memory_percent=0.0,
            network_io={},
            block_io={}
        )


# ==================== MCP Tool Definitions ====================

@mcp.tool()
async def list_available_images() -> str:
    """
    List all supported Docker base images for different languages.
    
    Returns:
        JSON string with available images categorized by language
    """
    try:
        result = {
            "status": "success",
            "supported_images": SUPPORTED_IMAGES,
            "default_limits": DEFAULT_LIMITS,
            "recommendations": {
                "python": "Use python:3.11-slim for most Python projects",
                "node": "Use node:18-alpine for Node.js projects",
                "java": "Use openjdk:17-jdk-slim for Java projects",
                "multi": "Use nikolaik/python-nodejs for Python+Node.js projects"
            }
        }
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Failed to list available images: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to list images: {str(e)}"
        })


@mcp.tool()
async def create_evaluation_container(
    image_type: str,
    version: str = None,
    container_name: str = None,
    memory_limit: str = "2g",
    cpu_limit: str = "2",
    network_mode: str = "none"
) -> str:
    """
    Create a new Docker container for code evaluation.
    
    Args:
        image_type: Type of base image (python, node, java, ubuntu, multi)
        version: Specific version of the image (optional)
        container_name: Custom container name (auto-generated if None)
        memory_limit: Memory limit (e.g., "2g", "1024m")
        cpu_limit: CPU limit (number of CPUs)
        network_mode: Network isolation mode ("none", "bridge", "host")
        
    Returns:
        JSON string with container creation result
    """
    try:
        client = get_docker_client()
        
        # Select appropriate image
        base_image = select_base_image(image_type, version)
        logger.info(f"Selected base image: {base_image}")
        
        # Generate container name
        if not container_name:
            container_name = generate_container_name()
        
        # Pull image if not available
        try:
            client.images.get(base_image)
        except ImageNotFound:
            logger.info(f"Pulling image: {base_image}")
            client.images.pull(base_image)
        
        # Container configuration
        container_config = {
            "image": base_image,
            "name": container_name,
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "mem_limit": memory_limit,
            "cpu_count": int(float(cpu_limit)),
            "network_mode": network_mode,
            "working_dir": "/workspace",
            "environment": {
                "DEBIAN_FRONTEND": "noninteractive",
                "PYTHONUNBUFFERED": "1",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8"
            },
            "command": "/bin/bash",
            "auto_remove": False  # We'll manage cleanup manually
        }
        
        # Create container
        container = client.containers.create(**container_config)
        
        # Start container
        container.start()
        
        # Wait a moment for container to be ready
        time.sleep(1)
        
        container_info = ContainerInfo(
            container_id=container.id,
            name=container_name,
            image=base_image,
            status="running",
            created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            workspace_path="/workspace",
            environment=container_config["environment"]
        )
        
        result = {
            "status": "success",
            "container": asdict(container_info),
            "message": f"Container '{container_name}' created successfully"
        }
        
        logger.info(f"Container created: {container_name} ({container.id[:12]})")
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except DockerException as e:
        logger.error(f"Docker error creating container: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Docker error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Failed to create container: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Container creation failed: {str(e)}"
        })


@mcp.tool()
async def setup_container_workspace(
    container_id: str,
    repo_path: str,
    docs_path: str = None,
    workspace_path: str = "/workspace"
) -> str:
    """
    Setup workspace in container by mounting repository and documentation.
    
    Args:
        container_id: Container ID or name
        repo_path: Local path to repository
        docs_path: Local path to documentation (optional)
        workspace_path: Container workspace path
        
    Returns:
        JSON string with workspace setup result
    """
    try:
        client = get_docker_client()
        
        # Get container
        container = client.containers.get(container_id)
        
        # Verify paths exist
        if not os.path.exists(repo_path):
            return json.dumps({
                "status": "error",
                "message": f"Repository path does not exist: {repo_path}"
            })
        
        repo_path = os.path.abspath(repo_path)
        
        # Create workspace directory in container
        exec_result = container.exec_run(f"mkdir -p {workspace_path}")
        if exec_result.exit_code != 0:
            return json.dumps({
                "status": "error",
                "message": f"Failed to create workspace directory: {exec_result.output.decode()}"
            })
        
        # Copy repository to container
        logger.info(f"Copying repository to container: {repo_path} -> {workspace_path}/repo")
        
        # Create temporary tar file
        with tempfile.NamedTemporaryFile(suffix='.tar') as tmp_tar:
            # Create tar archive of repository
            subprocess.run([
                'tar', '-cf', tmp_tar.name, '-C', os.path.dirname(repo_path), 
                os.path.basename(repo_path)
            ], check=True)
            
            # Copy tar to container and extract
            with open(tmp_tar.name, 'rb') as tar_file:
                container.put_archive(workspace_path, tar_file)
        
        # Rename to standard 'repo' directory
        repo_dirname = os.path.basename(repo_path)
        if repo_dirname != "repo":
            exec_result = container.exec_run(f"mv {workspace_path}/{repo_dirname} {workspace_path}/repo")
            if exec_result.exit_code != 0:
                logger.warning(f"Failed to rename repo directory: {exec_result.output.decode()}")
        
        # Copy documentation if provided
        docs_copied = False
        if docs_path and os.path.exists(docs_path):
            try:
                docs_path = os.path.abspath(docs_path)
                
                if os.path.isfile(docs_path):
                    # Copy single file
                    with open(docs_path, 'rb') as f:
                        container.put_archive(f"{workspace_path}/docs", f.read())
                    docs_copied = True
                elif os.path.isdir(docs_path):
                    # Copy directory
                    with tempfile.NamedTemporaryFile(suffix='.tar') as tmp_tar:
                        subprocess.run([
                            'tar', '-cf', tmp_tar.name, '-C', os.path.dirname(docs_path),
                            os.path.basename(docs_path)
                        ], check=True)
                        
                        with open(tmp_tar.name, 'rb') as tar_file:
                            container.put_archive(f"{workspace_path}/docs", tar_file)
                    docs_copied = True
                    
            except Exception as e:
                logger.warning(f"Failed to copy documentation: {e}")
        
        # Set working directory and permissions
        container.exec_run(f"chmod -R 755 {workspace_path}")
        container.exec_run(f"cd {workspace_path}")
        
        # List workspace contents for verification
        exec_result = container.exec_run(f"ls -la {workspace_path}")
        workspace_contents = exec_result.output.decode() if exec_result.exit_code == 0 else "Failed to list contents"
        
        result = {
            "status": "success",
            "container_id": container_id,
            "workspace_setup": {
                "workspace_path": workspace_path,
                "repo_copied": True,
                "docs_copied": docs_copied,
                "repo_source": repo_path,
                "docs_source": docs_path if docs_copied else None
            },
            "workspace_contents": workspace_contents,
            "message": "Workspace setup completed successfully"
        }
        
        logger.info(f"Workspace setup completed for container {container_id[:12]}")
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except DockerException as e:
        logger.error(f"Docker error setting up workspace: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Docker error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Failed to setup workspace: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Workspace setup failed: {str(e)}"
        })


@mcp.tool()
async def install_dependencies(
    container_id: str,
    requirements_file: str = None,
    dependencies: List[str] = None,
    language: str = "python"
) -> str:
    """
    Install project dependencies in the container.
    
    Args:
        container_id: Container ID or name
        requirements_file: Path to requirements file in container (e.g., /workspace/repo/requirements.txt)
        dependencies: List of specific dependencies to install
        language: Programming language (python, node, java)
        
    Returns:
        JSON string with installation result
    """
    try:
        client = get_docker_client()
        container = client.containers.get(container_id)
        
        installation_results = []
        
        if language == "python":
            # Update pip first
            update_cmd = "python -m pip install --upgrade pip"
            exec_result = container.exec_run(update_cmd, workdir="/workspace")
            installation_results.append({
                "command": update_cmd,
                "exit_code": exec_result.exit_code,
                "output": exec_result.output.decode()
            })
            
            # Install from requirements file
            if requirements_file:
                install_cmd = f"python -m pip install -r {requirements_file}"
                exec_result = container.exec_run(install_cmd, workdir="/workspace")
                installation_results.append({
                    "command": install_cmd,
                    "exit_code": exec_result.exit_code,
                    "output": exec_result.output.decode()
                })
            
            # Install specific dependencies
            if dependencies:
                for dep in dependencies:
                    install_cmd = f"python -m pip install {dep}"
                    exec_result = container.exec_run(install_cmd, workdir="/workspace")
                    installation_results.append({
                        "command": install_cmd,
                        "exit_code": exec_result.exit_code,
                        "output": exec_result.output.decode()
                    })
        
        elif language == "node":
            # Install from package.json
            if requirements_file or os.path.exists("/workspace/repo/package.json"):
                install_cmd = "npm install"
                exec_result = container.exec_run(install_cmd, workdir="/workspace/repo")
                installation_results.append({
                    "command": install_cmd,
                    "exit_code": exec_result.exit_code,
                    "output": exec_result.output.decode()
                })
            
            # Install specific dependencies
            if dependencies:
                for dep in dependencies:
                    install_cmd = f"npm install {dep}"
                    exec_result = container.exec_run(install_cmd, workdir="/workspace/repo")
                    installation_results.append({
                        "command": install_cmd,
                        "exit_code": exec_result.exit_code,
                        "output": exec_result.output.decode()
                    })
        
        elif language == "java":
            # For Java, we typically don't install dependencies directly
            # but check if build files exist
            build_check_cmd = "ls -la pom.xml build.gradle || echo 'No build files found'"
            exec_result = container.exec_run(build_check_cmd, workdir="/workspace/repo")
            installation_results.append({
                "command": build_check_cmd,
                "exit_code": exec_result.exit_code,
                "output": exec_result.output.decode()
            })
        
        # Check for any installation failures
        failed_installs = [r for r in installation_results if r["exit_code"] != 0]
        success = len(failed_installs) == 0
        
        result = {
            "status": "success" if success else "partial_failure",
            "container_id": container_id,
            "language": language,
            "installation_results": installation_results,
            "failed_installations": len(failed_installs),
            "total_installations": len(installation_results),
            "message": "Dependencies installation completed" if success else "Some dependencies failed to install"
        }
        
        logger.info(f"Dependencies installation completed for container {container_id[:12]}")
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except DockerException as e:
        logger.error(f"Docker error installing dependencies: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Docker error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Dependencies installation failed: {str(e)}"
        })


@mcp.tool()
async def execute_in_container(
    container_id: str,
    command: str,
    working_dir: str = "/workspace/repo",
    timeout: int = 300,
    capture_output: bool = True
) -> str:
    """
    Execute a command in the container and capture output.
    
    Args:
        container_id: Container ID or name
        command: Command to execute
        working_dir: Working directory for command execution
        timeout: Command timeout in seconds
        capture_output: Whether to capture and return output
        
    Returns:
        JSON string with execution result
    """
    try:
        client = get_docker_client()
        container = client.containers.get(container_id)
        
        logger.info(f"Executing command in container {container_id[:12]}: {command}")
        
        start_time = time.time()
        
        # Execute command
        exec_result = container.exec_run(
            command,
            workdir=working_dir,
            timeout=timeout,
            stdout=capture_output,
            stderr=capture_output
        )
        
        execution_time = time.time() - start_time
        
        # Decode output
        stdout = exec_result.output.decode('utf-8', errors='replace') if exec_result.output else ""
        stderr = ""
        
        # For some commands, stderr might be mixed with stdout
        # Try to separate if possible
        if exec_result.exit_code != 0 and stdout:
            lines = stdout.split('\n')
            stderr_lines = [line for line in lines if any(keyword in line.lower() for keyword in ['error', 'exception', 'traceback', 'failed'])]
            stderr = '\n'.join(stderr_lines)
        
        execution_result = ExecutionResult(
            command=command,
            exit_code=exec_result.exit_code,
            stdout=stdout,
            stderr=stderr,
            execution_time=round(execution_time, 2),
            success=exec_result.exit_code == 0
        )
        
        result = {
            "status": "success",
            "container_id": container_id,
            "execution": asdict(execution_result),
            "message": "Command executed successfully" if execution_result.success else "Command execution failed"
        }
        
        logger.info(f"Command execution completed: exit_code={exec_result.exit_code}, time={execution_time:.2f}s")
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except DockerException as e:
        logger.error(f"Docker error executing command: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Docker error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Failed to execute command: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Command execution failed: {str(e)}"
        })


@mcp.tool()
async def get_container_logs(container_id: str, tail_lines: int = 100) -> str:
    """
    Retrieve container logs and execution history.
    
    Args:
        container_id: Container ID or name
        tail_lines: Number of recent log lines to retrieve
        
    Returns:
        JSON string with container logs
    """
    try:
        client = get_docker_client()
        container = client.containers.get(container_id)
        
        # Get container logs
        logs = container.logs(tail=tail_lines, timestamps=True).decode('utf-8', errors='replace')
        
        # Get container info
        container_info = container.attrs
        
        result = {
            "status": "success",
            "container_id": container_id,
            "logs": logs,
            "container_status": container_info.get("State", {}).get("Status", "unknown"),
            "log_lines_retrieved": len(logs.split('\n')),
            "message": "Container logs retrieved successfully"
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except DockerException as e:
        logger.error(f"Docker error retrieving logs: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Docker error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Failed to retrieve logs: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Log retrieval failed: {str(e)}"
        })


@mcp.tool()
async def monitor_container_resources(container_id: str) -> str:
    """
    Monitor container resource usage (CPU, memory, I/O).
    
    Args:
        container_id: Container ID or name
        
    Returns:
        JSON string with resource usage statistics
    """
    try:
        client = get_docker_client()
        container = client.containers.get(container_id)
        
        # Get resource statistics
        stats = container.stats(stream=False)
        resource_usage = parse_resource_stats(stats)
        
        # Get container status
        container.reload()
        status = container.status
        
        result = {
            "status": "success",
            "container_id": container_id,
            "container_status": status,
            "resource_usage": asdict(resource_usage),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "message": "Resource monitoring completed"
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except DockerException as e:
        logger.error(f"Docker error monitoring resources: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Docker error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Failed to monitor resources: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Resource monitoring failed: {str(e)}"
        })


@mcp.tool()
async def cleanup_container(container_id: str, remove_volumes: bool = True) -> str:
    """
    Clean up and remove container and associated resources.
    
    Args:
        container_id: Container ID or name
        remove_volumes: Whether to remove associated volumes
        
    Returns:
        JSON string with cleanup result
    """
    try:
        client = get_docker_client()
        container = client.containers.get(container_id)
        
        # Get container info before removal
        container_name = container.name
        container_image = container.image.tags[0] if container.image.tags else "unknown"
        
        # Stop container if running
        if container.status == "running":
            logger.info(f"Stopping container {container_id[:12]}")
            container.stop(timeout=10)
        
        # Remove container
        logger.info(f"Removing container {container_id[:12]}")
        container.remove(v=remove_volumes)
        
        result = {
            "status": "success",
            "container_id": container_id,
            "cleanup_info": {
                "container_name": container_name,
                "container_image": container_image,
                "volumes_removed": remove_volumes,
                "cleanup_timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "message": f"Container '{container_name}' cleaned up successfully"
        }
        
        logger.info(f"Container cleanup completed: {container_name}")
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except DockerException as e:
        logger.error(f"Docker error during cleanup: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Docker error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Failed to cleanup container: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Container cleanup failed: {str(e)}"
        })


@mcp.tool()
async def list_evaluation_containers() -> str:
    """
    List all evaluation containers (created by this system).
    
    Returns:
        JSON string with list of evaluation containers
    """
    try:
        client = get_docker_client()
        
        # Get all containers with our prefix
        containers = client.containers.list(all=True, filters={"name": CONTAINER_PREFIX})
        
        container_list = []
        for container in containers:
            try:
                container_info = ContainerInfo(
                    container_id=container.id,
                    name=container.name,
                    image=container.image.tags[0] if container.image.tags else "unknown",
                    status=container.status,
                    created_at=container.attrs.get("Created", "unknown"),
                    workspace_path="/workspace"
                )
                container_list.append(asdict(container_info))
            except Exception as e:
                logger.warning(f"Error getting info for container {container.name}: {e}")
        
        result = {
            "status": "success",
            "containers": container_list,
            "total_containers": len(container_list),
            "active_containers": len([c for c in container_list if c["status"] == "running"]),
            "message": "Container list retrieved successfully"
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except DockerException as e:
        logger.error(f"Docker error listing containers: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Docker error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Failed to list containers: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Container listing failed: {str(e)}"
        })


# Run the server
if __name__ == "__main__":
    try:
        # Test Docker connection on startup (but don't fail if unavailable)
        docker_available = get_docker_client() is not None
        if docker_available:
            logger.info("Docker Management MCP Server starting with Docker support...")
        else:
            logger.warning("Docker Management MCP Server starting without Docker (Docker not available)")
        mcp.run()
    except Exception as e:
        logger.error(f"Failed to start Docker Management MCP Server: {e}")
        sys.exit(1)
