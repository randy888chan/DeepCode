#!/usr/bin/env python3
"""
Docker Sync Manager - Universal Docker Volume Mount Utility
Docker同步管理器 - 通用Docker卷挂载工具

🚀 Features:
- Automatic Docker environment detection
- Smart directory synchronization setup
- Real-time bidirectional file sync
- Easy integration with any workflow

📁 Usage:
    from utils.docker_sync_manager import DockerSyncManager
    
    sync_manager = DockerSyncManager()
    sync_info = await sync_manager.setup_sync()
    print(f"Sync directory: {sync_info['sync_directory']}")
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Union
import logging
import json
import platform

class DockerSyncManager:
    """
    Universal Docker synchronization manager for seamless local-container file sync.
    通用Docker同步管理器，实现本地-容器文件无缝同步
    """
    
    def __init__(
        self, 
        local_sync_dir: str = "deepcode_lab",
        docker_sync_dir: str = "/paper2code/deepcode_lab",
        docker_image: str = "deepcode:latest",
        container_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Docker Sync Manager
        
        Args:
            local_sync_dir: Local directory name for synchronization
            docker_sync_dir: Docker container directory path for mounting
            docker_image: Docker image name to use
            container_name: Optional container name (auto-generated if None)
            logger: Optional logger instance
        """
        self.local_sync_dir = local_sync_dir
        self.docker_sync_dir = docker_sync_dir
        self.docker_image = docker_image
        self.container_name = container_name or f"deepcode_sync_{os.getpid()}"
        
        # Setup logger
        self.logger = logger or self._setup_default_logger()
        
        # Runtime environment detection
        self.is_docker = self._detect_docker_environment()
        self.host_platform = platform.system().lower()
        
        # Path management
        self.current_dir = Path.cwd()
        self.local_sync_path = self.current_dir / self.local_sync_dir
        
    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logger for the sync manager"""
        logger = logging.getLogger('DockerSyncManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _detect_docker_environment(self) -> bool:
        """
        Detect if currently running inside a Docker container
        检测当前是否在Docker容器内运行
        
        Returns:
            bool: True if running in Docker, False otherwise
        """
        try:
            # Method 1: Check for /.dockerenv file
            if os.path.exists('/.dockerenv'):
                return True
            
            # Method 2: Check cgroup information
            if os.path.exists('/proc/1/cgroup'):
                with open('/proc/1/cgroup', 'r') as f:
                    content = f.read()
                    if 'docker' in content or 'containerd' in content:
                        return True
            
            # Method 3: Check hostname pattern
            hostname = os.uname().nodename
            if len(hostname) == 12 and all(c.isalnum() for c in hostname):
                return True
                
            return False
            
        except Exception as e:
            self.logger.warning(f"Could not detect Docker environment: {e}")
            return False
    
    def _check_docker_availability(self) -> bool:
        """
        Check if Docker is available on the system
        检查系统是否安装了Docker
        
        Returns:
            bool: True if Docker is available, False otherwise
        """
        try:
            result = subprocess.run(
                ['docker', '--version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def _check_docker_image_exists(self) -> bool:
        """
        Check if the specified Docker image exists locally
        检查指定的Docker镜像是否存在
        
        Returns:
            bool: True if image exists, False otherwise
        """
        try:
            result = subprocess.run(
                ['docker', 'images', '-q', self.docker_image],
                capture_output=True,
                text=True,
                timeout=10
            )
            return bool(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False
    
    def ensure_local_sync_directory(self) -> Path:
        """
        Ensure the local sync directory exists
        确保本地同步目录存在
        
        Returns:
            Path: The local sync directory path
        """
        try:
            self.local_sync_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"✅ Local sync directory ready: {self.local_sync_path}")
            return self.local_sync_path
        except Exception as e:
            self.logger.error(f"❌ Failed to create local sync directory: {e}")
            raise
    
    def ensure_docker_sync_directory(self) -> bool:
        """
        Ensure the Docker sync directory exists (when running in Docker)
        确保Docker同步目录存在（在Docker内运行时）
        
        Returns:
            bool: True if directory is ready, False otherwise
        """
        try:
            docker_path = Path(self.docker_sync_dir)
            docker_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"✅ Docker sync directory ready: {docker_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create Docker sync directory: {e}")
            return False
    
    def get_docker_run_command(
        self, 
        additional_volumes: Optional[List[str]] = None,
        additional_ports: Optional[List[str]] = None,
        command: Optional[str] = None
    ) -> List[str]:
        """
        Generate Docker run command with proper volume mounts
        生成带有正确卷挂载的Docker运行命令
        
        Args:
            additional_volumes: Additional volume mounts (format: "host:container")
            additional_ports: Additional port mappings (format: "host:container")
            command: Command to run in container (default: bash)
            
        Returns:
            List[str]: Docker command as list of strings
        """
        # Ensure local directory exists
        self.ensure_local_sync_directory()
        
        # Base Docker command
        docker_cmd = [
            'docker', 'run',
            '--rm',  # Remove container when it exits
            '-it',   # Interactive with TTY
        ]
        
        # Add main volume mount
        docker_cmd.extend([
            '-v', f'{self.local_sync_path.absolute()}:{self.docker_sync_dir}'
        ])
        
        # Add host code mount for development
        docker_cmd.extend([
            '-v', f'{self.current_dir.absolute()}:/paper2code/host_code'
        ])
        
        # Add additional volumes
        if additional_volumes:
            for volume in additional_volumes:
                docker_cmd.extend(['-v', volume])
        
        # Add default port mapping for Streamlit
        docker_cmd.extend(['-p', '8501:8501'])
        
        # Add additional ports
        if additional_ports:
            for port in additional_ports:
                docker_cmd.extend(['-p', port])
        
        # Add container name
        docker_cmd.extend(['--name', self.container_name])
        
        # Add image
        docker_cmd.append(self.docker_image)
        
        # Add command
        if command:
            if isinstance(command, str):
                docker_cmd.extend(command.split())
            else:
                docker_cmd.extend(command)
        else:
            docker_cmd.append('bash')
        
        return docker_cmd
    
    def start_docker_container(
        self, 
        additional_volumes: Optional[List[str]] = None,
        additional_ports: Optional[List[str]] = None,
        command: Optional[str] = None,
        detached: bool = False
    ) -> Dict[str, Union[str, bool, subprocess.Popen]]:
        """
        Start Docker container with volume mounts
        启动带有卷挂载的Docker容器
        
        Args:
            additional_volumes: Additional volume mounts
            additional_ports: Additional port mappings
            command: Command to run in container
            detached: Run container in background
            
        Returns:
            Dict: Container start result with status and process info
        """
        # Pre-flight checks
        if not self._check_docker_availability():
            raise RuntimeError("❌ Docker is not available. Please install Docker first.")
        
        if not self._check_docker_image_exists():
            raise RuntimeError(f"❌ Docker image '{self.docker_image}' not found. Please build the image first.")
        
        # Generate Docker command
        docker_cmd = self.get_docker_run_command(
            additional_volumes=additional_volumes,
            additional_ports=additional_ports,
            command=command
        )
        
        if detached:
            # Remove -it flags for detached mode
            docker_cmd = [arg for arg in docker_cmd if arg not in ['-it', '-i', '-t']]
            docker_cmd.insert(docker_cmd.index('--rm') + 1, '-d')
        
        self.logger.info(f"🚀 Starting Docker container with command:")
        self.logger.info(f"   {' '.join(docker_cmd)}")
        
        try:
            if detached:
                # Start detached container
                result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    container_id = result.stdout.strip()
                    self.logger.info(f"✅ Container started successfully: {container_id[:12]}")
                    return {
                        'status': 'success',
                        'container_id': container_id,
                        'detached': True,
                        'sync_directory': str(self.local_sync_path)
                    }
                else:
                    raise RuntimeError(f"Failed to start container: {result.stderr}")
            else:
                # Start interactive container
                self.logger.info("💡 Container will start in interactive mode")
                self.logger.info("🔄 Directory sync is now active:")
                self.logger.info(f"   Local: {self.local_sync_path}")
                self.logger.info(f"   Docker: {self.docker_sync_dir}")
                
                # Execute Docker command
                process = subprocess.Popen(docker_cmd)
                return {
                    'status': 'success',
                    'process': process,
                    'detached': False,
                    'sync_directory': str(self.local_sync_path)
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start Docker container: {e}")
            raise
    
    async def setup_sync(self, auto_start_docker: bool = False) -> Dict[str, Union[str, bool]]:
        """
        Main method to setup synchronization based on current environment
        根据当前环境设置同步的主要方法
        
        Args:
            auto_start_docker: Automatically start Docker if not in container
            
        Returns:
            Dict: Sync setup result with environment info and sync directory
        """
        self.logger.info("🔍 Setting up Docker synchronization...")
        
        if self.is_docker:
            # Running inside Docker container
            self.logger.info("🐳 Detected Docker environment")
            success = self.ensure_docker_sync_directory()
            return {
                'environment': 'docker',
                'sync_active': success,
                'sync_directory': self.docker_sync_dir,
                'message': 'Running in Docker container - sync directory ready'
            }
        else:
            # Running on local machine
            self.logger.info("💻 Detected local environment")
            self.ensure_local_sync_directory()
            
            if auto_start_docker:
                self.logger.info("🚀 Auto-starting Docker container for sync...")
                try:
                    result = self.start_docker_container(detached=True)
                    return {
                        'environment': 'local',
                        'sync_active': True,
                        'sync_directory': str(self.local_sync_path),
                        'container_id': result.get('container_id'),
                        'message': 'Docker container started with volume sync'
                    }
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to auto-start Docker: {e}")
                    return {
                        'environment': 'local',
                        'sync_active': False,
                        'sync_directory': str(self.local_sync_path),
                        'message': f'Local directory ready, Docker auto-start failed: {e}'
                    }
            else:
                return {
                    'environment': 'local',
                    'sync_active': False,
                    'sync_directory': str(self.local_sync_path),
                    'message': 'Local directory ready - use start_docker_container() for sync'
                }
    
    def get_sync_status(self) -> Dict[str, Union[str, bool]]:
        """
        Get current synchronization status
        获取当前同步状态
        
        Returns:
            Dict: Current sync status information
        """
        return {
            'is_docker_environment': self.is_docker,
            'local_sync_directory': str(self.local_sync_path),
            'docker_sync_directory': self.docker_sync_dir,
            'local_directory_exists': self.local_sync_path.exists(),
            'docker_available': self._check_docker_availability(),
            'docker_image_exists': self._check_docker_image_exists()
        }
    
    def print_usage_instructions(self):
        """Print helpful usage instructions for users"""
        print("\n" + "="*70)
        print("🔄 Docker Sync Manager - Usage Instructions")
        print("="*70)
        
        if self.is_docker:
            print("📁 Currently running in Docker container")
            print(f"   Sync directory: {self.docker_sync_dir}")
            print("💡 Any files created here will sync to your local machine")
        else:
            print("💻 Currently running on local machine")
            print(f"   Local sync directory: {self.local_sync_path}")
            print("\n🚀 To start Docker with sync:")
            docker_cmd = self.get_docker_run_command()
            print(f"   {' '.join(docker_cmd)}")
            print("\n📝 Or use the sync manager:")
            print("   sync_manager = DockerSyncManager()")
            print("   sync_manager.start_docker_container()")
        
        print("\n🔄 Real-time Synchronization:")
        print("   ✅ Local edits → Docker container (instant)")
        print("   ✅ Docker changes → Local files (instant)")
        print("   ✅ Works with any file operations")
        print("="*70)

    def get_sync_directory(self) -> str:
        """
        Get the appropriate sync directory based on current environment
        根据当前环境获取合适的同步目录
        
        Returns:
            str: The sync directory path to use
        """
        if self.is_docker:
            self.ensure_docker_sync_directory()
            return self.docker_sync_dir
        else:
            self.ensure_local_sync_directory()
            return str(self.local_sync_path)

# Convenience functions for easy integration
# 便捷函数，方便集成

async def setup_docker_sync(
    local_dir: str = "deepcode_lab",
    docker_dir: str = "/paper2code/deepcode_lab",
    auto_start: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Union[str, bool]]:
    """
    Convenience function to quickly setup Docker synchronization
    便捷函数，快速设置Docker同步
    
    Args:
        local_dir: Local directory name
        docker_dir: Docker directory path
        auto_start: Auto-start Docker if on local machine
        logger: Optional logger
        
    Returns:
        Dict: Sync setup result
    """
    sync_manager = DockerSyncManager(
        local_sync_dir=local_dir,
        docker_sync_dir=docker_dir,
        logger=logger
    )
    
    return await sync_manager.setup_sync(auto_start_docker=auto_start)

def get_sync_directory(
    local_dir: str = "deepcode_lab",
    docker_dir: str = "/paper2code/deepcode_lab"
) -> str:
    """
    Get the appropriate sync directory based on current environment
    根据当前环境获取合适的同步目录
    
    Args:
        local_dir: Local directory name
        docker_dir: Docker directory path
        
    Returns:
        str: The sync directory path to use
    """
    sync_manager = DockerSyncManager(
        local_sync_dir=local_dir,
        docker_sync_dir=docker_dir
    )
    
    return sync_manager.get_sync_directory()

# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Create sync manager
        sync_manager = DockerSyncManager()
        
        # Show current status
        status = sync_manager.get_sync_status()
        print("📊 Current Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # Setup sync
        result = await sync_manager.setup_sync()
        print("\n🔄 Sync Setup Result:")
        for key, value in result.items():
            print(f"   {key}: {value}")
        
        # Show usage instructions
        sync_manager.print_usage_instructions()
    
    # Run example
    asyncio.run(main()) 