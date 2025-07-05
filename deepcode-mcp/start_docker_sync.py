#!/usr/bin/env python3
"""
DeepCode Docker Sync Starter
启动Docker同步容器的便捷脚本

🚀 Usage:
    python start_docker_sync.py                    # Start interactive container
    python start_docker_sync.py --detached         # Start in background
    python start_docker_sync.py --streamlit        # Start with Streamlit app
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def get_docker_command(detached=False, run_streamlit=False):
    """Generate Docker run command with proper volume mounts"""
    
    current_dir = Path.cwd()
    deepcode_lab_dir = current_dir / "deepcode_lab"
    
    # Ensure deepcode_lab directory exists
    deepcode_lab_dir.mkdir(exist_ok=True)
    print(f"✅ Sync directory ready: {deepcode_lab_dir}")
    
    # Base Docker command
    docker_cmd = ['docker', 'run', '--rm']
    
    if detached:
        docker_cmd.extend(['-d'])
    else:
        docker_cmd.extend(['-it'])
    
    # Volume mounts
    docker_cmd.extend([
        '-v', f'{deepcode_lab_dir.absolute()}:/paper2code/deepcode_lab',
        '-v', f'{current_dir.absolute()}:/paper2code/host_code'
    ])
    
    # Port mapping
    docker_cmd.extend(['-p', '8501:8501'])
    
    # Container name
    docker_cmd.extend(['--name', f'deepcode_sync_{os.getpid()}'])
    
    # Image
    docker_cmd.append('deepcode:latest')
    
    # Command to run
    if run_streamlit:
        docker_cmd.extend(['python', 'paper_to_code.py'])
    else:
        docker_cmd.append('bash')
    
    return docker_cmd

def check_docker_image():
    """Check if deepcode image exists"""
    try:
        result = subprocess.run(
            ['docker', 'images', '-q', 'deepcode:latest'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return bool(result.stdout.strip())
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description='Start DeepCode Docker container with sync')
    parser.add_argument('--detached', '-d', action='store_true', 
                       help='Run container in background')
    parser.add_argument('--streamlit', '-s', action='store_true',
                       help='Start Streamlit app automatically')
    
    args = parser.parse_args()
    
    print("🔍 DeepCode Docker Sync Starter")
    print("=" * 50)
    
    # Check if Docker image exists
    if not check_docker_image():
        print("❌ DeepCode Docker image not found!")
        print("Please build the image first:")
        print("   docker build -f deepcode.Dockerfile -t deepcode:latest .")
        sys.exit(1)
    
    # Generate Docker command
    docker_cmd = get_docker_command(
        detached=args.detached,
        run_streamlit=args.streamlit
    )
    
    print(f"🚀 Starting Docker container...")
    print(f"Command: {' '.join(docker_cmd)}")
    print()
    
    if args.detached:
        print("🔄 Container will run in background")
        print("📁 Sync active between:")
        print(f"   Local:  ./deepcode_lab")
        print(f"   Docker: /paper2code/deepcode_lab")
        print()
        print("💡 To connect to the container:")
        container_name = f'deepcode_sync_{os.getpid()}'
        print(f"   docker exec -it {container_name} bash")
    else:
        print("🔄 Real-time sync active:")
        print(f"   Local:  ./deepcode_lab ↔ Docker: /paper2code/deepcode_lab")
        print("💡 Any file changes will sync immediately!")
        print("🛑 Press Ctrl+C to stop container")
    
    print("=" * 50)
    
    try:
        # Execute Docker command
        if args.detached:
            result = subprocess.run(docker_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                container_id = result.stdout.strip()
                print(f"✅ Container started: {container_id[:12]}")
                if args.streamlit:
                    print("🌐 Streamlit app available at: http://localhost:8501")
            else:
                print(f"❌ Failed to start container: {result.stderr}")
                sys.exit(1)
        else:
            subprocess.run(docker_cmd)
    except KeyboardInterrupt:
        print("\n🛑 Container stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 