#!/usr/bin/env python3
"""
PyInstaller Build Configuration for Snake Game
==============================================

This script provides automated build configuration for creating executable
distributions of the Snake Game using PyInstaller. It supports multiple
platforms and build options.

Usage:
    python build_config.py [options]

Options:
    --debug         Create debug build with console window
    --onefile       Create single executable file (slower startup)
    --windowed      Create windowed application (no console)
    --clean         Clean build directories before building
    --icon          Specify custom icon file path
    --name          Specify custom executable name

Examples:
    python build_config.py --windowed --clean
    python build_config.py --onefile --icon assets/images/snake_icon.ico
    python build_config.py --debug
"""

import os
import sys
import shutil
import argparse
import platform
from pathlib import Path
import subprocess
import json

# Build configuration constants
BUILD_DIR = "build"
DIST_DIR = "dist"
SPEC_DIR = "specs"
APP_NAME = "SnakeGame"
MAIN_SCRIPT = "main.py"

# Platform-specific settings
PLATFORM = platform.system().lower()
IS_WINDOWS = PLATFORM == "windows"
IS_MACOS = PLATFORM == "darwin"
IS_LINUX = PLATFORM == "linux"

# Default build options
DEFAULT_BUILD_OPTIONS = {
    "debug": False,
    "onefile": False,
    "windowed": True,
    "clean": False,
    "icon": None,
    "name": APP_NAME,
    "add_data": [],
    "hidden_imports": [],
    "exclude_modules": [],
    "optimize": True,
    "strip": False,
    "upx": False,
    "console": False
}

class SnakeGameBuilder:
    """
    PyInstaller build configuration and execution manager for Snake Game.
    """
    
    def __init__(self, options=None):
        """
        Initialize the builder with configuration options.
        
        Args:
            options (dict): Build configuration options
        """
        self.options = {**DEFAULT_BUILD_OPTIONS, **(options or {})}
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / BUILD_DIR
        self.dist_dir = self.project_root / DIST_DIR
        self.spec_dir = self.project_root / SPEC_DIR
        
        # Ensure directories exist
        self.spec_dir.mkdir(exist_ok=True)
        
    def get_data_files(self):
        """
        Collect data files that need to be included in the build.
        
        Returns:
            list: List of (source, destination) tuples for PyInstaller
        """
        data_files = []
        
        # Assets directory
        assets_dir = self.project_root / "assets"
        if assets_dir.exists():
            data_files.append((str(assets_dir), "assets"))
        
        # Data directory (config and scores)
        data_dir = self.project_root / "data"
        if data_dir.exists():
            data_files.append((str(data_dir), "data"))
        
        # Individual important files
        important_files = [
            "requirements.txt",
            "README.md"
        ]
        
        for file_name in important_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                data_files.append((str(file_path), "."))
        
        return data_files
    
    def get_hidden_imports(self):
        """
        Get list of hidden imports that PyInstaller might miss.
        
        Returns:
            list: List of module names to include
        """
        hidden_imports = [
            "pygame",
            "pygame.mixer",
            "pygame.font",
            "pygame.image",
            "pygame.transform",
            "numpy",
            "json5",
            "pathlib",
            "threading",
            "queue",
            "collections",
            "itertools",
            "functools",
            "operator"
        ]
        
        # Add user-specified hidden imports
        hidden_imports.extend(self.options.get("hidden_imports", []))
        
        return hidden_imports
    
    def get_exclude_modules(self):
        """
        Get list of modules to exclude from the build.
        
        Returns:
            list: List of module names to exclude
        """
        exclude_modules = [
            "tkinter",
            "matplotlib",
            "scipy",
            "pandas",
            "IPython",
            "jupyter",
            "notebook",
            "pytest",
            "unittest",
            "doctest",
            "pdb",
            "profile",
            "cProfile",
            "pstats"
        ]
        
        # Add user-specified exclusions
        exclude_modules.extend(self.options.get("exclude_modules", []))
        
        return exclude_modules
    
    def get_icon_path(self):
        """
        Get the icon file path for the executable.
        
        Returns:
            str or None: Path to icon file
        """
        if self.options.get("icon"):
            return str(Path(self.options["icon"]).resolve())
        
        # Look for default icons
        icon_paths = [
            self.project_root / "assets" / "images" / "snake_icon.ico",
            self.project_root / "assets" / "images" / "snake_icon.png",
            self.project_root / "assets" / "snake_icon.ico",
            self.project_root / "snake_icon.ico"
        ]
        
        for icon_path in icon_paths:
            if icon_path.exists():
                return str(icon_path)
        
        return None
    
    def clean_build_dirs(self):
        """
        Clean build and distribution directories.
        """
        print("🧹 Cleaning build directories...")
        
        dirs_to_clean = [self.build_dir, self.dist_dir]
        
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    print(f"   ✅ Cleaned {dir_path}")
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not clean {dir_path}: {e}")
    
    def generate_spec_file(self):
        """
        Generate PyInstaller spec file with current configuration.
        
        Returns:
            Path: Path to generated spec file
        """
        spec_content = self._create_spec_content()
        spec_file = self.spec_dir / f"{self.options['name']}.spec"
        
        with open(spec_file, 'w', encoding='utf-8') as f:
            f.write(spec_content)
        
        print(f"📝 Generated spec file: {spec_file}")
        return spec_file
    
    def _create_spec_content(self):
        """
        Create the content for the PyInstaller spec file.
        
        Returns:
            str: Spec file content
        """
        data_files = self.get_data_files()
        hidden_imports = self.get_hidden_imports()
        exclude_modules = self.get_exclude_modules()
        icon_path = self.get_icon_path()
        
        # Format data files for spec
        datas_str = "[\n"
        for src, dst in data_files:
            datas_str += f"    ('{src}', '{dst}'),\n"
        datas_str += "]"
        
        # Format hidden imports
        hiddenimports_str = str(hidden_imports)
        
        # Format excludes
        excludes_str = str(exclude_modules)
        
        # Icon parameter
        icon_str = f"'{icon_path}'" if icon_path else "None"
        
        spec_template = f"""# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Snake Game
# Generated automatically by build_config.py

import sys
from pathlib import Path

# Build configuration
block_cipher = None
app_name = '{self.options["name"]}'
debug = {self.options["debug"]}
onefile = {self.options["onefile"]}
windowed = {not self.options["console"] and self.options["windowed"]}

a = Analysis(
    ['{MAIN_SCRIPT}'],
    pathex=[],
    binaries=[],
    datas={datas_str},
    hiddenimports={hiddenimports_str},
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={excludes_str},
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if onefile:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name=app_name,
        debug=debug,
        bootloader_ignore_signals=False,
        strip={self.options["strip"]},
        upx={self.options["upx"]},
        upx_exclude=[],
        runtime_tmpdir=None,
        console=not windowed,
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon={icon_str},
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=app_name,
        debug=debug,
        bootloader_ignore_signals=False,
        strip={self.options["strip"]},
        upx={self.options["upx"]},
        console=not windowed,
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon={icon_str},
    )
    
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip={self.options["strip"]},
        upx={self.options["upx"]},
        upx_exclude=[],
        name=app_name,
    )

# Platform-specific configurations
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name=app_name + '.app',
        icon={icon_str},
        bundle_identifier='com.snakegame.app',
        info_plist={{
            'CFBundleName': app_name,
            'CFBundleDisplayName': 'Snake Game',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
        }},
    )
"""
        
        return spec_template
    
    def build(self):
        """
        Execute the PyInstaller build process.
        
        Returns:
            bool: True if build successful, False otherwise
        """
        try:
            # Clean if requested
            if self.options["clean"]:
                self.clean_build_dirs()
            
            # Generate spec file
            spec_file = self.generate_spec_file()
            
            # Build command
            cmd = [
                sys.executable, "-m", "PyInstaller",
                str(spec_file),
                "--distpath", str(self.dist_dir),
                "--workpath", str(self.build_dir)
            ]
            
            if self.options["debug"]:
                cmd.append("--debug=all")
            
            print(f"🔨 Building {self.options['name']}...")
            print(f"   Command: {' '.join(cmd)}")
            
            # Execute build
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Build completed successfully!")
                self._print_build_info()
                return True
            else:
                print("❌ Build failed!")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Build error: {e}")
            return False
    
    def _print_build_info(self):
        """
        Print information about the completed build.
        """
        print("\n📦 Build Information:")
        print(f"   App Name: {self.options['name']}")
        print(f"   Platform: {PLATFORM}")
        print(f"   Build Type: {'One File' if self.options['onefile'] else 'Directory'}")
        print(f"   Windowed: {self.options['windowed']}")
        print(f"   Debug: {self.options['debug']}")
        
        # Find output files
        output_dir = self.dist_dir / self.options['name']
        if output_dir.exists():
            print(f"   Output Directory: {output_dir}")
            
            # List main executable
            if IS_WINDOWS:
                exe_name = f"{self.options['name']}.exe"
            else:
                exe_name = self.options['name']
            
            exe_path = output_dir / exe_name
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / (1024 * 1024)
                print(f"   Executable: {exe_path} ({size_mb:.1f} MB)")
        
        print(f"\n🚀 To run the built application:")
        if IS_WINDOWS:
            print(f"   {output_dir / self.options['name']}.exe")
        else:
            print(f"   {output_dir / self.options['name']}")

def create_build_script():
    """
    Create a simple build script for common build scenarios.
    """
    script_content = '''#!/usr/bin/env python3
"""
Quick build script for Snake Game
Usage: python quick_build.py [release|debug|portable]
"""

import sys
from build_config import SnakeGameBuilder

def main():
    build_type = sys.argv[1] if len(sys.argv) > 1 else "release"
    
    if build_type == "release":
        options = {
            "windowed": True,
            "clean": True,
            "optimize": True,
            "strip": True
        }
    elif build_type == "debug":
        options = {
            "debug": True,
            "console": True,
            "clean": True
        }
    elif build_type == "portable":
        options = {
            "onefile": True,
            "windowed": True,
            "clean": True,
            "optimize": True
        }
    else:
        print("Usage: python quick_build.py [release|debug|portable]")
        return 1
    
    builder = SnakeGameBuilder(options)
    success = builder.build()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open("quick_build.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("📝 Created quick_build.py script")

def main():
    """
    Main entry point for the build configuration script.
    """
    parser = argparse.ArgumentParser(
        description="PyInstaller build configuration for Snake Game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--debug", action="store_true",
                       help="Create debug build with console window")
    parser.add_argument("--onefile", action="store_true",
                       help="Create single executable file")
    parser.add_argument("--windowed", action="store_true", default=True,
                       help="Create windowed application (default)")
    parser.add_argument("--console", action="store_true",
                       help="Show console window")
    parser.add_argument("--clean", action="store_true",
                       help="Clean build directories before building")
    parser.add_argument("--icon", type=str,
                       help="Specify custom icon file path")
    parser.add_argument("--name", type=str, default=APP_NAME,
                       help="Specify custom executable name")
    parser.add_argument("--create-script", action="store_true",
                       help="Create quick build script and exit")
    
    args = parser.parse_args()
    
    if args.create_script:
        create_build_script()
        return 0
    
    # Convert args to options dict
    options = {
        "debug": args.debug,
        "onefile": args.onefile,
        "windowed": args.windowed and not args.console,
        "console": args.console,
        "clean": args.clean,
        "icon": args.icon,
        "name": args.name
    }
    
    print("🐍 Snake Game Build Configuration")
    print("=" * 40)
    
    # Check if PyInstaller is available
    try:
        import PyInstaller
        print(f"✅ PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("❌ PyInstaller not found. Install with: pip install PyInstaller>=5.0")
        return 1
    
    # Create and run builder
    builder = SnakeGameBuilder(options)
    success = builder.build()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())