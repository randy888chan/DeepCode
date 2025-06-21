#!/usr/bin/env python3
"""
Code Indexer Runner Script

This script runs the code indexer with configuration file support.
It processes all repositories in the code_base directory and generates
relationship indexes for the target project structure.

使用方法:
    python run_indexer.py [--config CONFIG_FILE] [--target-structure STRUCTURE_FILE]

Examples:
    python run_indexer.py
    python run_indexer.py --config custom_config.yaml
    python run_indexer.py --target-structure my_structure.txt
"""

import argparse
import asyncio
import sys
from pathlib import Path
import yaml
from typing import Dict, Any

# Add tools directory to path
sys.path.append(str(Path(__file__).parent / "tools"))

from tools.code_indexer import CodeIndexer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        sys.exit(1)


def load_target_structure(structure_path: str) -> str:
    """Load target structure from file"""
    try:
        with open(structure_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error loading target structure file {structure_path}: {e}")
        sys.exit(1)


def extract_file_tree_from_plan(plan_content: str) -> str:
    """
    Extract file tree structure from initial_plan.txt content
    从initial_plan.txt内容中提取文件树结构

    Args:
        plan_content: Content of the initial_plan.txt file

    Returns:
        Extracted file tree structure as string
    """
    import re

    # Look for file structure section specifically in the format we see in initial_plan.txt
    # This matches the exact format: "## File Structure (≤30 files total)" followed by code block
    file_structure_pattern = r"## File Structure[^\n]*\n```[^\n]*\n(.*?)\n```"

    match = re.search(file_structure_pattern, plan_content, re.DOTALL)
    if match:
        file_tree = match.group(1).strip()
        lines = file_tree.split("\n")

        # Clean up the tree - remove empty lines and comments that aren't part of structure
        cleaned_lines = []
        for line in lines:
            # Keep lines that are part of the tree structure
            if line.strip() and (
                any(char in line for char in ["├──", "└──", "│"])
                or line.strip().endswith("/")
                or "." in line.split("/")[-1]  # has file extension
                or line.strip().endswith(".py")
                or line.strip().endswith(".txt")
                or line.strip().endswith(".md")
                or line.strip().endswith(".yaml")
            ):
                cleaned_lines.append(line)

        if len(cleaned_lines) >= 5:
            file_tree = "\n".join(cleaned_lines)
            print(
                f"📊 Extracted file tree structure from ## File Structure section ({len(cleaned_lines)} lines)"
            )
            return file_tree

    # Fallback: Look for any code block that contains project structure
    # This pattern looks for code blocks with common project names and tree structure
    code_block_patterns = [
        r"```[^\n]*\n(rice_framework/.*?(?:├──|└──).*?)\n```",
        r"```[^\n]*\n(project/.*?(?:├──|└──).*?)\n```",
        r"```[^\n]*\n(src/.*?(?:├──|└──).*?)\n```",
        r"```[^\n]*\n(.*?(?:├──|└──).*?(?:\.py|\.txt|\.md|\.yaml).*?)\n```",
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, plan_content, re.DOTALL)
        if match:
            file_tree = match.group(1).strip()
            lines = [line for line in file_tree.split("\n") if line.strip()]
            if len(lines) >= 5:
                print(
                    f"📊 Extracted file tree structure from code block ({len(lines)} lines)"
                )
                return file_tree

    # Final fallback: Extract file paths mentioned in the plan and create a basic structure
    print("⚠️ No standard file tree found, attempting to extract from file mentions...")

    # Look for file paths in backticks throughout the document
    file_mentions = re.findall(
        r"`([^`]*(?:\.py|\.txt|\.md|\.yaml|\.yml)[^`]*)`", plan_content
    )

    if file_mentions:
        # Organize files into a directory structure
        dirs = set()
        files_by_dir = {}

        for file_path in file_mentions:
            file_path = file_path.strip()
            if "/" in file_path:
                dir_path = "/".join(file_path.split("/")[:-1])
                filename = file_path.split("/")[-1]
                dirs.add(dir_path)
                if dir_path not in files_by_dir:
                    files_by_dir[dir_path] = []
                files_by_dir[dir_path].append(filename)
            else:
                if "root" not in files_by_dir:
                    files_by_dir["root"] = []
                files_by_dir["root"].append(file_path)

        # Create a tree structure
        structure_lines = []

        # Determine root directory name
        root_name = (
            "rice_framework" if any("rice" in f for f in file_mentions) else "project"
        )
        structure_lines.append(f"{root_name}/")

        # Add directories and files
        sorted_dirs = sorted(dirs) if dirs else []
        for i, dir_path in enumerate(sorted_dirs):
            is_last_dir = i == len(sorted_dirs) - 1
            prefix = "└──" if is_last_dir else "├──"
            structure_lines.append(f"{prefix} {dir_path}/")

            if dir_path in files_by_dir:
                files = sorted(files_by_dir[dir_path])
                for j, filename in enumerate(files):
                    is_last_file = j == len(files) - 1
                    if is_last_dir:
                        file_prefix = "    └──" if is_last_file else "    ├──"
                    else:
                        file_prefix = "│   └──" if is_last_file else "│   ├──"
                    structure_lines.append(f"{file_prefix} {filename}")

        # Add root files if any
        if "root" in files_by_dir:
            root_files = sorted(files_by_dir["root"])
            for i, filename in enumerate(root_files):
                is_last = (i == len(root_files) - 1) and not sorted_dirs
                prefix = "└──" if is_last else "├──"
                structure_lines.append(f"{prefix} {filename}")

        if len(structure_lines) >= 3:
            file_tree = "\n".join(structure_lines)
            print(
                f"📊 Generated file tree from file mentions ({len(structure_lines)} lines)"
            )
            return file_tree

    # If no file tree found, return None
    print("⚠️ No file tree structure found in initial plan")
    return None


def load_target_structure_from_plan(plan_path: str) -> str:
    """
    Load target structure from initial_plan.txt and extract file tree
    从initial_plan.txt加载目标结构并提取文件树

    Args:
        plan_path: Path to initial_plan.txt file

    Returns:
        Extracted file tree structure
    """
    try:
        # Load the full plan content
        with open(plan_path, "r", encoding="utf-8") as f:
            plan_content = f.read()

        print(f"📄 Loaded initial plan ({len(plan_content)} characters)")

        # Extract file tree structure
        file_tree = extract_file_tree_from_plan(plan_content)

        if file_tree:
            print("✅ Successfully extracted file tree from initial plan")
            print("📋 Preview of extracted structure:")
            # Show first few lines of the extracted tree
            preview_lines = file_tree.split("\n")[:8]
            for line in preview_lines:
                print(f"   {line}")
            if len(file_tree.split("\n")) > 8:
                print(f"   ... and {len(file_tree.split('\n')) - 8} more lines")
            return file_tree
        else:
            print("⚠️ Could not extract file tree from initial plan")
            print("🔄 Falling back to default target structure")
            return get_default_target_structure()

    except Exception as e:
        print(f"❌ Error loading initial plan file {plan_path}: {e}")
        print("🔄 Falling back to default target structure")
        return get_default_target_structure()


def get_default_target_structure() -> str:
    """Get the default target structure"""
    return """
project/
├── src/
│   ├── core/
│   │   ├── gcn.py        # GCN encoder
│   │   ├── diffusion.py  # forward/reverse processes
│   │   ├── denoiser.py   # denoising MLP
│   │   └── fusion.py     # fusion combiner
│   ├── models/           # model wrapper classes
│   │   └── recdiff.py
│   ├── utils/
│   │   ├── data.py       # loading & preprocessing
│   │   ├── predictor.py  # scoring functions
│   │   ├── loss.py       # loss functions
│   │   ├── metrics.py    # NDCG, Recall etc.
│   │   └── sched.py      # beta/alpha schedule utils
│   └── configs/
│       └── default.yaml  # hyperparameters, paths
├── tests/
│   ├── test_gcn.py
│   ├── test_diffusion.py
│   ├── test_denoiser.py
│   ├── test_loss.py
│   └── test_pipeline.py
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── README.md
├── experiments/
│   ├── run_experiment.py
│   └── notebooks/
│       └── analysis.ipynb
├── requirements.txt
└── setup.py
"""


def print_banner():
    """Print application banner"""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                      🔍 Code Indexer v1.0                            ║
║              Intelligent Code Relationship Analysis Tool              ║
╠═══════════════════════════════════════════════════════════════════════╣
║  📁 Analyzes existing codebases                                      ║
║  🔗 Builds intelligent relationships with target structure           ║
║  🤖 Powered by LLM analysis                                          ║
║  📊 Generates detailed JSON indexes                                   ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)


async def main():
    """Main function"""
    print_banner()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Code Indexer - Build intelligent relationships between existing codebase and target structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_indexer.py
    python run_indexer.py --config my_config.yaml
    python run_indexer.py --target-structure structure.txt
    python run_indexer.py --config my_config.yaml --target-structure structure.txt --output results/
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        default="tools/indexer_config.yaml",
        help="Path to configuration YAML file (default: tools/indexer_config.yaml)",
    )

    parser.add_argument(
        "--target-structure",
        "-t",
        help="Path to file containing target project structure",
    )

    parser.add_argument(
        "--output", "-o", help="Output directory for index files (overrides config)"
    )

    parser.add_argument(
        "--code-base", "-b", help="Path to code_base directory (overrides config)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"📋 Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Load target structure
    args.target_structure = "./agent_folders/papers/2/initial_plan.txt"
    if args.target_structure:
        print(f"📐 Loading target structure from: {args.target_structure}")
        target_structure = load_target_structure_from_plan(args.target_structure)
    else:
        print("📐 Using default target structure")
        target_structure = get_default_target_structure()

    # Override config with command line arguments
    code_base_path = args.code_base or config["paths"]["code_base_path"]
    output_dir = args.output or config["paths"]["output_dir"]

    print(f"📁 Code base path: {code_base_path}")
    print(f"📤 Output directory: {output_dir}")
    print()

    # Validate paths
    if not Path(code_base_path).exists():
        print(f"❌ Error: Code base path does not exist: {code_base_path}")
        sys.exit(1)

    # Create indexer
    try:
        indexer = CodeIndexer(
            code_base_path=code_base_path,
            target_structure=target_structure,
            output_dir=output_dir,
        )

        # Apply additional configuration settings
        if "file_analysis" in config:
            file_config = config["file_analysis"]
            if "supported_extensions" in file_config:
                indexer.supported_extensions = set(file_config["supported_extensions"])

        print("🚀 Starting code indexing process...")
        print("=" * 60)

        # Build all indexes
        output_files = await indexer.build_all_indexes()

        # Generate summary report
        summary_report = indexer.generate_summary_report(output_files)

        # Print results
        print("=" * 60)
        print("✅ Indexing completed successfully!")
        print(f"📊 Processed {len(output_files)} repositories")
        print()
        print("📁 Generated index files:")
        for repo_name, file_path in output_files.items():
            print(f"   📄 {repo_name}: {file_path}")
        print()
        print(f"📋 Summary report: {summary_report}")

        # Additional statistics if enabled
        if config.get("output", {}).get("generate_statistics", False):
            print("\n📈 Processing Statistics:")

            total_relationships = 0
            high_confidence_relationships = 0

            for file_path in output_files.values():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        index_data = yaml.safe_load(f)
                        relationships = index_data.get("relationships", [])
                        total_relationships += len(relationships)
                        high_confidence_relationships += len(
                            [
                                r
                                for r in relationships
                                if r.get("confidence_score", 0)
                                > config.get("relationships", {}).get(
                                    "high_confidence_threshold", 0.7
                                )
                            ]
                        )
                except Exception as e:
                    print(
                        f"   ⚠️ Warning: Could not load statistics from {file_path}: {e}"
                    )

            print(f"   🔗 Total relationships found: {total_relationships}")
            print(
                f"   ⭐ High confidence relationships: {high_confidence_relationships}"
            )
            print(
                f"   📊 Average relationships per repo: {total_relationships / len(output_files) if output_files else 0:.1f}"
            )

        print("\n🎉 Code indexing process completed successfully!")

    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Add a simple test mode for file tree extraction
    if len(sys.argv) > 1 and sys.argv[1] == "--test-extract":
        print("🧪 Testing file tree extraction from initial_plan.txt...")
        plan_path = "./agent_folders/papers/2/initial_plan.txt"
        if Path(plan_path).exists():
            try:
                result = load_target_structure_from_plan(plan_path)
                print("\n" + "=" * 60)
                print("📊 Final extracted structure:")
                print("=" * 60)
                print(result)
                print("=" * 60)
            except Exception as e:
                print(f"❌ Test failed: {e}")
        else:
            print(f"❌ Test file not found: {plan_path}")
    else:
        asyncio.run(main())
