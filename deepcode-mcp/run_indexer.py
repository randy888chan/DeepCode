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
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        sys.exit(1)


def load_target_structure(structure_path: str) -> str:
    """Load target structure from file"""
    try:
        with open(structure_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error loading target structure file {structure_path}: {e}")
        sys.exit(1)


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
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='tools/indexer_config.yaml',
        help='Path to configuration YAML file (default: tools/indexer_config.yaml)'
    )
    
    parser.add_argument(
        '--target-structure', '-t',
        help='Path to file containing target project structure'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory for index files (overrides config)'
    )
    
    parser.add_argument(
        '--code-base', '-b',
        help='Path to code_base directory (overrides config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"📋 Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Load target structure
    if args.target_structure:
        print(f"📐 Loading target structure from: {args.target_structure}")
        target_structure = load_target_structure(args.target_structure)
    else:
        print("📐 Using default target structure")
        target_structure = get_default_target_structure()
    
    # Override config with command line arguments
    code_base_path = args.code_base or config['paths']['code_base_path']
    output_dir = args.output or config['paths']['output_dir']
    
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
            output_dir=output_dir
        )
        
        # Apply additional configuration settings
        if 'file_analysis' in config:
            file_config = config['file_analysis']
            if 'supported_extensions' in file_config:
                indexer.supported_extensions = set(file_config['supported_extensions'])
        
        print("🚀 Starting code indexing process...")
        print("=" * 60)
        
        # Build all indexes
        output_files = await indexer.build_all_indexes()
        
        # Generate summary report
        summary_report = indexer.generate_summary_report(output_files)
        
        # Print results
        print("=" * 60)
        print(f"✅ Indexing completed successfully!")
        print(f"📊 Processed {len(output_files)} repositories")
        print()
        print("📁 Generated index files:")
        for repo_name, file_path in output_files.items():
            print(f"   📄 {repo_name}: {file_path}")
        print()
        print(f"📋 Summary report: {summary_report}")
        
        # Additional statistics if enabled
        if config.get('output', {}).get('generate_statistics', False):
            print("\n📈 Processing Statistics:")
            
            total_relationships = 0
            high_confidence_relationships = 0
            
            for file_path in output_files.values():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        index_data = yaml.safe_load(f)
                        relationships = index_data.get('relationships', [])
                        total_relationships += len(relationships)
                        high_confidence_relationships += len([
                            r for r in relationships 
                            if r.get('confidence_score', 0) > config.get('relationships', {}).get('high_confidence_threshold', 0.7)
                        ])
                except Exception as e:
                    print(f"   ⚠️ Warning: Could not load statistics from {file_path}: {e}")
            
            print(f"   🔗 Total relationships found: {total_relationships}")
            print(f"   ⭐ High confidence relationships: {high_confidence_relationships}")
            print(f"   📊 Average relationships per repo: {total_relationships / len(output_files) if output_files else 0:.1f}")
        
        print("\n🎉 Code indexing process completed successfully!")
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 