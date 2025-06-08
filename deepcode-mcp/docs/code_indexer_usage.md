# Code Indexer Usage Guide

## Overview

The Code Indexer is an intelligent tool that analyzes existing codebases and builds relationships with a target project structure using LLM-powered analysis. It helps developers understand how existing code can be leveraged when building new projects.

## Features

- 🔍 **Intelligent Analysis**: Uses LLM to understand code functionality and relationships
- 📁 **Recursive Traversal**: Analyzes all files in repository directories
- 🔗 **Relationship Mapping**: Maps existing code to target project structure
- 📊 **JSON Output**: Structured output with detailed metadata
- ⚙️ **Configurable**: Highly customizable through YAML configuration
- 🚀 **Async Processing**: Efficient concurrent processing

## Quick Start

### 1. Basic Usage

```bash
# Run with default settings
python run_indexer.py

# Use custom configuration
python run_indexer.py --config my_config.yaml

# Specify target structure file
python run_indexer.py --target-structure my_structure.txt

# Custom output directory
python run_indexer.py --output ./results/
```

### 2. Configuration

Edit `tools/indexer_config.yaml` to customize:

```yaml
# Paths Configuration
paths:
  code_base_path: "deepcode-mcp/agent_folders/papers/paper_3/code_base"
  output_dir: "deepcode-mcp/agent_folders/papers/paper_3/indexes"

# LLM Configuration
llm:
  model_provider: "anthropic"
  max_tokens: 4000
  temperature: 0.3

# Relationship Analysis Settings
relationships:
  min_confidence_score: 0.3
  high_confidence_threshold: 0.7
```

## Output Format

### Repository Index JSON Structure

Each repository generates a JSON file with the following structure:

```json
{
  "repo_name": "LightGCN-PyTorch",
  "total_files": 15,
  "file_summaries": [
    {
      "file_path": "LightGCN-PyTorch/code/model.py",
      "file_type": "Python ML model implementation",
      "main_functions": ["LightGCN", "BasicModel", "forward"],
      "key_concepts": ["graph neural network", "collaborative filtering", "embeddings"],
      "dependencies": ["torch", "torch_geometric"],
      "summary": "Implements the LightGCN model for collaborative filtering using graph neural networks.",
      "lines_of_code": 245,
      "last_modified": "2023-12-01T10:30:00"
    }
  ],
  "relationships": [
    {
      "repo_file_path": "LightGCN-PyTorch/code/model.py",
      "target_file_path": "src/core/gcn.py",
      "relationship_type": "direct_match",
      "confidence_score": 0.9,
      "helpful_aspects": [
        "Graph convolution implementation",
        "User-item embedding methods",
        "Forward pass logic"
      ],
      "potential_contributions": [
        "Can serve as base implementation for GCN encoder",
        "Provides tested embedding strategies",
        "Includes optimization techniques"
      ],
      "usage_suggestions": "This file can be directly adapted as the GCN encoder implementation. The LightGCN class provides a clean interface for graph-based collaborative filtering that aligns well with the target architecture."
    }
  ],
  "analysis_metadata": {
    "analysis_date": "2023-12-01T15:30:00",
    "target_structure_analyzed": "project/...",
    "total_relationships_found": 25,
    "high_confidence_relationships": 8,
    "analyzer_version": "1.0.0"
  }
}
```

### Relationship Types

1. **direct_match** (confidence: 0.8-1.0)
   - Direct implementation that can be used as-is or with minimal modifications

2. **partial_match** (confidence: 0.6-0.8)
   - Partial functionality match that requires adaptation

3. **reference** (confidence: 0.4-0.6)
   - Reference implementation or utility functions

4. **utility** (confidence: 0.3-0.4)
   - General utility or helper functions

### Summary Report

The system also generates `indexing_summary.json`:

```json
{
  "indexing_completion_time": "2023-12-01T15:45:00",
  "total_repositories_processed": 2,
  "output_files": {
    "LightGCN-PyTorch": "indexes/LightGCN-PyTorch_index.json",
    "neural_graph_collaborative_filtering": "indexes/neural_graph_collaborative_filtering_index.json"
  },
  "target_structure": "project/...",
  "code_base_path": "deepcode-mcp/agent_folders/papers/paper_3/code_base"
}
```

## Advanced Usage

### Custom Target Structure

Create a text file with your project structure:

```
my_project/
├── core/
│   ├── models.py       # ML models
│   ├── training.py     # Training logic
│   └── inference.py    # Inference engine
├── utils/
│   ├── data.py         # Data processing
│   └── metrics.py      # Evaluation metrics
└── configs/
    └── config.yaml     # Configuration
```

Then run:
```bash
python run_indexer.py --target-structure my_structure.txt
```

### Configuration Options

#### File Analysis Settings

```yaml
file_analysis:
  supported_extensions:
    - ".py"
    - ".js" 
    - ".cpp"
  skip_directories:
    - "__pycache__"
    - "node_modules"
  max_file_size: 1048576  # 1MB
  max_content_length: 3000
```

#### LLM Settings

```yaml
llm:
  model_provider: "anthropic"  # or "openai"
  max_tokens: 4000
  temperature: 0.3
  request_delay: 0.1  # Rate limiting
  max_retries: 3
```

#### Output Settings

```yaml
output:
  json_indent: 2
  ensure_ascii: false
  generate_summary: true
  generate_statistics: true
  index_filename_pattern: "{repo_name}_index.json"
```

## Integration with Your Workflow

### 1. Pre-Development Analysis

Use the indexer before starting development to:
- Identify reusable components
- Understand existing architectures
- Plan code adaptation strategies

### 2. Code Migration

The output helps with:
- Finding relevant implementations
- Understanding adaptation requirements
- Planning refactoring tasks

### 3. Knowledge Discovery

Use for:
- Learning from existing codebases
- Discovering best practices
- Finding utility functions

## Example Usage Scenarios

### Scenario 1: Building a Recommendation System

```bash
# Analyze existing recommendation systems
python run_indexer.py \
  --code-base ./existing_recommenders/ \
  --target-structure ./my_recsys_structure.txt \
  --output ./recsys_analysis/
```

### Scenario 2: ML Pipeline Development

```bash
# Analyze ML frameworks and tools
python run_indexer.py \
  --config ml_analysis_config.yaml \
  --target-structure ./ml_pipeline_structure.txt
```

### Scenario 3: Microservices Architecture

```bash
# Analyze existing microservices
python run_indexer.py \
  --code-base ./microservices_repos/ \
  --target-structure ./new_service_structure.txt \
  --verbose
```

## Tips and Best Practices

### 1. Target Structure Design

- Be specific about file purposes in comments
- Use clear, descriptive file names
- Organize by functionality, not just file type

### 2. Configuration Tuning

- Adjust `min_confidence_score` based on needs
- Use higher confidence for critical components
- Lower confidence for exploratory analysis

### 3. Result Analysis

- Focus on high-confidence relationships first
- Read usage suggestions carefully
- Consider adaptation effort vs. reuse benefits

### 4. Performance Optimization

```yaml
performance:
  enable_concurrent_analysis: true
  max_concurrent_files: 5
  enable_content_caching: false
```

## Troubleshooting

### Common Issues

1. **LLM API Errors**
   - Check API keys and quotas
   - Reduce request rate with `request_delay`
   - Use retry configuration

2. **File Access Errors**
   - Verify file permissions
   - Check file encoding issues
   - Use `max_file_size` limit

3. **Memory Issues**
   - Reduce `max_concurrent_files`
   - Enable content caching carefully
   - Process repos individually if needed

### Debug Mode

```bash
python run_indexer.py --verbose
```

Or enable in config:
```yaml
debug:
  save_raw_responses: true
  verbose_output: true
  mock_llm_responses: false  # For testing without API calls
```

## Output Analysis Tools

### Analyzing Results with Python

```python
import json

# Load index file
with open('indexes/LightGCN-PyTorch_index.json', 'r') as f:
    index = json.load(f)

# Find high-confidence relationships
high_conf = [r for r in index['relationships'] 
             if r['confidence_score'] > 0.7]

# Group by target file
from collections import defaultdict
by_target = defaultdict(list)
for rel in high_conf:
    by_target[rel['target_file_path']].append(rel)

# Print summary
for target, relationships in by_target.items():
    print(f"{target}: {len(relationships)} relationships")
```

### Filtering and Analysis

```python
# Find all files related to specific functionality
gcn_related = [r for r in index['relationships'] 
               if 'gcn' in r['target_file_path'].lower()]

# Get summary statistics
total_files = index['total_files']
total_relationships = len(index['relationships'])
avg_relationships = total_relationships / total_files

print(f"Average relationships per file: {avg_relationships:.2f}")
```

## Extending the System

### Adding New File Types

```yaml
file_analysis:
  supported_extensions:
    - ".py"
    - ".rs"      # Add Rust support
    - ".go"      # Add Go support
```

### Custom Relationship Types

Modify the `relationship_types` configuration:

```yaml
relationships:
  relationship_types:
    direct_match: 1.0
    partial_match: 0.8
    reference: 0.6
    utility: 0.4
    inspiration: 0.2    # New type for conceptual inspiration
```

## API Integration

The indexer can be used as a library:

```python
from tools.code_indexer import CodeIndexer

# Create indexer
indexer = CodeIndexer(
    code_base_path="./repos",
    target_structure=structure_text,
    output_dir="./results"
)

# Process repositories
output_files = await indexer.build_all_indexes()

# Access results
for repo_name, index_file in output_files.items():
    print(f"Processed {repo_name}: {index_file}")
```

This comprehensive system provides everything you need to build intelligent relationships between existing codebases and your target project structure. The LLM-powered analysis ensures accurate and contextual understanding of code relationships. 