from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
import os
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # 禁止生成.pyc文件

from prompts.code_prompts import (
    PLAN_ANALYZER_PROMPT,
    UNIVERSAL_STRUCTURE_GENERATOR_PROMPT,
    UNIVERSAL_MODULE_IMPLEMENTER_PROMPT,
    UNIVERSAL_INTEGRATION_SPECIALIST_PROMPT,
    UNIVERSAL_TESTING_ENGINEER_PROMPT,
    UNIVERSAL_OPTIMIZER_PROMPT,
    UNIVERSAL_DOCUMENTATION_WRITER_PROMPT,
    UNIVERSAL_VALIDATION_SPECIALIST_PROMPT,
    HIERARCHICAL_LAYER_IMPLEMENTER_PROMPT
)

def _get_project_file_tree(directory: str, max_depth: int = 3) -> str:
    """
    Generate a tree representation of the project file structure.
    
    Args:
        directory: Root directory to scan
        max_depth: Maximum depth to scan (to avoid too much output)
        
    Returns:
        String representation of the file tree
    """
    if not os.path.exists(directory):
        return f"Directory {directory} does not exist yet."
    
    def _build_tree(path: str, prefix: str = "", depth: int = 0) -> List[str]:
        if depth > max_depth:
            return []
            
        items = []
        try:
            entries = sorted(os.listdir(path))
            # Filter out hidden files and __pycache__
            entries = [e for e in entries if not e.startswith('.') and e != '__pycache__']
            
            for i, entry in enumerate(entries):
                entry_path = os.path.join(path, entry)
                is_last = i == len(entries) - 1
                
                if os.path.isdir(entry_path):
                    items.append(f"{prefix}{'└── ' if is_last else '├── '}{entry}/")
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    items.extend(_build_tree(entry_path, new_prefix, depth + 1))
                else:
                    # Show file size for Python files
                    if entry.endswith('.py'):
                        try:
                            size = os.path.getsize(entry_path)
                            items.append(f"{prefix}{'└── ' if is_last else '├── '}{entry} ({size} bytes)")
                        except:
                            items.append(f"{prefix}{'└── ' if is_last else '├── '}{entry}")
                    else:
                        items.append(f"{prefix}{'└── ' if is_last else '├── '}{entry}")
        except PermissionError:
            items.append(f"{prefix}[Permission Denied]")
            
        return items
    
    tree_lines = [f"Project Structure: {os.path.basename(directory)}/"]
    tree_lines.extend(_build_tree(directory))
    
    return "\n".join(tree_lines)

async def analyze_implementation_plan(initial_plan_path: str, logger) -> Dict[str, Any]:
    """
    Analyze the implementation plan to extract structured information about the project.
    
    Args:
        initial_plan_path: Path to the initial plan file
        logger: Logger instance
        
    Returns:
        Dict containing structured analysis of the implementation plan
    """
    # Check if analysis result already exists
    paper_dir = os.path.dirname(initial_plan_path)
    analysis_cache_path = os.path.join(paper_dir, "plan_analysis.json")
    
    if os.path.exists(analysis_cache_path):
        logger.info("Plan Analyzer: Found existing analysis result, loading from cache...")
        try:
            with open(analysis_cache_path, 'r', encoding='utf-8') as f:
                cached_analysis = json.load(f)
            
            # Verify cache validity by checking if original plan is included
            if 'original_plan' in cached_analysis:
                logger.info("✅ Using cached plan analysis result")
                return cached_analysis
            else:
                logger.warning("⚠️ Cached analysis missing original_plan, regenerating...")
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            logger.warning(f"⚠️ Invalid cache file: {e}, regenerating analysis...")
    
    # Read the initial plan
    with open(initial_plan_path, 'r', encoding='utf-8') as f:
        plan_content = f.read()
    
    plan_analyzer_agent = Agent(
        name="PlanAnalyzer",
        instruction=PLAN_ANALYZER_PROMPT,
    )
    
    async with plan_analyzer_agent:
        logger.info("Plan Analyzer: Analyzing implementation plan...")
        analyzer = await plan_analyzer_agent.attach_llm(AnthropicAugmentedLLM)
        
        # Set higher max_tokens to avoid truncation
        analysis_result = await analyzer.generate_str(
            message=f"Analyze this implementation plan and extract structured information:\n\n{plan_content}",
            request_params=RequestParams(max_tokens=4096)  # Increased from default 2048
        )
        
        # Parse the JSON response with improved error handling
        try:
            # Extract JSON from the response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', analysis_result, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group(1))
            else:
                # Try to find JSON block without markdown
                json_match = re.search(r'\{.*\}', analysis_result, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group(0))
                else:
                    # Last resort: try to parse the entire response as JSON
                    analysis_data = json.loads(analysis_result)
            
            # Add the original plan content for reference
            analysis_data['original_plan'] = plan_content
            
            # Save analysis result to cache
            with open(analysis_cache_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2)
            logger.info("💾 Saved analysis result to cache")
            
            logger.info(f"Successfully parsed plan analysis with confidence: {analysis_data.get('confidence_level', 'unknown')}")
            return analysis_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan analysis JSON: {e}")
            logger.debug(f"Raw response was: {analysis_result[:500]}...")
            
            # Enhanced fallback structure with intelligent content analysis
            project_name = "research_project"
            algorithm_type = "other"
            domain = "other"
            language = "python"
            frameworks = ["numpy"]
            hardware = "cpu"
            
            # Intelligent content analysis for fallback
            content_lower = plan_content.lower()
            
            # Language detection
            if any(lang in content_lower for lang in ["python", "py", "pip", "conda"]):
                language = "python"
            elif any(lang in content_lower for lang in ["r ", " r", "cran", "rstudio"]):
                language = "r"
            elif any(lang in content_lower for lang in ["java", "maven", "gradle"]):
                language = "java"
            elif any(lang in content_lower for lang in ["c++", "cpp", "cmake"]):
                language = "cpp"
            
            # Algorithm type detection
            if any(term in content_lower for term in ["neural", "deep", "cnn", "rnn", "transformer", "pytorch", "tensorflow", "keras"]):
                algorithm_type = "deep_learning"
                frameworks = ["torch", "numpy"] if language == "python" else ["numpy"]
            elif any(term in content_lower for term in ["svm", "random forest", "clustering", "scikit", "sklearn"]):
                algorithm_type = "traditional_ml"
                frameworks = ["scikit-learn", "numpy"]
            elif any(term in content_lower for term in ["optimization", "solver", "linear programming", "convex"]):
                algorithm_type = "optimization"
                frameworks = ["scipy", "numpy"]
            elif any(term in content_lower for term in ["image", "vision", "cv", "computer vision", "opencv"]):
                algorithm_type = "computer_vision"
                frameworks = ["opencv-python", "numpy"]
            elif any(term in content_lower for term in ["text", "nlp", "language", "bert", "gpt", "tokeniz"]):
                algorithm_type = "nlp"
                frameworks = ["transformers", "torch"] if "transform" in content_lower else ["nltk", "spacy"]
            elif any(term in content_lower for term in ["reinforcement", "rl", "agent", "policy", "reward"]):
                algorithm_type = "reinforcement_learning"
                frameworks = ["gym", "torch"]
            elif any(term in content_lower for term in ["graph", "network", "node", "edge", "gnn"]):
                algorithm_type = "graph_learning"
                frameworks = ["networkx", "torch"]
            elif any(term in content_lower for term in ["recommend", "collaborative", "rating", "user", "item"]):
                algorithm_type = "recommendation"
                domain = "recommendation"
                
            # Domain detection
            if domain == "other":  # Only if not already set
                if any(term in content_lower for term in ["medical", "health", "clinical", "patient"]):
                    domain = "healthcare"
                elif any(term in content_lower for term in ["financial", "finance", "trading", "stock"]):
                    domain = "finance"
                elif any(term in content_lower for term in ["robot", "autonomous", "control", "motion"]):
                    domain = "robotics"
                elif algorithm_type == "computer_vision":
                    domain = "computer_vision"
                elif algorithm_type == "nlp":
                    domain = "nlp"
                    
            # Hardware detection
            if any(term in content_lower for term in ["gpu", "cuda", "nvidia", "deep learning"]):
                hardware = "gpu"
            elif any(term in content_lower for term in ["tpu", "tensor processing"]):
                hardware = "tpu"
            elif any(term in content_lower for term in ["cluster", "distributed", "parallel"]):
                hardware = "cluster"
            
            # Try to extract project name from content
            import re
            # Look for title-like patterns or project names
            title_patterns = [
                r'title[:\s]*([^\n]+)',
                r'project[:\s]*([^\n]+)',
                r'paper[:\s]*([^\n]+)',
                r'algorithm[:\s]*([^\n]+)',
                r'method[:\s]*([^\n]+)'
            ]
            
            for pattern in title_patterns:
                match = re.search(pattern, content_lower)
                if match:
                    potential_name = match.group(1).strip().replace('"', '').replace("'", '')
                    if len(potential_name) > 3 and len(potential_name) < 50:
                        project_name = potential_name
                        break
            
            return {
                "project_info": {
                    "name": project_name,
                    "main_algorithm": "unknown",
                    "algorithm_type": algorithm_type,
                    "domain": domain
                },
                "technology_stack": {
                    "language": language,
                    "version": "latest",
                    "frameworks": frameworks,
                    "dependencies": frameworks + ["numpy"] if "numpy" not in frameworks else frameworks,
                    "hardware_requirements": hardware
                },
                "architecture": {
                    "main_components": [
                        {
                            "name": "main_algorithm",
                            "purpose": "core algorithm implementation",
                            "type": "core",
                            "dependencies": ["none"]
                        },
                        {
                            "name": "data_handler",
                            "purpose": "data processing and management",
                            "type": "data",
                            "dependencies": ["none"]
                        }
                    ],
                    "data_flow": "not_specified",
                    "entry_point": "not_specified"
                },
                "project_structure": {
                    "base_directories": ["src", "tests", "docs", "configs"],
                    "core_modules": [
                        {
                            "name": "main_module",
                            "file_name": "main.py",
                            "purpose": "main implementation",
                            "priority": 2
                        },
                        {
                            "name": "data_module", 
                            "file_name": "data.py",
                            "purpose": "data handling",
                            "priority": 1
                        },
                        {
                            "name": "utils_module",
                            "file_name": "utils.py", 
                            "purpose": "utility functions",
                            "priority": 1
                        }
                    ]
                },
                "implementation_phases": [
                    {
                        "phase": 1,
                        "name": "setup_and_data",
                        "tasks": ["project_setup", "data_processing"],
                        "can_parallelize": False
                    },
                    {
                        "phase": 2,
                        "name": "core_implementation",
                        "tasks": ["algorithm_implementation", "integration"],
                        "can_parallelize": True
                    }
                ],
                "confidence_level": "low",
                "analysis_raw": analysis_result,
                "original_plan": plan_content,
                "parsing_error": str(e),
                "fallback_analysis": {
                    "content_based_inference": True,
                    "detected_language": language,
                    "detected_algorithm_type": algorithm_type,
                    "detected_domain": domain,
                    "detected_hardware": hardware
                }
            }

async def create_universal_project_structure(analysis_data: Dict[str, Any], generate_code_dir: str, logger) -> Dict[str, str]:
    """
    Create a universal project structure based on the analyzed plan.
    
    Args:
        analysis_data: Structured analysis of the implementation plan
        generate_code_dir: Directory where code will be generated
        logger: Logger instance
        
    Returns:
        Dict containing the project structure information
    """
    # Check if structure result already exists
    structure_result_path = os.path.join(os.path.dirname(generate_code_dir), "structure_generation_result.txt")
    
    if os.path.exists(structure_result_path):
        logger.info("Universal Structure Generator: Found existing structure result, loading from cache...")
        try:
            with open(structure_result_path, 'r', encoding='utf-8') as f:
                cached_result = f.read()
            
            if cached_result.strip():  # Ensure it's not empty
                logger.info("✅ Using cached structure generation result")
                return {
                    "structure_result": cached_result,
                    "generate_code_dir": generate_code_dir,
                    "analysis_data": analysis_data
                }
            else:
                logger.warning("⚠️ Cached structure result is empty, regenerating...")
        except FileNotFoundError as e:
            logger.warning(f"⚠️ Error reading cache file: {e}, regenerating structure...")
    
    structure_agent = Agent(
        name="UniversalStructureGenerator",
        instruction=UNIVERSAL_STRUCTURE_GENERATOR_PROMPT + f"\n\nGenerate code in directory: {generate_code_dir}",
        server_names=["code-generator"],
    )
    
    async with structure_agent:
        logger.info("Universal Structure Generator: Implementing project structure from plan...")
        generator = await structure_agent.attach_llm(AnthropicAugmentedLLM)
        
        # Prepare the analysis data AND original plan as context
        context = f"""
        IMPLEMENTATION PLAN (Primary Reference):
        {analysis_data.get('original_plan', 'No original plan available')}
        
        ANALYSIS DATA (Secondary Reference):
        {json.dumps(analysis_data, indent=2)}
        
        TARGET DIRECTORY: {generate_code_dir}
        
        🚨 CRITICAL REQUIREMENT: USE MCP TOOLS TO CREATE ACTUAL PROJECT STRUCTURE
        
        TASK: Implement the project structure EXACTLY as specified in the implementation plan above by ACTUALLY CREATING FILES AND DIRECTORIES using the available MCP tools.
        
        ### Required Actions:
        1. **create_directory** or **create_project_structure**: Create the complete directory hierarchy
        2. **generate_python_file**: Create Python skeleton files with proper imports and structure
        3. **generate_requirements_file**: Create requirements.txt with all dependencies
        4. **generate_config_file**: Create configuration files as specified
        5. **write_file**: Create README.md, .gitignore, and other documentation files
        6. **validate_python_syntax**: Validate all created Python files
        
        ### Implementation Process:
        1. Analyze the implementation plan to understand the exact structure needed
        2. Use MCP tools to create the complete directory structure
        3. Generate Python skeleton files with proper imports, class stubs, and documentation
        4. Create configuration and documentation files
        5. Validate that all files are properly created and syntactically correct
        6. Report what was created and how it matches the plan
        
        Do not redesign or modify the structure - just create the files and directories as planned using MCP tools.
        Ensure all created Python files are importable and syntactically correct.
        """
        
        result = await generator.generate_str(
            message=context,
            request_params=RequestParams(max_tokens=8192)  # Increased for structure generation
        )
        
        # Save structure generation result
        with open(structure_result_path, 'w', encoding='utf-8') as f:
            f.write(result)
        logger.info("💾 Saved structure generation result to cache")
        
        return {
            "structure_result": result,
            "generate_code_dir": generate_code_dir,
            "analysis_data": analysis_data
        }

async def implement_hierarchical_codebase(structure_info: Dict[str, str], logger) -> Dict[str, str]:
    """
    Hierarchical implementation approach - implements codebase layer by layer with full context.
    This replaces the old module-by-module approach to ensure code coherence and avoid API limits.
    
    Args:
        structure_info: Information about the created project structure
        logger: Logger instance
        
    Returns:
        Dict containing implementation results for each layer
    """
    generate_code_dir = structure_info["generate_code_dir"]
    analysis_data = structure_info["analysis_data"]
    
    # Implementation layers (sequential, not parallel)
    implementation_layers = [
        {
            "name": "core_architecture",
            "description": "Core architecture and main algorithm class",
            "focus": "Implement the main algorithm class and core interfaces based on the plan",
            "files": ["main model class", "core interfaces", "base classes"],
            "context_priority": "architecture_design"
        },
        {
            "name": "data_layer", 
            "description": "Data processing and pipeline implementation",
            "focus": "Implement data loading, preprocessing, and pipeline components",
            "files": ["data loaders", "preprocessors", "data utilities"],
            "context_priority": "data_flow"
        },
        {
            "name": "algorithm_layer",
            "description": "Core algorithm implementation",
            "focus": "Implement the core algorithms following the main architecture",
            "files": ["algorithm modules", "mathematical operations", "core logic"],
            "context_priority": "algorithm_implementation"
        },
        {
            "name": "training_layer",
            "description": "Training and optimization implementation", 
            "focus": "Implement training loops, loss functions, and optimization",
            "files": ["training modules", "loss functions", "optimizers"],
            "context_priority": "training_pipeline"
        },
        {
            "name": "evaluation_layer",
            "description": "Evaluation and metrics implementation",
            "focus": "Implement evaluation metrics, testing, and validation",
            "files": ["evaluation modules", "metrics", "validators"],
            "context_priority": "evaluation_framework"
        },
        {
            "name": "integration_layer",
            "description": "Integration and utilities implementation",
            "focus": "Implement utilities, configurations, and final integration",
            "files": ["utilities", "configurations", "integration scripts"],
            "context_priority": "system_integration"
        }
    ]
    
    results = {}
    cumulative_context = f"""
    ORIGINAL IMPLEMENTATION PLAN (Primary Reference):
    {analysis_data.get('original_plan', 'No original plan available')}
    
    PROJECT ANALYSIS:
    - Algorithm Type: {analysis_data.get('project_info', {}).get('algorithm_type', 'unknown')}
    - Domain: {analysis_data.get('project_info', {}).get('domain', 'other')}
    - Language: {analysis_data.get('technology_stack', {}).get('language', 'python')}
    - Frameworks: {analysis_data.get('technology_stack', {}).get('frameworks', [])}
    """
    
    # Check if all layer results already exist
    all_cached = True
    for layer in implementation_layers:
        layer_result_path = os.path.join(os.path.dirname(generate_code_dir), f"layer_{layer['name']}_result.txt")
        if not os.path.exists(layer_result_path):
            all_cached = False
            break
    
    if all_cached:
        logger.info("Hierarchical Implementation: Found all layer results in cache, loading...")
        try:
            for layer in implementation_layers:
                layer_result_path = os.path.join(os.path.dirname(generate_code_dir), f"layer_{layer['name']}_result.txt")
                with open(layer_result_path, 'r', encoding='utf-8') as f:
                    cached_result = f.read()
                    if cached_result.strip():
                        results[layer['name']] = cached_result
                    else:
                        logger.warning(f"⚠️ Layer {layer['name']} cache is empty, will regenerate all layers...")
                        all_cached = False
                        results.clear()
                        break
            
            if all_cached:
                logger.info("✅ Using all cached layer implementation results")
                return results
        except Exception as e:
            logger.warning(f"⚠️ Error loading cached layer results: {e}, regenerating all layers...")
            results.clear()
    
    # Implement each layer sequentially (NO PARALLEL API CALLS)
    for i, layer in enumerate(implementation_layers):
        layer_result_path = os.path.join(os.path.dirname(generate_code_dir), f"layer_{layer['name']}_result.txt")
        
        # Check if this specific layer result exists
        if os.path.exists(layer_result_path):
            logger.info(f"Layer {i+1}/{len(implementation_layers)}: Found cached result for {layer['name']}, loading...")
            try:
                with open(layer_result_path, 'r', encoding='utf-8') as f:
                    cached_result = f.read()
                
                if cached_result.strip():
                    results[layer['name']] = cached_result
                    logger.info(f"✅ Using cached result for layer {layer['name']}")
                    continue
                else:
                    logger.warning(f"⚠️ Cached result for layer {layer['name']} is empty, regenerating...")
            except Exception as e:
                logger.warning(f"⚠️ Error reading cache for layer {layer['name']}: {e}, regenerating...")
        
        logger.info(f"Layer {i+1}/{len(implementation_layers)}: Implementing {layer['name']}...")
        
        # Create layer-specific agent
        layer_agent = Agent(
            name=f"LayerImplementer_{layer['name']}",
            instruction=HIERARCHICAL_LAYER_IMPLEMENTER_PROMPT,
            server_names=[ "code-generator"],
        )
        
        async with layer_agent:
            logger.info(f"  Focus: {layer['description']}")
            implementer = await layer_agent.attach_llm(AnthropicAugmentedLLM)
            
            # Build context with previous layer results
            layer_context = f"""
            IMPLEMENTATION PLAN (Primary Reference):
            {analysis_data.get('original_plan', 'No original plan available')}
            
            PROJECT CONTEXT:
            - Algorithm Type: {analysis_data.get('project_info', {}).get('algorithm_type', 'unknown')}
            - Domain: {analysis_data.get('project_info', {}).get('domain', 'other')}
            - Language: {analysis_data.get('technology_stack', {}).get('language', 'python')}
            - Frameworks: {analysis_data.get('technology_stack', {}).get('frameworks', [])}
            - Target Directory: {generate_code_dir}
            
            PROJECT FILE STRUCTURE (Already Created):
            {_get_project_file_tree(generate_code_dir)}
            
            PREVIOUS LAYER IMPLEMENTATIONS:
            {json.dumps({k: v[:400] + "..." if len(v) > 400 else v for k, v in results.items()}, indent=2)}
            
            CURRENT LAYER FOCUS:
            - Layer: {layer['name']} ({layer['description']})
            - Implementation Focus: {layer['focus']}
            - Target Components: {layer['files']}
            - Priority Context: {layer['context_priority']}
            
            🚨 CRITICAL REQUIREMENTS - MUST IMPLEMENT ACTUAL ALGORITHMS:
            
            You MUST use MCP tools to create COMPLETE, WORKING algorithm implementations in existing Python files. This is NOT about generating text descriptions - you must call MCP tools to create actual code.
            
            MANDATORY ACTIONS FOR THIS LAYER:
            1. **identify existing files** from the PROJECT FILE STRUCTURE above that need implementation for this layer
            2. **call generate_python_file(file_path, complete_algorithm_code, overwrite=True)** for EACH file
            3. **implement COMPLETE algorithms** - not stubs, not TODOs, but working Python code
            4. **validate syntax** with validate_python_syntax(file_path) for each file created
            
            ALGORITHMIC IMPLEMENTATION REQUIREMENTS:
            - Write complete functions that can be imported and executed
            - Implement actual mathematical operations based on the paper/plan
            - Include proper imports, class definitions, and method implementations
            - Add error handling, logging, and documentation
            - Ensure code follows the technology stack specified (frameworks: {analysis_data.get('technology_stack', {}).get('frameworks', [])})
            
            EXAMPLE OF WHAT YOU MUST DO:
            Based on the PROJECT FILE STRUCTURE above, if you see a file like:
            - {generate_code_dir}/src/core/classifier.py
            
            Then call:
            generate_python_file(file_path="{generate_code_dir}/src/core/classifier.py", content="[COMPLETE ALGORITHM CODE]", overwrite=True)
            
            The content must be a complete Python module with working classes and functions.
            
            Layer Integration Guidelines:
            - Build upon interfaces established by previous layers
            - Maintain consistent data structures and calling conventions
            - Implement robust error handling and validation
            - Add comprehensive logging for debugging
            - Ensure algorithmic correctness per the paper specifications
            
            YOU MUST CALL MCP TOOLS TO CREATE ACTUAL CODE FILES. DO NOT JUST DESCRIBE WHAT SHOULD BE IMPLEMENTED.
            
            Use the exact file paths shown in the PROJECT FILE STRUCTURE above.
            """
            
            # Sequential API call (one at a time) with increased tokens
            result = await implementer.generate_str(
                message=layer_context,
                request_params=RequestParams(max_tokens=8192)  # Increased for algorithm implementation
            )
            
            results[layer['name']] = result
            
            # Save layer result
            with open(layer_result_path, 'w', encoding='utf-8') as f:
                f.write(result)
            logger.info(f"💾 Saved layer {layer['name']} result to cache")
            
            # Add delay between layers to respect API limits
            if i < len(implementation_layers) - 1:
                logger.info(f"  Layer {layer['name']} completed. Waiting before next layer...")
                await asyncio.sleep(5)  # 5-second delay between layers
    
    logger.info("Hierarchical implementation completed successfully!")
    return results

async def integrate_universal_modules(layer_results: Dict[str, str], structure_info: Dict[str, str], logger) -> str:
    """
    Integrate all layers into a cohesive system.
    
    Args:
        layer_results: Results from hierarchical layer implementations
        structure_info: Project structure information
        logger: Logger instance
        
    Returns:
        Integration result
    """
    generate_code_dir = structure_info["generate_code_dir"]
    analysis_data = structure_info["analysis_data"]
    
    # Check if integration result already exists
    integration_result_path = os.path.join(os.path.dirname(generate_code_dir), "layer_integration_result.txt")
    
    if os.path.exists(integration_result_path):
        logger.info("Universal Integration Specialist: Found existing integration result, loading from cache...")
        try:
            with open(integration_result_path, 'r', encoding='utf-8') as f:
                cached_result = f.read()
            
            if cached_result.strip():
                logger.info("✅ Using cached integration result")
                return cached_result
            else:
                logger.warning("⚠️ Cached integration result is empty, regenerating...")
        except Exception as e:
            logger.warning(f"⚠️ Error reading integration cache: {e}, regenerating...")
    
    integration_agent = Agent(
        name="UniversalIntegrationSpecialist",
        instruction=UNIVERSAL_INTEGRATION_SPECIALIST_PROMPT + f"\n\nIntegrate layers in {generate_code_dir}",
        server_names=["code-generator"],
    )
    
    async with integration_agent:
        logger.info("Universal Integration Specialist: Connecting hierarchical layers...")
        integrator = await integration_agent.attach_llm(AnthropicAugmentedLLM)
        
        # Prepare integration context with layer-aware information
        integration_context = f"""
        ORIGINAL IMPLEMENTATION PLAN (Primary Reference):
        {analysis_data.get('original_plan', 'No original plan available')}
        
        IMPLEMENTED LAYERS: 
        {json.dumps({k: v[:800] + "..." if len(v) > 800 else v for k, v in layer_results.items()}, indent=2)}
        
        ANALYSIS SUMMARY:
        - Algorithm Type: {analysis_data.get('project_info', {}).get('algorithm_type', 'unknown')}
        - Main Algorithm: {analysis_data.get('project_info', {}).get('main_algorithm', 'unknown')}
        - Entry Point: {analysis_data.get('architecture', {}).get('entry_point', 'not_specified')}
        
        LAYER INTEGRATION TASK:
        You now have a complete set of hierarchical layers implemented. Your task is to:
        
        1. **Review layer interfaces**: Ensure all layers are properly connected
        2. **Create main entry points**: Implement main scripts and CLI interfaces
        3. **Verify data flow**: Ensure data flows correctly through all layers
        4. **Implement configuration**: Create comprehensive configuration management
        5. **Add orchestration**: Create main orchestration/coordination logic
        6. **Integration testing**: Add integration tests and validation
        
        The layers are implemented in this order:
        - core_architecture: Main algorithm class and core interfaces
        - data_layer: Data processing and pipeline
        - algorithm_layer: Core algorithm implementation
        - training_layer: Training and optimization
        - evaluation_layer: Evaluation and metrics
        - integration_layer: Utilities and final integration
        
        Create a cohesive, production-ready system that connects all these layers seamlessly.
        Focus on creating main execution scripts, configuration management, and ensuring the entire system works as intended.
        """
        
        result = await integrator.generate_str(
            message=integration_context,
            request_params=RequestParams(max_tokens=4096)
        )
        
        # Save integration result
        with open(integration_result_path, 'w', encoding='utf-8') as f:
            f.write(result)
        logger.info("💾 Saved integration result to cache")
        
        return result

async def create_universal_tests_and_documentation(integration_result: str, structure_info: Dict[str, str], logger) -> Tuple[str, str]:
    """
    Create comprehensive tests and documentation for any type of implementation.
    
    Args:
        integration_result: Result from module integration
        structure_info: Project structure information
        logger: Logger instance
        
    Returns:
        Tuple of (test_result, documentation_result)
    """
    generate_code_dir = structure_info["generate_code_dir"]
    analysis_data = structure_info["analysis_data"]
    
    # Check if both test and documentation results already exist
    test_result_path = os.path.join(os.path.dirname(generate_code_dir), "test_creation_result.txt")
    doc_result_path = os.path.join(os.path.dirname(generate_code_dir), "documentation_result.txt")
    
    test_cached = False
    doc_cached = False
    test_result = None
    doc_result = None
    
    # Check test cache
    if os.path.exists(test_result_path):
        logger.info("Universal Testing Engineer: Found existing test result, loading from cache...")
        try:
            with open(test_result_path, 'r', encoding='utf-8') as f:
                cached_test = f.read()
            if cached_test.strip():
                test_result = cached_test
                test_cached = True
                logger.info("✅ Using cached test creation result")
            else:
                logger.warning("⚠️ Cached test result is empty, will regenerate...")
        except Exception as e:
            logger.warning(f"⚠️ Error reading test cache: {e}, will regenerate...")
    
    # Check documentation cache
    if os.path.exists(doc_result_path):
        logger.info("Universal Documentation Writer: Found existing documentation result, loading from cache...")
        try:
            with open(doc_result_path, 'r', encoding='utf-8') as f:
                cached_doc = f.read()
            if cached_doc.strip():
                doc_result = cached_doc
                doc_cached = True
                logger.info("✅ Using cached documentation result")
            else:
                logger.warning("⚠️ Cached documentation result is empty, will regenerate...")
        except Exception as e:
            logger.warning(f"⚠️ Error reading documentation cache: {e}, will regenerate...")
    
    # If both are cached, return them
    if test_cached and doc_cached:
        logger.info("✅ Using both cached test and documentation results")
        return test_result, doc_result
    
    # Create universal testing agent
    testing_agent = Agent(
        name="UniversalTestingEngineer",
        instruction=UNIVERSAL_TESTING_ENGINEER_PROMPT + f"\n\nCreate tests in {generate_code_dir}/tests/",
        server_names=["code-generator"],
    )
    
    # Create universal documentation agent
    documentation_agent = Agent(
        name="UniversalDocumentationWriter",
        instruction=UNIVERSAL_DOCUMENTATION_WRITER_PROMPT + f"\n\nCreate documentation in {generate_code_dir}/docs/",
        server_names=["code-generator"],
    )
    
    # Run testing and documentation in parallel (only for uncached ones)
    async def create_tests():
        if test_cached:
            return test_result
            
        async with testing_agent:
            logger.info("Universal Testing Engineer: Creating test suite...")
            tester = await testing_agent.attach_llm(AnthropicAugmentedLLM)
            context = f"""
            Project Analysis: {json.dumps(analysis_data, indent=2)}
            Integration Result: {integration_result}
            
            Create comprehensive tests appropriate for this type of algorithm and implementation.
            """
            result = await tester.generate_str(message=context)
            
            # Save test result
            with open(test_result_path, 'w', encoding='utf-8') as f:
                f.write(result)
            logger.info("💾 Saved test creation result to cache")
            
            return result
    
    async def create_docs():
        if doc_cached:
            return doc_result
            
        async with documentation_agent:
            logger.info("Universal Documentation Writer: Creating documentation...")
            writer = await documentation_agent.attach_llm(AnthropicAugmentedLLM)
            context = f"""
            Project Analysis: {json.dumps(analysis_data, indent=2)}
            Integration Result: {integration_result}
            
            Create comprehensive documentation for this research implementation.
            """
            result = await writer.generate_str(message=context)
            
            # Save documentation result
            with open(doc_result_path, 'w', encoding='utf-8') as f:
                f.write(result)
            logger.info("💾 Saved documentation result to cache")
            
            return result
    
    test_result, doc_result = await asyncio.gather(create_tests(), create_docs())
    
    return test_result, doc_result

async def optimize_and_validate_universal(structure_info: Dict[str, str], logger) -> Tuple[str, str]:
    """
    Optimize and validate the implementation for any type of algorithm.
    
    Args:
        structure_info: Project structure information
        logger: Logger instance
        
    Returns:
        Tuple of (optimization_result, validation_result)
    """
    generate_code_dir = structure_info["generate_code_dir"]
    analysis_data = structure_info["analysis_data"]
    
    # Check if both optimization and validation results already exist
    opt_result_path = os.path.join(os.path.dirname(generate_code_dir), "optimization_result.txt")
    val_result_path = os.path.join(os.path.dirname(generate_code_dir), "validation_result.txt")
    
    opt_cached = False
    val_cached = False
    opt_result = None
    val_result = None
    
    # Check optimization cache
    if os.path.exists(opt_result_path):
        logger.info("Universal Optimizer: Found existing optimization result, loading from cache...")
        try:
            with open(opt_result_path, 'r', encoding='utf-8') as f:
                cached_opt = f.read()
            if cached_opt.strip():
                opt_result = cached_opt
                opt_cached = True
                logger.info("✅ Using cached optimization result")
            else:
                logger.warning("⚠️ Cached optimization result is empty, will regenerate...")
        except Exception as e:
            logger.warning(f"⚠️ Error reading optimization cache: {e}, will regenerate...")
    
    # Check validation cache
    if os.path.exists(val_result_path):
        logger.info("Universal Validation Specialist: Found existing validation result, loading from cache...")
        try:
            with open(val_result_path, 'r', encoding='utf-8') as f:
                cached_val = f.read()
            if cached_val.strip():
                val_result = cached_val
                val_cached = True
                logger.info("✅ Using cached validation result")
            else:
                logger.warning("⚠️ Cached validation result is empty, will regenerate...")
        except Exception as e:
            logger.warning(f"⚠️ Error reading validation cache: {e}, will regenerate...")
    
    # If both are cached, return them
    if opt_cached and val_cached:
        logger.info("✅ Using both cached optimization and validation results")
        return opt_result, val_result
    
    # Create universal optimization agent
    optimizer_agent = Agent(
        name="UniversalOptimizer",
        instruction=UNIVERSAL_OPTIMIZER_PROMPT + f"\n\nOptimize code in {generate_code_dir}",
        server_names=["code-generator"],
    )
    
    # Create universal validation agent
    validation_agent = Agent(
        name="UniversalValidationSpecialist",
        instruction=UNIVERSAL_VALIDATION_SPECIALIST_PROMPT + f"\n\nValidate implementation in {generate_code_dir}",
        server_names=["interpreter", "code-generator"],
    )
    
    # Run optimization and validation in parallel (only for uncached ones)
    async def optimize_code():
        if opt_cached:
            return opt_result
            
        async with optimizer_agent:
            logger.info("Universal Optimizer: Analyzing and optimizing code...")
            optimizer = await optimizer_agent.attach_llm(AnthropicAugmentedLLM)
            context = f"""
            Project Analysis: {json.dumps(analysis_data, indent=2)}
            
            Optimize this implementation considering the algorithm type and requirements.
            """
            result = await optimizer.generate_str(message=context)
            
            # Save optimization result
            with open(opt_result_path, 'w', encoding='utf-8') as f:
                f.write(result)
            logger.info("💾 Saved optimization result to cache")
            
            return result
    
    async def validate_code():
        if val_cached:
            return val_result
            
        async with validation_agent:
            logger.info("Universal Validation Specialist: Validating implementation...")
            validator = await validation_agent.attach_llm(AnthropicAugmentedLLM)
            context = f"""
            Project Analysis: {json.dumps(analysis_data, indent=2)}
            
            Validate this implementation against the paper's specifications and best practices.
            """
            result = await validator.generate_str(message=context)
            
            # Save validation result
            with open(val_result_path, 'w', encoding='utf-8') as f:
                f.write(result)
            logger.info("💾 Saved validation result to cache")
            
            return result
    
    opt_result, val_result = await asyncio.gather(optimize_code(), validate_code())
    
    return opt_result, val_result

async def execute_code_implementation(paper_dir: str, logger) -> Dict[str, Any]:
    """
    Execute the complete universal code implementation workflow.
    
    Args:
        paper_dir: Directory containing the paper and initial plan
        logger: Logger instance
        
    Returns:
        Dict containing all results from the implementation process
    """
    initial_plan_path = os.path.join(paper_dir, "initial_plan.txt")
    generate_code_dir = os.path.join(paper_dir, "generate_code")
    
    # Check if initial plan exists
    if not os.path.exists(initial_plan_path):
        raise FileNotFoundError(f"Initial plan not found at {initial_plan_path}")
    
    # Create generate_code directory if it doesn't exist
    os.makedirs(generate_code_dir, exist_ok=True)
    
    try:
        # Phase 0: Analyze the implementation plan
        logger.info("Phase 0: Analyzing implementation plan...")
        analysis_data = await analyze_implementation_plan(initial_plan_path, logger)
        
        # Phase 1: Create universal project structure
        logger.info("Phase 1: Creating universal project structure...")
        await asyncio.sleep(2)
        structure_info = await create_universal_project_structure(analysis_data, generate_code_dir, logger)
        
        # Phase 2: Implement modules hierarchically
        logger.info("Phase 2: Implementing modules hierarchically...")
        await asyncio.sleep(3)
        layer_results = await implement_hierarchical_codebase(structure_info, logger)
        
        # Phase 3: Integrate modules universally
        logger.info("Phase 3: Integrating modules...")
        await asyncio.sleep(3)
        integration_result = await integrate_universal_modules(layer_results, structure_info, logger)
        
        # Phase 4: Create tests and documentation
        logger.info("Phase 4: Creating tests and documentation...")
        await asyncio.sleep(3)
        test_result, doc_result = await create_universal_tests_and_documentation(integration_result, structure_info, logger)
        
        # Phase 5: Optimize and validate
        logger.info("Phase 5: Optimizing and validating implementation...")
        await asyncio.sleep(3)
        opt_result, val_result = await optimize_and_validate_universal(structure_info, logger)
        
        # Compile final results
        final_results = {
            "status": "success",
            "generate_code_dir": generate_code_dir,
            "project_info": analysis_data.get('project_info', {}),
            "phases_completed": {
                "plan_analysis": True,
                "structure_creation": True,
                "module_implementation": True,
                "integration": True,
                "testing_documentation": True,
                "optimization_validation": True
            },
            "modules_implemented": list(layer_results.keys()),
            "algorithm_type": analysis_data.get('project_info', {}).get('algorithm_type', 'unknown'),
            "summary": f"Successfully implemented {analysis_data.get('project_info', {}).get('main_algorithm', 'algorithm')} in {generate_code_dir}"
        }
        
        # Save final summary
        summary_path = os.path.join(paper_dir, "implementation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Universal code implementation completed successfully. Results saved to {paper_dir}")
        return final_results
        
    except Exception as e:
        logger.error(f"Error during universal code implementation: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "generate_code_dir": generate_code_dir,
            "phase_failed": "unknown"
        }

# Legacy function names for backward compatibility
create_project_structure = create_universal_project_structure
implement_core_modules = implement_hierarchical_codebase
integrate_modules = integrate_universal_modules
create_tests_and_documentation = create_universal_tests_and_documentation
optimize_and_validate = optimize_and_validate_universal 