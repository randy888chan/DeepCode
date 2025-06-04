from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
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
    UNIVERSAL_VALIDATION_SPECIALIST_PROMPT
)

async def analyze_implementation_plan(initial_plan_path: str, logger) -> Dict[str, Any]:
    """
    Analyze the implementation plan to extract structured information about the project.
    
    Args:
        initial_plan_path: Path to the initial plan file
        logger: Logger instance
        
    Returns:
        Dict containing structured analysis of the implementation plan
    """
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
        analysis_result = await analyzer.generate_str(
            message=f"Analyze this implementation plan and extract structured information:\n\n{plan_content}"
        )
        
        # Parse the JSON response
        try:
            # Extract JSON from the response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', analysis_result, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group(1))
            else:
                # Fallback: try to parse the entire response as JSON
                analysis_data = json.loads(analysis_result)
            
            # Add the original plan content for reference
            analysis_data['original_plan'] = plan_content
            
            return analysis_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan analysis JSON: {e}")
            # Return a fallback structure
            return {
                "project_info": {
                    "name": "extracted_project",
                    "main_algorithm": "unknown",
                    "algorithm_type": "other",
                    "domain": "other"
                },
                "technology_stack": {
                    "language": "python",
                    "version": "3.8+",
                    "frameworks": ["torch", "numpy"],
                    "dependencies": ["numpy", "torch"],
                    "hardware_requirements": "cpu"
                },
                "analysis_raw": analysis_result,
                "original_plan": plan_content
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
    structure_agent = Agent(
        name="UniversalStructureGenerator",
        instruction=UNIVERSAL_STRUCTURE_GENERATOR_PROMPT + f"\n\nGenerate code in directory: {generate_code_dir}",
        server_names=["filesystem", "code-generator"],
    )
    
    async with structure_agent:
        logger.info("Universal Structure Generator: Creating project structure...")
        generator = await structure_agent.attach_llm(AnthropicAugmentedLLM)
        
        # Prepare the analysis data as context
        analysis_context = json.dumps(analysis_data, indent=2)
        
        result = await generator.generate_str(
            message=f"Create universal project structure based on this analysis:\n\n{analysis_context}"
        )
        
        # Save structure generation result
        structure_result_path = os.path.join(os.path.dirname(generate_code_dir), "structure_generation_result.txt")
        with open(structure_result_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
        return {
            "structure_result": result,
            "generate_code_dir": generate_code_dir,
            "analysis_data": analysis_data
        }

async def implement_dynamic_modules(structure_info: Dict[str, str], logger) -> Dict[str, str]:
    """
    Dynamically implement modules based on the analyzed project structure.
    
    Args:
        structure_info: Information about the created project structure
        logger: Logger instance
        
    Returns:
        Dict containing implementation results for each module
    """
    generate_code_dir = structure_info["generate_code_dir"]
    analysis_data = structure_info["analysis_data"]
    
    # Extract modules from analysis, with fallback to generic modules
    modules_to_implement = []
    
    if 'project_structure' in analysis_data and 'core_modules' in analysis_data['project_structure']:
        # Use modules identified in the analysis
        for module_info in analysis_data['project_structure']['core_modules']:
            modules_to_implement.append((
                module_info.get('name', 'unknown_module'),
                module_info.get('file_name', 'module.py'),
                module_info.get('purpose', 'Core module implementation'),
                module_info.get('priority', 1)
            ))
    
    # If no modules found in analysis, use generic approach based on algorithm type
    if not modules_to_implement:
        algorithm_type = analysis_data.get('project_info', {}).get('algorithm_type', 'other')
        
        if algorithm_type in ['deep_learning', 'computer_vision', 'nlp']:
            modules_to_implement = [
                ("data_module", "data.py", "Data loading and preprocessing", 1),
                ("model_module", "model.py", "Main model implementation", 2),
                ("training_module", "training.py", "Training pipeline", 3),
                ("evaluation_module", "evaluation.py", "Evaluation and metrics", 4),
                ("utils_module", "utils.py", "Utility functions", 1)
            ]
        elif algorithm_type == 'traditional_ml':
            modules_to_implement = [
                ("data_module", "data.py", "Data processing and features", 1),
                ("algorithm_module", "algorithm.py", "Core algorithm implementation", 2),
                ("evaluation_module", "evaluation.py", "Model evaluation", 3),
                ("utils_module", "utils.py", "Utility functions", 1)
            ]
        elif algorithm_type == 'optimization':
            modules_to_implement = [
                ("problem_module", "problem.py", "Problem definition", 1),
                ("solver_module", "solver.py", "Optimization solver", 2),
                ("evaluation_module", "evaluation.py", "Solution evaluation", 3),
                ("utils_module", "utils.py", "Utility functions", 1)
            ]
        else:
            # Generic fallback
            modules_to_implement = [
                ("core_module", "core.py", "Core algorithm implementation", 2),
                ("data_module", "data.py", "Data handling", 1),
                ("utils_module", "utils.py", "Utility functions", 1),
                ("evaluation_module", "evaluation.py", "Evaluation methods", 3)
            ]
    
    # Sort modules by priority
    modules_to_implement.sort(key=lambda x: x[3])
    
    # Create agents for each module
    module_agents = []
    for module_name, file_name, description, priority in modules_to_implement:
        agent = Agent(
            name=f"{module_name}_implementer",
            instruction=UNIVERSAL_MODULE_IMPLEMENTER_PROMPT + f"\n\nImplement {description} in {generate_code_dir}/src/{file_name}",
            server_names=["filesystem", "code-generator"],
        )
        module_agents.append((agent, module_name, file_name, description))
    
    # Implement modules in parallel (grouped by priority for dependencies)
    results = {}
    
    # Group modules by priority
    priority_groups = {}
    for agent_info in module_agents:
        module_name = agent_info[1]
        # Find priority for this module
        priority = next((p for _, _, _, p in modules_to_implement if _ == module_name), 1)
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append(agent_info)
    
    # Implement in order of priority, with parallelization within each priority group
    for priority in sorted(priority_groups.keys()):
        logger.info(f"Implementing priority {priority} modules...")
        
        async def implement_module(agent, module_name, file_name, description):
            async with agent:
                logger.info(f"Implementing {module_name}...")
                implementer = await agent.attach_llm(AnthropicAugmentedLLM)
                
                # Prepare context
                context = f"""
                Analysis Data: {json.dumps(analysis_data, indent=2)}
                
                Module to implement: {module_name}
                File: {file_name}
                Description: {description}
                
                Please implement this module based on the paper's methodology and the implementation plan.
                """
                
                result = await implementer.generate_str(message=context)
                return module_name, result
        
        # Execute modules in this priority group in parallel
        tasks = []
        for agent_info in priority_groups[priority]:
            task = implement_module(*agent_info)
            tasks.append(task)
        
        if tasks:
            module_results = await asyncio.gather(*tasks)
            
            for module_name, result in module_results:
                results[module_name] = result
                # Save each module's implementation result
                result_path = os.path.join(os.path.dirname(generate_code_dir), f"{module_name}_result.txt")
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write(result)
        
        # Small delay between priority groups
        if priority < max(priority_groups.keys()):
            await asyncio.sleep(2)
    
    return results

async def integrate_universal_modules(module_results: Dict[str, str], structure_info: Dict[str, str], logger) -> str:
    """
    Integrate all modules into a cohesive system.
    
    Args:
        module_results: Results from module implementations
        structure_info: Project structure information
        logger: Logger instance
        
    Returns:
        Integration result
    """
    generate_code_dir = structure_info["generate_code_dir"]
    analysis_data = structure_info["analysis_data"]
    
    integration_agent = Agent(
        name="UniversalIntegrationSpecialist",
        instruction=UNIVERSAL_INTEGRATION_SPECIALIST_PROMPT + f"\n\nIntegrate modules in {generate_code_dir}",
        server_names=["filesystem", "code-generator"],
    )
    
    async with integration_agent:
        logger.info("Universal Integration Specialist: Connecting modules...")
        integrator = await integration_agent.attach_llm(AnthropicAugmentedLLM)
        
        # Prepare integration context
        integration_context = f"""
        Project Analysis: {json.dumps(analysis_data, indent=2)}
        
        Implemented Modules: {json.dumps(module_results, indent=2)}
        
        Create a cohesive system that integrates all modules. 
        Focus on the main algorithm class and training/inference pipelines.
        Consider the algorithm type: {analysis_data.get('project_info', {}).get('algorithm_type', 'unknown')}
        """
        
        result = await integrator.generate_str(message=integration_context)
        
        # Save integration result
        integration_result_path = os.path.join(os.path.dirname(generate_code_dir), "integration_result.txt")
        with open(integration_result_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
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
    
    # Create universal testing agent
    testing_agent = Agent(
        name="UniversalTestingEngineer",
        instruction=UNIVERSAL_TESTING_ENGINEER_PROMPT + f"\n\nCreate tests in {generate_code_dir}/tests/",
        server_names=["filesystem", "code-generator"],
    )
    
    # Create universal documentation agent
    documentation_agent = Agent(
        name="UniversalDocumentationWriter",
        instruction=UNIVERSAL_DOCUMENTATION_WRITER_PROMPT + f"\n\nCreate documentation in {generate_code_dir}/docs/",
        server_names=["filesystem", "code-generator"],
    )
    
    # Run testing and documentation in parallel
    async def create_tests():
        async with testing_agent:
            logger.info("Universal Testing Engineer: Creating test suite...")
            tester = await testing_agent.attach_llm(AnthropicAugmentedLLM)
            context = f"""
            Project Analysis: {json.dumps(analysis_data, indent=2)}
            Integration Result: {integration_result}
            
            Create comprehensive tests appropriate for this type of algorithm and implementation.
            """
            return await tester.generate_str(message=context)
    
    async def create_docs():
        async with documentation_agent:
            logger.info("Universal Documentation Writer: Creating documentation...")
            writer = await documentation_agent.attach_llm(AnthropicAugmentedLLM)
            context = f"""
            Project Analysis: {json.dumps(analysis_data, indent=2)}
            Integration Result: {integration_result}
            
            Create comprehensive documentation for this research implementation.
            """
            return await writer.generate_str(message=context)
    
    test_result, doc_result = await asyncio.gather(create_tests(), create_docs())
    
    # Save results
    test_result_path = os.path.join(os.path.dirname(generate_code_dir), "test_creation_result.txt")
    doc_result_path = os.path.join(os.path.dirname(generate_code_dir), "documentation_result.txt")
    
    with open(test_result_path, 'w', encoding='utf-8') as f:
        f.write(test_result)
    
    with open(doc_result_path, 'w', encoding='utf-8') as f:
        f.write(doc_result)
    
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
    
    # Create universal optimization agent
    optimizer_agent = Agent(
        name="UniversalOptimizer",
        instruction=UNIVERSAL_OPTIMIZER_PROMPT + f"\n\nOptimize code in {generate_code_dir}",
        server_names=["filesystem", "interpreter", "code-generator"],
    )
    
    # Create universal validation agent
    validation_agent = Agent(
        name="UniversalValidationSpecialist",
        instruction=UNIVERSAL_VALIDATION_SPECIALIST_PROMPT + f"\n\nValidate implementation in {generate_code_dir}",
        server_names=["filesystem", "interpreter", "code-generator"],
    )
    
    # Run optimization and validation in parallel
    async def optimize_code():
        async with optimizer_agent:
            logger.info("Universal Optimizer: Analyzing and optimizing code...")
            optimizer = await optimizer_agent.attach_llm(AnthropicAugmentedLLM)
            context = f"""
            Project Analysis: {json.dumps(analysis_data, indent=2)}
            
            Optimize this implementation considering the algorithm type and requirements.
            """
            return await optimizer.generate_str(message=context)
    
    async def validate_code():
        async with validation_agent:
            logger.info("Universal Validation Specialist: Validating implementation...")
            validator = await validation_agent.attach_llm(AnthropicAugmentedLLM)
            context = f"""
            Project Analysis: {json.dumps(analysis_data, indent=2)}
            
            Validate this implementation against the paper's specifications and best practices.
            """
            return await validator.generate_str(message=context)
    
    opt_result, val_result = await asyncio.gather(optimize_code(), validate_code())
    
    # Save results
    opt_result_path = os.path.join(os.path.dirname(generate_code_dir), "optimization_result.txt")
    val_result_path = os.path.join(os.path.dirname(generate_code_dir), "validation_result.txt")
    
    with open(opt_result_path, 'w', encoding='utf-8') as f:
        f.write(opt_result)
    
    with open(val_result_path, 'w', encoding='utf-8') as f:
        f.write(val_result)
    
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
        
        # Save analysis result
        analysis_path = os.path.join(paper_dir, "plan_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2)
        
        # Phase 1: Create universal project structure
        logger.info("Phase 1: Creating universal project structure...")
        await asyncio.sleep(2)
        structure_info = await create_universal_project_structure(analysis_data, generate_code_dir, logger)
        
        # Phase 2: Implement modules dynamically
        logger.info("Phase 2: Implementing modules dynamically...")
        await asyncio.sleep(3)
        module_results = await implement_dynamic_modules(structure_info, logger)
        
        # Phase 3: Integrate modules universally
        logger.info("Phase 3: Integrating modules...")
        await asyncio.sleep(3)
        integration_result = await integrate_universal_modules(module_results, structure_info, logger)
        
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
            "modules_implemented": list(module_results.keys()),
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
implement_core_modules = implement_dynamic_modules
integrate_modules = integrate_universal_modules
create_tests_and_documentation = create_universal_tests_and_documentation
optimize_and_validate = optimize_and_validate_universal 