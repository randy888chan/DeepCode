"""
Workflows package for paper-to-code implementation.
"""

from .initial_workflows import (
    run_paper_analyzer,
    run_paper_downloader,
    paper_code_analyzer,
    github_repo_download,
    paper_reference_analyzer,
    paper_code_preparation
)

from .code_implementation_workflow import (
    analyze_implementation_plan,
    create_universal_project_structure,
    implement_dynamic_modules,
    integrate_universal_modules,
    create_universal_tests_and_documentation,
    optimize_and_validate_universal,
    execute_code_implementation,
    # Legacy function names for backward compatibility
    create_project_structure,
    implement_core_modules,
    integrate_modules,
    create_tests_and_documentation,
    optimize_and_validate
)

__all__ = [
    # Initial workflows
    'run_paper_analyzer',
    'run_paper_downloader',
    'paper_code_analyzer',
    'github_repo_download',
    'paper_reference_analyzer',
    'paper_code_preparation',
    # Universal code implementation workflows
    'analyze_implementation_plan',
    'create_universal_project_structure',
    'implement_dynamic_modules',
    'integrate_universal_modules',
    'create_universal_tests_and_documentation',
    'optimize_and_validate_universal',
    'execute_code_implementation',
    # Legacy function names (backward compatibility)
    'create_project_structure',
    'implement_core_modules',
    'integrate_modules',
    'create_tests_and_documentation',
    'optimize_and_validate'
] 