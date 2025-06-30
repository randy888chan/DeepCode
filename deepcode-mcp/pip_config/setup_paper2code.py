import setuptools
from pathlib import Path


# Reading the long description from README.md
def read_long_description():
    try:
        return Path("README.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "A tool for converting research papers to executable code implementations."
        )


# Retrieving metadata from __init__.py
def retrieve_metadata():
    vars2find = ["__author__", "__version__", "__url__"]
    vars2readme = {}
    try:
        with open("./__init__.py") as f:
            for line in f.readlines():
                for v in vars2find:
                    if line.startswith(v):
                        line = (
                            line.replace(" ", "")
                            .replace('"', "")
                            .replace("'", "")
                            .strip()
                        )
                        vars2readme[v] = line.split("=")[1]
    except FileNotFoundError:
        raise FileNotFoundError("Metadata file './__init__.py' not found.")

    # Checking if all required variables are found
    missing_vars = [v for v in vars2find if v not in vars2readme]
    if missing_vars:
        raise ValueError(
            f"Missing required metadata variables in __init__.py: {missing_vars}"
        )

    return vars2readme


# Reading dependencies from requirements.txt
def read_requirements():
    deps = []
    try:
        with open("./requirements.txt") as f:
            deps = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(
            "Warning: 'requirements.txt' not found. No dependencies will be installed."
        )
    return deps


metadata = retrieve_metadata()
long_description = read_long_description()
requirements = read_requirements()

setuptools.setup(
    name="paper2code",
    url=metadata["__url__"],
    version=metadata["__version__"],
    author=metadata["__author__"],
    description="A comprehensive tool for analyzing research papers and generating executable code implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        exclude=("tests*", "docs*", "agent_folders*", "logs*", "test_files*", "schema*")
    ),  # Automatically find packages
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Markup",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    include_package_data=True,  # Includes non-code files from MANIFEST.in
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
        "prompts": ["*.py"],
        "schema": ["*.json", "*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "paper2code=paper2code.main:main",
        ],
    },
    project_urls={  # Additional project metadata
        "Documentation": metadata.get("__url__", ""),
        "Source": metadata.get("__url__", ""),
        "Tracker": f"{metadata.get('__url__', '')}/issues"
        if metadata.get("__url__")
        else "",
        "Bug Reports": f"{metadata.get('__url__', '')}/issues"
        if metadata.get("__url__")
        else "",
        "Homepage": metadata.get("__url__", ""),
    },
    keywords=[
        "research",
        "paper",
        "code-generation",
        "automation",
        "academic",
        "machine-learning",
        "AI",
        "natural-language-processing",
    ],
)
