from setuptools import setup, find_packages

setup(
    name="rice",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "gym>=0.17.0",
        "pandas",
        "matplotlib",
        "tensorboard"
    ],
    python_requires=">=3.7",
)