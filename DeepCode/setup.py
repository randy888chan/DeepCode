from setuptools import setup, find_packages

setup(
    name="deepcode",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "rich",
        # 其他依赖可在此添加
    ],
    entry_points={
        "console_scripts": [
            "deepcode=deepcode_cli:cli_entry",
        ]
    },
) 