import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # 禁止生成.pyc文件

import asyncio
import time
import json

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from workflows.initial_workflows import (
    paper_code_preparation,
    github_repo_download,
)
from utils.file_processor import FileProcessor

# Initialize the MCP application
app = MCPApp(name="paper_to_code")

async def main():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
        
        file_input_info = """Here's a structured list of the top 5 most relevant GitHub repositories for the referenced papers based on the specified criteria:

```json
{
    "repositories": [
        {
            "reference_number": "1",
            "paper_title": "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation",
            "original_reference": "[8] X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. In SIGIR, pages 639-648, 2020.",
            "github_url": "https://github.com/kuandeng/LightGCN",
            "repo_type": "official",
            "verification": {
                "is_verified": true,
                "last_updated": "2023",
                "stars": "1.2k+",
                "notes": "High quality implementation with comprehensive documentation and experimental code"
            }
        },
        {
            "reference_number": "2",
            "paper_title": "Neural Graph Collaborative Filtering",
            "original_reference": "[30] X. Wang, X. He, M. Wang, F. Feng, and T.-S. Chua. Neural Graph Collaborative Filtering. In SIGIR, pages 165-174, 2019.",
            "github_url": "https://github.com/xiangwang1223/neural_graph_collaborative_filtering",
            "repo_type": "official",
            "verification": {
                "is_verified": true,
                "last_updated": "2023",
                "stars": "900+",
                "notes": "Well-structured implementation with detailed documentation"
            }
        },
        {
            "reference_number": "3",
            "paper_title": "Graph Neural Networks for Social Recommendation",
            "original_reference": "[7] W. Fan, Y. Ma, Q. Li, Y. He, E. Zhao, J. Tang, and D. Yin. Graph Neural Networks for Social Recommendation. In WWW, pages 417-426, 2019.",
            "github_url": "https://github.com/wenqifan03/GraphRec-WWW19",
            "repo_type": "official",
            "verification": {
                "is_verified": true,
                "last_updated": "2023",
                "stars": "400+",
                "notes": "Clean implementation with social recommendation components"
            }
        },
        {
            "reference_number": "4",
            "paper_title": "Self-supervised Multi-channel Hypergraph Convolutional Network for Social Recommendation",
            "original_reference": "[45] J. Yu, H. Yin, J. Li, Q. Wang, N. Q. V. Hung, and X. Zhang. Self-supervised Multi-channel Hypergraph Convolutional Network for Social Recommendation. In WWW, pages 413-424, 2021.",
            "github_url": "https://github.com/librahu/MHCN",
            "repo_type": "official",
            "verification": {
                "is_verified": true,
                "last_updated": "2023",
                "stars": "200+",
                "notes": "Well-organized implementation with self-supervised components"
            }
        },
        {
            "reference_number": "5",
            "paper_title": "SGL: Self-supervised Graph Learning for Recommendation",
            "original_reference": "[36] J. Wu, X. Wang, F. Feng, X. He, L. Chen, J. Lian, and X. Xie. SGL: Self-supervised Graph Learning for Recommendation. In WWW, pages 800-801, 2021.",
            "github_url": "https://github.com/wujcan/SGL-Torch",
            "repo_type": "official",
            "verification": {
                "is_verified": true,
                "last_updated": "2023",
                "stars": "300+",
                "notes": "High-quality implementation with comprehensive training pipeline"
            }
        }
    ],
    "not_found": []
}
```

These repositories are official implementations by the authors and are well-maintained, with good documentation, making them useful for further exploration and replication of the methods described in the papers.
Please download the repositories to the following path: ./agent_folders/papers/paper_1/code_base
                            """         
        paper_dir = "./agent_folders/papers/paper_1"
        github_download_agent = Agent(
        name="GithubDownloadAgent",
        instruction="download github repo.",
        server_names=["filesystem", "github-downloader"],
    )
        async with github_download_agent:
            logger.info("GitHub downloader: Downloading repositories...")
            downloader = await github_download_agent.attach_llm(AnthropicAugmentedLLM)
            await downloader.generate_str(message=file_input_info)

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Total run time: {end - start:.2f}s")
    os.system('find . -type d -name "__pycache__" -exec rm -r {} +')