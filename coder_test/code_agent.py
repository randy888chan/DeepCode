# code_agent.py
import asyncio
import os
from dotenv import load_dotenv
from github_reader import GitHubReader
from code_generator import CodeGenerator, CodeRequest
from typing import List

class CodeAgent:
    """主代码代理 - 整合 GitHub 代码读取和 AI 代码生成功能"""
    
    def __init__(self, openai_api_key: str, github_token: str = None):
        """
        初始化代码代理
        :param openai_api_key: OpenAI API 密钥
        :param github_token: GitHub 访问令牌
        """
        self.github_reader = GitHubReader(github_token)
        self.code_generator = CodeGenerator(openai_api_key)
    
    async def analyze_and_generate(
        self, 
        description: str, 
        language: str,
        reference_repo: str = None,
        reference_files: List[str] = None
    ) -> str:
        """
        分析参考仓库并生成代码
        :param description: 代码需求描述
        :param language: 目标编程语言
        :param reference_repo: 参考仓库 (格式: "owner/repo")
        :param reference_files: 参考文件路径列表
        :return: 生成的代码
        """
        context = ""
        
        # 如果提供了参考仓库，读取相关文件作为上下文
        if reference_repo and reference_files:
            print(f"📖 正在分析参考仓库: {reference_repo}")
            
            try:
                owner, repo = reference_repo.split("/")
                
                # 获取仓库基本信息
                repo_info = await self.github_reader.get_repository_info(owner, repo)
                print(f"仓库信息: {repo_info['name']} - {repo_info['description']}")
                
                # 读取参考文件
                context_parts = [f"参考仓库: {reference_repo}"]
                context_parts.append(f"主要语言: {repo_info['language']}")
                
                for file_path in reference_files:
                    try:
                        print(f"📄 正在读取文件: {file_path}")
                        file_content = await self.github_reader.read_file(owner, repo, file_path)
                        context_parts.append(f"\n--- 文件: {file_path} ---")
                        context_parts.append(file_content[:1000])  # 限制长度避免超出 token 限制
                        
                    except Exception as e:
                        print(f"⚠️  读取文件 {file_path} 失败: {e}")
                        continue
                
                context = "\n".join(context_parts)
                
            except Exception as e:
                print(f"⚠️  分析仓库失败: {e}")
        
        # 创建代码生成请求
        request = CodeRequest(
            description=description,
            language=language,
            context=context,
            style_guide=f"遵循 {language} 最佳实践"
        )
        
        print("🤖 正在生成代码...")
        
        # 生成代码
        generated_code = await self.code_generator.generate_code(request)
        return generated_code
    
    async def explore_repository(self, owner: str, repo: str, path: str = "") -> None:
        """
        探索仓库结构（用于了解可用的文件）
        :param owner: 仓库所有者
        :param repo: 仓库名称
        :param path: 路径（默认为根目录）
        """
        try:
            print(f"📁 探索仓库 {owner}/{repo} 的目录: /{path}")
            files = await self.github_reader.list_files(owner, repo, path)
            
            for file_info in files[:10]:  # 只显示前 10 个文件
                icon = "📁" if file_info["type"] == "dir" else "📄"
                size = f"({file_info['size']} bytes)" if file_info["type"] == "file" else ""
                print(f"  {icon} {file_info['name']} {size}")
                
            if len(files) > 10:
                print(f"  ... 还有 {len(files) - 10} 个文件/目录")
                
        except Exception as e:
            print(f"❌ 探索仓库失败: {e}")

# 使用示例函数
async def example_usage():
    """演示如何使用代码代理"""
    
    # 加载环境变量
    load_dotenv()
    
    # 从环境变量获取 API 密钥
    openai_api_key = os.getenv("OPENAI_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not openai_api_key:
        print("❌ 请在 .env 文件中设置 OPENAI_API_KEY")
        return
    
    # 创建代码代理
    agent = CodeAgent(openai_api_key, github_token)
    
    print("🚀 代码代理启动成功！")
    print("=" * 50)
    
    # 示例 1: 探索一个流行的仓库结构
    print("\n📋 示例 1: 探索 仓库结构")
    await agent.explore_repository("HKUDS", "RecDiff", "")
    
    print("\n" + "=" * 50)
    
    # 示例 2: 基于 FastAPI 示例生成新的 API 代码
    print("\n📋 示例 2: 生成基于github的用户认证 API")
    
    generated_code = await agent.analyze_and_generate(
        description="Please rewrite the code to be more readable and efficient",
        language="Python",
        reference_repo="HKUDS/RecDiff",
        reference_files=[
            "DataHander.py",  
        ]
    )
    
    print("\n🎉 生成的代码:")
    print("-" * 30)
    print(generated_code)
    
    print("\n" + "=" * 50)
    
    # 示例 3: 生成前端组件（基于 React 仓库）
    print("\n📋 示例 3: 生成简单的 React 组件")
    
    simple_code = await agent.analyze_and_generate(
        description="创建一个简单的用户卡片组件，显示用户头像、姓名和邮箱",
        language="JavaScript",
        # 不使用参考仓库，直接生成
    )
    
    print("\n🎉 生成的 React 组件:")
    print("-" * 30)
    print(simple_code)

if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())