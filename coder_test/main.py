# main.py
import asyncio
import os
from dotenv import load_dotenv
from code_agent import CodeAgent

async def interactive_mode():
    """交互式模式 - 让用户输入需求并生成代码"""
    
    # 加载环境变量
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not openai_api_key:
        print("❌ 请在 .env 文件中设置 OPENAI_API_KEY")
        return
    
    # 创建代码代理
    agent = CodeAgent(openai_api_key, github_token)
    
    print("🤖 欢迎使用 AI 代码生成器！")
    print("输入 'quit' 退出程序")
    print("=" * 50)
    
    while True:
        try:
            # 获取用户输入
            print("\n请描述你想要生成的代码:")
            description = input("💭 需求描述: ").strip()
            
            if description.lower() == 'quit':
                break
            
            if not description:
                continue
                
            language = input("🔤 编程语言 (Python/JavaScript/Java/等): ").strip() or "Python"
            
            use_reference = input("🔗 是否使用参考仓库？(y/n): ").strip().lower() == 'y'
            
            reference_repo = None
            reference_files = None
            
            if use_reference:
                reference_repo = input("📚 参考仓库 (格式: owner/repo): ").strip()
                file_paths = input("📄 参考文件路径 (用逗号分隔): ").strip()
                if file_paths:
                    reference_files = [f.strip() for f in file_paths.split(",")]
            
            print("\n🔄 正在生成代码...")
            
            # 生成代码
            generated_code = await agent.analyze_and_generate(
                description=description,
                language=language,
                reference_repo=reference_repo,
                reference_files=reference_files
            )
            
            print("\n✅ 代码生成完成!")
            print("=" * 50)
            print(generated_code)
            print("=" * 50)
            
            # 询问是否需要改进
            improve = input("\n🔧 是否需要改进代码？(y/n): ").strip().lower() == 'y'
            if improve:
                feedback = input("💡 请提供改进建议: ").strip()
                if feedback:
                    print("\n🔄 正在改进代码...")
                    improved_code = await agent.code_generator.refine_code(generated_code, feedback)
                    print("\n✅ 改进后的代码:")
                    print("=" * 50)
                    print(improved_code)
                    print("=" * 50)
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
    
    print("\n👋 感谢使用！")

if __name__ == "__main__":
    asyncio.run(interactive_mode())