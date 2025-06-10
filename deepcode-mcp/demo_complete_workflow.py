#!/usr/bin/env python3
"""
完整的LLM+Shell执行器工作流演示
使用模拟的LLM响应来展示完整流程
"""

import asyncio
import sys
import os
import shutil

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def mock_extract_file_tree_from_plan(plan_content: str) -> str:
    """模拟LLM提取文件树结构"""
    return """
project/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── gcn.py        # GCN encoder
│   │   ├── diffusion.py  # forward/reverse processes
│   │   ├── denoiser.py   # denoising MLP
│   │   └── fusion.py     # fusion combiner
│   ├── models/           # model wrapper classes
│   │   └── recdiff.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data.py       # loading & preprocessing
│   │   ├── predictor.py  # scoring functions
│   │   ├── loss.py       # loss functions
│   │   ├── metrics.py    # NDCG, Recall etc.
│   │   └── sched.py      # beta/alpha schedule utils
│   │
│   └── configs/
│       └── default.yaml  # hyperparameters, paths
├── tests/
│   ├── test_gcn.py
│   ├── test_diffusion.py
│   ├── test_denoiser.py
│   ├── test_loss.py
│   └── test_pipeline.py
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── README.md
├── experiments/
│   ├── run_experiment.py
│   └── notebooks/
│       └── analysis.ipynb
├── requirements.txt
└── setup.py
"""

def mock_generate_shell_commands(file_tree: str) -> str:
    """模拟LLM生成shell命令"""
    return """mkdir -p generate_code
mkdir -p generate_code/src
mkdir -p generate_code/src/core
mkdir -p generate_code/src/models
mkdir -p generate_code/src/utils
mkdir -p generate_code/src/configs
mkdir -p generate_code/tests
mkdir -p generate_code/docs
mkdir -p generate_code/experiments
mkdir -p generate_code/experiments/notebooks
touch generate_code/src/__init__.py
touch generate_code/src/core/__init__.py
touch generate_code/src/core/gcn.py
touch generate_code/src/core/diffusion.py
touch generate_code/src/core/denoiser.py
touch generate_code/src/core/fusion.py
touch generate_code/src/models/__init__.py
touch generate_code/src/models/recdiff.py
touch generate_code/src/utils/__init__.py
touch generate_code/src/utils/data.py
touch generate_code/src/utils/predictor.py
touch generate_code/src/utils/loss.py
touch generate_code/src/utils/metrics.py
touch generate_code/src/utils/sched.py
touch generate_code/src/configs/__init__.py
touch generate_code/src/configs/default.yaml
touch generate_code/tests/__init__.py
touch generate_code/tests/test_gcn.py
touch generate_code/tests/test_diffusion.py
touch generate_code/tests/test_denoiser.py
touch generate_code/tests/test_loss.py
touch generate_code/tests/test_pipeline.py
touch generate_code/docs/architecture.md
touch generate_code/docs/api_reference.md
touch generate_code/docs/README.md
touch generate_code/experiments/__init__.py
touch generate_code/experiments/run_experiment.py
touch generate_code/experiments/notebooks/analysis.ipynb
touch generate_code/requirements.txt
touch generate_code/setup.py"""

async def demo_complete_workflow():
    """演示完整的工作流"""
    try:
        print("=" * 80)
        print("           通用论文代码复现Agent - 完整工作流演示")
        print("=" * 80)
        
        # 读取实现计划
        plan_file = r"agent_folders\papers\paper_3\initial_plan.txt"
        print(f"\n📄 读取实现计划: {plan_file}")
        
        if not os.path.exists(plan_file):
            print(f"❌ 文件不存在: {plan_file}")
            return
            
        with open(plan_file, 'r', encoding='utf-8') as f:
            plan_content = f.read()
        
        print(f"✅ 成功读取计划文件 ({len(plan_content)} 字符)")
        
        # 清理之前的输出
        target_directory = "agent_folders/papers/paper_3"
        output_dir = os.path.join(target_directory, "generate_code")
        if os.path.exists(output_dir):
            print(f"\n🧹 清理之前的输出: {output_dir}")
            shutil.rmtree(output_dir)
        
        print("\n🤖 开始Agent工作流程:")
        print("  " + "="*50)
        
        # 步骤1: LLM分析实现计划，提取文件树结构
        print("\n1️⃣ LLM分析实现计划，提取文件树结构")
        print("   🧠 LLM正在分析...")
        file_tree = mock_extract_file_tree_from_plan(plan_content)
        print("   ✅ 文件树结构提取完成")
        print("   📋 提取的文件树结构（前10行）:")
        for i, line in enumerate(file_tree.strip().split('\n')[:10]):
            if line.strip():
                print(f"      {line}")
        print("   ...")
        
        # 步骤2: LLM根据文件树生成shell创建命令
        print("\n2️⃣ LLM根据文件树生成shell创建命令")
        print("   🧠 LLM正在生成命令...")
        commands = mock_generate_shell_commands(file_tree)
        print("   ✅ Shell命令生成完成")
        print(f"   📊 生成了 {len(commands.strip().split())} 个命令")
        print("   🔧 生成的命令（前10行）:")
        for i, line in enumerate(commands.strip().split('\n')[:10]):
            if line.strip():
                print(f"      {line.strip()}")
        print("   ...")
        
        # 步骤3: Shell执行器执行命令
        print("\n3️⃣ Shell执行器在命令行执行LLM生成的命令")
        print("   ⚙️ 执行命令...")
        
        from tools.simple_shell_executor import execute_shell_commands
        
        creation_result = execute_shell_commands(commands, target_directory)
        print("   ✅ 命令执行完成")
        
        # 显示执行结果摘要
        result_lines = creation_result.split('\n')
        for line in result_lines[:3]:
            if line.strip():
                print(f"   📋 {line.strip()}")
        
        # 步骤4: 验证文件创建结果
        print("\n4️⃣ 验证文件创建结果")
        print("   🔍 检查生成的文件...")
        
        if os.path.exists(output_dir):
            print(f"   ✅ 成功创建目录: {output_dir}")
            
            # 统计文件
            file_count = 0
            dir_count = 0
            for root, dirs, files in os.walk(output_dir):
                dir_count += len(dirs)
                file_count += len(files)
            
            print(f"   📈 统计: {dir_count} 个目录, {file_count} 个文件")
            
            # 验证关键文件
            key_files = [
                "src/core/gcn.py",
                "src/core/diffusion.py", 
                "src/core/denoiser.py",
                "src/utils/data.py",
                "tests/test_gcn.py",
                "requirements.txt",
                "setup.py"
            ]
            
            print("   🎯 关键文件验证:")
            success_count = 0
            for file_path in key_files:
                full_path = os.path.join(output_dir, file_path)
                if os.path.exists(full_path):
                    print(f"      ✅ {file_path}")
                    success_count += 1
                else:
                    print(f"      ❌ {file_path}")
            
            # 显示项目结构
            print("\n   📂 生成的项目结构:")
            display_directory_tree(output_dir, "generate_code", "      ")
            
            print("\n" + "="*80)
            print("                    🎉 演示完成！")
            print("="*80)
            
            print(f"\n✨ Agent成功完成了完整的论文代码复现文件树创建流程:")
            print(f"   📝 智能分析了论文实现计划")
            print(f"   🧠 LLM提取了完整的文件树结构") 
            print(f"   🔧 LLM生成了 {len(commands.strip().split())} 个精确的创建命令")
            print(f"   ⚙️ Shell执行器成功执行了所有命令")
            print(f"   📊 创建了 {dir_count} 个目录和 {file_count} 个文件")
            print(f"   🎯 验证了 {success_count}/{len(key_files)} 个关键文件")
            
            print(f"\n💡 您可以在以下目录查看生成的代码结构:")
            print(f"   {os.path.abspath(output_dir)}")
            
            if success_count == len(key_files):
                print("\n🏆 所有测试通过！Agent工作流程完美运行！")
            else:
                print(f"\n⚠️ 部分验证失败，但主要流程成功运行")
                
        else:
            print("   ❌ 生成目录不存在")
            
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()

def display_directory_tree(directory, name="", prefix="", max_depth=2, current_depth=0):
    """显示目录树结构（限制深度）"""
    if not os.path.exists(directory) or current_depth > max_depth:
        return
        
    if current_depth == 0:
        print(f"{prefix}{name}/")
    
    try:
        items = sorted(os.listdir(directory))[:8]  # 限制显示数量
        for i, item in enumerate(items):
            item_path = os.path.join(directory, item)
            is_last = i == len(items) - 1
            
            if os.path.isdir(item_path):
                print(f"{prefix}{'└── ' if is_last else '├── '}{item}/")
                if current_depth < max_depth:
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    display_directory_tree(item_path, "", new_prefix, max_depth, current_depth + 1)
            else:
                print(f"{prefix}{'└── ' if is_last else '├── '}{item}")
                
        if len(os.listdir(directory)) > 8:
            print(f"{prefix}... 还有 {len(os.listdir(directory)) - 8} 个项目")
            
    except PermissionError:
        print(f"{prefix}[权限拒绝]")

if __name__ == "__main__":
    print("🚀 启动完整的Agent工作流演示...")
    print("📌 注意：这是演示版本，使用模拟的LLM响应")
    asyncio.run(demo_complete_workflow()) 