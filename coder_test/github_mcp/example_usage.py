# usage_examples.py
import asyncio
import json
from repository_client import RepositoryCodeReader

async def analyze_popular_repositories():
    """Analyze some popular repositories"""
    
    reader = RepositoryCodeReader()
    
    repositories = [
        ("microsoft", "vscode"),
        ("facebook", "react"),
        ("tensorflow", "tensorflow"),
        ("python", "cpython")
    ]
    
    for owner, repo in repositories:
        try:
            print(f"\n{'='*60}")
            print(f"Analyzing {owner}/{repo}")
            print(f"{'='*60}")
            
            # Get summary first
            summary = await reader.get_repo_summary(owner, repo)
            print(f"📊 Total files: {summary['summary']['total_files']}")
            print(f"📈 Total lines: {summary['summary']['total_lines']:,}")
            print(f"💾 Total size: {summary['summary']['total_size_bytes']:,} bytes")
            
            print("\n🔍 Top file types:")
            sorted_types = sorted(
                summary['summary']['file_types'].items(), 
                key=lambda x: x[1]['count'], 
                reverse=True
            )
            
            for ext, info in sorted_types[:5]:
                print(f"  {ext:12} {info['count']:6} files  {info['lines']:8,} lines")
            
            # Read specific file types
            if '.py' in [ext for ext, _ in sorted_types]:
                python_files = await reader.read_by_extension(owner, repo, [".py"])
                print(f"\n🐍 Python files: {python_files['matched_files']}")
            
        except Exception as e:
            print(f"❌ Error analyzing {owner}/{repo}: {e}")
            continue

async def deep_dive_small_repo():
    """Do a complete analysis of a smaller repository"""
    
    reader = RepositoryCodeReader()
    
    # Use a smaller repository for complete analysis
    owner, repo = "octocat", "Hello-World"
    
    print(f"🔍 Complete analysis of {owner}/{repo}")
    print("="*60)
    
    # Step 1: Complete repository read
    complete_data = await reader.read_entire_repo(owner, repo)
    
    print(f"📚 Repository contains {complete_data['total_files']} files")
    
    # Step 2: Show all files with content
    for file_path, file_info in complete_data['files'].items():
        print(f"\n📄 {file_path}")
        print(f"   📏 {file_info['lines']} lines, {file_info['size']} bytes")
        print("   📝 Content:")
        print("   " + "─" * 50)
        
        # Show content with line numbers
        lines = file_info['content'].splitlines()
        for i, line in enumerate(lines[:20], 1):  # Show first 20 lines
            print(f"   {i:3} | {line}")
        
        if len(lines) > 20:
            print(f"   ... ({len(lines) - 20} more lines)")
        print()

async def extract_specific_patterns():
    """Extract specific code patterns from repositories"""
    
    reader = RepositoryCodeReader()
    
    # Look for FastAPI patterns
    print("🔍 Looking for FastAPI patterns...")
    
    try:
        fastapi_files = await reader.read_by_extension("tiangolo", "fastapi", [".py"])
        
        print(f"Found {fastapi_files['matched_files']} Python files in FastAPI repo")
        
        # Look for specific patterns
        patterns = {
            "FastAPI()": [],
            "@app.get": [],
            "@app.post": [],
            "async def": [],
            "Depends(": []
        }
        
        for file_path, file_info in fastapi_files['files'].items():
            content = file_info['content']
            
            for pattern in patterns:
                if pattern in content:
                    patterns[pattern].append(file_path)
        
        print("\n📊 Pattern analysis:")
        for pattern, files in patterns.items():
            print(f"  {pattern:15} found in {len(files)} files")
            for file_path in files[:3]:  # Show first 3 files
                print(f"    📄 {file_path}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more files")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Choose an example to run:")
    print("1. Analyze popular repositories")
    print("2. Deep dive into a small repository")
    print("3. Extract specific code patterns")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(analyze_popular_repositories())
    elif choice == "2":
        asyncio.run(deep_dive_small_repo())
    elif choice == "3":
        asyncio.run(extract_specific_patterns())
    else:
        print("Invalid choice")