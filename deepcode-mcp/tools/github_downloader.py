import os
import json
import subprocess
import shutil
from typing import Dict, List, Optional

class GitHubDownloader:
    def __init__(self, base_dir: str):
        """
        初始化 GitHub 下载器
        
        Args:
            base_dir (str): 基础目录路径，用于存储下载的代码
        """
        self.base_dir = base_dir
        
    def parse_github_urls(self, content: str) -> List[Dict[str, str]]:
        """
        从文本内容中解析 GitHub 仓库信息
        
        Args:
            content (str): 包含 GitHub 仓库信息的文本内容
            
        Returns:
            List[Dict[str, str]]: 包含仓库信息的列表
        """
        try:
            repos = []
            lines = content.split('\n')
            ref_num = 1
            
            for line in lines:
                # 查找包含 github.com 的行
                if 'github.com' in line:
                    # 提取 URL（假设URL包含在文本中）
                    words = line.split()
                    for word in words:
                        if 'github.com' in word:
                            # 清理 URL（移除可能的标点符号）
                            url = word.strip(',.()[]"\'')
                            # 确保 URL 以 https:// 开头
                            if not url.startswith('http'):
                                url = 'https://' + url
                            
                            repos.append({
                                "url": url,
                                "ref_num": str(ref_num),
                                "title": f"Reference Implementation {ref_num}"
                            })
                            ref_num += 1
                            break
            
            return repos
        except Exception as e:
            print(f"Error parsing content: {str(e)}")
            return []
    
    def clean_github_url(self, url: str) -> str:
        """
        清理和规范化 GitHub URL
        
        Args:
            url (str): 原始 GitHub URL
            
        Returns:
            str: 清理后的 URL
        """
        # 移除 URL 开头的 @ 符号
        url = url.lstrip('@')
        
        # 确保 URL 以 https:// 或 http:// 开头
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        # 移除 URL 末尾的斜杠
        url = url.rstrip('/')
        
        # 如果 URL 不以 .git 结尾，添加 .git
        if not url.endswith('.git'):
            url = url + '.git'
            
        return url

    def clone_repository(self, repo_url: str, target_dir: str) -> bool:
        """
        克隆 GitHub 仓库到指定目录
        
        Args:
            repo_url (str): GitHub 仓库 URL
            target_dir (str): 目标目录
            
        Returns:
            bool: 是否成功克隆
        """
        try:
            # 清理和规范化 URL
            cleaned_url = self.clean_github_url(repo_url)
            
            # 如果目录已存在，先删除
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            
            print(f"Cloning from: {cleaned_url}")
            # 克隆仓库
            result = subprocess.run(
                ["git", "clone", cleaned_url, target_dir],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # 删除 .git 目录
                git_dir = os.path.join(target_dir, ".git")
                if os.path.exists(git_dir):
                    shutil.rmtree(git_dir)
                return True
            else:
                print(f"Error cloning repository: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error during cloning: {str(e)}")
            return False
    
    def create_readme(self, target_dir: str, repo_info: Dict[str, str]):
        """
        在目标目录创建 README.md 文件
        
        Args:
            target_dir (str): 目标目录
            repo_info (Dict[str, str]): 仓库信息
        """
        readme_content = f"""# {repo_info['title']}

This repository was downloaded as part of the reference implementation analysis.

- Reference Number: {repo_info['ref_num']}
- Original Repository: {repo_info['url']}
- Paper Title: {repo_info['title']}

Note: This is a copy of the original repository with the .git directory removed.
"""
        try:
            with open(os.path.join(target_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(readme_content)
        except Exception as e:
            print(f"Error creating README: {str(e)}")
    
    def process_file(self, file_path: str) -> Dict[str, List[str]]:
        """
        处理包含 GitHub 仓库信息的文件
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            Dict[str, List[str]]: 处理结果，包含成功和失败的仓库列表
        """
        results = {
            "success": [],
            "failed": []
        }
        
        try:
            # 读取文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 解析仓库信息
            repos = self.parse_github_urls(content)
            
            # 创建 github_codes 目录
            github_codes_dir = os.path.join(self.base_dir, "github_codes")
            os.makedirs(github_codes_dir, exist_ok=True)
            
            # 克隆每个仓库
            for repo in repos:
                target_dir = os.path.join(github_codes_dir, f"ref_{repo['ref_num']}")
                
                if self.clone_repository(repo['url'], target_dir):
                    self.create_readme(target_dir, repo)
                    results["success"].append(repo['url'])
                else:
                    results["failed"].append(repo['url'])
            
            return results
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return results

def main():
    """
    主函数，用于测试
    """
    # 示例用法
    paper_dir = "./agent_folders/papers/paper_1"
    downloader = GitHubDownloader(paper_dir)
    
    search_result_path = os.path.join(paper_dir, "github_search.txt")
    if os.path.exists(search_result_path):
        results = downloader.process_file(search_result_path)
        print("\nDownload Results:")
        print(f"Successfully downloaded: {len(results['success'])} repositories")
        print(f"Failed to download: {len(results['failed'])} repositories")
        
        if results['success']:
            print("\nSuccessfully downloaded repositories:")
            for url in results['success']:
                print(f"- {url}")
        
        if results['failed']:
            print("\nFailed to download repositories:")
            for url in results['failed']:
                print(f"- {url}")
    else:
        print(f"Error: File not found at {search_result_path}")

if __name__ == "__main__":
    main() 