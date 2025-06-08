# github_reader.py
import httpx
import base64
import json
from typing import Dict, List, Any, Optional

class GitHubReader:
    """GitHub 代码读取器 - 用于从 GitHub 仓库读取代码文件"""
    
    def __init__(self, github_token: Optional[str] = None):
        """
        初始化 GitHub 读取器
        :param github_token: GitHub 访问令牌（可选，但建议提供以避免 API 限制）
        """
        self.github_token = github_token
        self.base_url = "https://api.github.com"
        
    def _get_headers(self) -> Dict[str, str]:
        """
        获取 HTTP 请求头
        :return: 包含认证信息的请求头
        """
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "CodeAgent/1.0"
        }
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        return headers
    
    async def read_file(self, owner: str, repo: str, path: str, branch: str = "main") -> str:
        """
        读取 GitHub 仓库中的单个文件
        :param owner: 仓库所有者
        :param repo: 仓库名称
        :param path: 文件路径
        :param branch: 分支名称
        :return: 文件内容
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": branch}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._get_headers(), params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("type") == "file":
                    # GitHub API 返回的内容是 base64 编码的，需要解码
                    content = base64.b64decode(data["content"]).decode("utf-8")
                    return content
                else:
                    raise ValueError(f"{path} 不是一个文件")
            else:
                raise Exception(f"读取文件失败: {response.status_code} - {response.text}")
    
    async def list_files(self, owner: str, repo: str, path: str = "", branch: str = "main") -> List[Dict[str, Any]]:
        """
        列出仓库目录中的文件
        :param owner: 仓库所有者
        :param repo: 仓库名称
        :param path: 目录路径
        :param branch: 分支名称
        :return: 文件列表
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": branch}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._get_headers(), params=params)
            
            if response.status_code == 200:
                data = response.json()
                return [
                    {
                        "name": item["name"],
                        "type": item["type"],  # "file" 或 "dir"
                        "path": item["path"],
                        "size": item.get("size", 0)
                    }
                    for item in data
                ]
            else:
                raise Exception(f"列出文件失败: {response.status_code} - {response.text}")
    
    async def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        获取仓库基本信息
        :param owner: 仓库所有者
        :param repo: 仓库名称
        :return: 仓库信息
        """
        url = f"{self.base_url}/repos/{owner}/{repo}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._get_headers())
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "name": data["name"],
                    "description": data.get("description", ""),
                    "language": data.get("language", ""),
                    "stars": data["stargazers_count"],
                    "default_branch": data["default_branch"]
                }
            else:
                raise Exception(f"获取仓库信息失败: {response.status_code} - {response.text}")