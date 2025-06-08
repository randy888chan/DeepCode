# code_generator.py
import openai
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class CodeRequest:
    """代码生成请求的数据结构"""
    description: str      # 代码描述和需求
    language: str        # 编程语言
    context: str = ""    # 上下文信息（从 GitHub 读取的代码）
    style_guide: str = ""  # 代码风格指南

class CodeGenerator:
    """代码生成器 - 使用 OpenAI GPT 生成代码"""
    
    def __init__(self, api_key: str):
        """
        初始化代码生成器
        :param api_key: OpenAI API 密钥
        """
        self.client = openai.OpenAI(api_key=api_key)
    
    async def generate_code(self, request: CodeRequest) -> str:
        """
        根据请求生成代码
        :param request: 代码生成请求
        :return: 生成的代码
        """
        # 构建提示词
        prompt = self._build_prompt(request)
        
        try:
            # 调用 OpenAI API 生成代码
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # 使用 GPT-3.5，成本更低
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一个专业的软件开发工程师，擅长编写高质量、可维护的代码。请根据用户需求生成代码。"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # 较低的温度确保代码更加确定性
                max_tokens=2000   # 限制输出长度
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            raise Exception(f"代码生成失败: {str(e)}")
    
    def _build_prompt(self, request: CodeRequest) -> str:
        """
        构建发送给 GPT 的提示词
        :param request: 代码生成请求
        :return: 完整的提示词
        """
        prompt_parts = [
            f"请用 {request.language} 编写代码，满足以下需求：",
            f"需求描述：{request.description}",
        ]
        
        # 如果有上下文信息（参考代码），添加到提示词中
        if request.context:
            prompt_parts.extend([
                "\n参考代码示例：",
                "```",
                request.context,
                "```",
                "\n请参考上述代码的风格和模式来编写新代码。"
            ])
        
        # 如果有风格指南，添加到提示词中
        if request.style_guide:
            prompt_parts.extend([
                f"\n代码风格要求：{request.style_guide}"
            ])
        
        prompt_parts.extend([
            "\n请确保代码：",
            "1. 具有良好的可读性和注释",
            "2. 遵循最佳实践",
            "3. 包含必要的错误处理",
            "4. 结构清晰，易于维护"
        ])
        
        return "\n".join(prompt_parts)
    
    async def refine_code(self, original_code: str, feedback: str) -> str:
        """
        根据反馈改进代码
        :param original_code: 原始代码
        :param feedback: 改进建议
        :return: 改进后的代码
        """
        prompt = f"""
请根据以下反馈改进代码：

原始代码："""