"""
RAG Configuration Management

Handles reading and managing RAG-related configurations from
mcp_agent.config.yaml and mcp_agent.secrets.yaml files.
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """RAG configuration data class"""

    # Core RAG settings
    enabled: bool = True
    parser: str = "mineru"  # mineru or docling
    parse_method: str = "auto"  # auto, ocr, or txt
    enable_image_processing: bool = True
    enable_table_processing: bool = True
    enable_equation_processing: bool = True
    working_dir: str = "./rag_storage"

    # Embedding settings
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072
    max_token_size: int = 8192

    # LLM settings
    llm_model: str = "gpt-4o-mini"

    # Vision model settings
    vision_model: str = "gpt-4o"

    # API credentials
    openai_api_key: str = ""
    openai_base_url: str = ""


class RAGConfigManager:
    """RAG Configuration Manager"""

    def __init__(
        self,
        config_path: str = "mcp_agent.config.yaml",
        secrets_path: str = "mcp_agent.secrets.yaml",
    ):
        """
        Initialize RAG configuration manager

        Args:
            config_path: Path to the main configuration file
            secrets_path: Path to the secrets configuration file
        """
        self.config_path = config_path
        self.secrets_path = secrets_path
        self._config: Optional[RAGConfig] = None

    def load_config(self) -> RAGConfig:
        """
        Load RAG configuration from YAML files

        Returns:
            RAGConfig: Complete RAG configuration object
        """
        if self._config is not None:
            return self._config

        # Load main configuration
        main_config = self._load_yaml_file(self.config_path)
        rag_config = main_config.get("rag", {})

        # Load secrets configuration
        secrets_config = self._load_yaml_file(self.secrets_path)
        rag_secrets = secrets_config.get("rag", {})

        # Create RAGConfig with merged settings
        config_dict = {
            # Core settings
            "enabled": rag_config.get("enabled", True),
            "parser": rag_config.get("parser", "mineru"),
            "parse_method": rag_config.get("parse_method", "auto"),
            "enable_image_processing": rag_config.get("enable_image_processing", True),
            "enable_table_processing": rag_config.get("enable_table_processing", True),
            "enable_equation_processing": rag_config.get(
                "enable_equation_processing", True
            ),
            "working_dir": rag_config.get("working_dir", "./rag_storage"),
            # Embedding settings
            "embedding_model": rag_config.get("embedding", {}).get(
                "model", "text-embedding-3-large"
            ),
            "embedding_dimension": rag_config.get("embedding", {}).get(
                "dimension", 3072
            ),
            "max_token_size": rag_config.get("embedding", {}).get(
                "max_token_size", 8192
            ),
            # LLM settings
            "llm_model": rag_config.get("llm", {}).get("model", "gpt-4o-mini"),
            # Vision model settings
            "vision_model": rag_config.get("vision", {}).get("model", "gpt-4o"),
            # API credentials from secrets
            "openai_api_key": rag_secrets.get("openai_api_key", ""),
            "openai_base_url": rag_secrets.get("openai_base_url", ""),
        }

        self._config = RAGConfig(**config_dict)
        return self._config

    def _load_yaml_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load YAML configuration file

        Args:
            file_path: Path to the YAML file

        Returns:
            Dict: Parsed YAML content
        """
        try:
            if not os.path.exists(file_path):
                print(f"⚠️ Configuration file {file_path} not found, using defaults")
                return {}

            with open(file_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                return content if content is not None else {}

        except Exception as e:
            print(f"❌ Error loading configuration file {file_path}: {e}")
            return {}

    def is_rag_enabled(self) -> bool:
        """
        Check if RAG is enabled and properly configured

        Returns:
            bool: True if RAG is enabled and has required configuration
        """
        config = self.load_config()

        if not config.enabled:
            return False

        # Check if required API key is available
        if not config.openai_api_key:
            print("⚠️ RAG is enabled but OpenAI API key is not configured")
            return False

        return True

    def get_raganything_config(self) -> Dict[str, Any]:
        """
        Get configuration dict compatible with RAGAnything

        Returns:
            Dict: Configuration for RAGAnything initialization
        """
        config = self.load_config()

        return {
            "working_dir": config.working_dir,
            "parser": config.parser,
            "parse_method": config.parse_method,
            "enable_image_processing": config.enable_image_processing,
            "enable_table_processing": config.enable_table_processing,
            "enable_equation_processing": config.enable_equation_processing,
        }

    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration for RAGAnything

        Returns:
            Dict: LLM configuration parameters
        """
        config = self.load_config()

        return {
            "model": config.llm_model,
            "api_key": config.openai_api_key,
            "base_url": config.openai_base_url if config.openai_base_url else None,
        }

    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get embedding configuration for RAGAnything

        Returns:
            Dict: Embedding configuration parameters
        """
        config = self.load_config()

        return {
            "model": config.embedding_model,
            "embedding_dim": config.embedding_dimension,
            "max_token_size": config.max_token_size,
            "api_key": config.openai_api_key,
            "base_url": config.openai_base_url if config.openai_base_url else None,
        }

    def get_vision_config(self) -> Dict[str, Any]:
        """
        Get vision model configuration for RAGAnything

        Returns:
            Dict: Vision model configuration parameters
        """
        config = self.load_config()

        return {
            "model": config.vision_model,
            "api_key": config.openai_api_key,
            "base_url": config.openai_base_url if config.openai_base_url else None,
        }
