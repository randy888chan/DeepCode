"""
RAG Manager for DeepCode

Simplified interface for RAG operations, focusing on single query generation for agents.
"""

import os
from typing import Optional

# RAGAnything imports
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️ RAGAnything not available. RAG functionality will be disabled.")

from .config import RAGConfigManager

# Query generator import
from .query_generator import QueryGenerator, QueryGenerationContext


class RAGManager:
    """
    Simplified RAG Manager for DeepCode workflow

    Provides single query generation for different agent types.
    """

    def __init__(
        self,
        config_path: str = "mcp_agent.config.yaml",
        secrets_path: str = "mcp_agent.secrets.yaml",
    ):
        """
        Initialize RAG Manager

        Args:
            config_path: Path to main configuration file
            secrets_path: Path to secrets configuration file
        """
        self.config_manager = RAGConfigManager(config_path, secrets_path)
        self.rag_instance: Optional[RAGAnything] = None
        self.is_initialized = False

        # Initialize query generator
        self.query_generator: Optional[QueryGenerator] = None

        # Check if RAG is available and enabled
        self.rag_enabled = RAG_AVAILABLE and self.config_manager.is_rag_enabled()

        if not RAG_AVAILABLE:
            print("📚 RAGAnything not installed. Install with: pip install raganything")
        elif not self.rag_enabled:
            print("🔶 RAG is disabled in configuration")

        # Initialize query generator if available
        if self.rag_enabled:
            self._initialize_query_generator()

    def _initialize_query_generator(self):
        """Initialize the query generator with LLM function"""
        try:
            # Create a simple LLM function for query generation
            async def query_generation_llm(prompt: str) -> str:
                # Use the same LLM function as RAG instance
                return await self.rag_instance.llm_model_func(prompt)

            self.query_generator = QueryGenerator(query_generation_llm)
            print("🤖 Query generator initialized")

        except Exception as e:
            print(f"⚠️ Failed to initialize query generator: {e}")
            self.query_generator = None

    async def initialize_rag(self, paper_path: str) -> bool:
        """
        Initialize RAG system and index the research paper

        Args:
            paper_path: Path to the research paper file

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self.rag_enabled:
            print("🔶 RAG not enabled, skipping initialization")
            return False

        try:
            print("🧠 Initializing RAG system for paper analysis...")

            # Validate paper file exists
            if not os.path.exists(paper_path):
                print(f"❌ Paper file not found: {paper_path}")
                return False

            # Get configurations
            rag_config = self.config_manager.get_raganything_config()
            llm_config = self.config_manager.get_llm_config()
            embedding_config = self.config_manager.get_embedding_config()
            vision_config = self.config_manager.get_vision_config()

            # Create RAGAnything configuration
            raganything_config = RAGAnythingConfig(
                working_dir=rag_config["working_dir"],
                parser=rag_config["parser"],
                parse_method=rag_config["parse_method"],
                enable_image_processing=rag_config["enable_image_processing"],
                enable_table_processing=rag_config["enable_table_processing"],
                enable_equation_processing=rag_config["enable_equation_processing"],
            )

            # Define LLM model function
            def llm_model_func(
                prompt, system_prompt=None, history_messages=[], **kwargs
            ):
                return openai_complete_if_cache(
                    llm_config["model"],
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=llm_config["api_key"],
                    base_url=llm_config["base_url"],
                    **kwargs,
                )

            # Define vision model function for image processing
            def vision_model_func(
                prompt,
                system_prompt=None,
                history_messages=[],
                image_data=None,
                **kwargs,
            ):
                if image_data:
                    return openai_complete_if_cache(
                        vision_config["model"],
                        "",
                        system_prompt=None,
                        history_messages=[],
                        messages=[
                            {"role": "system", "content": system_prompt}
                            if system_prompt
                            else None,
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data}"
                                        },
                                    },
                                ],
                            }
                            if image_data
                            else {"role": "user", "content": prompt},
                        ],
                        api_key=vision_config["api_key"],
                        base_url=vision_config["base_url"],
                        **kwargs,
                    )
                else:
                    return llm_model_func(
                        prompt, system_prompt, history_messages, **kwargs
                    )

            # Define embedding function
            embedding_func = EmbeddingFunc(
                embedding_dim=embedding_config["embedding_dim"],
                max_token_size=embedding_config["max_token_size"],
                func=lambda texts: openai_embed(
                    texts,
                    model=embedding_config["model"],
                    api_key=embedding_config["api_key"],
                    base_url=embedding_config["base_url"],
                ),
            )

            # Initialize RAGAnything
            self.rag_instance = RAGAnything(
                config=raganything_config,
                llm_model_func=llm_model_func,
                vision_model_func=vision_model_func,
                embedding_func=embedding_func,
            )

            # Process document
            output_dir = os.path.join(rag_config["working_dir"], "parsed")
            os.makedirs(output_dir, exist_ok=True)

            print(f"📄 Processing document: {paper_path}")
            await self.rag_instance.process_document_complete(
                file_path=paper_path,
                output_dir=output_dir,
                parse_method=rag_config["parse_method"],
            )

            self.is_initialized = True

            print("✅ RAG system initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            print("🔄 Falling back to traditional paper processing")
            self.rag_enabled = False
            return False

    async def generate_query_for_agent(
        self, agent_type: str, requirements: str = ""
    ) -> str:
        """
        Generate a single optimized query for the specified agent type

        Args:
            agent_type: Type of agent (research, concept, algorithm, reference, planning)
            requirements: Optional specific requirements for the query

        Returns:
            str: Single optimized query for the agent
        """
        # Use dynamic query generation to create a single optimal query
        context = QueryGenerationContext(
            agent_type=agent_type,
            user_requirements=requirements,
            analysis_depth="focused",  # Generate fewer, more focused queries
        )

        query = await self.query_generator.generate_query(context)

        print(f"🎯 Generated optimized query for {agent_type}")
        return query

    async def query_paper(self, query: str) -> Optional[str]:
        """
        Query the indexed research paper with a single query

        Args:
            query: Query text

        Returns:
            Optional[str]: Query result or None if failed
        """
        if not self.is_initialized or not self.rag_instance:
            print("⚠️ RAG system not initialized")
            return None

        try:
            print(f"🔍 Querying RAG system: {query[:100]}...")

            result = await self.rag_instance.aquery(query, mode="mix")

            if result:
                print("✅ RAG query completed successfully")
                return result
            else:
                print("⚠️ RAG query returned empty result")
                return None

        except Exception as e:
            print(f"❌ RAG query failed: {e}")
            return None

    async def analyze_with_agent(
        self, agent_type: str, requirements: str = ""
    ) -> Optional[str]:
        """
        Complete analysis workflow: generate query + execute query

        Args:
            agent_type: Type of agent analysis needed
            requirements: Specific requirements for the analysis

        Returns:
            Optional[str]: Analysis result
        """
        if not self.is_initialized:
            print("⚠️ RAG system not initialized")
            return None

        # Generate optimized query for the agent
        query = await self.generate_query_for_agent(agent_type, requirements)

        # Execute the query
        return await self.query_paper(query)

    def is_rag_available(self) -> bool:
        """
        Check if RAG functionality is available and enabled

        Returns:
            bool: True if RAG can be used
        """
        return self.rag_enabled and self.is_initialized
