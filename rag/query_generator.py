"""
Dynamic RAG Query Generator

Simplified intelligent query generation using LLM to create single optimal queries
for specific agent types based on context and requirements.
"""

import asyncio
from typing import List, Optional, Callable
from dataclasses import dataclass

# Import prompts for query generation
from prompts.code_prompts import (
    RAG_RESEARCH_ANALYSIS_PROMPT,
    RAG_CONCEPT_ANALYSIS_PROMPT,
    RAG_ALGORITHM_EXTRACTION_PROMPT,
    RAG_REFERENCE_ANALYSIS_PROMPT,
    RAG_CODE_PLANNING_PROMPT,
)


@dataclass
class QueryGenerationContext:
    """Context information for dynamic query generation"""

    agent_type: str
    user_requirements: Optional[str] = None
    analysis_depth: str = "focused"  # focused, comprehensive


class QueryGenerator:
    """
    Simplified dynamic query generator that creates single optimal queries
    """

    def __init__(self, llm_function: Optional[Callable] = None):
        """
        Initialize dynamic query generator

        Args:
            llm_function: Function to call LLM for query generation
        """
        self.llm_function = llm_function
        self.prompt_map = {
            "research": RAG_RESEARCH_ANALYSIS_PROMPT,
            "concept": RAG_CONCEPT_ANALYSIS_PROMPT,
            "algorithm": RAG_ALGORITHM_EXTRACTION_PROMPT,
            "reference": RAG_REFERENCE_ANALYSIS_PROMPT,
            "planning": RAG_CODE_PLANNING_PROMPT,
        }

    async def generate_query(self, context: QueryGenerationContext) -> List[str]:
        """
        Generate a single optimal query based on context using LLM

        Args:
            context: Query generation context

        Returns:
            str: Single optimal query
        """
        if not self.llm_function:
            print("⚠️ No LLM function provided, falling back to static query")
            return self._get_fallback_query(
                context.agent_type, context.user_requirements
            )

        try:
            # Build context-aware prompt
            dynamic_prompt = self._build_dynamic_prompt(context)

            # Generate query using LLM
            response = await self._call_llm_async(dynamic_prompt)

            # Parse and validate generated query
            query = self._parse_single_query_response(response)

            if query:
                print(f"✅ Generated optimal query for {context.agent_type}")
                return query
            else:
                print("⚠️ LLM generated empty query, falling back to static")
                return self._get_fallback_query(
                    context.agent_type, context.user_requirements
                )

        except Exception as e:
            print(f"❌ Dynamic query generation failed: {e}")
            print("🔄 Falling back to static query")
            return self._get_fallback_query(
                context.agent_type, context.user_requirements
            )

    def _build_dynamic_prompt(self, context: QueryGenerationContext) -> str:
        """
        Build a dynamic prompt based on context information

        Args:
            context: Query generation context

        Returns:
            str: Complete prompt for LLM query generation
        """
        base_prompt = self.prompt_map.get(context.agent_type, "")

        # Add context information to the prompt
        context_info = []

        if context.user_requirements:
            context_info.append(f"User Requirements: {context.user_requirements}")

        # Depth-specific instructions
        if context.analysis_depth == "focused":
            instruction = "Generate 1 highly focused, optimal query for the most important information."
        else:
            instruction = (
                "Generate 1 comprehensive query that captures all essential aspects."
            )

        # Combine everything into a complete prompt
        full_prompt = f"""{base_prompt}

            Context Information:
            {chr(10).join(context_info) if context_info else "No specific context provided."}

            Instructions: {instruction}

            Based on the above context, generate the single most effective query that will extract the most relevant information for {context.agent_type} analysis. Return only the query text as a single string, not as JSON or list.

            Query:"""

        return full_prompt

    async def _call_llm_async(self, prompt: str) -> str:
        """
        Call LLM function asynchronously

        Args:
            prompt: Prompt for LLM

        Returns:
            str: LLM response
        """
        if asyncio.iscoroutinefunction(self.llm_function):
            return await self.llm_function(prompt)
        else:
            # If it's a sync function, run it in a thread
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.llm_function, prompt)

    def _parse_single_query_response(self, response: str) -> Optional[str]:
        """
        Parse LLM response to extract single query

        Args:
            response: Raw LLM response

        Returns:
            Optional[str]: Extracted query or None if failed
        """
        try:
            # Clean up the response
            query = response.strip()

            # Remove common LLM artifacts
            if query.startswith("Query:"):
                query = query[6:].strip()

            # Remove quotes if present
            if (query.startswith('"') and query.endswith('"')) or (
                query.startswith("'") and query.endswith("'")
            ):
                query = query[1:-1]

            # Take first line if multi-line response
            if "\n" in query:
                query = query.split("\n")[0].strip()

            # Ensure it's a reasonable query length
            if len(query) > 20 and len(query) < 500:
                return query
            else:
                print(f"⚠️ Generated query has unusual length: {len(query)}")
                return None

        except Exception as e:
            print(f"❌ Failed to parse query response: {e}")
            return None

    def _get_fallback_query(
        self, agent_type: str, requirements: Optional[str] = None
    ) -> str:
        """
        Get fallback query when dynamic generation fails

        Args:
            agent_type: Type of agent
            requirements: User requirements

        Returns:
            str: Fallback query
        """
        basic_queries = {
            "research": "What is the main contribution and innovation of this research paper?",
            "concept": "What is the overall system architecture and framework design?",
            "algorithm": "What are all the algorithms presented in this paper with their pseudocode?",
            "reference": "What papers are cited in the References or Bibliography section?",
            "planning": "What is the complete technical implementation roadmap for this paper?",
        }

        query = basic_queries.get(agent_type, "Analyze this research paper.")
        if requirements:
            query += f" Focus on: {requirements}"
        return query

    def set_llm_function(self, llm_function: Callable):
        """
        Set or update the LLM function for query generation

        Args:
            llm_function: Function to call LLM
        """
        self.llm_function = llm_function
